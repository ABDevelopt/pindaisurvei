"""
seruti_pipeline/channel_extractor.py
====================================
Ekstraksi kanal monokrom optimal dan normalisasi pencahayaan (flat-fielding).
Mendeteksi apakah halaman menggunakan kertas biru (sampul luar Seruti) atau putih,
serta mengisolasi goresan pensil dan kisi tabel menjadi citra monokrom bersih.
Terintegrasi dengan Advanced Filter untuk VLM dan OCR, serta Magic Filter CamScanner Berkontras Tinggi.
"""
from typing import Tuple, Optional, Union, Dict, Any
import cv2
import numpy as np

from .advanced_filter import enhance_for_vlm_and_ocr


def is_blue_paper(img: np.ndarray) -> Tuple[bool, float]:
    """
    Mendeteksi apakah kertas berwarna biru (seperti sampul kuesioner BPS Hal 1, 7, 8).
    Memeriksa selisih rata-rata kanal Blue terhadap Red (B - R).
    Mengembalikan (is_blue: bool, blue_diff: float).
    """
    if len(img.shape) != 3 or img.shape[2] < 3:
        return False, 0.0

    H, W = img.shape[:2]
    step = max(1, H // 200)
    b_mean = float(img[::step, ::step, 0].mean())
    r_mean = float(img[::step, ::step, 2].mean())
    blue_diff = b_mean - r_mean
    is_blue = blue_diff > 12.0
    return is_blue, round(blue_diff, 2)


def flatten_illumination_flat_fielding(
    img: np.ndarray,
    ksize: Optional[int] = None
) -> np.ndarray:
    """
    Menghilangkan gradasi bayangan dan iluminasi tidak merata (flat-fielding).
    Mengestimasi permukaan latar belakang kertas melalui operasi morfologi penutupan/dilasi skala besar,
    sehingga goresan pensil dan huruf tidak tergerus oleh pembagian latar belakang.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    H, W = gray.shape[:2]
    scale = max(1, min(H, W) // 250)
    small_w = max(1, W // scale)
    small_h = max(1, H // scale)
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Kernel dilasi yang cukup besar untuk melompati seluruh teks dan kisi tabel
    k_small = max(9, int(min(small_h, small_w) * 0.04) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_small, k_small))
    bg_small = cv2.dilate(small, kernel)
    k_blur = 21 if (21 < small_h and 21 < small_w) else 5
    bg_small = cv2.medianBlur(bg_small, k_blur)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=max(2.0, k_small * 0.8))

    bg = cv2.resize(bg_small, (W, H), interpolation=cv2.INTER_LINEAR)
    flat = cv2.divide(gray.astype(np.float32), np.maximum(bg.astype(np.float32), 1.0), scale=255.0)
    return np.clip(flat, 0, 255).astype(np.uint8)


def apply_scurve_lut(img: np.ndarray) -> np.ndarray:
    """
    Menerapkan transformasi kontras dinamis S-Curve via Lookup Table (LUT).
    Mempertegas goresan pensil grafit 2B dan batas kisi, sekaligus mengangkat
    abu-abu terang latar belakang mendekati 255 putih.
    """
    x = np.arange(256, dtype=np.float32)
    xn = x / 255.0
    # Sigmoid kurva S bergradien tegas untuk menarik goresan pensil ke arah gelap
    k = 8.5
    x0 = 0.50
    s_curve = 1.0 / (1.0 + np.exp(-k * (xn - x0)))
    s_min, s_max = s_curve[0], s_curve[255]
    lut = ((s_curve - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
    return cv2.LUT(img, lut)


def apply_camscanner_magic_filter(
    img: np.ndarray,
    unsharp_strength: float = 0.50,
    white_cutoff: int = 222,
    gamma: float = 2.0,
    bp_offset: float = 22.0
) -> np.ndarray:
    """
    Filter CamScanner Magic Mutakhir Berkontras Tinggi & Huruf Pekat Tebal:
    1. Dynamic Black-Point Clamping: Mengidentifikasi intensitas tinta/pensil tergelap,
       lalu menariknya langsung ke level hitam pekat (0 - 30).
    2. Power-Law Gamma Expansion: Mempertebal tulisan tangan dan angka tipis agar
       terbaca jelas dan tebal tanpa terfragmentasi.
    3. High-Frequency Unsharp Masking: Menajamkan batas tepi karakter dan kisi tabel.
    4. White Point Snapping: Menghapus seluruh bayangan kabur dan serat kertas menjadi 255 putih salju.
    """
    H, W = img.shape[:2]

    # Blank page protection (sampul kosong / non-teks)
    std_val = float(np.std(img))
    dark_ratio = float(np.mean(img < 180))
    if dark_ratio < 0.0002 and std_val < 4.5:
        return np.full((H, W), 255, dtype=np.uint8)

    # 1. Estimasi dinamis Black Point (tinta pensil/cetak)
    p1 = float(np.percentile(img, 0.8))
    p95 = float(np.percentile(img, 95.0))
    if p1 < 160.0:
        bp = max(15.0, min(p1 + bp_offset, 140.0))
        wp = min(240.0, max(p95 - 4.0, 205.0))
        if wp > bp:
            norm = np.clip((img.astype(np.float32) - bp) / (wp - bp), 0.0, 1.0)
            stretched = np.clip(np.power(norm, gamma) * 255.0, 0, 255).astype(np.uint8)
        else:
            stretched = img.copy()
    else:
        stretched = img.copy()

    # 3. Soft unsharp masking untuk ketajaman goresan (Masked Sharpening)
    if unsharp_strength > 0:
        blur = cv2.GaussianBlur(stretched, (0, 0), sigmaX=1.2)
        sharp = cv2.addWeighted(stretched, 1.0 + unsharp_strength, blur, -unsharp_strength, 0)
        # Terapkan penajaman HANYA pada goresan teks/kisi, latar kertas selalu 255 murni bebas derau
        stroke_mask = (stretched < white_cutoff)
        out = np.where(stroke_mask, np.clip(sharp, 0, 255), 255).astype(np.uint8)
    else:
        out = stretched.copy()
        out[out >= white_cutoff] = 255
    return out


def clean_scanner_margins(img: np.ndarray, margin_px: int = 65) -> np.ndarray:
    """
    Membersihkan derau margin scanner (roller, tepi bayangan kaca pemindai)
    dengan menetapkan perimeter sebesar margin_px menjadi 255 putih murni.
    """
    out = img.copy()
    H, W = out.shape[:2]
    m = min(margin_px, min(H, W) // 10)
    if m <= 0:
        return out
    out[:m, :] = 255
    out[-m:, :] = 255
    out[:, :m] = 255
    out[:, -m:] = 255
    return out


def extract_optimal_channel(
    page_bgr: np.ndarray,
    page_id: Optional[str] = None,
    side: Optional[str] = None,
    return_vlm: bool = False
) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool, np.ndarray]]:
    """
    Mengekstrak kanal monokrom optimal dari citra halaman A4 BGR:
    1. Mendeteksi warna kertas biru vs putih:
       Jika kertas biru (sampul), kanal merah (R) memberikan kontras pensil dan cetakan tertinggi.
    2. Menjalankan Retinex flat-fielding, de-shading lipatan tengah, dan adaptive stroke booster.
    3. Jika return_vlm=True, mengembalikan (mono_image, is_blue, vlm_image).
       Jika return_vlm=False, mengembalikan (mono_image, is_blue).
    """
    # GUARDRAIL 1: Kertas biru HANYA berlaku untuk sampul (hal_1 dan hal_8).
    # Halaman dalam (hal_2 s.d. hal_7) mutlak kertas putih kuesioner dan dilarang di-bleaching biru.
    if page_id is not None and page_id not in ["hal_1", "hal_8"]:
        is_blue = False
    else:
        is_blue, blue_diff = is_blue_paper(page_bgr)
        if page_id in ["hal_1", "hal_8"] and blue_diff > 5.0:
            is_blue = True

    # Lipatan tengah berada di sisi kanan untuk left_a4, dan sisi kiri untuk right_a4
    fold_side = "right" if side == "left" else ("left" if side == "right" else None)

    # Hanya hal_8 (atau tanpa page_id) yang diizinkan di-blank jika terdeteksi kosong
    allow_blank = (page_id == "hal_8" or page_id is None)
    vlm_bgr, ocr_mono, diag = enhance_for_vlm_and_ocr(
        page_bgr, is_blue=is_blue, fold_side=fold_side, allow_blank=allow_blank
    )

    if return_vlm:
        return ocr_mono, is_blue, vlm_bgr
    return ocr_mono, is_blue


def pad_to_canonical_a4(
    img: np.ndarray,
    target_h: int = 3507,
    target_w: int = 2480
) -> np.ndarray:
    """
    Memusatkan citra halaman A4 ke dalam kanvas kanonikal standar 300 DPI
    (Portrait: 3507 x 2480, Landscape: 2480 x 3507).
    Latar penambahan (padding) selalu 255 putih murni.
    """
    H, W = img.shape[:2]
    is_landscape = W > H
    th = target_w if is_landscape else target_h
    tw = target_h if is_landscape else target_w

    if H == th and W == tw:
        return img

    if len(img.shape) == 3:
        canvas = np.full((th, tw, img.shape[2]), 255, dtype=img.dtype)
    else:
        canvas = np.full((th, tw), 255, dtype=img.dtype)

    src_h = min(H, th)
    src_w = min(W, tw)
    src_top = max(0, (H - th) // 2)
    src_left = max(0, (W - tw) // 2)

    pad_top = max(0, (th - H) // 2)
    pad_left = max(0, (tw - W) // 2)

    canvas[pad_top:pad_top + src_h, pad_left:pad_left + src_w] = img[
        src_top:src_top + src_h, src_left:src_left + src_w
    ]
    return canvas
