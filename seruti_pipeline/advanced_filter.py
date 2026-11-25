"""
seruti_pipeline/advanced_filter.py
==================================
Modul Penyaringan Mutakhir Citra Kuesioner Statistik BPS:
Dioptimalkan secara ganda untuk Vision-Language Models (VLM) dan Traditional OCR/ICR.

Karakteristik & Keunggulan Teknik:
1. CIELAB Perceptual Retinex Flat-Fielding:
   - Menghilangkan gradasi bayangan scanner, kertas menguning, dan vignetting.
   - Latar belakang kertas menjadi putih bersih seragam (~250-255).
   - Menjaga saturasi warna asli (tinta cetak hitam, garis kisi biru, dan goresan pensil abu-abu).
2. Adaptive Gutter Crease De-shading:
   - Mendeteksi dan mengeliminasi bayangan lengkungan jilidan/lipatan buku (spine shadow)
     pada sisi jahitan tengah tanpa batas potongan kasar.
3. Anti-Aliased Stroke Enhancement (Anti-Fragmentation):
   - TIDAK menggunakan binarisasi 1-bit kasar (Otsu/Sauvola keras) yang merusak gradien tepi
     dan memutus goresan pensil grafit 2B.
   - Menggunakan soft unsharp-masking adaptif berbasis masker densitas sehingga angka tulisan
     tangan tetap utuh dan mudah dibaca oleh model ViT (Vision Transformer) dan CNN OCR.
4. Blue Substrate Spectral Whitening:
   - Menetralkan warna kertas sampul luar (Halaman 1 dan 8) pada citra monokrom OCR via
     proyeksi spektral kanal merah (R), sekaligus menjaga warna natural pada citra VLM.
5. Punctuation-Safe Despeckling:
   - Membersihkan debu kaca scanner dengan analisis komponen terhubung tanpa menghapus
     titik tanda baca sah (., :, -, tanda koma).
"""
from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


def deshade_gutter_edge(img_bgr: np.ndarray, fold_side: str = "right") -> np.ndarray:
    """
    Menghilangkan bayangan lengkungan lipatan tengah (gutter crease)
    pada tepi lipatan lembar A3 (kanan untuk left_a4, kiri untuk right_a4).
    """
    H, W = img_bgr.shape[:2]
    margin = min(180, int(W * 0.08))

    if fold_side == "right":
        strip = img_bgr[:, -margin:]
        # Estimasi profil intensitas kolom rata-rata
        gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        col_means = gray_strip.mean(axis=0)
        baseline = np.percentile(col_means, 85)

        # Hitung faktor penguatan (gain) invers terhadap bayangan
        diff = np.maximum(0.0, baseline - col_means)
        if diff.max() > 10.0:
            gain_curve = 1.0 + (diff / (baseline + 1e-5)) * 0.8
            # Haluskan kurva gain
            gain_curve = cv2.GaussianBlur(gain_curve.reshape(1, -1), (15, 1), 0).ravel()
            gain_map = np.tile(gain_curve, (H, 1))[:, :, np.newaxis]
            enhanced_strip = np.clip(strip.astype(np.float32) * gain_map, 0, 255).astype(np.uint8)
            img_bgr[:, -margin:] = enhanced_strip

    elif fold_side == "left":
        strip = img_bgr[:, :margin]
        gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        col_means = gray_strip.mean(axis=0)
        baseline = np.percentile(col_means, 85)

        diff = np.maximum(0.0, baseline - col_means)
        if diff.max() > 10.0:
            gain_curve = 1.0 + (diff / (baseline + 1e-5)) * 0.8
            gain_curve = cv2.GaussianBlur(gain_curve.reshape(1, -1), (15, 1), 0).ravel()
            gain_map = np.tile(gain_curve, (H, 1))[:, :, np.newaxis]
            enhanced_strip = np.clip(strip.astype(np.float32) * gain_map, 0, 255).astype(np.uint8)
            img_bgr[:, :margin] = enhanced_strip

    return img_bgr


def enhance_for_vlm_and_ocr(
    img_bgr: np.ndarray,
    is_blue: bool = False,
    fold_side: Optional[str] = None,
    allow_blank: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Memproses citra pindaian kuesioner menjadi dua format optimal:
    1. vlm_bgr: Citra 3-kanal (BGR/RGB) dengan latar putih murni dan warna asli terjaga untuk VLM.
    2. ocr_gray: Citra monokrom (8-bit grayscale) anti-aliased kontras tinggi untuk mesin OCR/ICR.
    
    Mengembalikan (vlm_bgr, ocr_gray, diagnostics).
    """
    H, W = img_bgr.shape[:2]
    work_bgr = img_bgr.copy()

    # 1. De-shading bayangan lipatan tengah jika diketahui sisinya
    if fold_side in ["left", "right"]:
        work_bgr = deshade_gutter_edge(work_bgr, fold_side=fold_side)

    # 2. Perceptual Retinex Flat-Fielding dalam CIELAB
    lab = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Downsampling untuk estimasi permukaan latar belakang secara cepat dan halus
    scale = max(1, min(W, H) // 250)
    small_w, small_h = W // scale, H // scale
    L_small = cv2.resize(L, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Morfologi dilasi besar untuk membuang teks dan goresan pensil dari estimasi latar
    k_dil = max(3, int(small_h * 0.04))
    bg_dil = cv2.dilate(L_small, np.ones((k_dil, k_dil), np.uint8))
    bg_med = cv2.medianBlur(bg_dil, 21 if 21 % 2 == 1 else 23)
    bg_blur = cv2.GaussianBlur(bg_med, (31, 31), 0)
    bg_surface = cv2.resize(bg_blur, (W, H), interpolation=cv2.INTER_LINEAR)

    # Pembagian bidang pencahayaan datar
    L_flat = cv2.divide(L, bg_surface, scale=252)

    # 3. Penguatan Goresan Pensil Adaptif (Soft Anti-Aliased Stroke Boost)
    # Mempertajam tulisan pensil tipis tanpa merusak teks cetak atau menimbulkan noise
    blurred_L = cv2.GaussianBlur(L_flat, (3, 3), 0)
    high_freq = cv2.subtract(L_flat, blurred_L)

    # Masker goresan: aktif pada piksel tulisan/garis (< 225)
    stroke_mask = np.clip((225.0 - L_flat.astype(np.float32)) / 75.0, 0.0, 1.0)
    L_boosted = np.clip(
        L_flat.astype(np.float32) - 1.25 * high_freq.astype(np.float32) * stroke_mask,
        0, 255
    ).astype(np.uint8)

    # 4. Rekonstruksi Citra VLM (Enhanced RGB)
    if is_blue:
        # Menetralkan tint biru pada kertas kosong sampul agar bersih di VLM
        # B channel LAB: < 128 = biru. Tarik piksel kertas terang ke 128 (netral)
        paper_mask = (L_boosted > 215).astype(np.float32)
        B_neutral = np.clip(
            B.astype(np.float32) * (1.0 - 0.75 * paper_mask) + 128.0 * (0.75 * paper_mask),
            0, 255
        ).astype(np.uint8)
        vlm_bgr = cv2.cvtColor(cv2.merge([L_boosted, A, B_neutral]), cv2.COLOR_LAB2BGR)
    else:
        vlm_bgr = cv2.cvtColor(cv2.merge([L_boosted, A, B]), cv2.COLOR_LAB2BGR)

    # 5. Rekonstruksi Citra OCR/ICR Monokrom (High-Contrast Grayscale)
    if is_blue:
        # Pada sampul biru, kanal Merah (Red channel 2 pada format BGR) memberikan
        # kontras pensil dan tinta cetak tertinggi karena pigmen kertas biru menyerap spektrum merah.
        R_raw = work_bgr[:, :, 2]
        R_small = cv2.resize(R_raw, (small_w, small_h), interpolation=cv2.INTER_AREA)
        bg_r = cv2.resize(
            cv2.GaussianBlur(cv2.medianBlur(cv2.dilate(R_small, np.ones((k_dil, k_dil), np.uint8)), 21), (31, 31), 0),
            (W, H)
        )
        ocr_raw = cv2.divide(R_raw, bg_r, scale=255)
    else:
        ocr_raw = L_boosted.copy()

    # Perlindungan halaman kosong (hanya berlaku jika allow_blank=True, misal untuk Hal 8)
    std_val = float(np.std(ocr_raw))
    dark_ratio = float(np.mean(ocr_raw < 180))
    if allow_blank and dark_ratio < 0.0002 and std_val < 4.5:
        ocr_gray = np.full((H, W), 255, dtype=np.uint8)
    else:
        # Dynamic Black-Point & Gamma Stretch untuk Huruf Pekat Tebal & Kontras Maksimal
        p1 = float(np.percentile(ocr_raw, 0.8))
        p95 = float(np.percentile(ocr_raw, 95.0))
        if p1 < 160.0:
            bp = max(15.0, min(p1 + 18.0, 140.0))
            wp = min(240.0, max(p95 - 4.0, 205.0))
            if wp > bp:
                norm = np.clip((ocr_raw.astype(np.float32) - bp) / (wp - bp), 0.0, 1.0)
                ocr_stretched = np.clip(np.power(norm, 1.75) * 255.0, 0, 255).astype(np.uint8)
            else:
                ocr_stretched = ocr_raw.copy()
        else:
            ocr_stretched = ocr_raw.copy()

        # High-frequency unsharp sharpening (Masked Sharpening: hanya pada goresan teks/kisi)
        blur = cv2.GaussianBlur(ocr_stretched, (0, 0), sigmaX=1.2)
        sharp = cv2.addWeighted(ocr_stretched, 1.40, blur, -0.40, 0)
        stroke_mask = (ocr_stretched < 225)
        ocr_gray = np.where(stroke_mask, np.clip(sharp, 0, 255), 255).astype(np.uint8)

    # 6. Pembersihan Margin Tepi Kaca Scanner (0.8% batas luar)
    pad_h = int(H * 0.008)
    pad_w = int(W * 0.008)
    vlm_bgr[:pad_h, :] = (255, 255, 255)
    vlm_bgr[-pad_h:, :] = (255, 255, 255)
    vlm_bgr[:, :pad_w] = (255, 255, 255)
    vlm_bgr[:, -pad_w:] = (255, 255, 255)

    ocr_gray[:pad_h, :] = 255
    ocr_gray[-pad_h:, :] = 255
    ocr_gray[:, :pad_w] = 255
    ocr_gray[:, -pad_w:] = 255

    diagnostics = {
        "is_blue": is_blue,
        "mean_vlm_l": round(float(L_boosted.mean()), 2),
        "mean_ocr_gray": round(float(ocr_gray.mean()), 2),
        "white_paper_coverage": round(float((L_boosted > 240).mean()), 3),
        "fold_deshaded": fold_side is not None,
    }

    return vlm_bgr, ocr_gray, diagnostics


def remove_comb_box_border(img_bgr: np.ndarray, bg_color: int = 255) -> np.ndarray:
    """
    Menghilangkan garis batas kotak fisik comb-box pada citra mikro sel kuesioner
    secara menyeluruh (garis horizontal, garis vertikal, dan noktah residu tepi)
    tanpa merusak goresan tulisan tangan di dalamnya dan mempertahankan 100%
    dimensi resolusi dan margin asli citra crop.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    cleaned = img_bgr.copy()
    is_color = (len(cleaned.shape) == 3 and cleaned.shape[2] == 3)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY) if is_color else cleaned.copy()
    h, w = gray.shape

    # 1. Binarisasi adaptif untuk mendeteksi tinta/garis gelap
    bw = (gray < 185).astype(np.uint8) * 255

    # 2. Definisikan zona inti (core/digit zone) dan zona perimeter luar
    top_band = max(4, int(h * 0.18))
    bot_band = max(4, int(h * 0.18))
    left_band = max(4, int(w * 0.18))
    right_band = max(4, int(w * 0.18))

    core_mask = np.zeros((h, w), dtype=bool)
    core_mask[top_band:h - bot_band, left_band:w - right_band] = True

    # 3. Analisis Connected Components pada seluruh elemen gelap
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    remove_mask = np.zeros((h, w), dtype=bool)

    for i in range(1, n_labels):
        x, y, bw_comp, bh_comp, area = stats[i]
        comp_mask = (labels == i)

        intersects_core = np.any(comp_mask & core_mask)

        # Garis horizontal (lebar >= 48% W di pita atas/bawah)
        is_h_line = (bw_comp >= int(w * 0.48)) and (y <= top_band or (y + bh_comp) >= (h - bot_band))

        # Garis vertikal (tinggi >= 48% H di pita kiri/kanan)
        is_v_line = (bh_comp >= int(h * 0.48)) and (x <= left_band or (x + bw_comp) >= (w - right_band))

        # Residu tepi (di luar core_mask)
        is_edge_noise = not intersects_core and (
            y < top_band or (y + bh_comp) > (h - bot_band) or
            x < left_band or (x + bw_comp) > (w - right_band)
        )

        if is_h_line or is_v_line:
            if intersects_core:
                remove_mask |= (comp_mask & ~core_mask)
            else:
                remove_mask |= comp_mask
        elif is_edge_noise and area < (h * w * 0.15):
            remove_mask |= comp_mask

    # Dilasi 1 px agar anti-aliasing garis batas terhapus mulus tanpa mengenai core_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    remove_mask_dil = cv2.dilate(remove_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    remove_mask_final = (remove_mask_dil & ~core_mask) | remove_mask

    if is_color:
        cleaned[remove_mask_final] = [bg_color, bg_color, bg_color]
    else:
        cleaned[remove_mask_final] = bg_color

    return cleaned


def is_blank_cell(cell_img: np.ndarray) -> bool:
    """
    Deteksi cerdas multi-faktor sel kosong (blank comb-box / zero phantom digits):
    1. Membersihkan garis kotak comb-box via remove_comb_box_border.
    2. Menghitung profil kontras dan densitas piksel grafit di zona inti (core zone).
    3. Menganalisis komponen terhubung (connected component) untuk membedakan goresan
       tulisan tangan nyata (angka/tanda baca) vs bintik debu/tekstur scanner.
    """
    if cell_img is None or cell_img.size == 0:
        return True

    cl = remove_comb_box_border(cell_img)
    gray = cv2.cvtColor(cl, cv2.COLOR_BGR2GRAY) if len(cl.shape) == 3 else cl.copy()
    h, w = gray.shape

    # Abaikan batas luar perimeter (pad 5%)
    pad_h = max(2, int(h * 0.05))
    pad_w = max(2, int(w * 0.05))
    inner = gray[pad_h:h - pad_h, pad_w:w - pad_w]
    if inner.size == 0:
        return True

    min_v = int(inner.min())
    mean_v = float(inner.mean())
    contrast = mean_v - min_v

    # 1. Kertas seragam tanpa coretan (kontras sangat rendah atau tidak ada goresan gelap)
    if min_v > 150 or contrast < 45:
        return True

    # 2. Analisis Komponen Terhubung pada goresan gelap (< 180)
    bw = (inner < 180).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    max_area = 0
    max_h = 0
    max_w = 0
    total_stroke_area = 0

    for i in range(1, n_labels):
        _, _, cw, ch, area = stats[i]
        if area >= 4:
            total_stroke_area += area
        if area > max_area:
            max_area = area
            max_h = ch
            max_w = cw

    # Sel kosong yang hanya berisi noise acak/tekstur/debu:
    # Angka terkecil (angka '1' tipis atau lengkungan '0' pudar) memiliki:
    # tinggi >= 8 px, lebar >= 2 px, dan area komponen >= 22 px
    if max_area < 22 or max_h < 8 or max_w < 2:
        return True

    # Jika total area akumulasi goresan sangat minim (< 28 px)
    if total_stroke_area < 28:
        return True

    return False


def clean_cell_mnist(cell_img: np.ndarray, target_dim: int = 24) -> Optional[np.ndarray]:
    """
    Standarisasi sel karakter ke kanvas MNIST 28x28 dengan letterboxing proporsional
    dan multi-factor blank cell detection:
    1. Hapus garis kisi comb-box di ruang koordinat asli via remove_comb_box_border().
    2. Deteksi sel kosong secara presisi (Zero Phantom Digits). Mengembalikan None jika sel kosong.
    3. Letterbox proporsional menjaga rasio aspek asli tanpa distorsi pergeseran massa.
    """
    if cell_img is None or cell_img.size == 0:
        return None

    cl = remove_comb_box_border(cell_img)
    if is_blank_cell(cl):
        return None

    return letterbox_cell_28x28(cl, target_dim=target_dim)


def letterbox_cell_28x28(
    cell_img: np.ndarray,
    target_dim: int = 26
) -> np.ndarray:
    """
    Mengonversi citra sel yang telah dibersihkan garis kotaknya menjadi kanvas 28x28
    format MNIST (0 = latar hitam, 255 = goresan putih) dengan mempertahankan rasio aspek
    dan proporsi margin asli tanpa distorsi peregangan atau pemangkasan tight-box.
    """
    if cell_img is None or cell_img.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()

    h, w = gray.shape

    # Inversi soft anti-aliased: piksel latar kertas (terang) menjadi 0, goresan pensil menjadi terang
    inv = np.clip((240.0 - gray.astype(np.float32)) * (255.0 / 120.0), 0, 255).astype(np.uint8)

    # Skala proporsional menjaga rasio aspek (asli)
    scale = min(float(target_dim) / h, float(target_dim) / w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(inv, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - nw) // 2
    y_off = (28 - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas

