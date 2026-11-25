"""
seruti_pipeline/orientation_handler.py
======================================
Koreksi kemiringan (deskewing) dan orientasi kanonikal per halaman A4:
- Halaman Portrait (Hal 1, 2, 3, 4, 8): Penegakan orientasi tegak (0° vs 180°).
- Halaman Landscape (Hal 5, 6, 7): Auto-rotate ke orientasi landscape kanonikal
  (90° CW vs 270° CW / 90° CCW) sehingga judul tabel selalu di bagian atas.
Mendukung pemrosesan tandem untuk citra OCR (monokrom) dan VLM (RGB).
"""
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr


def estimate_skew_angle(
    gray: np.ndarray,
    max_angle_deg: float = 4.0,
    is_landscape: Optional[bool] = None
) -> float:
    """
    Mengestimasi sudut kemiringan dokumen (skew angle) dengan presisi sub-derajat:
    1. Blank check: Halaman kosong / non-kisi (seperti Hal 8) menghasilkan 0.0 derajat.
    2. Deteksi adaptif orientasi landscape (W > H) vs portrait (H > W).
    3. Isolasi garis kisi horizontal melalui opening morfologis selektif.
    4. Hierarchical Two-Stage Search (Coarse-to-Fine):
       - Coarse Search (skala 4): mencari kisaran puncak global dengan resolusi 0.2°.
       - Fine Search (skala 2): mencari sudut optimal dengan resolusi presisi tinggi
         (0.005° untuk landscape/Blok V, 0.010° untuk portrait) menggunakan fungsi objektif
         momen ke-4 sum(P^4) yang sangat tajam dan unimodal.
    5. Mengembalikan sudut putar dalam derajat (presisi sub-derajat hingga 0.005°).
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape[:2]
    # Blank-page check (sampul kosong / non-teks)
    if float(np.std(gray)) < 10.0 or float(np.mean(gray < 160)) < 0.005:
        return 0.0

    if is_landscape is None:
        is_landscape = (W > H)

    if is_landscape:
        # Area kisi tabel Blok V (Halaman 5, 6, 7): tabel membentang lebar y: [8%, 94%], x: [5%, 95%]
        sample = gray[int(0.08 * H):int(0.94 * H), int(0.05 * W):int(0.95 * W)]
    else:
        # Area konten tabel portrait: y: [12%, 88%], x: [8%, 92%]
        sample = gray[int(0.12 * H):int(0.88 * H), int(0.08 * W):int(0.92 * W)]

    sH_orig, sW_orig = sample.shape[:2]

    # 1. Tahap Coarse Search (Downsampling skala 4 untuk kecepatan komputasi ~40 ms)
    scale_coarse = 4
    c_w = max(1, sW_orig // scale_coarse)
    c_h = max(1, sH_orig // scale_coarse)
    small_c = cv2.resize(sample, (c_w, c_h), interpolation=cv2.INTER_AREA)
    _, bw_c = cv2.threshold(small_c, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kh_c_len = max(12, int(c_w * (0.038 if is_landscape else 0.030)))
    kh_c = cv2.getStructuringElement(cv2.MORPH_RECT, (kh_c_len, 1))
    lines_c = cv2.morphologyEx(bw_c, cv2.MORPH_OPEN, kh_c)

    center_c = (c_w / 2.0, c_h / 2.0)
    coarse_angles = np.arange(-max_angle_deg, max_angle_deg + 0.05, 0.2)
    scores_c = []
    for ang in coarse_angles:
        M = cv2.getRotationMatrix2D(center_c, ang, 1.0)
        rot = cv2.warpAffine(lines_c, M, (c_w, c_h), flags=cv2.INTER_NEAREST)
        prof = rot.sum(axis=1).astype(np.float64)
        scores_c.append(float(np.sum(prof ** 4)))

    best_coarse = coarse_angles[np.argmax(scores_c)]

    # 2. Tahap Fine Search (Downsampling skala 2 untuk resolusi tinggi & interpolasi sub-derajat)
    scale_fine = 2
    f_w = max(1, sW_orig // scale_fine)
    f_h = max(1, sH_orig // scale_fine)
    small_f = cv2.resize(sample, (f_w, f_h), interpolation=cv2.INTER_AREA)
    _, bw_f = cv2.threshold(small_f, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kh_f_len = max(24, int(f_w * (0.038 if is_landscape else 0.030)))
    kh_f = cv2.getStructuringElement(cv2.MORPH_RECT, (kh_f_len, 1))
    lines_f = cv2.morphologyEx(bw_f, cv2.MORPH_OPEN, kh_f)

    center_f = (f_w / 2.0, f_h / 2.0)
    fine_step = 0.005 if is_landscape else 0.010
    fine_window = 0.25
    fine_angles = np.arange(best_coarse - fine_window, best_coarse + fine_window + fine_step / 2.0, fine_step)
    scores_f = []
    for ang in fine_angles:
        M = cv2.getRotationMatrix2D(center_f, ang, 1.0)
        rot = cv2.warpAffine(lines_f, M, (f_w, f_h), flags=cv2.INTER_LINEAR)
        prof = rot.sum(axis=1).astype(np.float64)
        scores_f.append(float(np.sum(prof ** 4)))

    best_fine = fine_angles[np.argmax(scores_f)]
    return round(float(best_fine), 3)


def rotate_image(img: np.ndarray, angle: float, interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Memutar gambar sebesar angle derajat dengan padding putih menggunakan interpolasi Bicubic tajam."""
    if abs(angle) < 0.025:
        return img
    H, W = img.shape[:2]
    center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_val = (255, 255, 255) if len(img.shape) == 3 else 255
    rotated = cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return rotated


PORTRAIT_KEYWORDS = {
    "hal_1": ["VSERUTI", "SURVEI", "REPUBLIK", "BLOK I", "BLOK II", "TRIWULAN"],
    "hal_2": ["BLOK", "KETERANGAN", "RUMAH TANGGA", "ART", "STATUS PERKAWINAN"],
    "hal_3": ["BLOK", "PENGELUARAN", "BAHAN PEMBERSIH", "401", "415", "MINYAK TANAH"],
    "hal_4": ["BLOK", "KESEHATAN", "PENDIDIKAN", "PERSEPSI", "470", "OOP"],
}


def orient_and_partition_page(
    mono: np.ndarray,
    page_id: str,
    vlm_image: Optional[np.ndarray] = None,
    is_already_deskewed: bool = False,
    force_rotation: Optional[int] = None
) -> Dict[str, Any]:
    """
    Menyesuaikan orientasi halaman A4 secara deterministik:
    1. Deskew sudut kemiringan dokumen (dilewati jika is_already_deskewed=True).
    2. Jika force_rotation ditentukan (0, 90, 180, 270), terapkan langsung tanpa evaluasi OCR lambat.
    3. Jika halaman Portrait (Hal 1, 2, 3, 4, 8): tentukan 0° vs 180° agar judul di atas.
    4. Jika halaman Landscape (Hal 5, 6, 7): tentukan 90° CW vs 270° CW agar tabel landscape terbaca normal.
    Jika vlm_image disertakan, rotasi yang sama diterapkan secara tandem pada citra VLM RGB.
    """
    H, W = mono.shape[:2]

    # Halaman 8 (blank sampul belakang)
    if page_id == "hal_8" or float(mono.std()) < 12.0:
        return {
            "image": mono,
            "full": mono,
            "vlm": vlm_image if vlm_image is not None else mono,
            "angle": 0.0,
            "rotation_applied": 0,
            "page_id": page_id,
            "is_upright": True,
            "is_landscape": False,
        }

    # 1. Koreksi kemiringan (deskew)
    if is_already_deskewed:
        skew = 0.0
        deskewed = mono
        deskewed_vlm = vlm_image
    else:
        skew = estimate_skew_angle(mono)
        deskewed = rotate_image(mono, skew, interp=cv2.INTER_CUBIC) if abs(skew) >= 0.025 else mono
        deskewed_vlm = rotate_image(vlm_image, skew, interp=cv2.INTER_CUBIC) if (vlm_image is not None and abs(skew) >= 0.025) else vlm_image

    # 2. Fast Path: jika force_rotation ditentukan secara deterministik
    if force_rotation is not None:
        rotation_applied = force_rotation
        if force_rotation == 0:
            final_img = deskewed
            final_vlm = deskewed_vlm
            is_landscape = False
        elif force_rotation == 90:
            final_img = cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE)
            final_vlm = cv2.rotate(deskewed_vlm, cv2.ROTATE_90_CLOCKWISE) if deskewed_vlm is not None else None
            is_landscape = True
        elif force_rotation == 180:
            final_img = cv2.rotate(deskewed, cv2.ROTATE_180)
            final_vlm = cv2.rotate(deskewed_vlm, cv2.ROTATE_180) if deskewed_vlm is not None else None
            is_landscape = False
        elif force_rotation == 270:
            final_img = cv2.rotate(deskewed, cv2.ROTATE_90_COUNTERCLOCKWISE)
            final_vlm = cv2.rotate(deskewed_vlm, cv2.ROTATE_90_COUNTERCLOCKWISE) if deskewed_vlm is not None else None
            is_landscape = True
        else:
            final_img = deskewed
            final_vlm = deskewed_vlm
            is_landscape = False

        # Pass 2: Landscape High-Precision Auto-Deskew
        # Khusus untuk halaman landscape (Hal 5, 6, 7), setelah rotasi ke landscape kanonikal,
        # lakukan evaluasi kemiringan presisi tinggi (0.005°) pada orientasi landscape.
        land_skew = 0.0
        if is_landscape:
            land_skew = estimate_skew_angle(final_img, max_angle_deg=3.0, is_landscape=True)
            if abs(land_skew) >= 0.025:
                final_img = rotate_image(final_img, land_skew, interp=cv2.INTER_CUBIC)
                if final_vlm is not None:
                    final_vlm = rotate_image(final_vlm, land_skew, interp=cv2.INTER_CUBIC)

        return {
            "image": final_img,
            "full": final_img,
            "vlm": final_vlm if final_vlm is not None else final_img,
            "angle": round(skew + land_skew, 3) if is_landscape else round(skew, 3),
            "rotation_applied": rotation_applied,
            "page_id": page_id,
            "is_upright": True,
            "is_landscape": is_landscape,
            "pass1_skew_deg": round(skew, 3),
            "landscape_skew_deg": round(land_skew, 3) if is_landscape else 0.0,
        }

    ocr_engine = get_ocr()

    # 2. Penanganan Halaman Portrait (Hal 1, 2, 3, 4)
    if page_id in ["hal_1", "hal_2", "hal_3", "hal_4"]:
        opt_0 = deskewed
        opt_180 = cv2.rotate(deskewed, cv2.ROTATE_180)

        # Uji area atas (top 30%) dengan downscale cepat (~0.04s)
        slice_0 = opt_0[:int(0.30 * H), :]
        slice_180 = opt_180[:int(0.30 * H), :]
        s0_fast = cv2.resize(slice_0, (800, int(slice_0.shape[0] * 800 / slice_0.shape[1])))
        s180_fast = cv2.resize(slice_180, (800, int(slice_180.shape[0] * 800 / slice_180.shape[1])))
        res_0, _ = ocr_engine(s0_fast)
        res_180, _ = ocr_engine(s180_fast)

        txt_0 = " ".join(x[1] for x in res_0).upper() if res_0 else ""
        txt_180 = " ".join(x[1] for x in res_180).upper() if res_180 else ""

        kws = PORTRAIT_KEYWORDS.get(page_id, ["BLOK"])
        sc_0 = sum(3.0 for k in kws if k in txt_0) + (len(res_0) if res_0 else 0) * 0.1
        sc_180 = sum(3.0 for k in kws if k in txt_180) + (len(res_180) if res_180 else 0) * 0.1

        if sc_180 > sc_0:
            final_img = opt_180
            final_vlm = cv2.rotate(deskewed_vlm, cv2.ROTATE_180) if deskewed_vlm is not None else None
            rotation_applied = 180
        else:
            final_img = opt_0
            final_vlm = deskewed_vlm
            rotation_applied = 0

        is_landscape = False
        land_skew = 0.0

    # 3. Penanganan Halaman Landscape (Hal 5, 6, 7)
    elif page_id in ["hal_5", "hal_6", "hal_7"]:
        opt_90 = cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE)
        opt_270 = cv2.rotate(deskewed, cv2.ROTATE_90_COUNTERCLOCKWISE)

        def score_landscape(candidate_img):
            rH = candidate_img.shape[0]
            slice_top = candidate_img[:int(0.35 * rH), :]
            top_fast = cv2.resize(slice_top, (800, int(slice_top.shape[0] * 800 / slice_top.shape[1])))
            res, _ = ocr_engine(top_fast)
            if not res:
                return -10.0
            txt = " ".join(x[1] for x in res).upper()
            score = 0.0
            if "BLOK" in txt:
                score += 15.0
            if page_id == "hal_5" and any(k in txt for k in ["BLOK V", "PENDAPATAN", "501", "UPAH", "GAJI"]):
                score += 20.0
            if page_id == "hal_6" and any(k in txt for k in ["BLOK V", "504", "TRANSFER", "PENSIUN"]):
                score += 25.0
            if page_id == "hal_7" and any(k in txt for k in ["BLOK VI", "POSISI ASET", "KEWAJIBAN", "RINCIAN", "NILAI"]):
                score += 20.0

            # Penalti tegas jika area atas memuat teks catatan kaki / footer bawah tabel
            if any(k in txt for k in ["DAUR ULANG", "REMEDIASI", "AIR LIMBAH", "CATATAN: KODE", "KODE '0'"]):
                score -= 50.0
            if any(k in txt for k in ["WAKTU SELESAI", "SELESAI WAWANCARA"]):
                score -= 50.0
            if "JUMLAH" in txt and page_id == "hal_5":
                score -= 25.0
            return score

        sc_90 = score_landscape(opt_90)
        sc_270 = score_landscape(opt_270)

        if sc_90 >= sc_270:
            final_img = opt_90
            final_vlm = cv2.rotate(deskewed_vlm, cv2.ROTATE_90_CLOCKWISE) if deskewed_vlm is not None else None
            rotation_applied = 90
        else:
            final_img = opt_270
            final_vlm = cv2.rotate(deskewed_vlm, cv2.ROTATE_90_COUNTERCLOCKWISE) if deskewed_vlm is not None else None
            rotation_applied = 270

        is_landscape = True

        # Pass 2: Landscape High-Precision Auto-Deskew
        land_skew = estimate_skew_angle(final_img, max_angle_deg=3.0, is_landscape=True)
        if abs(land_skew) >= 0.025:
            final_img = rotate_image(final_img, land_skew, interp=cv2.INTER_CUBIC)
            if final_vlm is not None:
                final_vlm = rotate_image(final_vlm, land_skew, interp=cv2.INTER_CUBIC)

    else:
        final_img = deskewed
        final_vlm = deskewed_vlm
        rotation_applied = 0
        is_landscape = False
        land_skew = 0.0

    return {
        "image": final_img,
        "full": final_img,
        "vlm": final_vlm if final_vlm is not None else final_img,
        "angle": round(skew + land_skew, 3) if is_landscape else round(skew, 3),
        "rotation_applied": rotation_applied,
        "page_id": page_id,
        "is_upright": True,
        "is_landscape": is_landscape,
        "pass1_skew_deg": round(skew, 3),
        "landscape_skew_deg": round(land_skew, 3) if is_landscape else 0.0,
    }


def auto_deskew(
    mono: np.ndarray,
    max_angle_deg: float = 5.0,
    is_landscape: Optional[bool] = None
) -> Tuple[np.ndarray, float]:
    """
    Mengoreksi kemiringan (deskewing) dokumen dengan toleransi hingga max_angle_deg derajat.
    Dapat menerima citra BGR (raw) maupun monokrom. Menggunakan interpolasi Bicubic tajam.
    Mengembalikan (deskewed_image, skew_angle).
    """
    skew = estimate_skew_angle(mono, max_angle_deg=max_angle_deg, is_landscape=is_landscape)
    if abs(skew) > max_angle_deg:
        skew = 0.0
    if abs(skew) >= 0.025:
        deskewed = rotate_image(mono, skew, interp=cv2.INTER_CUBIC)
        return deskewed, skew
    return mono.copy(), 0.0


def detect_hal7_split_y(rotated_mono: np.ndarray) -> int:
    """
    Mendeteksi posisi vertikal pembatas antara Blok VI (Tabel Finansial)
    dan Blok VII (Catatan Lapangan) pada Halaman 7 (Landscape).
    Memanfaatkan deteksi teks banner 'BLOK VII. CATATAN' (~y=1700)
    atau proyeksi garis pemisah horizontal.
    """
    H, W = rotated_mono.shape[:2]
    default_split = int(round(H * 0.685))

    ocr = get_ocr()
    crop_y1 = int(round(H * 0.55))
    crop_y2 = int(round(H * 0.80))
    crop = rotated_mono[crop_y1:crop_y2, :]

    try:
        res, _ = ocr(crop)
        if res:
            for r in res:
                text = r[1].upper()
                if "BLOK" in text and ("VII" in text or "CATATAN" in text or "ML" in text):
                    box = r[0]
                    box_y = min(pt[1] for pt in box)
                    detected_y = crop_y1 + int(box_y)
                    # Ambil margin batas atas banner
                    return max(100, detected_y - 10)
    except Exception:
        pass

    # Fallback morfologis: cari garis horizontal panjang di area y ~ [0.60 H, 0.75 H]
    try:
        search_strip = rotated_mono[int(0.60 * H):int(0.75 * H), int(0.05 * W):int(0.95 * W)]
        inv = 255 - search_strip
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.15), 1))
        lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh)
        prof = lines.sum(axis=1)
        if prof.max() > 0:
            rel_y = int(np.argmax(prof))
            return int(0.60 * H) + rel_y
    except Exception:
        pass

    return default_split

