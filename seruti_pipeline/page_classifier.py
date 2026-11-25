"""
seruti_pipeline/page_classifier.py
==================================
Klasifikasi pasangan halaman pada satu lembar spread A3 Seruti Inti.
Menegakkan invarian saddle-stitch L + R = 9.
"""
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Inisialisasi OCR lokal
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr

PAGE_KEYWORD_MAP = {
    1: [
        "VSERUTI26.INTI", "VSERUTI26INTI", "VSERUTI", "SURVEI EKONOMI",
        "KETERANGAN PENGELUARAN", "BLOK I", "BLOK II", "TRIWULAN", "REPUBLIK INDONESIA"
    ],
    2: [
        "KETERANGAN RUMAH TANGGA", "KETERANGAN RUMAHTANGGA", "KETERANGANRUMAHTANGGA",
        "BLOK III", "BLOKIII", "BLOKIL", "BLOKHL", "BLOK111", "BLOK 3",
        "UNTUK ART YANG", "UNTUKART", "STATUS PERKAWINAN", "HUBUNGAN DENGAN",
        "VSERUTI.MAK", "VSERUTI_MAK"
    ],
    3: [
        "BLOK IV.2.A", "IV.2.A", "PENGELUARAN BUKAN MAKANAN", "BAHAN PEMBERSIH",
        "PEMELIHARAAN RUMAH", "MINYAK TANAH", "SABUN CUCI", "KARBOL"
    ],
    4: [
        "BLOK IV.2.B", "IV.2.B", "BLOK IV.3", "PENGELUARAN KESEHATAN",
        "KESEHATAN DAN PENDIDIKAN", "PERSEPSI TERHADAP KONDISI", "KONDISI EKONOMI",
        "SELAMA TIGA BULAN TERAKHIR"
    ],
    5: [
        "BLOK V.1", "BELUM TERJUAL", "LEMBUR", "HONORARIUM", "THR",
        "BAHAN BAKU", "NILAI PRODUKSI", "STOK AKHIR", "PENDAPATAN DAN PENERIMAAN"
    ],
    6: [
        "BLOK V.2", "ASET RUMAH TANGGA", "USAHA RUMAH TANGGA",
        "PENAMBAHAN DAN PENGURANGAN ASET", "PENAMBAHAN ASET"
    ],
    7: [
        "BLOK VI", "BLOK VII", "BLOKVI", "BLOKVII", "POSISI ASET FINANSIAL",
        "BLOK VII.CATATAN", "CATATAN", "LEMBAGA PEMBIAYAAN", "MODAL USAHA",
        "UTANG", "PINJAMAN", "612", "609"
    ],
}

SADDLE_STITCH_PAIRS = {
    1: 8, 8: 1,
    2: 7, 7: 2,
    3: 6, 6: 3,
    4: 5, 5: 4,
}


def _extract_page_scores(half_img: np.ndarray) -> Dict[int, float]:
    """Mengekstrak skor kemiripan halaman berdasarkan kata kunci header/footer."""
    # Deteksi halaman sampul belakang kosong (blank)
    if float(half_img.std()) < 12.0:
        return {8: 15.0}

    H, W = half_img.shape[:2]
    # Ambil sampel area atas 30% dan bawah 30%
    top = half_img[:int(0.30 * H), :]
    bot = half_img[int(0.70 * H):, :]

    ocr_engine = get_ocr()
    res_t, _ = ocr_engine(top)
    res_b, _ = ocr_engine(bot)

    txt = ""
    if res_t:
        txt += " " + " ".join(r[1] for r in res_t)
    if res_b:
        txt += " " + " ".join(r[1] for r in res_b)
    txt_u = txt.upper()

    scores: Dict[int, float] = {}
    for p_num, kws in PAGE_KEYWORD_MAP.items():
        score = sum(3.0 if kw in txt_u else 0.0 for kw in kws)
        if score > 0:
            scores[p_num] = score

    return scores


def classify_spread(left_a4: np.ndarray, right_a4: np.ndarray) -> Tuple[str, str, float, Dict[str, Any]]:
    """
    Mengklasifikasikan pasangan halaman A4 pada satu lembar spread A3.
    Mengembalikan (left_id, right_id, confidence, diagnostics).
    """
    # Pastikan dalam grayscale untuk evaluasi fitur teks
    if len(left_a4.shape) == 3:
        left_gray = cv2.cvtColor(left_a4, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_a4

    if len(right_a4.shape) == 3:
        right_gray = cv2.cvtColor(right_a4, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right_a4

    scores_l = _extract_page_scores(left_gray)
    scores_r = _extract_page_scores(right_gray)

    best_l = max(scores_l, key=scores_l.get) if scores_l else None
    best_r = max(scores_r, key=scores_r.get) if scores_r else None

    conf_l = scores_l.get(best_l, 0.0) if best_l else 0.0
    conf_r = scores_r.get(best_r, 0.0) if best_r else 0.0

    left_num: Optional[int] = None
    right_num: Optional[int] = None
    overall_conf = 0.5

    # 1. Jika kedua sisi terdeteksi dan memenuhi invarian
    if best_l and best_r and best_l + best_r == 9:
        left_num = best_l
        right_num = best_r
        overall_conf = min(0.99, max(conf_l, conf_r) / 10.0 + 0.5)

    # 2. Jika salah satu sisi lebih dominan, gunakan invarian saddle-stitch L + R = 9
    elif best_l and (conf_l >= conf_r or not best_r):
        left_num = best_l
        right_num = SADDLE_STITCH_PAIRS.get(best_l, 9 - best_l)
        overall_conf = min(0.95, conf_l / 10.0 + 0.4)
    elif best_r:
        right_num = best_r
        left_num = SADDLE_STITCH_PAIRS.get(best_r, 9 - best_r)
        overall_conf = min(0.95, conf_r / 10.0 + 0.4)
    else:
        # Fallback default: anggap spread [hal_4, hal_5] jika tidak ada teks yang cocok
        left_num = 4
        right_num = 5
        overall_conf = 0.3

    left_id = f"hal_{left_num}"
    right_id = f"hal_{right_num}"

    diag = {
        "scores_left": scores_l,
        "scores_right": scores_r,
        "chosen_left": left_id,
        "chosen_right": right_id,
        "confidence": round(overall_conf, 3),
    }

    return left_id, right_id, overall_conf, diag
