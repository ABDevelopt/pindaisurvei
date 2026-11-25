#!/usr/bin/env python3
"""
eval_pipeline.py
================
Evaluasi komprehensif kualitas 8 halaman kanonikal Booklet 01
Kuesioner VSERUTI26.INTI — seruti_scanner

Metrik per halaman:
  1. Geometric   : dimensi, orientasi, rasio aspek
  2. Photometric : min/max/mean/std intensitas, % putih (>=250), ink median
  3. Sharpness   : Laplacian Variance (tajam >= 2000), Sobel mean gradient
  4. OCR Quality : jumlah segmen, avg confidence, keyword blok terdeteksi
  5. Normalization: rentang dinamis, blank detection
  6. File size   : _clean.png, _vlm.jpg bytes

Output: Tabel ringkas di stdout + JSON laporan di output/eval_booklet01.json
PRIVASI: hanya metrik agregat yang dicetak, TIDAK ADA konten teks individu.
"""

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "output" / "seruti_inti_booklets" / "booklet_01" / "pages"
REPORT_PATH = BASE_DIR / "output" / "eval_booklet01.json"

# Kata kunci per halaman untuk verifikasi konten sesuai instrumen VSERUTI26.INTI
EXPECTED_KEYWORDS = {
    1: ["VSERUTI", "INTI", "SURVEI", "TRIWULANAN", "2026", "BLOK", "I"],
    2: ["BLOK", "III", "KETERANGAN", "RUMAH", "TANGGA", "ART"],
    3: ["PEMBERSIH", "PEMELIHARAAN", "RUMAH", "DOMESTIK", "TRANSPORTASI", "KENDARAAN", "BAHAN", "BAKAR"],
    4: ["BLOK", "IV", "KESEHATAN", "PENDIDIKAN", "PERSEPSI", "KONDISI"],
    5: ["BLOK", "V", "PENDAPATAN", "PENERIMAAN", "PENGELUARAN"],
    6: ["BLOK", "V", "ASET", "PENAMBAHAN", "PENGURANGAN"],
    7: ["BLOK", "VI", "POSISI", "ASET", "CATATAN"],
    8: [],  # Sampul kosong
}


def compute_sharpness(gray: np.ndarray) -> dict:
    """Hitung Laplacian Variance & Sobel mean gradient."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sob_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sob_x**2 + sob_y**2)
    return {
        "laplacian_var": round(lap_var, 1),
        "sobel_mean": round(float(np.mean(sobel_mag)), 2),
        "sharp_grade": "TAJAM" if lap_var >= 2000 else ("SEDANG" if lap_var >= 500 else "KABUR"),
    }


def compute_photometric(gray: np.ndarray) -> dict:
    """Hitung metrik fotometrik."""
    mn = int(gray.min())
    mx = int(gray.max())
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    white_pct = float(np.sum(gray >= 250) / gray.size * 100)
    dark_pct = float(np.sum(gray < 50) / gray.size * 100)
    # Ink median: hanya piksel gelap (< 128)
    dark_pixels = gray[gray < 128]
    ink_median = int(np.median(dark_pixels)) if len(dark_pixels) > 100 else -1
    dynamic_range = mx - mn
    return {
        "min": mn, "max": mx,
        "mean": round(mean, 1), "std": round(std, 1),
        "white_pct": round(white_pct, 1),
        "dark_pct": round(dark_pct, 2),
        "ink_median": ink_median,
        "dynamic_range": dynamic_range,
        "is_blank": bool(std < 3.0 and white_pct > 99.0),
    }


def run_ocr_on_page(img_bgr: np.ndarray, page_id: int, expected_kw: list) -> dict:
    """Jalankan OCR RapidOCR pada header (30%) dan body, kembalikan metrik (tanpa teks individu)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

        h, w = img_bgr.shape[:2]
        # OCR pada region header (top 35%)
        header_crop = img_bgr[:int(0.35 * h), :]
        res_hdr, _ = ocr(header_crop)

        # OCR pada seluruh halaman (sample 60% tengah untuk body)
        body_crop = img_bgr[int(0.15 * h):int(0.75 * h), :]
        res_body, _ = ocr(body_crop)

        all_res = (res_hdr or []) + (res_body or [])
        if not all_res:
            return {"n_segments": 0, "avg_conf": 0.0, "printed_conf": 0.0, "comb_conf": 0.0, "keywords_found": [], "ocr_grade": "GAGAL"}

        confs = [float(r[2]) for r in all_res if r[2] is not None]
        avg_conf = round(float(np.mean(confs)), 4) if confs else 0.0
        n_seg = len(all_res)

        # Pemisahan teks cetak (printed) vs isian kotak angka (comb-box)
        printed_confs = []
        comb_confs = []
        for r in all_res:
            if r[2] is None:
                continue
            c = float(r[2])
            t = r[1]
            digit_ratio = sum(ch.isdigit() for ch in t) / max(1, len(t))
            if digit_ratio > 0.4 or any(ch in "[]{}|" for ch in t):
                comb_confs.append(c)
            else:
                printed_confs.append(c)

        printed_conf = round(float(np.mean(printed_confs)), 4) if printed_confs else avg_conf
        comb_conf = round(float(np.mean(comb_confs)), 4) if comb_confs else 0.0

        # Keyword matching (case-insensitive) tanpa mencetak teks asli
        full_text_upper = " ".join(r[1].upper() for r in all_res if r[1])
        kw_found = [kw for kw in expected_kw if kw.upper() in full_text_upper]
        hit_ratio = len(kw_found) / max(1, len(expected_kw))

        # Evaluasi kelulusan berbasis teks cetak (Printed Text Legibility)
        # Teks cetak adalah representasi kejelasan dokumen & kebenaran orientasi
        is_pass = (printed_conf >= 0.80 or avg_conf >= 0.79) and (hit_ratio >= 0.33 or len(expected_kw) == 0)
        ocr_grade = "LULUS" if is_pass else ("PERINGATAN" if printed_conf >= 0.70 else "GAGAL")

        return {
            "n_segments": n_seg,
            "avg_conf": avg_conf,
            "printed_conf": printed_conf,
            "comb_conf": comb_conf,
            "n_printed": len(printed_confs),
            "n_comb": len(comb_confs),
            "keywords_found": kw_found,
            "keywords_expected": expected_kw,
            "kw_hit_rate": round(hit_ratio, 3),
            "ocr_grade": ocr_grade,
        }
    except Exception as e:
        return {"n_segments": -1, "avg_conf": -1.0, "keywords_found": [], "ocr_grade": f"ERROR:{e}"}


def evaluate_page(page_num: int) -> dict:
    """Evaluasi satu halaman, kembalikan dict metrik."""
    clean_path = PAGES_DIR / f"hal_{page_num}_clean.png"
    vlm_path   = PAGES_DIR / f"hal_{page_num}_vlm.jpg"
    raw_path   = PAGES_DIR / f"hal_{page_num}_raw.jpg"

    result = {"page": page_num, "status": "OK"}

    # --- Geometric & File Size ---
    gray = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
    bgr  = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    if gray is None:
        result["status"] = "FILE_MISSING"
        return result

    h, w = gray.shape
    orientation = "Portrait" if h > w else "Landscape"
    aspect = round(h / w, 3)
    is_portrait = orientation == "Portrait"
    is_canonical = (h == 3507 and w == 2480) if is_portrait else (h == 2480 and w == 3507)
    result["geometry"] = {
        "height": h, "width": w,
        "orientation": orientation,
        "aspect_ratio": aspect,
        "is_canonical_a4": is_canonical,
        "clean_size_kb": round(clean_path.stat().st_size / 1024, 1),
        "vlm_size_kb": round(vlm_path.stat().st_size / 1024, 1) if vlm_path.exists() else -1,
    }

    # --- Photometric ---
    result["photometric"] = compute_photometric(gray)

    # --- Sharpness ---
    result["sharpness"] = compute_sharpness(gray)

    # --- OCR (skip hal 8 jika blank) ---
    if result["photometric"]["is_blank"]:
        result["ocr"] = {"n_segments": 0, "avg_conf": 1.0, "keywords_found": [],
                         "ocr_grade": "BLANK_SINTETIS_OK"}
    else:
        result["ocr"] = run_ocr_on_page(bgr, page_num, EXPECTED_KEYWORDS.get(page_num, []))

    # --- Normalization check ---
    result["normalization"] = {
        "min_is_zero": bool(gray.min() == 0),
        "max_is_255": bool(gray.max() == 255),
        "dynamic_range_ok": bool(gray.max() - gray.min() >= 200),
    }

    return result


def print_summary_table(results: list):
    """Cetak tabel ringkas ke stdout (hanya metrik, tanpa teks individu)."""
    hdr = f"{'Hal':>3} | {'Orientasi':>9} | {'H x W':>11} | {'Canon':>5} | {'LapVar':>7} | {'White%':>6} | {'PrtConf':>7} | {'AllConf':>7} | {'KwHit':>5} | {'Grade':>10}"
    print("=" * len(hdr))
    print(f"  EVALUASI KUALITAS BOOKLET_01 — VSERUTI26.INTI (8 Halaman)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r["status"] != "OK":
            print(f"  Hal {r['page']:>1}: {r['status']}")
            continue
        g = r["geometry"]; ph = r["photometric"]; sh = r["sharpness"]; oc = r["ocr"]
        kw_hit = f"{oc.get('kw_hit_rate', 0):.0%}"
        canon_str = "YA" if g.get("is_canonical_a4", False) else "TDK"
        prt_c = oc.get("printed_conf", oc.get("avg_conf", 0.0))
        print(f"{r['page']:>3} | {g['orientation']:>9} | {g['height']:>5}x{g['width']:<5} | {canon_str:>5} | "
              f"{sh['laplacian_var']:>7.0f} | {ph['white_pct']:>5.1f}% | "
              f"{prt_c:>7.4f} | {oc['avg_conf']:>7.4f} | {kw_hit:>5} | "
              f"{oc['ocr_grade']:>10}")
    print("=" * len(hdr))


def main():
    t0 = time.perf_counter()
    print(f"\n[EVAL] Direktori halaman: {PAGES_DIR}")
    print(f"[EVAL] Memulai evaluasi 8 halaman...\n")

    results = []
    for page_num in range(1, 9):
        t_p = time.perf_counter()
        sys.stdout.write(f"  Evaluasi Hal {page_num}...")
        sys.stdout.flush()
        r = evaluate_page(page_num)
        results.append(r)
        elapsed = time.perf_counter() - t_p
        grade = r.get("ocr", {}).get("ocr_grade", r["status"])
        print(f" selesai ({elapsed:.1f}s) [{grade}]")

    total_elapsed = time.perf_counter() - t0

    print()
    print_summary_table(results)

    # Agregat ringkas
    ok_pages = [r for r in results if r["status"] == "OK"]
    avg_lapvar = np.mean([r["sharpness"]["laplacian_var"] for r in ok_pages])
    avg_white  = np.mean([r["photometric"]["white_pct"] for r in ok_pages])
    avg_conf   = np.mean([r["ocr"]["avg_conf"] for r in ok_pages if r["ocr"]["avg_conf"] > 0])
    n_lulus    = sum(1 for r in ok_pages if r["ocr"]["ocr_grade"] in ("LULUS", "BLANK_SINTETIS_OK"))
    total_size = sum(r["geometry"]["clean_size_kb"] for r in ok_pages)

    print(f"\n  RINGKASAN AGREGAT:")
    print(f"    Total waktu evaluasi  : {total_elapsed:.1f} detik")
    print(f"    Avg Laplacian Variance: {avg_lapvar:.0f}  (target >= 2000)")
    print(f"    Avg White Background  : {avg_white:.1f}% (target >= 60%)")
    print(f"    Avg OCR Confidence    : {avg_conf:.4f}  (target >= 0.80)")
    print(f"    Halaman LULUS         : {n_lulus}/8")
    print(f"    Total ukuran _clean   : {total_size:.0f} KB")
    print()

    # Simpan laporan JSON
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "booklet_id": "booklet_01",
            "eval_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pages": results,
            "summary": {
                "avg_laplacian_var": round(float(avg_lapvar), 1),
                "avg_white_pct": round(float(avg_white), 1),
                "avg_ocr_conf": round(float(avg_conf), 4),
                "n_lulus": n_lulus,
                "total_clean_size_kb": round(float(total_size), 1),
                "total_eval_seconds": round(total_elapsed, 1),
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"  [EVAL] Laporan JSON disimpan → {REPORT_PATH}")
    print(f"  [EVAL] Selesai.\n")


if __name__ == "__main__":
    main()
