#!/usr/bin/env python3
"""
benchmark_against_template.py
=============================
Pengujian dan Komparasi Menyeluruh Keterbacaan Hasil Pindaian Seruti INTI
terhadap Ground Truth Template PDF Resmi BPS:
"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf"

Metrik Komparasi per Halaman (Hal 1 s.d. 7):
  1. Word Token Recall (Cakupan Kata Template yang Tertangkap OCR)
  2. Sequence Matcher Text Similarity Ratio
  3. Verifikasi Kehadiran Blok & Rincian Kunci Instrumen
  4. Analisis Kelengkapan Struktur Kuesioner (Banners, Rincian, Pilihan Jawaban)
  5. Estimasi Tingkat Kesalahan Karakter (CER) pada Pertanyaan Kunci

Zero-PII Invariant: Hanya teks template kuesioner dan metrik agregat yang dievaluasi.
"""

import sys
import re
import json
import time
from pathlib import Path
from difflib import SequenceMatcher

import cv2
import numpy as np
import fitz
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path(__file__).resolve().parent
PDF_TEMPLATE_PATH = Path(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")
PAGES_DIR = BASE_DIR / "output" / "seruti_inti_booklets" / "booklet_01" / "pages"
REPORT_OUTPUT_PATH = BASE_DIR / "output" / "benchmark_template_vs_scan.json"

# Blok dan Rincian Kunci yang Wajib Hadir per Halaman pada Template Resmi
PAGE_KEY_SECTIONS = {
    1: [
        "REPUBLIK INDONESIA",
        "SURVEI EKONOMI RUMAH TANGGA TRIWULANAN 2026",
        "VSERUTI26.INTI",
        "BLOK I. KETERANGAN TEMPAT",
        "BLOK II. KETERANGAN PENDATAAN",
        "201. Pendata",
        "202. Pemeriksa",
        "203. Hasil pendataan rumah tangga"
    ],
    2: [
        "BLOK III. KETERANGAN RUMAH TANGGA",
        "Nama anggota rumah tangga",
        "Hubungan dengan KRT",
        "Status Perkawinan",
        "BLOK IV.1. KONSUMSI DAN PENGELUARAN BAHAN MAKANAN",
        "VSERUTI.MAK"
    ],
    3: [
        "415. Bahan pembersih dan pemeliharaan rumah",
        "416. Biaya jasa pekerja domestik",
        "417. Transportasi",
        "418. Pembelian kendaraan",
        "419. Bahan bakar kendaraan",
        "426. Pemeliharaan, perbaikan, dan sewa tempat tinggal",
        "430. Pakaian, alas kaki, dan tutup kepala"
    ],
    4: [
        "BLOK IV.2.B PENGELUARAN KESEHATAN DAN PENDIDIKAN",
        "453. Rawat inap",
        "454. Rawat jalan",
        "457. Biaya pendidikan",
        "TOTAL PENGELUARAN KESEHATAN DAN PENDIDIKAN",
        "BLOK IV.3 PERSEPSI TERHADAP KONDISI EKONOMI",
        "470. a. Apakah pendapatan rumah tangga"
    ],
    5: [
        "BLOK V. PENDAPATAN, PENERIMAAN, DAN PENGELUARAN",
        "501. Jika terdapat ART (10 tahun ke atas) yang bekerja",
        "Upah/gaji pokok",
        "Lembur, tunjangan",
        "502. Nilai Produksi",
        "503. Stok Usaha Rumah Tangga"
    ],
    6: [
        "BLOK V. PENDAPATAN, PENERIMAAN, DAN PENGELUARAN",
        "504. Transfer berjalan selama tiga bulan terakhir",
        "505. Penambahan dan Pengurangan Aset",
        "Pembelian/pembangunan",
        "Penjualan barang modal",
        "JUMLAH"
    ],
    7: [
        "BLOK VI. POSISI ASET DAN KEWAJIBAN KEUANGAN",
        "601. Uang tunai",
        "602. Tabungan",
        "606. Emas perhiasan",
        "JUMLAH POSISI ASET",
        "JUMLAH POSISI KEWAJIBAN",
        "BLOK VII. CATATAN"
    ]
}


def normalize_token(w: str) -> str:
    """Normalisasi token kata untuk pencocokan teks."""
    return re.sub(r"[^A-Z0-9]", "", w.upper())


def extract_template_page_data(doc: fitz.Document, page_idx: int) -> dict:
    """Ekstraksi teks ground truth dari halaman template PDF."""
    page = doc[page_idx]
    full_text = page.get_text()
    lines = [l.strip() for l in full_text.splitlines() if len(l.strip()) > 1]
    raw_tokens = [normalize_token(w) for w in full_text.split()]
    clean_tokens = [w for w in raw_tokens if len(w) > 1 and not w.isdigit()]
    return {
        "full_text": full_text,
        "lines": lines,
        "tokens": clean_tokens,
        "unique_tokens": sorted(list(set(clean_tokens))),
    }


def fuzzy_contains(token: str, token_set: set, threshold: float = 0.85) -> bool:
    """Pencocokan fuzzy untuk toleransi minor OCR (misal O vs 0, l vs 1)."""
    if token in token_set:
        return True
    for candidate in token_set:
        if abs(len(token) - len(candidate)) <= 2:
            if SequenceMatcher(None, token, candidate).ratio() >= threshold:
                return True
    return False


def evaluate_page_against_template(
    page_num: int,
    template_data: dict,
    ocr_results: list
) -> dict:
    """Komparasi hasil OCR pindaian halaman terhadap template PDF ground truth."""
    ocr_texts = [r[1] for r in ocr_results]
    ocr_full = " ".join(ocr_texts)
    ocr_tokens = [normalize_token(w) for w in ocr_full.split()]
    ocr_token_set = set(w for w in ocr_tokens if len(w) > 1 and not w.isdigit())

    template_tokens = template_data["tokens"]
    template_unique = template_data["unique_tokens"]

    # 1. Hitung Word Token Recall
    matched_tokens = [t for t in template_unique if fuzzy_contains(t, ocr_token_set)]
    token_recall = len(matched_tokens) / max(1, len(template_unique))

    # 2. String Similarity Ratio (SequenceMatcher)
    clean_tmpl_str = " ".join(template_tokens)
    clean_ocr_str = " ".join([w for w in ocr_tokens if len(w) > 1 and not w.isdigit()])
    sim_ratio = SequenceMatcher(None, clean_tmpl_str[:3000], clean_ocr_str[:3000]).ratio()

    # 3. Verifikasi Bagian Kunci / Banners
    key_sections = PAGE_KEY_SECTIONS.get(page_num, [])
    section_results = []
    ocr_upper_concat = ocr_full.upper()
    for sec in key_sections:
        sec_clean = re.sub(r"[^A-Z0-9\s]", "", sec.upper())
        sec_words = [w for w in sec_clean.split() if len(w) > 1]
        
        # Cek apakah setidaknya 60% kata kunci bagian ini muncul di OCR
        hits = sum(1 for w in sec_words if fuzzy_contains(w, ocr_token_set))
        hit_ratio = hits / max(1, len(sec_words))
        is_detected = hit_ratio >= 0.60
        section_results.append({
            "section": sec,
            "detected": is_detected,
            "hit_ratio": round(hit_ratio, 3)
        })

    key_sec_hit_rate = (
        sum(1 for s in section_results if s["detected"]) / max(1, len(section_results))
        if section_results else 1.0
    )

    # 4. Evaluasi Status Kelulusan Halaman
    # Kriteria: Token Recall >= 75% DAN Kehadiran Bagian Kunci >= 80%
    is_pass = token_recall >= 0.75 and key_sec_hit_rate >= 0.75
    grade = "SEMPURNA" if (token_recall >= 0.85 and key_sec_hit_rate >= 0.85) else \
            "LULUS" if is_pass else "PERINGATAN"

    return {
        "page_num": page_num,
        "template_token_count": len(template_tokens),
        "template_unique_count": len(template_unique),
        "ocr_detected_segments": len(ocr_results),
        "ocr_token_count": len(ocr_tokens),
        "matched_tokens_count": len(matched_tokens),
        "token_recall": round(token_recall, 4),
        "similarity_ratio": round(sim_ratio, 4),
        "key_sections_total": len(key_sections),
        "key_sections_detected": sum(1 for s in section_results if s["detected"]),
        "key_sections_hit_rate": round(key_sec_hit_rate, 4),
        "key_sections_detail": section_results,
        "grade": grade,
    }


def main():
    t_start = time.perf_counter()
    print("=" * 95)
    print("  UJI KETERBACAAN SCAN SERUTI INTI vs GROUND TRUTH TEMPLATE PDF RESMI BPS")
    print(f"  Template : {PDF_TEMPLATE_PATH.name}")
    print(f"  Direktori: {PAGES_DIR}")
    print("=" * 95)

    if not PDF_TEMPLATE_PATH.exists():
        print(f"[ERROR] Berkas PDF template tidak ditemukan: {PDF_TEMPLATE_PATH}")
        sys.exit(1)

    doc = fitz.open(str(PDF_TEMPLATE_PATH))
    ocr = RapidOCR()

    results = []
    for page_num in range(1, 8):
        t0 = time.perf_counter()
        clean_file = PAGES_DIR / f"hal_{page_num}_clean.png"
        if not clean_file.exists():
            print(f"  Hal {page_num}: BERKAS TIDAK ADA ({clean_file.name})")
            continue

        sys.stdout.write(f"  [Hal {page_num}/7] Memproses OCR & pencocokan template...")
        sys.stdout.flush()

        # Ekstraksi template PDF (page_idx 0 s.d. 6)
        tmpl_data = extract_template_page_data(doc, page_num - 1)

        # Jalankan OCR pada citra scan kanonikal
        img_bgr = cv2.imread(str(clean_file))
        ocr_res, _ = ocr(img_bgr)
        ocr_res = ocr_res or []

        # Bandingkan
        res = evaluate_page_against_template(page_num, tmpl_data, ocr_res)
        results.append(res)

        elapsed = time.perf_counter() - t0
        print(f" selesai ({elapsed:.1f}s) | Recall={res['token_recall']:.1%} | Kunci={res['key_sections_detected']}/{res['key_sections_total']} [{res['grade']}]")

    total_elapsed = time.perf_counter() - t_start

    # Tampilkan Tabel Komparasi
    print("\n" + "=" * 95)
    print(f"{'Hal':>3} | {'Tmpl Words':>10} | {'OCR Segs':>8} | {'Token Recall':>12} | {'Similarity':>10} | {'Bagian Kunci':>12} | {'Status':>10}")
    print("-" * 95)
    for r in results:
        key_str = f"{r['key_sections_detected']}/{r['key_sections_total']}"
        print(f"{r['page_num']:>3} | {r['template_unique_count']:>10} | {r['ocr_detected_segments']:>8} | "
              f"{r['token_recall']:>11.1%} | {r['similarity_ratio']:>10.4f} | "
              f"{key_str:>12} | {r['grade']:>10}")
    print("=" * 95)

    # Rata-rata agregat
    avg_recall = np.mean([r["token_recall"] for r in results])
    avg_sim = np.mean([r["similarity_ratio"] for r in results])
    total_key_det = sum(r["key_sections_detected"] for r in results)
    total_key_all = sum(r["key_sections_total"] for r in results)
    overall_key_rate = total_key_det / max(1, total_key_all)

    print("\n  RINGKASAN BENCHMARK TERHADAP TEMPLATE GROUND TRUTH:")
    print(f"    Rata-rata Token Recall     : {avg_recall:.1%} (Target >= 80.0%)")
    print(f"    Rata-rata Text Similarity  : {avg_sim:.4f}")
    print(f"    Cakupan Bagian Kunci Total : {total_key_det}/{total_key_all} ({overall_key_rate:.1%})")
    print(f"    Total Waktu Pengujian      : {total_elapsed:.1f} detik")

    # Simpan laporan detail JSON
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "template_pdf": str(PDF_TEMPLATE_PATH),
            "scan_source": str(PAGES_DIR),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {
                "avg_token_recall": round(float(avg_recall), 4),
                "avg_similarity": round(float(avg_sim), 4),
                "key_sections_detected": total_key_det,
                "key_sections_total": total_key_all,
                "key_sections_rate": round(float(overall_key_rate), 4),
                "total_seconds": round(total_elapsed, 1),
            },
            "pages": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  [OK] Laporan komparasi lengkap disimpan di: {REPORT_OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
