#!/usr/bin/env python3
"""
batch_extract_all.py
====================
Ekstraksi OCR massal untuk seluruh 76 buklet menggunakan multiprocessing terisolasi.
Kepatuhan mutlak UU No. 16/1997 & UU PDP No. 27/2022 (Zero PII Cloud Egress, Zero-Jitter Invariant).
"""

import time
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from run_seruti_ocr import (
    load_ocr_model,
    build_variable_registry,
    process_booklet_ocr,
    DEFAULT_MODEL_PATH,
    DEFAULT_CROPS_DIR,
    DEFAULT_OUTPUT_DIR
)

def process_single(booklet_id: str):
    """Worker fungsi untuk memproses 1 buklet."""
    try:
        # Load model per worker process
        clf = load_ocr_model(DEFAULT_MODEL_PATH)
        registry = build_variable_registry()
        res = process_booklet_ocr(
            booklet_id=booklet_id,
            crops_base_dir=DEFAULT_CROPS_DIR,
            output_base_dir=DEFAULT_OUTPUT_DIR,
            clf=clf,
            registry=registry
        )
        s = res["summary"]
        v = s["semantic_validation"]
        return {
            "booklet_id": booklet_id,
            "success": True,
            "elapsed": s["elapsed_seconds"],
            "total_fields": s["total_fields"],
            "total_boxes": s["total_boxes_scanned"],
            "filled_boxes": s["total_filled_boxes"],
            "avg_conf": s["average_confidence"],
            "high_conf_pct": s["high_confidence_pct"],
            "review_flag_count": s["low_confidence_count"],
            "roster_art": s["roster_art_count"],
            "rules_passed": v["rules_passed"],
            "total_rules": v["total_rules"],
            "pass_rate": v["pass_rate_pct"]
        }
    except Exception as e:
        return {
            "booklet_id": booklet_id,
            "success": False,
            "error": str(e)
        }

def main():
    crops_dir = DEFAULT_CROPS_DIR
    booklets = sorted([d.name for d in crops_dir.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
    total_booklets = len(booklets)
    print(f"[*] Memulai batch ekstraksi OCR untuk {total_booklets} buklet...")
    
    t0 = time.time()
    num_workers = min(4, os.cpu_count() or 2)
    print(f"[*] Menggunakan {num_workers} worker paralel...")

    completed = []
    failed = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single, b_id): b_id for b_id in booklets}
        for future in as_completed(futures):
            res = future.result()
            if res["success"]:
                completed.append(res)
                print(f"[+] {res['booklet_id']} SELESAI ({res['elapsed']}s) | Terisi: {res['filled_boxes']} | Conf: {res['avg_conf']}% | Validasi: {res['rules_passed']}/{res['total_rules']} ({res['pass_rate']}%)")
            else:
                failed.append(res)
                print(f"[!] {res['booklet_id']} GAGAL: {res.get('error')}")

    total_time = round(time.time() - t0, 2)
    print("\n" + "=" * 70)
    print("  RINGKASAN EKSEKUSI BATCH EKSTRAKSI OCR")
    print("=" * 70)
    print(f"  Total Buklet Diproses : {total_booklets}")
    print(f"  Sukses                : {len(completed)}")
    print(f"  Gagal                 : {len(failed)}")
    print(f"  Total Waktu           : {total_time} detik ({total_time/total_booklets:.2f} s/buklet)")
    
    if completed:
        total_filled = sum(c["filled_boxes"] for c in completed)
        avg_conf_all = sum(c["avg_conf"] * c["filled_boxes"] for c in completed) / max(1, total_filled)
        print(f"  Total Karakter Terisi : {total_filled:,} kotak")
        print(f"  Rata-rata Akurasi Conf: {avg_conf_all:.2f}%")
        
        # Simpan ringkasan ke JSON
        summary_path = DEFAULT_OUTPUT_DIR / "batch_extraction_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_booklets": total_booklets,
                "successful": len(completed),
                "failed": len(failed),
                "total_time_seconds": total_time,
                "total_filled_boxes": total_filled,
                "weighted_avg_confidence": round(avg_conf_all, 2),
                "booklets": sorted(completed, key=lambda x: x["booklet_id"])
            }, f, indent=2)
        print(f"  Ringkasan disimpan ke : {summary_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
