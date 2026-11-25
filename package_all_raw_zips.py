#!/usr/bin/env python3
"""
package_all_raw_zips.py
========================
Mengemas seluruh citra RAW mikro crops dari setiap buklet (76 buklet)
ke dalam berkas ZIP individual pada satu folder terpusat:
`output/crops_raw_zips/booklet_XX_raw.zip`.

Fitur:
- Menggunakan citra RAW murni (output/crops/booklet_XX/hal_Y/mikro/raw/).
- Multi-threaded compression (ThreadPoolExecutor) untuk kecepatan tinggi.
- Verifikasi integritas ZIP pasca-kompresi (zipfile.testzip()).
"""

import concurrent.futures
import glob
import os
import sys
import time
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CROPS_DIR = BASE_DIR / "output" / "crops"
DEST_DIR = BASE_DIR / "output" / "crops_raw_zips"
DEST_DIR.mkdir(parents=True, exist_ok=True)


def zip_single_booklet(b_path: Path) -> dict:
    b_name = b_path.name
    raw_files = sorted(b_path.glob("hal_*/mikro/raw/**/*.png"))
    
    if not raw_files:
        return {"booklet": b_name, "count": 0, "size_mb": 0.0, "status": "NO_RAW_FILES"}
    
    zip_path = DEST_DIR / f"{b_name}_raw.zip"
    
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in raw_files:
            # Simpan path relatif terhadap folder booklet: hal_X/mikro/raw/<field>/<box>.png
            rel_path = f.relative_to(b_path)
            zf.write(f, str(rel_path).replace("\\", "/"))
            
    # Verifikasi integritas zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        corrupted = zf.testzip()
        
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    
    return {
        "booklet": b_name,
        "count": len(raw_files),
        "size_mb": size_mb,
        "elapsed": elapsed,
        "status": "CORRUPTED" if corrupted else "OK",
        "zip_path": str(zip_path)
    }


def main():
    print("=" * 80)
    print("  PENGEMASAN ZIP RAW CROPS DARI SEMUA BUKLET SERUTI")
    print(f"  Direktori Target: {DEST_DIR}")
    print("=" * 80)
    
    booklet_dirs = sorted([p for p in CROPS_DIR.glob("booklet_*") if p.is_dir()])
    print(f"[*] Ditemukan {len(booklet_dirs)} direktori buklet di {CROPS_DIR}")
    
    t_global_start = time.time()
    results = []
    
    # Eksekusi paralel dengan 8 worker thread
    max_workers = min(8, os.cpu_count() or 4)
    print(f"[*] Menjalankan kompresi paralel ({max_workers} threads)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_b = {executor.submit(zip_single_booklet, b): b for b in booklet_dirs}
        
        done_count = 0
        total_b = len(booklet_dirs)
        
        for future in concurrent.futures.as_completed(future_to_b):
            res = future.result()
            results.append(res)
            done_count += 1
            if done_count % 10 == 0 or done_count == total_b:
                print(f"    -> Progres: [{done_count:2d}/{total_b:2d}] buklet terkemas...")
                
    # Urutkan hasil berdasarkan nama booklet
    results.sort(key=lambda x: x["booklet"])
    
    total_files = sum(r["count"] for r in results)
    total_size_mb = sum(r["size_mb"] for r in results)
    total_time = time.time() - t_global_start
    corrupted_count = sum(1 for r in results if r["status"] != "OK")
    
    print("\n" + "=" * 80)
    print("  HASIL PENGEMASAN ZIP RAW CROPS")
    print("=" * 80)
    print(f"  Total Buklet Diproses    : {len(results)} buklet")
    print(f"  Total Citra RAW Terkemas : {total_files:,} citra")
    print(f"  Total Ukuran Berkas ZIP  : {total_size_mb:.2f} MB")
    print(f"  Integritas Arsip         : {'100% Valid (0 corrupt)' if corrupted_count == 0 else f'{corrupted_count} CORRUPTED'}")
    print(f"  Total Waktu Kompresi     : {total_time:.2f} detik ({total_files/total_time:.1f} citra/detik)")
    print(f"  Lokasi Folder Output     : {DEST_DIR}")
    print("=" * 80)
    
    # Tampilkan preview 5 buklet pertama
    print("\nContoh berkas yang dihasilkan di output/crops_raw_zips/:")
    for r in results[:5]:
        print(f"  - {r['booklet']}_raw.zip : {r['count']:,} citra ({r['size_mb']:.2f} MB)")
    print(f"  ... dan {len(results)-5} berkas ZIP buklet lainnya.")


if __name__ == "__main__":
    main()
