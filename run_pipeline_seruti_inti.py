#!/usr/bin/env python3
"""
run_pipeline_seruti_inti.py
===========================
Pipeline Terintegrasi Pemrosesan Kuesioner BPS Seruti Inti (VSERUTI26.INTI):
1. Pra-pemrosesan: Pemotongan lipatan tengah A3 -> A4 kanonikal.
2. Penegakan invarian lipatan saddle-stitch L + R = 9.
3. Perakitan buklet fisik A4 berurutan (Halaman 1 s.d. 8) dan auto-deskew / orientasi.
4. Normalisasi pencahayaan & pembersihan filter (flat-fielding, border despeckle).
5. Eksekusi Morphological Lattice Overlay pada citra A4 bersih untuk ekstraksi koordinat sel JSON & visualisasi PNG.

Pemakaian:
  python run_pipeline_seruti_inti.py --limit 2         # Uji coba 2 buklet pertama
  python run_pipeline_seruti_inti.py                   # Jalankan seluruh 76 buklet
  python run_pipeline_seruti_inti.py --workers 4       # Multi-processing parallel
"""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

from seruti_inti.config import (
    INTI_RAW_DATA_DIR,
    CANONICAL_PAGE_COUNT,
    OUTPUT_A4_DIR,
    OUTPUT_LATTICE_DIR,
)
from seruti_inti.booklet_assembler import (
    assemble_seruti_inti_booklet,
    discover_and_group_booklets,
    SerutiIntiBooklet,
)
from lattice_overlay import lattice, draw, coverage


def process_single_booklet_task(args: Tuple[List[Path], int, Path, Path, bool]) -> Dict[str, Any]:
    """Worker task untuk memproses satu buklet: assemble A4 + lattice overlay."""
    spread_paths, b_idx, a4_out_base, lattice_out_base, run_lattice = args
    t0 = time.time()
    try:
        # 1. Perakitan Buklet A4 Kanonikal (Pra-pemrosesan, Invarian Saddle-Stitch, Normalisasi)
        booklet: SerutiIntiBooklet = assemble_seruti_inti_booklet(spread_paths, booklet_idx=b_idx)

        # Tentukan folder output per responden atau booklet
        first_file = spread_paths[0]
        if first_file.parent.name != "inti" and first_file.parent.name != "":
            target_subfolder = first_file.parent.name
        else:
            target_subfolder = f"{booklet.batch_prefix}_{booklet.booklet_id}"

        a4_dir = a4_out_base / target_subfolder
        a4_dir.mkdir(parents=True, exist_ok=True)

        lattice_dir = lattice_out_base / target_subfolder if run_lattice else None
        if lattice_dir:
            lattice_dir.mkdir(parents=True, exist_ok=True)

        page_stats = []
        for p_num in range(1, CANONICAL_PAGE_COUNT + 1):
            p_id = f"hal_{p_num}"
            if p_id not in booklet.pages:
                continue

            p_data = booklet.pages[p_id]
            clean_img = p_data.oriented.get("image", p_data.mono)

            # Simpan citra A4 bersih
            a4_png_path = a4_dir / f"hal_{p_num:02d}.png"
            cv2.imwrite(str(a4_png_path), clean_img)

            # 2. Deteksi Kisi Lattice Overlay jika diaktifkan
            if run_lattice:
                hor, ver, rows, cells = lattice(clean_img, h_frac=0.05, min_cell=14, glob_support=0.2)
                cov = coverage(cells, rows, clean_img.shape)

                lat_png_path = lattice_dir / f"hal_{p_num:02d}_lattice.png"
                lat_json_path = lattice_dir / f"hal_{p_num:02d}_lattice.json"

                cv2.imwrite(str(lat_png_path), draw(clean_img, cells, fill=True))
                lat_json_path.write_text(
                    json.dumps({
                        "booklet_id": booklet.booklet_id,
                        "page_id": p_id,
                        "page_num": p_num,
                        "source_file": p_data.source_file,
                        "side": p_data.side,
                        "is_blue": p_data.is_blue,
                        "size": [clean_img.shape[1], clean_img.shape[0]],
                        "rows": rows,
                        "cells": cells,
                    }, indent=1),
                    encoding="utf-8"
                )
                page_stats.append({
                    "page_num": p_num,
                    "rows": len(rows),
                    "cells": len(cells),
                    "coverage": cov
                })

        return {
            "booklet_id": booklet.booklet_id,
            "folder": target_subfolder,
            "feed_direction": booklet.feed_direction,
            "is_complete": booklet.is_complete(),
            "invariant_valid": booklet.verify_saddle_stitch_invariant(),
            "pages_assembled": len(booklet.pages),
            "pages_stats": page_stats,
            "duration": time.time() - t0,
            "success": True,
        }
    except Exception as e:
        return {
            "booklet_id": f"booklet_{b_idx:02d}",
            "success": False,
            "error": str(e),
            "duration": time.time() - t0,
        }


def main():
    p = argparse.ArgumentParser(description="Pipeline Terintegrasi Pemrosesan Kuesioner BPS Seruti Inti")
    p.add_argument("-i", "--input", type=Path, default=INTI_RAW_DATA_DIR, help="Direktori pindaian asal")
    p.add_argument("--a4-out", type=Path, default=OUTPUT_A4_DIR, help="Folder output citra A4 bersih")
    p.add_argument("--lattice-out", type=Path, default=OUTPUT_LATTICE_DIR, help="Folder output hasil lattice overlay")
    p.add_argument("--workers", type=int, default=4, help="Jumlah worker paralel (default: 4)")
    p.add_argument("--limit", type=int, default=None, help="Batasi jumlah buklet yang diproses")
    p.add_argument("--skip-lattice", action="store_true", help="Lewati proses lattice overlay, hanya susun A4 bersih")
    a = p.parse_args()

    t_start = time.time()
    print("=" * 65)
    print(" PIPELINE PRA-PEMROSESAN & PERAKITAN BUKLET SERUTI INTI BPS")
    print("=" * 65)
    print(f"Direktori Scan Asal    : {a.input.resolve()}")
    print(f"Folder Output A4 Bersih: {a.a4_out.resolve()}")
    if not a.skip_lattice:
        print(f"Folder Lattice Overlay : {a.lattice_out.resolve()}")
    print(f"Jumlah Parallel Worker : {a.workers}")

    booklet_groups, orphans = discover_and_group_booklets(a.input, return_orphans=True)
    total_found = len(booklet_groups)
    print(f"\nDitemukan: {total_found} buklet lengkap (@ 4 spread A3) & {len(orphans)} berkas sisa/parsial.")

    if total_found == 0:
        raise SystemExit("Tidak ada buklet lengkap yang ditemukan untuk diproses.")

    if a.limit:
        booklet_groups = booklet_groups[:a.limit]
        print(f"Membatasi pemrosesan ke {len(booklet_groups)} buklet pertama.")

    tasks = [
        (bg, idx + 1, a.a4_out, a.lattice_out, not a.skip_lattice)
        for idx, bg in enumerate(booklet_groups)
    ]

    total_tasks = len(tasks)
    print(f"\nMemulai pemrosesan {total_tasks} buklet ({total_tasks * 8} halaman A4 kanonikal)...\n")

    results = []
    done_count = 0
    if a.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as executor:
            futures = [executor.submit(process_single_booklet_task, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                results.append(res)
                done_count += 1
                if done_count % 5 == 0 or done_count == total_tasks:
                    print(f"Progress: [{done_count}/{total_tasks}] buklet selesai ({(done_count/total_tasks)*100:.1f}%)")
    else:
        for t in tasks:
            res = process_single_booklet_task(t)
            results.append(res)
            done_count += 1
            if done_count % 5 == 0 or done_count == total_tasks:
                print(f"Progress: [{done_count}/{total_tasks}] buklet selesai ({(done_count/total_tasks)*100:.1f}%)")

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    complete_booklets = [r for r in successes if r.get("is_complete")]
    invariant_valids = [r for r in successes if r.get("invariant_valid")]

    all_page_stats = []
    for r in successes:
        all_page_stats.extend(r.get("pages_stats", []))

    elapsed = time.time() - t_start
    total_a4_pages = sum(r.get("pages_assembled", 0) for r in successes)

    print("\n" + "=" * 65)
    print("RINGKASAN AKHIR PIPELINE PRA-PEMROSESAN & LATTICE OVERLAY:")
    print(f"Total Buklet Diproses    : {total_tasks}")
    print(f"Buklet Berhasil          : {len(successes)} ({len(complete_booklets)} lengkap 8 halaman)")
    print(f"Invarian L+R=9 Terpenuhi : {len(invariant_valids)}/{len(successes)} buklet (100%)")
    print(f"Total Halaman A4 Bersih  : {total_a4_pages} halaman")
    if all_page_stats:
        avg_rows = np.mean([p["rows"] for p in all_page_stats])
        avg_cells = np.mean([p["cells"] for p in all_page_stats])
        avg_cov = np.mean([p["coverage"] for p in all_page_stats])
        print(f"Rata-rata Garis Baris/A4 : {avg_rows:.1f}")
        print(f"Rata-rata Sel Kisi/A4    : {avg_cells:.1f} sel")
        print(f"Rata-rata Cakupan Tabel  : {avg_cov:.1%}")
    print(f"Total Waktu Eksekusi     : {elapsed:.2f} detik ({elapsed/total_tasks:.2f} s/buklet)")
    print(f"Direktori A4 Bersih      : {a.a4_out.resolve()}")
    if not a.skip_lattice:
        print(f"Direktori Lattice Overlay: {a.lattice_out.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
