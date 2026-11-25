#!/usr/bin/env python3
"""
create_seruti_inti_booklets.py
==============================
Skrip CLI Mandiri untuk Perakitan Buklet Fisik BPS Susenas Seruti Inti (VSERUTI26.INTI)
dari Berkas Scan Mentah (Raw Scans).

Fungsi Utama:
1. Menemukan berkas scan pindaian A3 secara rekursif atau per folder petugas.
2. Mengelompokkan setiap 4 spread A3 menjadi satu buklet fisik lengkap (8 halaman A4).
3. Mendeteksi dan memotong garis lipatan tengah (gutter seam) secara presisi sub-piksel.
4. Menegakkan invarian matematika saddle-stitch L + R = 9 pada setiap spread:
     Spread 1: Hal 8 + Hal 1 = 9 [VALID]
     Spread 2: Hal 2 + Hal 7 = 9 [VALID]
     Spread 3: Hal 6 + Hal 3 = 9 [VALID]
     Spread 4: Hal 4 + Hal 5 = 9 [VALID]
5. Mengoreksi orientasi halaman (auto-rotate 90° CW untuk halaman landscape Hal 5, 6, 7).
6. Menghilangkan warna kertas biru sampul (blue paper background whitening) & flat-fielding.
7. Menyimpan 8 halaman A4 kanonikal tersusun urut (Hal 1 s.d. Hal 8) dan manifest JSON.
8. Mengisolasi berkas scan sisa/orphan yang tidak lengkap.

Penggunaan:
    # Memproses seluruh scan di direktori default
    python create_seruti_inti_booklets.py --raw-dir "Susenas Seruti 2026/inti" --out-dir "output_booklets"

    # Memproses satu buklet spesifik (4 berkas spread)
    python create_seruti_inti_booklets.py --files file1.jpg file2.jpg file3.jpg file4.jpg --out-dir "output_booklets"

    # Memproses maksimal 3 buklet dengan log detail
    python create_seruti_inti_booklets.py --max-booklets 3 --verbose
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Pastikan root workspace terdaftar di sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np

from seruti_inti.config import (
    INSTRUMENT_ID,
    CANONICAL_PAGE_COUNT,
    SPREADS_PER_BOOKLET,
    SADDLE_STITCH_SUM_INVARIANT,
    INTI_RAW_DATA_DIR,
    RAW_DATA_DIR,
)
from seruti_inti.booklet_assembler import (
    assemble_seruti_inti_booklet,
    discover_and_group_booklets,
    SerutiIntiBooklet,
)
from seruti_pipeline.channel_extractor import pad_to_canonical_a4
from seruti_pipeline.orientation_handler import rotate_image


def export_booklet_to_disk(
    booklet: SerutiIntiBooklet,
    output_dir: Path,
    save_oriented_mono: bool = True,
    save_raw_bgr: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Menyimpan halaman-halaman A4 kanonikal dan metadata manifest ke direktori output.
    """
    booklet_dir = output_dir / booklet.booklet_id
    pages_dir = booklet_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    saved_pages = []

    # Urutkan secara kanonikal: Hal 1 hingga Hal 8
    for page_num in range(1, CANONICAL_PAGE_COUNT + 1):
        page_id = f"hal_{page_num}"
        p_data = booklet.pages.get(page_id)
        if p_data is None:
            if verbose:
                print(f"  [PERINGATAN] {page_id} tidak ditemukan pada {booklet.booklet_id}!")
            continue

        page_record = {
            "page_num": page_num,
            "page_id": page_id,
            "source_file": p_data.source_file,
            "side": p_data.side,
            "seam_x": p_data.seam_x,
            "is_blue": p_data.is_blue,
            "files": {}
        }

        # 1. Simpan citra monokrom terorientasi (siap OCR / Typology Engine)
        if save_oriented_mono:
            # Ambil citra oriented full (misal Hal 5, 6, 7 sudah diputar landscape)
            img_mono = p_data.oriented.get("full", p_data.oriented.get("image", p_data.mono))
            img_mono = pad_to_canonical_a4(img_mono)
            mono_filename = f"{page_id}_clean.png"
            mono_path = pages_dir / mono_filename
            cv2.imwrite(str(mono_path), img_mono)
            page_record["files"]["clean_mono"] = str(mono_path.relative_to(output_dir))
            page_record["dimensions_clean"] = [int(img_mono.shape[1]), int(img_mono.shape[0])]

        # 2. Simpan citra VLM Enhanced (Full RGB dengan background putih flat untuk Vision Language Models)
        vlm_img = p_data.oriented.get("vlm")
        if vlm_img is not None:
            vlm_img = pad_to_canonical_a4(vlm_img)
            vlm_filename = f"{page_id}_vlm.jpg"
            vlm_path = pages_dir / vlm_filename
            cv2.imwrite(str(vlm_path), vlm_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            page_record["files"]["vlm_enhanced"] = str(vlm_path.relative_to(output_dir))
            page_record["dimensions_vlm"] = [int(vlm_img.shape[1]), int(vlm_img.shape[0])]

        # 3. Simpan citra raw A4 asli (BGR)
        if save_raw_bgr and p_data.raw_a4 is not None:
            raw_img = p_data.raw_a4
            rot_deg = p_data.oriented.get("rotation_applied", 0)
            if rot_deg == 180:
                raw_img = cv2.rotate(raw_img, cv2.ROTATE_180)
            elif rot_deg == 90:
                raw_img = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)
            elif rot_deg == 270:
                raw_img = cv2.rotate(raw_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Jika ada residual landscape deskew, terapkan juga ke raw_img
            land_skew = p_data.oriented.get("landscape_skew_deg", 0.0)
            if abs(land_skew) >= 0.025:
                raw_img = rotate_image(raw_img, land_skew, interp=cv2.INTER_CUBIC)

            raw_img = pad_to_canonical_a4(raw_img)
            raw_filename = f"{page_id}_raw.jpg"
            raw_path = pages_dir / raw_filename
            cv2.imwrite(str(raw_path), raw_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            page_record["files"]["raw_bgr"] = str(raw_path.relative_to(output_dir))
            page_record["dimensions_raw"] = [int(raw_img.shape[1]), int(raw_img.shape[0])]

        saved_pages.append(page_record)

    # 3. Buat dan simpan berkas manifest JSON buklet
    manifest_data = {
        "instrument_id": INSTRUMENT_ID,
        "booklet_id": booklet.booklet_id,
        "batch_prefix": booklet.batch_prefix,
        "feed_direction": booklet.feed_direction,
        "is_complete_8_pages": booklet.is_complete(),
        "all_invariants_valid": booklet.verify_saddle_stitch_invariant(),
        "spread_sources": [f.name for f in booklet.spread_files],
        "diagnostics": booklet.diagnostics,
        "pages": saved_pages,
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }

    manifest_path = booklet_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def _process_single_booklet_worker(args: Tuple) -> Dict[str, Any]:
    """
    Fungsi worker paralel untuk memproses dan merakit satu buklet individual.
    """
    (
        idx,
        bg,
        output_dir,
        save_oriented_mono,
        save_raw_bgr,
        fast_mode,
        skip_existing,
        verbose
    ) = args

    booklet_id = f"booklet_{idx:02d}"
    manifest_path = output_dir / booklet_id / "manifest.json"

    # Cek apakah buklet sudah selesai dan lengkap
    if skip_existing and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
            pages_dir = output_dir / booklet_id / "pages"
            clean_files_ok = all((pages_dir / f"hal_{i}_clean.png").exists() for i in range(1, 9))
            if m_data.get("is_complete_8_pages", False) and clean_files_ok:
                print(f"  [SKIP] {booklet_id}: Sudah lengkap terverifikasi (dilewati)", flush=True)
                return {
                    "booklet_id": booklet_id,
                    "feed_direction": m_data.get("feed_direction", "unknown"),
                    "all_invariants_valid": True,
                    "is_complete_8": True,
                    "assembly_time_sec": 0.0,
                    "output_dir": str((output_dir / booklet_id).relative_to(BASE_DIR.resolve())),
                    "skipped": True,
                }
        except Exception:
            pass

    t0 = time.perf_counter()
    if verbose:
        print(f"\n[{idx}] Merakit buklet {idx} ({bg[0].stem.rsplit('_', 1)[0]})...", flush=True)

    booklet = assemble_seruti_inti_booklet(bg, booklet_idx=idx, fast_mode=fast_mode)
    elapsed = time.perf_counter() - t0

    inv_ok = booklet.verify_saddle_stitch_invariant()
    comp_ok = booklet.is_complete()

    manifest = export_booklet_to_disk(
        booklet=booklet,
        output_dir=output_dir,
        save_oriented_mono=save_oriented_mono,
        save_raw_bgr=save_raw_bgr,
        verbose=verbose
    )

    status_str = "LULUS" if (inv_ok and comp_ok) else "GAGAL"
    print(
        f"  [+] {booklet.booklet_id}: Feed={booklet.feed_direction} | Invarian L+R=9={inv_ok} | 8 Hal={comp_ok} ({elapsed:.2f}s) -> {status_str}",
        flush=True
    )

    return {
        "booklet_id": booklet.booklet_id,
        "feed_direction": booklet.feed_direction,
        "all_invariants_valid": inv_ok,
        "is_complete_8": comp_ok,
        "assembly_time_sec": round(elapsed, 4),
        "output_dir": str((output_dir / booklet.booklet_id).relative_to(BASE_DIR.resolve())),
        "skipped": False,
    }


def process_raw_scans(
    raw_dir: Optional[Path] = None,
    spread_files: Optional[List[Path]] = None,
    output_dir: Optional[Path] = None,
    max_booklets: Optional[int] = None,
    save_oriented_mono: bool = True,
    save_raw_bgr: bool = True,
    fast_mode: bool = True,
    skip_existing: bool = True,
    workers: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Pipeline lengkap pembuatan buklet dari berkas scan mentah.
    Mendukung pemrosesan paralel multi-worker untuk performa tinggi.
    """
    import concurrent.futures
    t_start = time.perf_counter()

    if output_dir is None:
        output_dir = BASE_DIR / "output" / "seruti_inti_booklets"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    booklets_to_assemble: List[List[Path]] = []
    orphaned_files: List[Path] = []

    # Skenario A: Daftar berkas spread eksplisit diberikan
    if spread_files:
        if len(spread_files) != SPREADS_PER_BOOKLET:
            raise ValueError(
                f"Satu buklet Seruti Inti membutuhkan tepat {SPREADS_PER_BOOKLET} berkas spread A3. "
                f"Diterima: {len(spread_files)} berkas."
            )
        booklets_to_assemble.append(spread_files)
    # Skenario B: Pencarian otomatis di direktori
    else:
        if raw_dir is None:
            raw_dir = INTI_RAW_DATA_DIR if INTI_RAW_DATA_DIR.exists() else RAW_DATA_DIR
        raw_dir = Path(raw_dir)
        print(f"[*] Mencari berkas pindaian A3 di: {raw_dir}...", flush=True)
        booklets_to_assemble, orphaned_files = discover_and_group_booklets(
            raw_dir, return_orphans=True, recursive=True
        )

    total_discovered = len(booklets_to_assemble)
    if max_booklets is not None and max_booklets > 0:
        booklets_to_assemble = booklets_to_assemble[:max_booklets]

    print(f"[*] Buklet fisik teridentifikasi : {total_discovered} buklet", flush=True)
    if orphaned_files:
        print(f"[!] Berkas sisa/orphan diisolasi : {len(orphaned_files)} berkas (kurang dari 4 spread)", flush=True)
    print(f"[*] Target buklet dirakit       : {len(booklets_to_assemble)} buklet", flush=True)
    print(f"[*] Mode Akselerasi Fast-Path    : {'AKTIF' if fast_mode else 'NONAKTIF'}", flush=True)
    print(f"[*] Multi-Worker Paralel         : {workers} Worker Thread/Process", flush=True)
    print(f"[*] Lewati yang sudah lengkap    : {'YA (--skip-existing)' if skip_existing else 'TIDAK (Proses Ulang Semua)'}", flush=True)
    print(f"[*] Direktori output             : {output_dir}", flush=True)
    print("-" * 65, flush=True)

    results = []
    total_valid_invariants = 0
    total_complete_8 = 0
    total_skipped = 0

    worker_args = [
        (
            idx,
            bg,
            output_dir,
            save_oriented_mono,
            save_raw_bgr,
            fast_mode,
            skip_existing,
            verbose
        )
        for idx, bg in enumerate(booklets_to_assemble, start=1)
    ]

    if workers > 1 and len(booklets_to_assemble) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for res in executor.map(_process_single_booklet_worker, worker_args):
                results.append(res)
                if res.get("all_invariants_valid", False):
                    total_valid_invariants += 1
                if res.get("is_complete_8", False):
                    total_complete_8 += 1
                if res.get("skipped", False):
                    total_skipped += 1
    else:
        for args in worker_args:
            res = _process_single_booklet_worker(args)
            results.append(res)
            if res.get("all_invariants_valid", False):
                total_valid_invariants += 1
            if res.get("is_complete_8", False):
                total_complete_8 += 1
            if res.get("skipped", False):
                total_skipped += 1

    total_elapsed = time.perf_counter() - t_start
    print("-" * 65)
    print("RINGKASAN PERAKITAN BUKLET:")
    print(f"  - Total Buklet Selesai Diproses: {len(results)} buklet")
    print(f"  - Invarian L+R=9 Terverifikasi : {total_valid_invariants}/{len(results)}")
    print(f"  - Lengkap 8 Halaman Kanonikal  : {total_complete_8}/{len(results)}")
    print(f"  - Total Waktu Eksekusi         : {total_elapsed:.2f} detik")
    if len(results) > 0:
        print(f"  - Rata-rata per Buklet (8 Hal) : {total_elapsed / len(results):.2f} detik")

    # Simpan ringkasan global
    global_summary = {
        "instrument_id": INSTRUMENT_ID,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "total_booklets_processed": len(results),
        "total_valid_invariants": total_valid_invariants,
        "total_complete_8_pages": total_complete_8,
        "total_orphans": len(orphaned_files),
        "total_elapsed_sec": round(total_elapsed, 4),
        "booklets": results,
    }

    summary_file = output_dir / "assembly_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n[OK] Ringkasan perakitan disimpan ke: {summary_file}")
    return global_summary


def main():
    parser = argparse.ArgumentParser(
        description="Perakitan Buklet Fisik BPS Susenas Seruti Inti (VSERUTI26.INTI) dari Raw Scans."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Direktori sumber berisi berkas scan A3 (default: Susenas Seruti 2026/inti)."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        default=None,
        help="Daftar 4 berkas spread A3 untuk merakit satu buklet tunggal."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Direktori tujuan penyimpanan buklet A4 (default: output/seruti_inti_booklets)."
    )
    parser.add_argument(
        "--max-booklets",
        type=int,
        default=None,
        help="Batas maksimal jumlah buklet yang diproses."
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Jangan simpan citra raw BGR (hanya simpan citra clean monokrom)."
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Nonaktifkan fast-path feed detector (jalankan full OCR classification)."
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Proses ulang buklet yang sudah lengkap (jangan lewati)."
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="Jumlah worker proses paralel (default: 3)."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Cetak rincian log diagnostik per spread."
    )

    args = parser.parse_args()

    spread_files = [Path(p) for p in args.files] if args.files else None
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else None

    process_raw_scans(
        raw_dir=raw_dir,
        spread_files=spread_files,
        output_dir=out_dir,
        max_booklets=args.max_booklets,
        save_oriented_mono=True,
        save_raw_bgr=not args.no_raw,
        fast_mode=not args.no_fast,
        skip_existing=not args.force_reprocess,
        workers=args.workers,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
