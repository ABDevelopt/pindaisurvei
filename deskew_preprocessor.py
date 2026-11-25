#!/usr/bin/env python3
"""
deskew_preprocessor.py
======================
Modul & CLI Mandiri Pra-Pemrosesan Deskew (Koreksi Sudut Kemiringan Pindaian):
Kuesioner BPS Susenas Seruti (VSERUTI26.INTI & VSERUTI26.MAK).

Keunggulan Algoritma:
1. Isolasi Kisi Morfologis Selektif: Memfilter seluruh teks, angka, dan goresan tulisan tangan,
   hanya mempertahankan garis-garis tabel horizontal kanonikal.
2. Hierarchical Two-Stage Search (Coarse-to-Fine):
   - Coarse search (skala 4): mencari kisaran global ±max_angle dengan langkah 0.2°.
   - Fine search (skala 2): mencari sudut optimal dengan langkah presisi tinggi (0.005° untuk
     landscape/Blok V, 0.010° untuk portrait) memanfaatkan fungsi objektif momen ke-4 sum(P^4).
3. Tandem Triple-Asset Synchronisation: Memutar citra `_clean.png` (monokrom OCR),
   `_vlm.jpg` (RGB enhanced VLM), dan `_raw.jpg` (RGB mentah) secara bersamaan dengan
   sudut rotasi dan pusat geometri identik.
4. Interpolasi Sub-Piksel Bicubic (cv2.INTER_CUBIC) dengan kanvas padding putih murni (255).
5. Blank Page Safeguard: Halaman kosong / non-kisi (seperti Hal 8 sampul belakang)
   secara otomatis dideteksi dan dilindungi dari rotasi sembarangan (angle = 0.000°).
6. Multi-Processing Parallel Support untuk pemrosesan direktori massal.

Pemakaian CLI:
  # 1. Memproses 1 berkas gambar
  python deskew_preprocessor.py path/to/halaman.png -o output/deskewed

  # 2. Memproses seluruh berkas dalam folder
  python deskew_preprocessor.py output/seruti_inti_booklets/booklet_01/pages -o output/deskewed_pages --workers 4

  # 3. Batch deskewing in-place pada buklet seruti inti (khusus Halaman 5 atau landscape 5, 6, 7):
  python deskew_preprocessor.py --booklets-dir output/seruti_inti_booklets --pages 5 6 7 --in-place
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np

from seruti_pipeline.orientation_handler import estimate_skew_angle, rotate_image, auto_deskew


def compute_line_sharpness(img_gray: np.ndarray) -> Tuple[float, float]:
    """
    Menghitung metrik ketajaman garis horizontal tabel:
    1. peak_max: Nilai puncak proyeksi horizontal tertinggi.
    2. variance: Varians dari profil proyeksi horizontal.
    Semakin tegak garis tabel, semakin tinggi nilai puncak dan variansnya.
    """
    H, W = img_gray.shape[:2]
    # Downsample untuk kalkulasi metrik cepat
    scale = max(1, min(H, W) // 400)
    small = cv2.resize(img_gray, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(small.shape[1] * 0.035)), 1))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
    prof = lines.sum(axis=1).astype(np.float64)
    peak = float(np.max(prof)) if len(prof) > 0 else 0.0
    var = float(np.var(prof)) if len(prof) > 0 else 0.0
    return peak, var


def process_single_image_deskew(
    file_path: Path,
    out_dir: Path,
    max_angle: float = 4.0,
    quality: str = "cubic",
    in_place: bool = False
) -> Dict[str, Any]:
    """
    Memproses deskew pada satu berkas citra pindaian.
    """
    t0 = time.perf_counter()
    img = cv2.imread(str(file_path))
    if img is None:
        return {"file": file_path.name, "success": False, "error": "Gagal membaca citra"}

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Hitung metrik ketajaman awal
    peak_before, var_before = compute_line_sharpness(gray)

    # Estimasi sudut kemiringan presisi sub-derajat
    is_land = (W > H)
    skew_angle = estimate_skew_angle(gray, max_angle_deg=max_angle, is_landscape=is_land)

    interp = cv2.INTER_CUBIC if quality == "cubic" else cv2.INTER_LINEAR
    was_rotated = False

    if abs(skew_angle) >= 0.025:
        deskewed = rotate_image(img, skew_angle, interp=interp)
        was_rotated = True
    else:
        deskewed = img.copy()

    # Hitung metrik ketajaman akhir
    gray_deskewed = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY) if len(deskewed.shape) == 3 else deskewed
    peak_after, var_after = compute_line_sharpness(gray_deskewed)

    sharpness_gain = (var_after / max(1.0, var_before)) if was_rotated else 1.0

    # Simpan hasil
    if in_place:
        out_path = file_path
        out_filename = file_path.name
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"{file_path.stem}_deskewed{file_path.suffix}"
        out_path = out_dir / out_filename

    cv2.imwrite(str(out_path), deskewed)

    dur = time.perf_counter() - t0
    return {
        "file": file_path.name,
        "out_file": out_filename,
        "out_path": str(out_path),
        "skew_angle_deg": round(skew_angle, 3),
        "was_rotated": was_rotated,
        "sharpness_gain": round(sharpness_gain, 2),
        "elapsed_sec": round(dur, 3),
        "shape": [W, H],
        "success": True,
    }


def process_booklet_page_tandem(
    booklet_dir: Path,
    page_num: int,
    max_angle: float = 3.5,
    quality: str = "cubic",
    in_place: bool = True,
    out_base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Memproses deskew tandem pada satu halaman buklet (memutar _clean.png, _vlm.jpg, dan _raw.jpg bersamaan).
    """
    t0 = time.perf_counter()
    page_id = f"hal_{page_num}"
    pages_dir = booklet_dir / "pages"
    if not pages_dir.exists():
        pages_dir = booklet_dir

    clean_file = pages_dir / f"{page_id}_clean.png"
    vlm_file = pages_dir / f"{page_id}_vlm.jpg"
    raw_file = pages_dir / f"{page_id}_raw.jpg"

    if not clean_file.exists():
        return {
            "booklet": booklet_dir.name,
            "page": page_id,
            "success": False,
            "error": f"Berkas {clean_file.name} tidak ditemukan",
            "was_rotated": False,
        }

    img_clean = cv2.imread(str(clean_file), cv2.IMREAD_GRAYSCALE)
    if img_clean is None:
        return {
            "booklet": booklet_dir.name,
            "page": page_id,
            "success": False,
            "error": "Gagal membaca berkas citra clean",
            "was_rotated": False,
        }

    H, W = img_clean.shape[:2]
    is_land = (W > H)
    peak_before, var_before = compute_line_sharpness(img_clean)

    # Estimasi sudut kemiringan presisi tinggi
    skew_angle = estimate_skew_angle(img_clean, max_angle_deg=max_angle, is_landscape=is_land)
    interp = cv2.INTER_CUBIC if quality == "cubic" else cv2.INTER_LINEAR

    was_rotated = False
    affected_files = []

    if abs(skew_angle) >= 0.025:
        was_rotated = True
        # 1. Rotasi clean mono
        rot_clean = rotate_image(img_clean, skew_angle, interp=interp)
        target_clean = clean_file if in_place else (out_base_dir / booklet_dir.name / "pages" / clean_file.name)
        target_clean.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target_clean), rot_clean)
        affected_files.append(target_clean.name)

        # 2. Rotasi vlm enhanced (RGB) jika ada
        if vlm_file.exists():
            img_vlm = cv2.imread(str(vlm_file))
            if img_vlm is not None:
                rot_vlm = rotate_image(img_vlm, skew_angle, interp=interp)
                target_vlm = vlm_file if in_place else (out_base_dir / booklet_dir.name / "pages" / vlm_file.name)
                cv2.imwrite(str(target_vlm), rot_vlm, [cv2.IMWRITE_JPEG_QUALITY, 95])
                affected_files.append(target_vlm.name)

        # 3. Rotasi raw BGR jika ada
        if raw_file.exists():
            img_raw = cv2.imread(str(raw_file))
            if img_raw is not None:
                rot_raw = rotate_image(img_raw, skew_angle, interp=interp)
                target_raw = raw_file if in_place else (out_base_dir / booklet_dir.name / "pages" / raw_file.name)
                cv2.imwrite(str(target_raw), rot_raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
                affected_files.append(target_raw.name)

        peak_after, var_after = compute_line_sharpness(rot_clean)
        gain = var_after / max(1.0, var_before)
    else:
        gain = 1.0

    dur = time.perf_counter() - t0
    return {
        "booklet": booklet_dir.name,
        "page": page_id,
        "skew_angle_deg": round(skew_angle, 3),
        "was_rotated": was_rotated,
        "sharpness_gain": round(gain, 2),
        "affected_files": affected_files,
        "elapsed_sec": round(dur, 3),
        "success": True,
    }


def run_booklets_batch_deskew(
    booklets_dir: Path,
    pages: List[int],
    booklet_filter: Optional[List[str]] = None,
    max_angle: float = 3.5,
    quality: str = "cubic",
    in_place: bool = True,
    out_dir: Optional[Path] = None,
    workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Menjalankan pra-pemrosesan deskew tandem massal untuk seluruh buklet.
    """
    all_booklet_dirs = sorted([d for d in booklets_dir.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
    if booklet_filter:
        filter_set = set(booklet_filter)
        all_booklet_dirs = [d for d in all_booklet_dirs if d.name in filter_set]

    print("=" * 72)
    print(" BATCH TANDEM DESKEW BUKLET SERUTI INTI (PRESISI TINGGI)")
    print("=" * 72)
    print(f"Direktori Buklet : {booklets_dir.resolve()}")
    print(f"Total Buklet     : {len(all_booklet_dirs)}")
    print(f"Target Halaman   : {pages}")
    print(f"Mode Penyimpanan : {'IN-PLACE (Timpa Langsung)' if in_place else f'OUTPUT -> {out_dir}'}")
    print(f"Max Sudut Rotasi : ±{max_angle}°")
    print(f"Worker Paralel   : {workers}")
    print("-" * 72)

    tasks = []
    for b_dir in all_booklet_dirs:
        for p_num in pages:
            tasks.append((b_dir, p_num))

    results = []
    t_start = time.perf_counter()

    if workers <= 1:
        for b_dir, p_num in tasks:
            res = process_booklet_page_tandem(
                b_dir, p_num, max_angle=max_angle, quality=quality, in_place=in_place, out_base_dir=out_dir
            )
            results.append(res)
            _print_task_result(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_booklet_page_tandem,
                    b_dir, p_num, max_angle, quality, in_place, out_dir
                ): (b_dir.name, p_num)
                for b_dir, p_num in tasks
            }
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                    _print_task_result(res)
                except Exception as e:
                    b_name, p_num = futures[fut]
                    print(f"  [-] {b_name} hal_{p_num}: ERROR {e}")

    total_time = time.perf_counter() - t_start
    rotated_count = sum(1 for r in results if r.get("was_rotated"))

    print("-" * 72)
    print("RINGKASAN BATCH DESKEW:")
    print(f"  - Total Halaman Diproses: {len(results)}")
    print(f"  - Halaman Terkoreksi    : {rotated_count}/{len(results)}")
    print(f"  - Total Durasi          : {total_time:.2f} detik")
    print("=" * 72)
    return results


def _print_task_result(res: Dict[str, Any]):
    b_name = res["booklet"]
    p_name = res["page"]
    ang = res.get("skew_angle_deg", 0.0)
    rotated = res.get("was_rotated", False)
    gain = res.get("sharpness_gain", 1.0)
    dur = res.get("elapsed_sec", 0.0) * 1000

    if rotated:
        aff = ", ".join(res.get("affected_files", []))
        print(f"  [+] {b_name} {p_name}: {ang:+6.3f}° | Gain={gain:4.2f}x | Updated: [{aff}] ({dur:5.0f} ms)")
    else:
        print(f"  [.] {b_name} {p_name}: {ang:+6.3f}° | Lurus Sempurna (Preserved) ({dur:5.0f} ms)")


def main():
    parser = argparse.ArgumentParser(
        description="Pra-Pemrosesan Deskew Presisi Sub-Derajat Kuesioner BPS Seruti"
    )
    parser.add_argument("input", nargs="?", type=Path, default=None, help="Berkas gambar atau direktori gambar pindaian")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/deskewed"), help="Direktori output")
    parser.add_argument("--booklets-dir", type=Path, default=None, help="Direktori kumpulan buklet seruti (output/seruti_inti_booklets)")
    parser.add_argument("--pages", type=int, nargs="+", default=[5], help="Daftar nomor halaman yang diproses (default: 5)")
    parser.add_argument("--booklets", type=str, nargs="+", default=None, help="Filter buklet tertentu (misal: booklet_01 booklet_07)")
    parser.add_argument("--in-place", action="store_true", help="Timpa berkas citra asli secara in-place (Bicubic)")
    parser.add_argument("--max-angle", type=float, default=3.5, help="Batas maksimum sudut kemiringan (default: 3.5°)")
    parser.add_argument("--quality", choices=["cubic", "linear"], default="cubic", help="Interpolasi rotasi (default: cubic)")
    parser.add_argument("--workers", type=int, default=4, help="Jumlah worker paralel (default: 4)")
    args = parser.parse_args()

    # Mode 1: Batch Booklets Directory
    if args.booklets_dir:
        if not args.booklets_dir.exists():
            sys.exit(f"Direktori buklet tidak ditemukan: {args.booklets_dir}")
        run_booklets_batch_deskew(
            booklets_dir=args.booklets_dir,
            pages=args.pages,
            booklet_filter=args.booklets,
            max_angle=args.max_angle,
            quality=args.quality,
            in_place=args.in_place,
            out_dir=args.output,
            workers=args.workers
        )
        return

    # Mode 2: Input File atau Folder Tradisional
    if args.input is None:
        parser.print_help()
        sys.exit("\nKesalahan: Harap tentukan 'input' atau gunakan '--booklets-dir'.")

    args.output.mkdir(parents=True, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    if args.input.is_file():
        files = [args.input]
    elif args.input.is_dir():
        files = sorted([f for f in args.input.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    else:
        sys.exit(f"Input tidak ditemukan: {args.input}")

    if not files:
        sys.exit(f"Tidak ada berkas citra yang valid di: {args.input}")

    print("=" * 68)
    print(" PRA-PEMROSESAN DESKEW DOKUMEN SERUTI BPS (SUB-DERAJAT PRESISI)")
    print("=" * 68)
    print(f"Input          : {args.input.resolve()}")
    print(f"Output         : {args.output.resolve() if not args.in_place else 'IN-PLACE'}")
    print(f"Total Berkas   : {len(files)} gambar")
    print(f"Max Angle      : ±{args.max_angle}°")
    print(f"Interpolasi    : {args.quality.upper()}")
    print(f"Parallel Worker: {args.workers}")
    print("-" * 68)

    t_start = time.perf_counter()
    results = []

    if len(files) == 1:
        res = process_single_image_deskew(files[0], args.output, max_angle=args.max_angle, quality=args.quality, in_place=args.in_place)
        results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_image_deskew, f, args.output, args.max_angle, args.quality, args.in_place): f
                for f in files
            }
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                    status_str = f"Angle={res['skew_angle_deg']:+6.3f}°" if res.get('was_rotated') else "Tegak (0.000°)"
                    gain_str = f"Gain={res['sharpness_gain']:4.2f}x" if res.get('was_rotated') else "Preserved"
                    print(f"  [+] {res['file']:<28} | {status_str} | {gain_str} ({res['elapsed_sec']*1000:5.1f} ms)")
                except Exception as e:
                    f = futures[fut]
                    print(f"  [-] {f.name:<28} | ERROR: {str(e)}")

    total_time = time.perf_counter() - t_start
    rotated_count = sum(1 for r in results if r.get("was_rotated"))

    # Simpan ringkasan JSON
    summary_path = (args.output if not args.in_place else args.input.parent) / "deskew_summary.json"
    summary_path.write_text(json.dumps({
        "timestamp": time.time(),
        "total_files": len(files),
        "rotated_count": rotated_count,
        "elapsed_total_sec": round(total_time, 2),
        "results": results
    }, indent=2), encoding="utf-8")

    print("-" * 68)
    print("RINGKASAN PRA-PEMROSESAN DESKEW:")
    print(f"  - Total Berkas Diproses : {len(files)}")
    print(f"  - Berkas Terkoreksi     : {rotated_count}/{len(files)}")
    print(f"  - Total Durasi          : {total_time:.2f} detik (rerata {total_time/max(1, len(files)):.3f} s/berkas)")
    print(f"  - Ringkasan JSON        : {summary_path.name}")
    print(f"[OK] Seluruh berkas hasil deskew selesai diproses.")


if __name__ == "__main__":
    main()
