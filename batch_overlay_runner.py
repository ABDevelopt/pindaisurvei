import sys, os, time, json
sys.path.insert(0, r'd:\01_Kedinasan_BPS\seruti_scanner')
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from render_scan_overlays import (
    align_template_to_image,
    render_scan_grid_overlay,
    render_scan_template_overlay,
    render_scan_subblock_overlay,
    render_scan_block_overlay
)

def process_single_booklet(booklet_id: str, image_type: str = "clean", jpeg_quality: int = 75, target_pages: list = None) -> dict:
    """Memproses halaman kuesioner untuk 1 buklet dan mengembalikan metrik robustness."""
    start_time = time.time()
    booklet_dir = Path("output/seruti_inti_booklets") / booklet_id
    pages_dir = booklet_dir / "pages"
    out_dir = Path("output/scan_overlays") / booklet_id / image_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pages_stats = []
    total_snapped = 0
    PAGE_EXPECTED = {1: 54, 2: 216, 3: 350, 4: 231, 5: 516, 6: 411, 7: 295}
    has_failure = False
    
    if target_pages is None:
        pages_to_process = list(range(1, 8))
        total_expected_digits = 2073
    else:
        pages_to_process = target_pages
        total_expected_digits = sum(PAGE_EXPECTED.get(p, 300) for p in pages_to_process)

    for p in pages_to_process:
        clean_file = pages_dir / f"hal_{p}_clean.png"
        if not clean_file.exists():
            has_failure = True
            pages_stats.append({"page": p, "status": "FILE_NOT_FOUND"})
            continue
            
        clean_img = cv2.imread(str(clean_file))
        H, W = clean_img.shape[:2]
        
        try:
            # 1. Aligment & Snapping
            alignment = align_template_to_image(clean_img, page_num=p)
            snapped_digits = alignment.enable_physical_cell_snapping(clean_img)
            total_snapped += snapped_digits
            
            # 2. Render 4 overlays
            grid_overlay = render_scan_grid_overlay(clean_img, alignment, booklet_id, p, image_type)
            full_overlay = render_scan_template_overlay(clean_img, alignment, booklet_id, p, image_type)
            sub_overlay = render_scan_subblock_overlay(clean_img, alignment, booklet_id, p, image_type)
            blk_overlay = render_scan_block_overlay(clean_img, alignment, booklet_id, p, image_type)
            
            # 3. Save JPEGs (optimized quality to save disk space)
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            cv2.imwrite(str(out_dir / f"hal_{p}_scan_grid_overlay.jpg"), grid_overlay, encode_param)
            cv2.imwrite(str(out_dir / f"hal_{p}_scan_template_overlay.jpg"), full_overlay, encode_param)
            cv2.imwrite(str(out_dir / f"hal_{p}_scan_subblock_overlay.jpg"), sub_overlay, encode_param)
            cv2.imwrite(str(out_dir / f"hal_{p}_scan_block_overlay.jpg"), blk_overlay, encode_param)
            
            sx = round(alignment.scale_x, 4)
            sy = round(alignment.scale_y, 4)
            aspect_dev = round(abs(sy - sx) / sx * 100.0, 2)
            phys_count = len(alignment.physical_cells or [])
            
            pages_stats.append({
                "page": p,
                "status": "PASS" if snapped_digits > 30 or (p == 1 and snapped_digits >= 20) else "WARN",
                "dims": [W, H],
                "scale_x": sx,
                "scale_y": sy,
                "aspect_dev_pct": aspect_dev,
                "physical_cells": phys_count,
                "snapped_digits": snapped_digits
            })
        except Exception as e:
            has_failure = True
            pages_stats.append({"page": p, "status": "ERROR", "error": str(e)})
            
    elapsed = round(time.time() - start_time, 2)
    snapped_pct = round((total_snapped / float(total_expected_digits)) * 100.0, 2)
    
    result = {
        "booklet_id": booklet_id,
        "status": "PASS" if (not has_failure and snapped_pct >= 95.0) else ("WARN" if snapped_pct >= 85.0 else "FAIL"),
        "total_snapped_digits": total_snapped,
        "snapped_pct": snapped_pct,
        "elapsed_sec": elapsed,
        "pages": pages_stats
    }
    
    # Simpan booklet summary JSON
    with open(out_dir / "scan_overlay_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch Runner Overlaying Seruti Inti")
    parser.add_argument("--limit", type=int, default=None, help="Maksimal jumlah buklet yang diproses")
    parser.add_argument("--workers", type=int, default=4, help="Jumlah parallel workers")
    parser.add_argument("--pages", nargs="+", type=int, default=None, help="Halaman spesifik yang ingin diproses (misal: --pages 3)")
    args = parser.parse_args()
    
    from seruti_pipeline.stage_gate_validator import DUPLICATE_BOOKLET_IDS, REJECTED_BOOKLET_IDS

    booklets_base = Path("output/seruti_inti_booklets")
    all_bids = sorted([
        d.name for d in booklets_base.iterdir()
        if d.is_dir() and d.name.startswith("booklet_")
        and d.name not in DUPLICATE_BOOKLET_IDS
        and d.name not in REJECTED_BOOKLET_IDS
    ])
    if args.limit:
        all_bids = all_bids[:args.limit]
        
    print(f"\n{'='*80}")
    print(f"  SERUTI INTI 2026 — BATCH OVERLAYING ROBUSTNESS RUNNER")
    print(f"  Target Buklet Bersih: {len(all_bids)} buklet (Duplikat & Cacat Pindaian Diisolasi)")
    print(f"  Parallel Mode : {args.workers} workers (ProcessPoolExecutor)")
    print(f"  Output Base   : output/scan_overlays/<booklet_id>/clean")
    print(f"{'='*80}\n", flush=True)
    
    start_total = time.time()
    batch_results = []
    
    # Jalankan secara paralel dengan kualitas JPEG 85 (visual bersih & tajam)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_b = {executor.submit(process_single_booklet, b, "clean", 85, args.pages): b for b in all_bids}
        done_count = 0
        for future in as_completed(future_to_b):
            b_id = future_to_b[future]
            try:
                res = future.result()
                batch_results.append(res)
                done_count += 1
                status_badge = f"[{res['status']}]"
                print(f"[{done_count:02d}/{len(all_bids):02d}] {res['booklet_id']}: {res['total_snapped_digits']}/2073 ({res['snapped_pct']}%) | Waktu: {res['elapsed_sec']}s {status_badge}", flush=True)
            except Exception as exc:
                print(f"[{done_count:02d}/{len(all_bids):02d}] {b_id} FAILED with exception: {exc}", flush=True)
                
    total_elapsed = round(time.time() - start_total, 2)
    # Sort results by booklet_id
    batch_results.sort(key=lambda r: r["booklet_id"])
    
    # Simpan batch summary
    summary_path = Path("output/scan_overlays/batch_robustness_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_booklets": len(batch_results),
            "total_elapsed_sec": total_elapsed,
            "avg_time_per_booklet_sec": round(total_elapsed / max(1, len(batch_results)), 2),
            "pass_count": sum(1 for r in batch_results if r["status"] == "PASS"),
            "warn_count": sum(1 for r in batch_results if r["status"] == "WARN"),
            "fail_count": sum(1 for r in batch_results if r["status"] == "FAIL"),
            "booklets": batch_results
        }, f, indent=2)
        
    print(f"\n{'='*80}")
    print(f"  BATCH COMPLETED in {total_elapsed}s ({total_elapsed/60:.1f} menit)!")
    print(f"  Total Buklet: {len(batch_results)} | PASS: {sum(1 for r in batch_results if r['status'] == 'PASS')} | WARN: {sum(1 for r in batch_results if r['status'] == 'WARN')} | FAIL: {sum(1 for r in batch_results if r['status'] == 'FAIL')}")
    print(f"  Summary tersimpan ke: {summary_path}")
    print(f"{'='*80}\n")
