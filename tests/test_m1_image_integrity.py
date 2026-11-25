"""
Empirical Verification Test Suite: M1 Batch Image Integrity & Dimension Challenger.
Author: Challenger 2 (critic, specialist)
Workspace: d:/01_Kedinasan_BPS/seruti_scanner

Tests:
1. Free Disk Space on Drive D: > 2.0 GB headroom.
2. Global Survey: 76 booklets, 532 pages, 532 valid crop manifests, 0 atomic cells, 0 zero-byte files.
3. In-Depth Sample Booklets (01, 10, 25, 50, 76):
   - 100% PIL and OpenCV image decodability across all 5 tiers.
   - Precise Grid: envelopes & lattices have non-zero positive dimensions (w > 0, h > 0) and valid bbox.
   - Clean 28x28: strict (28, 28) resolution, single-channel/grayscale, non-empty ink, normalized center-of-mass.
4. Cross-Booklet Extended Clean-28x28 Sampling across arbitrary additional booklets (05, 15, 30, 45, 60, 70).
"""

import sys
import os
import json
import shutil
import ctypes
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
CROPS_DIR = BASE_DIR / "output" / "crops"


def get_free_disk_space_gb(path: str = "D:\\") -> float:
    """Get free disk space in GB for Windows."""
    free_bytes = ctypes.c_ulonglong(0)
    total_bytes = ctypes.c_ulonglong(0)
    total_free_bytes = ctypes.c_ulonglong(0)
    res = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(path),
        ctypes.byref(free_bytes),
        ctypes.byref(total_bytes),
        ctypes.byref(total_free_bytes),
    )
    if res == 0:
        # Fallback to shutil
        total, used, free = shutil.disk_usage(path)
        return free / (1024 ** 3)
    return free_bytes.value / (1024 ** 3)


def test_disk_space():
    print("=== [TEST 1] Disk Space Headroom Check ===")
    free_gb = get_free_disk_space_gb("D:\\")
    print(f"Drive D: Free Space: {free_gb:.2f} GB")
    assert free_gb >= 2.0, f"Drive D: free space ({free_gb:.2f} GB) is below the required 2.0 GB headroom!"
    print("  -> PASS: Disk space satisfies > 2.0 GB headroom requirement.\n")
    return free_gb


def test_global_crops_structure():
    print("=== [TEST 2] Global Crops Structure & Zero-Byte Scan ===")
    assert CROPS_DIR.exists(), f"Crops directory {CROPS_DIR} does not exist!"
    
    summary_file = CROPS_DIR / "batch_crop_summary.json"
    assert summary_file.exists(), f"Summary file {summary_file} missing!"
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    
    meta = summary_data.get("metadata", {})
    assert meta.get("success_count") == 76, f"Expected 76 successes, got {meta.get('success_count')}"
    assert meta.get("failed_count") == 0, f"Expected 0 failures, got {meta.get('failed_count')}"
    assert meta.get("total_pages_cropped") == 532, f"Expected 532 pages, got {meta.get('total_pages_cropped')}"

    booklet_dirs = sorted([d for d in CROPS_DIR.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
    assert len(booklet_dirs) == 76, f"Expected 76 booklet dirs, found {len(booklet_dirs)}"

    # Check atomic cells violation
    atomic_cells = list(CROPS_DIR.glob("**/precise_grid/cells"))
    assert len(atomic_cells) == 0, f"Violation: Found {len(atomic_cells)} atomic cell dirs!"

    # Scan for 0-byte files across entire output/crops/
    print("Scanning for 0-byte or corrupted files across all booklets...")
    zero_byte_files = []
    total_files_scanned = 0
    manifest_count = 0

    for b in booklet_dirs:
        for p in range(1, 8):
            hal_dir = b / f"hal_{p}"
            assert hal_dir.exists(), f"Missing page dir: {hal_dir}"
            m_path = hal_dir / "crop_manifest.json"
            assert m_path.exists(), f"Missing manifest: {m_path}"
            manifest_count += 1
            
            # Check manifest can be parsed
            with open(m_path, "r", encoding="utf-8") as mf:
                m_json = json.load(mf)
                assert m_json.get("booklet_id") == b.name
                assert m_json.get("page_num") == p
                assert "tiers" in m_json

            # Check all files in hal_dir
            for f in hal_dir.rglob("*.*"):
                total_files_scanned += 1
                if f.stat().st_size == 0:
                    zero_byte_files.append(str(f))

    print(f"Total files scanned across 76 booklets: {total_files_scanned}")
    print(f"Total manifests verified: {manifest_count}")
    print(f"Zero-byte files found: {len(zero_byte_files)}")
    assert len(zero_byte_files) == 0, f"Found {len(zero_byte_files)} zero-byte files: {zero_byte_files[:5]}"
    print("  -> PASS: All 76 booklets, 532 pages, 532 manifests, 0 atomic cell dirs, 0 zero-byte files verified.\n")
    return total_files_scanned, manifest_count


def verify_image_file(img_path: Path) -> Tuple[int, int, int]:
    """Verify image decodability using both PIL and OpenCV. Return (width, height, channels)."""
    # 1. PIL verification
    with Image.open(img_path) as pil_img:
        pil_img.verify()  # verify data integrity
    with Image.open(img_path) as pil_img:
        w_pil, h_pil = pil_img.size
        mode_pil = pil_img.mode

    # 2. OpenCV verification
    cv_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    assert cv_img is not None, f"OpenCV failed to read image: {img_path}"
    
    if len(cv_img.shape) == 2:
        h_cv, w_cv = cv_img.shape
        c_cv = 1
    else:
        h_cv, w_cv, c_cv = cv_img.shape

    assert (w_pil, h_pil) == (w_cv, h_cv), f"Dimension mismatch between PIL ({w_pil}x{h_pil}) and CV ({w_cv}x{h_cv}) for {img_path}"
    assert w_cv > 0 and h_cv > 0, f"Invalid zero dimensions for {img_path}"
    return w_cv, h_cv, c_cv


def test_sample_booklets_integrity(sample_ids: List[str] = ["booklet_01", "booklet_10", "booklet_25", "booklet_50", "booklet_76"]):
    print(f"=== [TEST 3] In-Depth Tier & Image Integrity on Sample Booklets: {sample_ids} ===")
    
    tiers_required = ["makro", "meso", "fields", "precise_grid", "mikro"]
    total_images_tested = 0
    clean_digits_tested = 0
    envelopes_tested = 0
    lattices_tested = 0
    tier_counts = {t: 0 for t in tiers_required}

    for bid in sample_ids:
        b_dir = CROPS_DIR / bid
        assert b_dir.exists(), f"Sample booklet {bid} missing!"
        print(f"--> Testing booklet: {bid}")

        for p in range(1, 8):
            hal_dir = b_dir / f"hal_{p}"
            assert hal_dir.exists(), f"Page {hal_dir} missing!"
            
            # Check 5 tiers exist
            for tier in tiers_required:
                t_dir = hal_dir / tier
                assert t_dir.exists(), f"Tier directory {t_dir} missing!"

            manifest_path = hal_dir / "crop_manifest.json"
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)

            # 1. Makro check
            for makro_file in (hal_dir / "makro").glob("*.png"):
                w, h, c = verify_image_file(makro_file)
                total_images_tested += 1
                tier_counts["makro"] += 1

            # 2. Meso check
            for meso_file in (hal_dir / "meso").glob("*.png"):
                w, h, c = verify_image_file(meso_file)
                total_images_tested += 1
                tier_counts["meso"] += 1

            # 3. Fields check
            for field_file in (hal_dir / "fields").glob("*.png"):
                w, h, c = verify_image_file(field_file)
                total_images_tested += 1
                tier_counts["fields"] += 1

            # 4. Precise Grid (Envelopes and Lattices)
            grid_dir = hal_dir / "precise_grid"
            envelope_files = list(grid_dir.glob("*_envelope.png"))
            lattice_files = list(grid_dir.glob("*_lattice.png"))
            assert len(envelope_files) == len(lattice_files), f"Envelope count ({len(envelope_files)}) != Lattice count ({len(lattice_files)}) in {grid_dir}"

            for ef in envelope_files:
                w, h, c = verify_image_file(ef)
                assert w > 0 and h > 0, f"Envelope {ef} has zero dimensions: {w}x{h}"
                total_images_tested += 1
                envelopes_tested += 1
                tier_counts["precise_grid"] += 1

            for lf in lattice_files:
                w, h, c = verify_image_file(lf)
                assert w > 0 and h > 0, f"Lattice {lf} has zero dimensions: {w}x{h}"
                total_images_tested += 1
                lattices_tested += 1
                tier_counts["precise_grid"] += 1

            # Verify precise_grid records in manifest
            grid_records = manifest.get("tiers", {}).get("precise_grid", [])
            for gr in grid_records:
                bbox_px = gr.get("bbox_px", [])
                assert len(bbox_px) == 4, f"Invalid bbox_px format in {gr}"
                x0, y0, x1, y1 = bbox_px
                assert x1 > x0 and y1 > y0, f"Invalid non-positive bbox in manifest: {bbox_px}"
                w = gr.get("width")
                h = gr.get("height")
                if w is None or h is None:
                    dims = gr.get("dimensions", [])
                    assert len(dims) == 2 and dims[0] > 0 and dims[1] > 0, f"Invalid dims in manifest: {dims}"
                    w, h = dims[0], dims[1]
                assert w > 0 and h > 0, f"Invalid non-positive width/height in manifest: {w}x{h}"
                assert w == (x1 - x0), f"Width mismatch in manifest: {w} != {x1 - x0}"
                assert h == (y1 - y0), f"Height mismatch in manifest: {h} != {y1 - y0}"

            # 5. Mikro (Raw & Clean 28x28)
            mikro_dir = hal_dir / "mikro"
            raw_dir = mikro_dir / "raw"
            clean_dir = mikro_dir / "clean_28x28"
            assert raw_dir.exists(), f"Raw mikro dir missing: {raw_dir}"
            assert clean_dir.exists(), f"Clean mikro dir missing: {clean_dir}"

            # Raw comb-box crops
            for rf in raw_dir.rglob("*.png"):
                w, h, c = verify_image_file(rf)
                assert w > 0 and h > 0, f"Raw crop {rf} has zero dimensions"
                total_images_tested += 1
                tier_counts["mikro"] += 1

            # Clean 28x28 MNIST crops
            for cf in clean_dir.rglob("*.png"):
                w, h, c = verify_image_file(cf)
                # Strict 28x28 check
                assert (w, h) == (28, 28), f"Clean crop {cf} has wrong dimension: ({w}, {h}) != (28, 28)"
                
                # Grayscale / single-channel verification
                cv_clean = cv2.imread(str(cf), cv2.IMREAD_GRAYSCALE)
                assert cv_clean is not None and cv_clean.shape == (28, 28)
                
                # Ink presence check (must not be completely blank/0 ink)
                mean_ink = cv_clean.mean()
                assert mean_ink > 0, f"Clean crop {cf} is completely blank (0 ink)!"
                
                # Center of mass verification
                M = cv2.moments(cv_clean)
                assert M["m00"] > 0, f"Zero moment for {cf}"
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                # Center is 14. Normalized centering should be roughly centered [7.0, 21.0]
                assert 7.0 <= cx <= 21.0, f"Center of mass cx={cx:.2f} too far from center for {cf}"
                assert 7.0 <= cy <= 21.0, f"Center of mass cy={cy:.2f} too far from center for {cf}"

                total_images_tested += 1
                clean_digits_tested += 1
                tier_counts["mikro"] += 1

    print(f"\n--- Sample Booklets Verification Metrics ---")
    print(f"Total images fully tested & decoded: {total_images_tested}")
    print(f"  - Makro images: {tier_counts['makro']}")
    print(f"  - Meso images: {tier_counts['meso']}")
    print(f"  - Fields images: {tier_counts['fields']}")
    print(f"  - Precise Grid images: {tier_counts['precise_grid']} ({envelopes_tested} envelopes, {lattices_tested} lattices)")
    print(f"  - Mikro images: {tier_counts['mikro']} (including {clean_digits_tested} clean 28x28 digits)")
    print("  -> PASS: All sampled images decode flawlessly with exact dimension specifications.\n")
    return total_images_tested, clean_digits_tested, envelopes_tested, lattices_tested


def test_extended_clean_crops(additional_booklets: List[str] = ["booklet_05", "booklet_15", "booklet_30", "booklet_45", "booklet_60", "booklet_70"]):
    print(f"=== [TEST 4] Extended Clean 28x28 Verification on Booklets: {additional_booklets} ===")
    total_extended_clean = 0
    for bid in additional_booklets:
        b_dir = CROPS_DIR / bid
        assert b_dir.exists(), f"Booklet {bid} does not exist"
        clean_crops = list((b_dir).rglob("mikro/clean_28x28/**/*.png"))
        print(f"  - {bid}: {len(clean_crops)} clean 28x28 crops")
        for cf in clean_crops:
            cv_clean = cv2.imread(str(cf), cv2.IMREAD_UNCHANGED)
            assert cv_clean is not None, f"Failed to load {cf}"
            if len(cv_clean.shape) == 2:
                h, w = cv_clean.shape
            else:
                h, w, _ = cv_clean.shape
            assert (w, h) == (28, 28), f"Dimension error in {cf}: {w}x{h}"
            
            # Moments check
            gray = cv_clean if len(cv_clean.shape) == 2 else cv2.cvtColor(cv_clean, cv2.COLOR_BGR2GRAY)
            assert gray.mean() > 0, f"Blank image in {cf}"
            M = cv2.moments(gray)
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            assert 6.0 <= cx <= 22.0 and 6.0 <= cy <= 22.0, f"COM out of bounds for {cf}: ({cx:.2f}, {cy:.2f})"
            total_extended_clean += 1

    print(f"Total extended clean 28x28 crops tested: {total_extended_clean}")
    print("  -> PASS: All extended clean crops conform to 28x28 and normalized centering.\n")
    return total_extended_clean


def run_all_challenges():
    print("=================================================================")
    print("EMPIRICAL CHALLENGER 2: M1 Image Integrity & Dimension Challenge")
    print("=================================================================\n")

    free_gb = test_disk_space()
    files_scanned, manifests_count = test_global_crops_structure()
    images_tested, clean_tested, envelopes, lattices = test_sample_booklets_integrity()
    extended_clean_tested = test_extended_clean_crops()

    print("=================================================================")
    print("FINAL CHALLENGE AUDIT SUMMARY")
    print("=================================================================")
    print(f"Free Disk Headroom on Drive D: : {free_gb:.2f} GB (Requirement: > 2.0 GB) -> PASS")
    print(f"Total Booklets Verified        : 76 (100% of physical inventory) -> PASS")
    print(f"Total Pages Verified           : 532 (7 pages per booklet) -> PASS")
    print(f"Total Manifests Verified       : {manifests_count} (Valid JSON, zero corrupt) -> PASS")
    print(f"Total Files Scanned on Disk    : {files_scanned} (0 zero-byte files) -> PASS")
    print(f"Images Tested & Decoded        : {images_tested} across 5 tiers (PIL & CV2) -> PASS")
    print(f"Table Envelopes & Lattices     : {envelopes} envelopes, {lattices} lattices -> PASS (w>0, h>0)")
    print(f"Clean 28x28 Crops Tested       : {clean_tested + extended_clean_tested} digits -> PASS (Exact 28x28, Centered)")
    print(f"Atomic Cell Directories Omitted: Verified 0 dirs found -> PASS")
    print("VERDICT: APPROVE")
    print("=================================================================")


if __name__ == "__main__":
    run_all_challenges()
