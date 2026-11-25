"""
Comprehensive empirical verification script for Seruti Scanner Milestone M1.
Challenger 1: M1 Manifest & Directory Verification Challenger.

This script independently verifies:
1. Exactly 76 booklets (booklet_01 through booklet_76).
2. Exactly 532 pages (hal_1 through hal_7 for each booklet).
3. JSON validity, non-empty, non-truncated status of all 532 crop_manifest.json files.
4. Schema conformance of all 532 crop_manifest.json files (keys, types, booklet_id, page_num).
5. Internal consistency (manifest summary counts match length of tier lists).
6. File existence and non-zero size for 100% of referenced image files across all tiers:
   - makro
   - meso
   - fields
   - precise_grid (envelope_file, lattice_file)
   - mikro (raw_file, and clean_file when not blank)
   - preview_file (if present)
7. Absence of leaked atomic grid cells (precise_grid/cells/).
8. Page 8 absence/presence verification.
9. Verification of batch_crop_summary.json against empirical aggregation.
10. Detection of any 0-byte files in output/crops/.
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict


def run_manifest_verification(base_dir="output/crops", source_dir="output/seruti_inti_booklets"):
    start_time = time.time()
    base_path = Path(base_dir)
    source_path = Path(source_dir)

    results = {
        "verified_booklets_count": 0,
        "verified_pages_count": 0,
        "manifest_files_checked": 0,
        "total_referenced_images": 0,
        "missing_files": [],
        "zero_byte_files": [],
        "schema_errors": [],
        "summary_count_mismatches": [],
        "unmatched_ids": [],
        "unexpected_page_dirs": [],
        "atomic_cells_dirs": [],
        "manifest_sizes_bytes": [],
        "tier_counts_aggregate": defaultdict(int),
        "anomalies_detected": 0
    }

    print(f"=== Starting Empirical Verification of {base_dir} ===")

    # 1. Verify booklet directories
    if not base_path.exists():
        print(f"FATAL: Base path {base_dir} does not exist!")
        sys.exit(1)

    booklet_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
    results["verified_booklets_count"] = len(booklet_dirs)
    print(f"Found {len(booklet_dirs)} booklet directories in {base_dir}.")

    expected_booklets = [f"booklet_{i:02d}" for i in range(1, 77)]
    actual_booklets = [d.name for d in booklet_dirs]

    if actual_booklets != expected_booklets:
        missing_b = set(expected_booklets) - set(actual_booklets)
        extra_b = set(actual_booklets) - set(expected_booklets)
        if missing_b:
            results["anomalies_detected"] += len(missing_b)
            print(f"ERROR: Missing expected booklets: {sorted(list(missing_b))}")
        if extra_b:
            print(f"INFO: Extra booklets found: {sorted(list(extra_b))}")
    else:
        print("PASS: Exactly 76 booklets found (booklet_01 through booklet_76).")

    # 2. Check for atomic cells directories
    atomic_cells = list(base_path.glob("**/precise_grid/cells"))
    results["atomic_cells_dirs"] = [str(p) for p in atomic_cells]
    if atomic_cells:
        results["anomalies_detected"] += len(atomic_cells)
        print(f"ERROR: Detected {len(atomic_cells)} atomic cells directories (suppression failed)!")
    else:
        print("PASS: Zero atomic cells directories detected. Disk safety confirmed.")

    # 3. Check for any 0-byte files in output/crops/
    print("Scanning for 0-byte files across output/crops/...")
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            fp = os.path.join(root, f)
            try:
                sz = os.path.getsize(fp)
                if sz == 0:
                    results["zero_byte_files"].append(fp)
                    results["anomalies_detected"] += 1
            except OSError as e:
                results["zero_byte_files"].append(f"{fp} (error: {e})")
                results["anomalies_detected"] += 1

    if results["zero_byte_files"]:
        print(f"ERROR: Found {len(results['zero_byte_files'])} zero-byte files!")
    else:
        print("PASS: Zero 0-byte files found in output/crops/.")

    # 4. Check each booklet and each page
    expected_pages = [f"hal_{p}" for p in range(1, 8)]

    for b_idx, b_dir in enumerate(booklet_dirs, 1):
        b_name = b_dir.name
        
        # Check subdirectories in booklet
        subdirs = [d.name for d in b_dir.iterdir() if d.is_dir()]
        
        # Check for unexpected page dirs (e.g. hal_8)
        extra_pages = [d for d in subdirs if d not in expected_pages]
        if extra_pages:
            results["unexpected_page_dirs"].append({"booklet": b_name, "extra_pages": extra_pages})
            print(f"WARNING: Booklet {b_name} has extra page directories: {extra_pages}")

        missing_pages = [d for d in expected_pages if d not in subdirs]
        if missing_pages:
            results["anomalies_detected"] += len(missing_pages)
            print(f"ERROR: Booklet {b_name} missing pages: {missing_pages}")
            continue

        for p_num in range(1, 8):
            page_dir = b_dir / f"hal_{p_num}"
            manifest_file = page_dir / "crop_manifest.json"
            
            if not manifest_file.exists():
                results["missing_files"].append(str(manifest_file))
                results["anomalies_detected"] += 1
                continue

            results["manifest_files_checked"] += 1
            sz = manifest_file.stat().st_size
            results["manifest_sizes_bytes"].append(sz)

            if sz == 0:
                results["zero_byte_files"].append(str(manifest_file))
                results["anomalies_detected"] += 1
                print(f"ERROR: 0-byte manifest: {manifest_file}")
                continue

            # Load JSON
            try:
                with open(manifest_file, "r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
            except Exception as e:
                results["schema_errors"].append(f"Corrupt JSON in {manifest_file}: {e}")
                results["anomalies_detected"] += 1
                continue

            # Schema checks
            required_top_keys = ["booklet_id", "page_num", "image_file", "dimensions", "scale", "translation", "snapped_digits", "summary", "tiers"]
            missing_top = [k for k in required_top_keys if k not in manifest]
            if missing_top:
                results["schema_errors"].append(f"{manifest_file} missing top keys: {missing_top}")
                results["anomalies_detected"] += 1

            if manifest.get("booklet_id") != b_name:
                results["unmatched_ids"].append(f"{manifest_file}: booklet_id {manifest.get('booklet_id')} != {b_name}")
                results["anomalies_detected"] += 1

            if manifest.get("page_num") != p_num:
                results["unmatched_ids"].append(f"{manifest_file}: page_num {manifest.get('page_num')} != {p_num}")
                results["anomalies_detected"] += 1

            # Check tiers existence
            tiers = manifest.get("tiers", {})
            required_tiers = ["makro", "meso", "fields", "precise_grid", "mikro"]
            missing_tiers = [t for t in required_tiers if t not in tiers]
            if missing_tiers:
                results["schema_errors"].append(f"{manifest_file} missing tiers: {missing_tiers}")
                results["anomalies_detected"] += 1

            # Summary checks
            summary = manifest.get("summary", {})
            makro_list = tiers.get("makro", [])
            meso_list = tiers.get("meso", [])
            grid_list = tiers.get("precise_grid", [])
            fields_list = tiers.get("fields", [])
            mikro_list = tiers.get("mikro", [])

            clean_mikro_count = sum(1 for m in mikro_list if m.get("clean_file") is not None)

            if summary.get("makro_count") != len(makro_list):
                results["summary_count_mismatches"].append(f"{manifest_file}: makro_count {summary.get('makro_count')} != {len(makro_list)}")
                results["anomalies_detected"] += 1

            if summary.get("meso_count") != len(meso_list):
                results["summary_count_mismatches"].append(f"{manifest_file}: meso_count {summary.get('meso_count')} != {len(meso_list)}")
                results["anomalies_detected"] += 1

            if summary.get("precise_grid_count") != len(grid_list):
                results["summary_count_mismatches"].append(f"{manifest_file}: precise_grid_count {summary.get('precise_grid_count')} != {len(grid_list)}")
                results["anomalies_detected"] += 1

            if summary.get("fields_count") != len(fields_list):
                results["summary_count_mismatches"].append(f"{manifest_file}: fields_count {summary.get('fields_count')} != {len(fields_list)}")
                results["anomalies_detected"] += 1

            if summary.get("mikro_boxes_count") != len(mikro_list):
                results["summary_count_mismatches"].append(f"{manifest_file}: mikro_boxes_count {summary.get('mikro_boxes_count')} != {len(mikro_list)}")
                results["anomalies_detected"] += 1

            if summary.get("mikro_clean_count") != clean_mikro_count:
                results["summary_count_mismatches"].append(f"{manifest_file}: mikro_clean_count {summary.get('mikro_clean_count')} != {clean_mikro_count}")
                results["anomalies_detected"] += 1

            # Aggregate stats
            results["tier_counts_aggregate"]["makro"] += len(makro_list)
            results["tier_counts_aggregate"]["meso"] += len(meso_list)
            results["tier_counts_aggregate"]["precise_grid"] += len(grid_list)
            results["tier_counts_aggregate"]["fields"] += len(fields_list)
            results["tier_counts_aggregate"]["mikro_boxes"] += len(mikro_list)
            results["tier_counts_aggregate"]["mikro_clean"] += clean_mikro_count
            results["tier_counts_aggregate"]["snapped_digits"] += manifest.get("snapped_digits", 0)

            # 5. Check all referenced files for existence and non-zero size
            # Makro
            for item in makro_list:
                fpath = item.get("file")
                if not fpath:
                    results["missing_files"].append(f"{manifest_file}: makro empty file path")
                    results["anomalies_detected"] += 1
                else:
                    results["total_referenced_images"] += 1
                    if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                        results["missing_files"].append(fpath)
                        results["anomalies_detected"] += 1

            # Meso
            for item in meso_list:
                fpath = item.get("file")
                if not fpath:
                    results["missing_files"].append(f"{manifest_file}: meso empty file path")
                    results["anomalies_detected"] += 1
                else:
                    results["total_referenced_images"] += 1
                    if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                        results["missing_files"].append(fpath)
                        results["anomalies_detected"] += 1

            # Fields
            for item in fields_list:
                fpath = item.get("file")
                if not fpath:
                    results["missing_files"].append(f"{manifest_file}: field empty file path")
                    results["anomalies_detected"] += 1
                else:
                    results["total_referenced_images"] += 1
                    if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                        results["missing_files"].append(fpath)
                        results["anomalies_detected"] += 1

            # Precise grid
            for item in grid_list:
                env_file = item.get("envelope_file")
                lat_file = item.get("lattice_file")
                if env_file:
                    results["total_referenced_images"] += 1
                    if not os.path.exists(env_file) or os.path.getsize(env_file) == 0:
                        results["missing_files"].append(env_file)
                        results["anomalies_detected"] += 1
                if lat_file:
                    results["total_referenced_images"] += 1
                    if not os.path.exists(lat_file) or os.path.getsize(lat_file) == 0:
                        results["missing_files"].append(lat_file)
                        results["anomalies_detected"] += 1

            # Mikro
            for item in mikro_list:
                raw_file = item.get("raw_file")
                clean_file = item.get("clean_file")
                if not raw_file:
                    results["missing_files"].append(f"{manifest_file}: mikro empty raw_file path")
                    results["anomalies_detected"] += 1
                else:
                    results["total_referenced_images"] += 1
                    if not os.path.exists(raw_file) or os.path.getsize(raw_file) == 0:
                        results["missing_files"].append(raw_file)
                        results["anomalies_detected"] += 1

                if clean_file:
                    results["total_referenced_images"] += 1
                    if not os.path.exists(clean_file) or os.path.getsize(clean_file) == 0:
                        results["missing_files"].append(clean_file)
                        results["anomalies_detected"] += 1

            # Optional preview_file
            prev_file = manifest.get("preview_file")
            if prev_file:
                results["total_referenced_images"] += 1
                if not os.path.exists(prev_file) or os.path.getsize(prev_file) == 0:
                    results["missing_files"].append(prev_file)
                    results["anomalies_detected"] += 1

            results["verified_pages_count"] += 1

        if b_idx % 15 == 0 or b_idx == len(booklet_dirs):
            print(f"Verified {b_idx}/{len(booklet_dirs)} booklets ({results['verified_pages_count']} pages, {results['total_referenced_images']} images checked)...")

    # 6. Compare with batch_crop_summary.json
    summary_path = base_path / "batch_crop_summary.json"
    batch_summary_match = True
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as sf:
            b_summary = json.load(sf)
        meta = b_summary.get("metadata", {})
        agg_summary = b_summary.get("aggregate_counts", {})
        agg = results["tier_counts_aggregate"]

        print("\n--- Cross-Check with batch_crop_summary.json ---")
        print(f"Total booklets: Manifest Sum = {results['verified_booklets_count']}, Summary = {meta.get('total_booklets')}")
        print(f"Total pages: Manifest Sum = {results['verified_pages_count']}, Summary = {meta.get('total_pages_cropped')}")
        print(f"Snapped digits: Manifest Sum = {agg['snapped_digits']}, Summary = {agg_summary.get('total_snapped_digits')}")
        print(f"Makro blocks: Manifest Sum = {agg['makro']}, Summary = {agg_summary.get('makro_blocks')}")
        print(f"Meso blocks: Manifest Sum = {agg['meso']}, Summary = {agg_summary.get('meso_sub_blocks')}")
        print(f"Precise grid tables: Manifest Sum = {agg['precise_grid']}, Summary = {agg_summary.get('precise_grid_tables')}")
        print(f"Fields: Manifest Sum = {agg['fields']}, Summary = {agg_summary.get('fields')}")
        print(f"Mikro raw boxes: Manifest Sum = {agg['mikro_boxes']}, Summary = {agg_summary.get('mikro_boxes')}")
        print(f"Mikro clean digits: Manifest Sum = {agg['mikro_clean']}, Summary = {agg_summary.get('mikro_clean_mnist')}")

        mismatches = []
        if results['verified_booklets_count'] != meta.get('total_booklets'):
            mismatches.append(f"total_booklets ({results['verified_booklets_count']} != {meta.get('total_booklets')})")
        if results['verified_pages_count'] != meta.get('total_pages_cropped'):
            mismatches.append(f"total_pages_cropped ({results['verified_pages_count']} != {meta.get('total_pages_cropped')})")
        if agg['snapped_digits'] != agg_summary.get('total_snapped_digits'):
            mismatches.append(f"total_snapped_digits ({agg['snapped_digits']} != {agg_summary.get('total_snapped_digits')})")
        if agg['makro'] != agg_summary.get('makro_blocks'):
            mismatches.append(f"makro_blocks ({agg['makro']} != {agg_summary.get('makro_blocks')})")
        if agg['meso'] != agg_summary.get('meso_sub_blocks'):
            mismatches.append(f"meso_sub_blocks ({agg['meso']} != {agg_summary.get('meso_sub_blocks')})")
        if agg['precise_grid'] != agg_summary.get('precise_grid_tables'):
            mismatches.append(f"precise_grid_tables ({agg['precise_grid']} != {agg_summary.get('precise_grid_tables')})")
        if agg['fields'] != agg_summary.get('fields'):
            mismatches.append(f"fields ({agg['fields']} != {agg_summary.get('fields')})")
        if agg['mikro_boxes'] != agg_summary.get('mikro_boxes'):
            mismatches.append(f"mikro_boxes ({agg['mikro_boxes']} != {agg_summary.get('mikro_boxes')})")
        if agg['mikro_clean'] != agg_summary.get('mikro_clean_mnist'):
            mismatches.append(f"mikro_clean_mnist ({agg['mikro_clean']} != {agg_summary.get('mikro_clean_mnist')})")

        if mismatches:
            print(f"ERROR: Discrepancy between empirical manifests and batch_crop_summary.json: {mismatches}")
            results["anomalies_detected"] += len(mismatches)
            batch_summary_match = False
        else:
            print("PASS: 100% exact match between empirical manifest sums and batch_crop_summary.json!")
    else:
        print("ERROR: batch_crop_summary.json not found!")
        results["anomalies_detected"] += 1
        batch_summary_match = False

    elapsed = time.time() - start_time
    print(f"\nVerification finished in {elapsed:.2f} seconds.")
    print(f"Summary:")
    print(f"- Total booklets verified: {results['verified_booklets_count']}/76")
    print(f"- Total pages verified: {results['verified_pages_count']}/532")
    print(f"- Total manifests checked: {results['manifest_files_checked']}/532")
    print(f"- Total referenced image files checked: {results['total_referenced_images']}")
    print(f"- Missing / 0-byte image files: {len(results['missing_files'])}")
    print(f"- 0-byte files in output/crops/: {len(results['zero_byte_files'])}")
    print(f"- Schema errors: {len(results['schema_errors'])}")
    print(f"- Manifest summary count mismatches: {len(results['summary_count_mismatches'])}")
    print(f"- Booklet/Page ID mismatches: {len(results['unmatched_ids'])}")
    print(f"- Atomic cell leaks: {len(results['atomic_cells_dirs'])}")
    print(f"- Total anomalies detected: {results['anomalies_detected']}")

    manifest_sizes = results["manifest_sizes_bytes"]
    if manifest_sizes:
        print(f"Manifest sizes: min={min(manifest_sizes)} bytes, max={max(manifest_sizes)} bytes, avg={sum(manifest_sizes)/len(manifest_sizes):.1f} bytes")

    verdict = "APPROVE" if results["anomalies_detected"] == 0 and results["verified_pages_count"] == 532 and results["verified_booklets_count"] == 76 else "REQUEST_CHANGES"
    print(f"\nVERDICT: {verdict}")

    return verdict, results


if __name__ == "__main__":
    verdict, res = run_manifest_verification()
    sys.exit(0 if verdict == "APPROVE" else 1)
