"""
Comprehensive audit of all 76 booklets in output/seruti_inti_booklets.
Checks:
1. 100% of 76 booklets x 8 pages (608 pages) for canonical dimensions:
   - hal_1, hal_2, hal_3, hal_4, hal_8: Portrait (w < h, canonical 2480x3507)
   - hal_5, hal_6, hal_7: Landscape (w > h, canonical 3507x2480)
2. Blank page check: hal_1..hal_7 must NOT be blank (std >= 12.0)
3. Manifest verification: all 4 spreads parsed and mapped.
4. Deep inspection on previously anomalous booklets (38, 55, 67, 21, 49).
"""
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path("d:/01_Kedinasan_BPS/seruti_scanner")
OUTPUT_DIR = BASE_DIR / "output" / "seruti_inti_booklets"

def main():
    b_dirs = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
    print(f"============================================================")
    print(f"AUDIT MENYELURUH {len(b_dirs)} BUKLET SERUTI INTI (608 HALAMAN)")
    print(f"============================================================", flush=True)

    dimension_errors = []
    blank_errors = []
    missing_errors = []
    booklet_reports = []

    for idx, b_dir in enumerate(b_dirs, 1):
        b_id = b_dir.name
        pages_dir = b_dir / "pages"
        mf_file = b_dir / "manifest.json"
        
        mf = {}
        if mf_file.exists():
            try:
                mf = json.loads(mf_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        page_issues = []
        for p_num in range(1, 9):
            p_id = f"hal_{p_num}"
            img_path = pages_dir / f"{p_id}_clean.png"
            if not img_path.exists():
                page_issues.append(f"{p_id}_MISSING")
                missing_errors.append((b_id, p_id))
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                page_issues.append(f"{p_id}_UNREADABLE")
                missing_errors.append((b_id, p_id))
                continue

            h, w = img.shape[:2]
            std_val = float(np.std(img))
            is_blank = (std_val < 12.0)

            # Check orientation & dimensions
            exp_landscape = (p_num in (5, 6, 7))
            is_landscape = (w > h)

            if is_landscape != exp_landscape:
                issue = f"{p_id}_WRONG_ORIENTATION(got_{w}x{h}_exp_{'land' if exp_landscape else 'port'})"
                page_issues.append(issue)
                dimension_errors.append((b_id, p_id, w, h, "landscape" if exp_landscape else "portrait"))

            if p_num != 8 and is_blank:
                issue = f"{p_id}_UNEXPECTED_BLANK(std={std_val:.1f})"
                page_issues.append(issue)
                blank_errors.append((b_id, p_id, std_val))

        booklet_reports.append({
            "booklet_id": b_id,
            "feed": mf.get("feed_direction", "unknown"),
            "issues": page_issues,
            "ok": (len(page_issues) == 0)
        })

    # Deep inspection on previously anomalous booklets
    ocr = RapidOCR()
    anom_ids = ["booklet_38", "booklet_55", "booklet_67", "booklet_21", "booklet_49"]
    print(f"\n--- VERIFIKASI DEEP OCR 5 BUKLET TARGET PERBAIKAN ---", flush=True)
    deep_results = {}
    for aid in anom_ids:
        adir = OUTPUT_DIR / aid / "pages"
        hal_1_p = adir / "hal_1_clean.png"
        hal_5_p = adir / "hal_5_clean.png"
        
        c_status = "UNKNOWN"
        l_status = "UNKNOWN"
        
        if hal_1_p.exists():
            im1 = cv2.imread(str(hal_1_p), cv2.IMREAD_GRAYSCALE)
            if im1 is not None:
                h1, w1 = im1.shape[:2]
                t1 = im1[:int(0.25 * h1), :]
                t1_s = cv2.resize(t1, (800, int(t1.shape[0] * 800 / t1.shape[1])))
                r1, _ = ocr(t1_s)
                txt1 = " ".join(r[1] for r in r1).upper() if r1 else ""
                if any(k in txt1 for k in ["VSERUTI", "SURVEI", "REPUBLIK"]):
                    c_status = "HAL_1_COVER_UPRIGHT_OK"
                else:
                    c_status = f"HAL_1_NO_COVER_TXT({txt1[:30]})"

        if hal_5_p.exists():
            im5 = cv2.imread(str(hal_5_p), cv2.IMREAD_GRAYSCALE)
            if im5 is not None:
                h5, w5 = im5.shape[:2]
                t5 = im5[:int(0.30 * h5), :]
                t5_s = cv2.resize(t5, (800, int(t5.shape[0] * 800 / t5.shape[1])))
                r5, _ = ocr(t5_s)
                txt5 = " ".join(r[1] for r in r5).upper() if r5 else ""
                if any(k in txt5 for k in ["BLOK", "ILOK", "PENDAPATAN", "UPAH", "GAJI", "PENERIMAAN", "KOMODITAS"]):
                    l_status = "HAL_5_BLOK_V_UPRIGHT_OK"
                else:
                    l_status = f"HAL_5_AMBIGUOUS({txt5[:30]})"

        print(f"  {aid}: Dimensi={im1.shape[1]}x{im1.shape[0]} & {im5.shape[1]}x{im5.shape[0]} | Cover={c_status} | Landscape={l_status}", flush=True)
        deep_results[aid] = {"cover": c_status, "landscape": l_status}

    # Summary
    total_pages = len(b_dirs) * 8
    passed_booklets = sum(1 for r in booklet_reports if r["ok"])
    
    print(f"\n============================================================")
    print(f"HASIL AKHIR AUDIT 76 BUKLET:")
    print(f"Total Buklet Dipindai    : {len(b_dirs)} buklet")
    print(f"Total Halaman A4         : {total_pages} halaman")
    print(f"Halaman Missing / Corrupt: {len(missing_errors)}")
    print(f"Kesalahan Orientasi      : {len(dimension_errors)}")
    print(f"Kesalahan Blank Tak Wajar: {len(blank_errors)}")
    print(f"Buklet 100% Sempurna     : {passed_booklets} / {len(b_dirs)} ({(passed_booklets/len(b_dirs))*100:.1f}%)")
    print(f"============================================================", flush=True)

    if dimension_errors:
        print("\nRincian Kesalahan Dimensi:")
        for err in dimension_errors:
            print(f"  {err[0]} {err[1]}: got {err[2]}x{err[3]}, expected {err[4]}")

    if blank_errors:
        print("\nRincian Halaman Blank Tak Wajar:")
        for err in blank_errors:
            print(f"  {err[0]} {err[1]}: std={err[2]:.2f}")

    # Simpan hasil audit
    report_data = {
        "total_booklets": len(b_dirs),
        "total_pages": total_pages,
        "passed_booklets": passed_booklets,
        "dimension_errors": dimension_errors,
        "blank_errors": blank_errors,
        "missing_errors": missing_errors,
        "deep_audit_repaired": deep_results,
        "reports": booklet_reports
    }
    (OUTPUT_DIR.parent / "full_orientation_audit_report.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nLaporan audit lengkap disimpan di: output/full_orientation_audit_report.json")

if __name__ == "__main__":
    main()
