"""
Fast deterministic audit of all 76 booklets in output/seruti_inti_booklets.
Checks:
1. Physical page dimensions (Portrait 2480x3507 vs Landscape 3507x2480).
2. Blank page anomalies (e.g. hal_1 or hal_4 being blank instead of hal_8).
3. Cover page placement: Is 'VSERUTI26.INTI' on hal_1? Or on hal_5/hal_8?
4. Landscape orientation: Is hal_5 upright (Blok V table header at top) or upside down?
5. Fast summary of all anomalous booklets.
"""
import json
import cv2
import numpy as np
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path("d:/01_Kedinasan_BPS/seruti_scanner")
OUTPUT_DIR = BASE_DIR / "output" / "seruti_inti_booklets"

ocr = RapidOCR()

def audit_booklet(b_dir: Path):
    b_id = b_dir.name
    mf_path = b_dir / "manifest.json"
    manifest = {}
    if mf_path.exists():
        try:
            manifest = json.loads(mf_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    feed_dir = manifest.get("feed_direction", "unknown")
    pages_dir = b_dir / "pages"

    anomalies = []
    page_stats = {}

    # 1. Periksa dimensi dan blank check untuk semua halaman
    for p_num in range(1, 9):
        p_id = f"hal_{p_num}"
        clean_path = pages_dir / f"{p_id}_clean.png"
        if not clean_path.exists():
            anomalies.append(f"{p_id}_MISSING")
            continue

        img = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            anomalies.append(f"{p_id}_UNREADABLE")
            continue

        h, w = img.shape[:2]
        std_val = float(np.std(img))
        is_blank = std_val < 12.0

        # Ekspektasi orientasi
        expected_landscape = (p_num in (5, 6, 7))
        is_landscape = (w > h)

        if is_landscape != expected_landscape:
            anomalies.append(f"{p_id}_DIMENSION_MISMATCH(got_{w}x{h}_exp_{'land' if expected_landscape else 'port'})")

        if p_num == 8:
            if not is_blank:
                # Hal 8 biasanya blank, jika tidak blank catat
                page_stats[p_id] = {"std": round(std_val, 1), "blank": False, "w": w, "h": h}
            else:
                page_stats[p_id] = {"std": round(std_val, 1), "blank": True, "w": w, "h": h}
        else:
            if is_blank:
                anomalies.append(f"{p_id}_UNEXPECTED_BLANK(std={std_val:.1f})")
            page_stats[p_id] = {"std": round(std_val, 1), "blank": is_blank, "w": w, "h": h}

    # 2. Verifikasi posisi Cover Depan ('VSERUTI26.INTI')
    # Cek hal_1 top
    hal_1_path = pages_dir / "hal_1_clean.png"
    cover_loc = "unknown"
    if hal_1_path.exists():
        img1 = cv2.imread(str(hal_1_path), cv2.IMREAD_GRAYSCALE)
        if img1 is not None and float(np.std(img1)) >= 12.0:
            h1, w1 = img1.shape[:2]
            top1 = img1[:int(0.25 * h1), :]
            top1_s = cv2.resize(top1, (800, int(top1.shape[0] * 800 / top1.shape[1])))
            res1_t, _ = ocr(top1_s)
            txt1_t = " ".join(r[1] for r in res1_t).upper() if res1_t else ""

            if any(k in txt1_t for k in ["VSERUTI", "SURVEI", "REPUBLIK"]):
                cover_loc = "hal_1_top_OK"
            else:
                # Cek apakah hal_1 upside down (180 deg)
                bot1 = img1[int(0.75 * h1):, :]
                bot1_s = cv2.resize(bot1, (800, int(bot1.shape[0] * 800 / bot1.shape[1])))
                res1_b, _ = ocr(bot1_s)
                txt1_b = " ".join(r[1] for r in res1_b).upper() if res1_b else ""
                if any(k in txt1_b for k in ["VSERUTI", "SURVEI", "REPUBLIK"]):
                    cover_loc = "hal_1_UPSIDE_DOWN"
                    anomalies.append("hal_1_UPSIDE_DOWN_180")
                else:
                    cover_loc = "hal_1_NO_COVER_TEXT"

    # Jika cover tidak di hal_1, cek hal_5 (kasus booklet terbalik reverse feed)
    if cover_loc in ("unknown", "hal_1_NO_COVER_TEXT"):
        hal_5_path = pages_dir / "hal_5_clean.png"
        if hal_5_path.exists():
            img5 = cv2.imread(str(hal_5_path), cv2.IMREAD_GRAYSCALE)
            if img5 is not None and float(np.std(img5)) >= 12.0:
                h5, w5 = img5.shape[:2]
                top5 = img5[:int(0.35 * h5), :]
                top5_s = cv2.resize(top5, (800, int(top5.shape[0] * 800 / top5.shape[1])))
                res5, _ = ocr(top5_s)
                txt5 = " ".join(r[1] for r in res5).upper() if res5 else ""
                if any(k in txt5 for k in ["VSERUTI", "SURVEI", "REPUBLIK"]):
                    cover_loc = "FOUND_IN_HAL_5(REVERSE_INVERTED)"
                    anomalies.append("SPREAD_INVERTED_COVER_IN_HAL_5")

    # 3. Verifikasi orientasi Landscape Halaman 5 (Blok V)
    land_status = "unknown"
    hal_5_path = pages_dir / "hal_5_clean.png"
    if hal_5_path.exists():
        img5 = cv2.imread(str(hal_5_path), cv2.IMREAD_GRAYSCALE)
        if img5 is not None and float(np.std(img5)) >= 12.0:
            h5, w5 = img5.shape[:2]
            if w5 > h5:  # Landscape
                top5 = img5[:int(0.30 * h5), :]
                top5_s = cv2.resize(top5, (800, int(top5.shape[0] * 800 / top5.shape[1])))
                res5_t, _ = ocr(top5_s)
                txt5_t = " ".join(r[1] for r in res5_t).upper() if res5_t else ""

                keywords = ["BLOK", "ILOK", "PENDAPATAN", "UPAH", "GAJI", "PENERIMAAN", "KOMODITAS"]
                has_blok_top = any(k in txt5_t for k in keywords)
                
                if has_blok_top:
                    land_status = "hal_5_UPRIGHT_OK"
                else:
                    bot5 = img5[int(0.70 * h5):, :]
                    bot5_s = cv2.resize(bot5, (800, int(bot5.shape[0] * 800 / bot5.shape[1])))
                    res5_b, _ = ocr(bot5_s)
                    txt5_b = " ".join(r[1] for r in res5_b).upper() if res5_b else ""
                    has_blok_bot = any(k in txt5_b for k in keywords)
                    if has_blok_bot:
                        land_status = "hal_5_UPSIDE_DOWN_270"
                        anomalies.append("hal_5_UPSIDE_DOWN_270")
                    else:
                        land_status = "hal_5_AMBIGUOUS_NO_KEYWORDS"

    return {
        "booklet_id": b_id,
        "feed_direction": feed_dir,
        "cover_location": cover_loc,
        "landscape_status": land_status,
        "anomalies": anomalies,
        "is_ok": (len(anomalies) == 0)
    }

def main():
    b_dirs = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
    print(f"Memulai fast audit pada {len(b_dirs)} buklet...", flush=True)

    results = []
    anomalous_booklets = []

    for idx, b_dir in enumerate(b_dirs, 1):
        r = audit_booklet(b_dir)
        results.append(r)
        if not r["is_ok"]:
            anomalous_booklets.append(r)
            print(f"  [ANOMALI] {r['booklet_id']} (feed={r['feed_direction']}): {r['anomalies']} | Cover={r['cover_location']} | Land={r['landscape_status']}", flush=True)
        else:
            if idx % 5 == 0 or idx == len(b_dirs):
                print(f"  [OK] Progres: {idx}/{len(b_dirs)} buklet diperiksa...", flush=True)

    out_file = BASE_DIR / "output" / "fast_audit_report.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "="*60, flush=True)
    print(f"RINGKASAN AUDIT CEPAT:", flush=True)
    print(f"Total Buklet: {len(b_dirs)}", flush=True)
    print(f"Buklet Lulus (Normal): {len(b_dirs) - len(anomalous_booklets)}", flush=True)
    print(f"Buklet Bermasalah: {len(anomalous_booklets)}", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    main()
