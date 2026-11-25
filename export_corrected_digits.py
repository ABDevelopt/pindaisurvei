#!/usr/bin/env python3
"""
export_corrected_digits.py
==========================
Mengekstrak dan mendokumentasikan secara rinci 649 sel digit yang diluruskan
oleh AI Antigravity dari tebakan model baseline BPS.

Keluaran:
1. dataset/eval/ai_corrected_649_digits.csv (Berkas CSV lengkap 649 baris)
2. dataset/eval/ai_corrected_649_digits.json (Berkas JSON terstruktur)
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EXTRACTIONS_DIR = BASE_DIR / "output" / "extractions"
CROPS_DIR = BASE_DIR / "output" / "crops"
GT_DIR = BASE_DIR / "dataset" / "ground_truth"
OUT_CSV = BASE_DIR / "dataset" / "eval" / "ai_corrected_649_digits.csv"
OUT_JSON = BASE_DIR / "dataset" / "eval" / "ai_corrected_649_digits.json"

def main():
    print("[*] Mengekstrak data 649 digit yang diluruskan oleh AI...")
    
    # 1. Muat seluruh ground truth yang sudah dikurasi AI
    curated_records = {}
    for gt_path in GT_DIR.glob("*.json"):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        b_id = gt["booklet_id"]
        # Simpan fields ground truth
        curated_records[b_id] = gt

    # 2. Muat hasil ekstraksi awal baseline dan bandingkan
    corrections = []

    for b_dir in sorted(EXTRACTIONS_DIR.iterdir()):
        if not b_dir.is_dir() or not b_dir.name.startswith("booklet_"):
            continue
        b_id = b_dir.name
        ext_path = b_dir / f"{b_id}_extracted.json"
        if not ext_path.exists():
            continue

        with open(ext_path, "r", encoding="utf-8") as f:
            ext = json.load(f)

        # Muat crop lookup untuk booklet ini
        crops_b_dir = CROPS_DIR / b_id
        image_lookup = {}
        if crops_b_dir.exists():
            for p_num in range(1, 8):
                m_path = crops_b_dir / f"hal_{p_num}" / "crop_manifest.json"
                if not m_path.exists():
                    continue
                try:
                    with open(m_path, "r", encoding="utf-8") as mf:
                        man = json.load(mf)
                    for item in man.get("tiers", {}).get("mikro", []):
                        fid = item.get("field_id")
                        b_idx = item.get("box_index")
                        cfile_rel = item.get("clean_file")
                        if fid and b_idx is not None and cfile_rel:
                            c_path = BASE_DIR / cfile_rel
                            if c_path.exists():
                                image_lookup[(p_num, fid, b_idx)] = str(cfile_rel)
                except Exception:
                    continue

        # Muat manifest split untuk tahu split dan arbitration reason
        # Kita baca dari manifest CSV
        pass

    # Lebih presisi: baca langsung dari 3 split manifest yang baru saja digenerasi oleh build_ai_ground_truth.py
    splits_dir = BASE_DIR / "dataset" / "splits"
    all_manifests = []
    for split_name in ["train", "val", "test"]:
        m_file = splits_dir / f"{split_name}_manifest.csv"
        if m_file.exists():
            df_m = pd.read_csv(m_file)
            df_m["split"] = split_name
            all_manifests.append(df_m)

    df_all = pd.concat(all_manifests, ignore_index=True)

    # Dapatkan baseline char dari extracted.json
    baseline_lookup = {}
    for b_dir in sorted(EXTRACTIONS_DIR.iterdir()):
        if not b_dir.is_dir() or not b_dir.name.startswith("booklet_"):
            continue
        b_id = b_dir.name
        ext_path = b_dir / f"{b_id}_extracted.json"
        if not ext_path.exists():
            continue
        with open(ext_path, "r", encoding="utf-8") as f:
            ext = json.load(f)
        pages = ext.get("pages", {})
        for p_key, p_data in pages.items():
            p_num = int(p_key.split("_")[1]) if p_key.startswith("hal_") else (int(p_key) if p_key.isdigit() else 1)
            for f_key, f_obj in p_data.get("fields", {}).items():
                fid = f_obj.get("field_id", f_key)
                var_name = f_obj.get("var_name", f_key)
                for bx in f_obj.get("box_details", []):
                    if not bx.get("is_blank", False):
                        b_idx = bx.get("box_index")
                        b_char = str(bx.get("char", ""))
                        b_conf = float(bx.get("conf", 1.0))
                        baseline_lookup[(b_id, p_num, fid, b_idx)] = (b_char, b_conf)

    # Bandingkan dan kumpulkan baris yang berbeda
    corrected_rows = []
    for idx, row in df_all.iterrows():
        b_id = row["booklet_id"]
        p_num = row["page"]
        fid = row["field_id"]
        b_idx = row["box_index"]
        ai_label = int(row["label"])
        ai_conf = float(row["conf"])
        reason = row["arbitration_reason"]
        img_p = row["image_path"]
        split = row["split"]
        var_name = row["var_name"]

        base_info = baseline_lookup.get((b_id, p_num, fid, b_idx))
        if base_info:
            base_char, base_conf = base_info
            if base_char.isdigit() and int(base_char) != ai_label:
                corrected_rows.append({
                    "no": len(corrected_rows) + 1,
                    "split": split,
                    "booklet_id": b_id,
                    "page": p_num,
                    "field_id": fid,
                    "var_name": var_name,
                    "box_index": b_idx,
                    "baseline_digit": int(base_char),
                    "baseline_conf": round(base_conf, 4),
                    "ai_corrected_digit": ai_label,
                    "ai_conf": round(ai_conf, 4),
                    "confusion_pair": f"{base_char} -> {ai_label}",
                    "arbitration_reason": reason,
                    "image_path": img_p,
                })

    df_corr = pd.DataFrame(corrected_rows)
    print(f"[+] Total digit diluruskan yang terdeteksi: {len(df_corr)} baris")

    # Ekspor ke CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(OUT_CSV, index=False)
    print(f"[+] Berkas CSV disimpan ke: {OUT_CSV}")

    # Ekspor ke JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(corrected_rows, f, indent=2)
    print(f"[+] Berkas JSON disimpan ke: {OUT_JSON}")

    # Ringkasan Statistik
    print("\n" + "=" * 70)
    print("  RINGKASAN STATISTIK 649 DIGIT YANG DILURUSKAN")
    print("=" * 70)
    print("\n1. Distribusi Alasan Arbitrase AI:")
    print(df_corr["arbitration_reason"].value_counts().to_string())

    print("\n2. Top 10 Pasangan Koreksi (Baseline -> AI):")
    print(df_corr["confusion_pair"].value_counts().head(10).to_string())

    print("\n3. Distribusi per Split Dataset:")
    print(df_corr["split"].value_counts().to_string())

    print("\n4. Top 10 Variabel Paling Sering Diluruskan:")
    print(df_corr["var_name"].value_counts().head(10).to_string())
    print("=" * 70)

if __name__ == "__main__":
    main()
