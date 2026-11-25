#!/usr/bin/env python3
"""
generate_corrected_crops_showcase.py
====================================
Menghasilkan aset visual (citra crop resolusi tinggi dan lembar montase grid)
untuk membuktikan pelurusan 649 sel digit oleh AI Antigravity.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = Path(r"C:\Users\ajian\.gemini\antigravity\brain\ffb6c438-eb71-4258-a214-3baaa5a8baad")
SHOWCASE_DIR = ARTIFACTS_DIR / "crops_showcase"
SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_DIR / "dataset" / "eval" / "ai_corrected_649_digits.csv"
ARRAYS_DIR = BASE_DIR / "dataset" / "data_arrays"
CROPS_DIR = BASE_DIR / "output" / "crops"


def main():
    print("[*] Memuat daftar digit yang diluruskan...")
    df = pd.read_csv(CSV_PATH)
    print(f"    Total data: {len(df)} baris")

    # Muat npz untuk train, val, test
    npz_data = {
        "train": np.load(ARRAYS_DIR / "train.npz"),
        "val": np.load(ARRAYS_DIR / "val.npz"),
        "test": np.load(ARRAYS_DIR / "test.npz"),
    }

    # Muat manifest split untuk pemetaan indeks
    splits_dir = BASE_DIR / "dataset" / "splits"
    manifests = {
        "train": pd.read_csv(splits_dir / "train_manifest.csv"),
        "val": pd.read_csv(splits_dir / "val_manifest.csv"),
        "test": pd.read_csv(splits_dir / "test_manifest.csv"),
    }

    # Bangun lookup (split, booklet_id, page, field_id, box_index) -> sample_idx
    lookup_idx = {}
    for sp_name, m_df in manifests.items():
        for s_idx, row in m_df.iterrows():
            key = (sp_name, row["booklet_id"], int(row["page"]), str(row["field_id"]), int(row["box_index"]))
            lookup_idx[key] = int(row["sample_idx"])

    # Pilih 20 contoh representatif yang beragam:
    # 1. Kasus ART 01 Hub KRT (8 -> 1 dan 4 -> 1)
    # 2. Kasus Aritmetika Total OOP Kesehatan & Pendidikan (8 -> 0 dan 6 -> 0)
    # 3. Kasus 0 -> 8
    # 4. Kasus 4 -> 9 dan 9 -> 4
    # 5. Kasus 8 -> 2 dan 2 -> 0
    # 6. Kasus 3 -> 8 dan 5 -> 8
    selected_indices = []

    # Filter per kategori
    krt_cases = df[df["var_name"] == "b3_art_01_hub_krt"].head(3).index.tolist()
    oop_cases = df[df["var_name"].str.contains("oop|kes_pend", na=False)].head(4).index.tolist()
    subtotal_cases = df[df["var_name"].str.contains("subtotal|total_diterima", na=False)].head(3).index.tolist()
    c_8_0 = df[df["confusion_pair"] == "8 -> 0"].head(3).index.tolist()
    c_0_8 = df[df["confusion_pair"] == "0 -> 8"].head(3).index.tolist()
    c_4_9 = df[df["confusion_pair"].isin(["4 -> 9", "9 -> 4", "4 -> 2"])].head(3).index.tolist()
    c_other = df[df["arbitration_reason"] == "V2_V3_CONSENSUS_OVERRIDE_BASELINE"].head(3).index.tolist()

    # Ambil 20 sampel unik teratas dari berbagai kategori
    candidates = list(dict.fromkeys(krt_cases + oop_cases + subtotal_cases + c_8_0 + c_0_8 + c_4_9 + c_other))
    # Tambah sampel lain jika kurang dari 20
    for idx in df.index:
        if idx not in candidates:
            candidates.append(idx)
        if len(candidates) >= 20:
            break

    df_sample = df.loc[candidates[:20]].reset_index(drop=True)
    print(f"[*] Terpilih {len(df_sample)} sampel visual representatif.")

    showcase_records = []
    grid_tiles = []

    for i, row in df_sample.iterrows():
        sp = row["split"]
        b_id = row["booklet_id"]
        p_num = int(row["page"])
        fid = str(row["field_id"])
        b_idx = int(row["box_index"])
        base_d = int(row["baseline_digit"])
        ai_d = int(row["ai_corrected_digit"])
        pair = row["confusion_pair"]
        reason = row["arbitration_reason"]
        var_name = row["var_name"]

        # Ambil citra 28x28 dari npz
        key = (sp, b_id, p_num, fid, b_idx)
        s_idx = lookup_idx.get(key)
        
        im_28 = None
        if s_idx is not None and s_idx < len(npz_data[sp]["X"]):
            im_28 = npz_data[sp]["X"][s_idx]
        
        if im_28 is None:
            # Fallback jika tidak ketemu
            im_28 = np.zeros((28, 28), dtype=np.uint8)

        # Upscale 5x (140x140 px) dengan nearest neighbor agar goresan pensil tajam
        im_large = cv2.resize(im_28, (140, 140), interpolation=cv2.INTER_NEAREST)

        # Buat versi kontras putih (kertas) dan hitam (pensil) agar menyerupai kuesioner asli
        im_paper = 255 - im_large

        # Beri border abu-abu kotak sel kuesioner
        im_boxed = cv2.copyMakeBorder(im_paper, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=180)

        # Simpan citra crop individual ke SHOWCASE_DIR
        crop_filename = f"crop_{i+1:02d}_{b_id}_p{p_num}_{pair.replace(' -> ', '_to_')}.png"
        crop_path = SHOWCASE_DIR / crop_filename
        cv2.imwrite(str(crop_path), im_boxed)

        # Siapkan kartu visual untuk montase grid
        # Buat kartu ukuran 180x220 px dengan label
        card = np.ones((220, 180, 3), dtype=np.uint8) * 245
        # Tempel im_boxed (148x148)
        im_boxed_bgr = cv2.cvtColor(im_boxed, cv2.COLOR_GRAY2BGR)
        card[10:158, 16:164] = im_boxed_bgr

        # Beri teks keterangan
        # Judul: Base -> AI
        cv2.putText(card, f"#{i+1} [{pair}]", (16, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(card, f"Base: {base_d} -> AI: {ai_d}", (16, 193), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 100, 0), 1, cv2.LINE_AA)
        short_var = var_name[:16] + ".." if len(var_name) > 18 else var_name
        cv2.putText(card, f"{short_var}", (16, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

        grid_tiles.append(card)

        showcase_records.append({
            "no": i + 1,
            "booklet_id": b_id,
            "page": p_num,
            "var_name": var_name,
            "box_index": b_idx,
            "baseline_digit": base_d,
            "ai_corrected_digit": ai_d,
            "confusion_pair": pair,
            "arbitration_reason": reason,
            "crop_filename": crop_filename,
            "crop_abs_path": str(crop_path).replace("\\", "/"),
        })

    # Susun Montase Grid Komposit (4 kolom x 5 baris = 20 kartu)
    # Ukuran total: (5 baris * 220) = 1100 px tinggi, (4 kolom * 180) = 720 px lebar
    rows = []
    for r in range(5):
        row_cards = grid_tiles[r*4 : (r+1)*4]
        rows.append(np.hstack(row_cards))
    grid_composite = np.vstack(rows)

    composite_path = SHOWCASE_DIR / "composite_corrected_digits_grid.png"
    cv2.imwrite(str(composite_path), grid_composite)
    print(f"[+] Lembar Grid Komposit disimpan ke: {composite_path}")

    # Simpan metadata showcase
    json_path = SHOWCASE_DIR / "showcase_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(showcase_records, f, indent=2)
    print(f"[+] Metadata showcase disimpan ke: {json_path}")


if __name__ == "__main__":
    main()
