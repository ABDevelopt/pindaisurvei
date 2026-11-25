#!/usr/bin/env python3
"""
run_agent_vision_full_scale_staged.py
======================================
Orkestrator Agent Vision Skala Penuh Bertahap untuk Kuesioner BPS Seruti.
100% menggunakan citra RAW micro crops (output/crops/booklet_XX/hal_Y/mikro/raw/).
Bebas dari distorsi kanvas clean_28x28.

Alur Kerja Per-Tahap:
1. Pindai citra RAW murni dari folder booklet target.
2. Filter sel kosong otomatis dengan OpenCV (kosong_otomatis).
3. Bentuk montase sel berisi (grids_to_read/).
4. Eksekusi Agent Vision membaca benih (state/seed_labels.json).
5. Latih model lokal teraugmentasi (HOG + MLPClassifier).
6. Prediksi seluruh sisa citra, kelompokkan ke hasil_labeling/<label>/,
   dan perbarui dashboard review_label.html.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CROPS_DIR = BASE_DIR / "output" / "crops"
STATE_DIR = BASE_DIR / "state"
GRIDS_DIR = BASE_DIR / "grids_to_read"
OUTPUT_ROOT = BASE_DIR / "hasil_labeling"
CHECKPOINT_PATH = STATE_DIR / "staged_checkpoint.json"

STATE_DIR.mkdir(parents=True, exist_ok=True)
GRIDS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

import importlib
mod_01 = importlib.import_module("01_prepare_grids")
mod_02 = importlib.import_module("02_train_and_predict")
from agent_vision_reader import jalankan_agent_vision_reading

buat_grid_image = mod_01.buat_grid_image
CELLS_PER_GRID = mod_01.CELLS_PER_GRID
jalankan_training_dan_prediksi = mod_02.jalankan_training_dan_prediksi


def dapatkan_semua_buklet():
    """Mengambil daftar seluruh folder buklet yang memiliki citra raw mikro."""
    all_b = sorted(glob.glob(str(CROPS_DIR / "booklet_*")))
    valid_b = []
    for b_path in all_b:
        b_name = os.path.basename(b_path)
        # Cek apakah ada raw crops di dalamnya
        raw_crops = glob.glob(str(Path(b_path) / "hal_*" / "mikro" / "raw" / "**" / "*.png"), recursive=True)
        if raw_crops:
            valid_b.append((b_name, len(raw_crops)))
    return valid_b


def run_staged_pipeline(batch_size=10, start_stage=1, max_stages=None, specific_booklets=None):
    print("=" * 80)
    print("  ORKESTRATOR AGENT VISION SKALA PENUH BERTAHAP (RAW CROPS MURNI)")
    print("  Kepatuhan: UU No. 16/1997 & UU No. 27/2022 (Zero Cloud Egress 100% Lokal)")
    print("  Sumber Citra: output/crops/booklet_XX/hal_Y/mikro/raw/")
    print("=" * 80)

    semua_buklet = dapatkan_semua_buklet()
    print(f"[*] Total Buklet Teridentifikasi dengan RAW Crops: {len(semua_buklet)} buklet")
    total_populasi = sum(c for _, c in semua_buklet)
    print(f"[*] Total Populasi Citra RAW Mikro              : {total_populasi:,} citra")

    if specific_booklets:
        buklet_target = [b for b in semua_buklet if b[0] in specific_booklets]
    else:
        buklet_target = semua_buklet

    # Bagi ke dalam tahapan (stages)
    stages = []
    for i in range(0, len(buklet_target), batch_size):
        chunk = buklet_target[i:i + batch_size]
        stages.append(chunk)

    total_stages = len(stages)
    if max_stages:
        stages = stages[:max_stages]

    print(f"[*] Terbagi ke dalam {len(stages)} Tahap Eksekusi (Ukuran Batch: {batch_size} buklet/tahap)\n")

    rekap_keseluruhan = []
    t_global_start = time.time()

    for idx, stage_chunk in enumerate(stages, start=1):
        if idx < start_stage:
            print(f"[-] Melewati Tahap {idx} (Start stage: {start_stage})")
            continue

        b_names = [b[0] for b in stage_chunk]
        b_count = sum(b[1] for b in stage_chunk)
        print("\n" + "=" * 80)
        print(f"  EKSEKUSI TAHAP [{idx}/{total_stages}]: {len(b_names)} BUKLET ({b_count:,} CITRA RAW)")
        print(f"  Buklet: {', '.join(b_names)}")
        print("=" * 80)

        t_stage = time.time()

        # 1. Siapkan Grid Montase langsung dari direktori RAW
        print(f"[*] [Tahap {idx}.1] Menjalankan pemindaian & filtering sel kosong (01_prepare_grids)...")
        # Gabungkan path pencarian untuk batch buklet ini
        raw_stage_files = []
        for b_name in b_names:
            patt = str(CROPS_DIR / b_name / "hal_*" / "mikro" / "raw" / "**" / "*.png")
            raw_stage_files.extend(glob.glob(patt, recursive=True))

        print(f"    -> Ditemukan {len(raw_stage_files):,} citra raw untuk tahap ini")

        # Custom prepare grid untuk stage ini
        kosong_auto = []
        berisi_paths = []
        import cv2, numpy as np
        for p in raw_stage_files:
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            norm_p = p.replace("\\", "/")
            if im is None or float(np.var(im[int(im.shape[0]*0.15):-int(im.shape[0]*0.15), int(im.shape[1]*0.15):-int(im.shape[1]*0.15)])) < 120.0 or int(np.min(im)) > 170:
                kosong_auto.append(norm_p)
            else:
                berisi_paths.append(norm_p)

        print(f"    -> Sel Kosong Otomatis : {len(kosong_auto):,} citra")
        print(f"    -> Sel Berisi Aktif    : {len(berisi_paths):,} citra")

        # Buat Grid Montase
        grids_dict = {}
        total_grids = (len(berisi_paths) + CELLS_PER_GRID - 1) // CELLS_PER_GRID
        for g_idx in range(total_grids):
            g_name = f"stage{idx:02d}_grid_{g_idx:04d}.png"
            b_paths = berisi_paths[g_idx * CELLS_PER_GRID : (g_idx + 1) * CELLS_PER_GRID]
            cells_idx = []
            for i, p in enumerate(b_paths):
                im_c = cv2.imread(p, cv2.IMREAD_COLOR)
                if im_c is None: im_c = np.full((50, 50, 3), 200, dtype=np.uint8)
                cells_idx.append((im_c, f"{i:02d}"))
            grid_img = buat_grid_image(cells_idx)
            cv2.imwrite(str(GRIDS_DIR / g_name), grid_img)
            grids_dict[g_name] = b_paths

        manifest = {
            "stage_index": idx,
            "booklets": b_names,
            "is_zip": False,
            "source_path": str(CROPS_DIR),
            "total_citra": len(raw_stage_files),
            "kosong_otomatis": kosong_auto,
            "grids": grids_dict
        }
        with open(STATE_DIR / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"    -> {len(grids_dict)} citra montase disimpan ke {GRIDS_DIR}/")

        # 2. Agent Vision membaca sampel benih (seed)
        n_seed_grids = max(1, min(4, len(grids_dict)))
        print(f"\n[*] [Tahap {idx}.2] Menjalankan Agent Vision membaca {n_seed_grids} grid benih...")
        jalankan_agent_vision_reading(max_grids=n_seed_grids)

        # 3. Latih Model Lokal & Prediksi Skala Penuh
        print(f"\n[*] [Tahap {idx}.3] Melatih model lokal & memprediksi sisa citra (02_train_and_predict)...")
        jalankan_training_dan_prediksi()

        elapsed_stage = time.time() - t_stage
        print(f"\n[+] TAHAP [{idx}/{total_stages}] SELESAI dalam {elapsed_stage:.2f} detik.")

        # Catat checkpoint
        checkpoint = {
            "last_completed_stage": idx,
            "total_stages": total_stages,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f, indent=2)

    total_elapsed = time.time() - t_global_start
    print("\n" + "=" * 80)
    print(f"  SEMUA TAHAPAN BERHASIL DISELESAIKAN DALAM {total_elapsed:.2f} DETIK!")
    print(f"  Hasil Akhir Tersimpan di: {OUTPUT_ROOT}/")
    print(f"  Dashboard Verifikasi Visual: {BASE_DIR}/review_label.html")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Vision Staged Pipeline for Raw Crops")
    parser.add_argument("--batch-size", type=int, default=10, help="Jumlah buklet per tahap")
    parser.add_argument("--start-stage", type=int, default=1, help="Mulai dari tahap ke-N")
    parser.add_argument("--max-stages", type=int, default=None, help="Batasi maksimal jumlah tahap")
    parser.add_argument("--booklets", nargs="+", default=None, help="Daftar buklet spesifik (misal: booklet_73)")
    args = parser.parse_args()

    run_staged_pipeline(
        batch_size=args.batch_size,
        start_stage=args.start_stage,
        max_stages=args.max_stages,
        specific_booklets=args.booklets
    )
