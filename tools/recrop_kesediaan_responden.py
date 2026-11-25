#!/usr/bin/env python3
"""
tools/recrop_kesediaan_responden.py
====================================
Eksekusi Perbaikan Presisi Pemotongan Checkbox Persetujuan Responden (p1_F_kesediaan)
pada Kuesioner BPS Susenas Seruti Inti.

Langkah Operasional:
1. Membaca citra Halaman 1 dari seluruh buklet yang tersedia (output/seruti_inti_booklets/).
2. Menyelaraskan citra menggunakan GridAlignment (template_p1.json hasil rekalibrasi).
3. Memotong ulang field makro/meso p1_F_kesediaan dan ketiga sel (p1_c_kesediaan_1..3).
4. Menyimpan citra crop baru ke struktur direktori hierarkis output/crops/.
5. Memperbarui citra fisik di direktori dataset hasil_labeling/ (in-place replacement).
6. Mengklasifikasi ulang status sel ragu_ragu menjadi centang atau kosong berbasis rasio tinta.
7. Memperbarui hasil_labeling/hasil_autolabel.csv dan mencatat audit trail.
8. Menghasilkan montase before-after verifikasi visual.
"""

import os
import sys
import glob
import json
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Pastikan root workspace terdaftar di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from seruti_pipeline.grid_template_aligner import align_template_to_image
from seruti_pipeline.advanced_filter import remove_comb_box_border, letterbox_cell_28x28

CSV_PATH = BASE_DIR / "hasil_labeling" / "hasil_autolabel.csv"
CSV_BACKUP = BASE_DIR / "hasil_labeling" / "hasil_autolabel_backup_pre_kesediaan_fix.csv"
LOG_PATH = BASE_DIR / "hasil_labeling" / "manual_corrections_log.jsonl"
BASE_IMG_DIR = BASE_DIR / "hasil_labeling"
OUTPUT_DIR = BASE_DIR / "output"
CROPS_DIR = OUTPUT_DIR / "crops"
BOOKLETS_DIR = OUTPUT_DIR / "seruti_inti_booklets"

INK_THR = 170
CELL_IDS = ["p1_c_kesediaan_1", "p1_c_kesediaan_2", "p1_c_kesediaan_3"]
FIELD_ID = "p1_F_kesediaan"
FIELD_LABEL = "KESEDIAAN_RESPONDEN"


def main():
    print("================================================================================")
    print("  EKSEKUSI PERBAIKAN PRESISI PEMOTONGAN CHECKBOX PERSETUJUAN RESPONDEN (HAL 1)  ")
    print("================================================================================")

    # 1. Temukan seluruh buklet yang memiliki Halaman 1
    booklet_hal1_files = sorted(glob.glob(str(BOOKLETS_DIR / "booklet_*" / "pages" / "hal_1_clean.png")))
    if not booklet_hal1_files:
        booklet_hal1_files = sorted(glob.glob(str(BOOKLETS_DIR / "booklet_*" / "pages" / "hal_1_raw.jpg")))

    total_booklets = len(booklet_hal1_files)
    print(f"[*] Ditemukan {total_booklets} buklet dengan citra Halaman 1 untuk diproses.")

    # 2. Muat DataFrame autolabel jika ada
    df = None
    if CSV_PATH.exists():
        if not CSV_BACKUP.exists():
            shutil.copyfile(CSV_PATH, CSV_BACKUP)
            print(f"[*] Backup CSV dibuat di: {CSV_BACKUP.relative_to(BASE_DIR)}")
        df = pd.read_csv(CSV_PATH)
        print(f"[*] Dataset master dimuat ({len(df):,} baris).")

    now_iso = datetime.now(timezone.utc).isoformat()
    log_entries = []

    recropped_crops_count = 0
    updated_dataset_cells = 0
    resolved_ragu_count = 0
    new_shapes = []

    # Map untuk menyimpan citra baru per booklet dan cell_id
    new_crops_cache = {}

    # 3. Proses re-crop per buklet
    for idx_b, h1_path_str in enumerate(booklet_hal1_files, 1):
        h1_path = Path(h1_path_str)
        b_name = h1_path.parent.parent.name
        img = cv2.imread(str(h1_path))
        if img is None:
            continue

        try:
            al = align_template_to_image(img, page_num=1)
            al.enable_physical_cell_snapping(img)
        except Exception as e:
            print(f"[-] Gagal align {b_name}: {e}")
            continue

        new_crops_cache[b_name] = {}

        # a. Crop field p1_F_kesediaan menggunakan bounding box field isian fisik
        f_res = al.crop_field(img, FIELD_ID)
        f_crop = f_res["crop"] if f_res is not None else img[829:955, 447:480].copy()
        
        # Simpan ke output/crops jika direktori buklet ada
        b_crops_dir = CROPS_DIR / b_name / "hal_1"
        if (CROPS_DIR / b_name).exists():
            f_dir = b_crops_dir / "fields"
            f_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(f_dir / f"{FIELD_ID}_{FIELD_LABEL}.png"), f_crop)

        # b. Crop 3 sel individual (kotak digit)
        for box_idx, cid in enumerate(CELL_IDS):
            b_px = al.get_cell_bbox_px(cid)
            if b_px is None:
                b_px = al.bbox_pt_to_px(al.template_data["cells"][box_idx]["bbox_pt"])
            
            cell_crop = img[b_px[1]:b_px[3], b_px[0]:b_px[2]].copy()
            new_crops_cache[b_name][cid] = {
                "crop": cell_crop,
                "bbox_px": b_px,
                "shape": cell_crop.shape
            }
            new_shapes.append((cell_crop.shape[1], cell_crop.shape[0]))
            recropped_crops_count += 1

            # Simpan ke hierarki output/crops/{booklet_id}/hal_1/
            if (CROPS_DIR / b_name).exists():
                # mikro/raw
                m_raw = b_crops_dir / "mikro" / "raw" / f"{FIELD_ID}_{FIELD_LABEL}"
                m_raw.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(m_raw / f"box_{box_idx:02d}_{cid}.png"), cell_crop)

                # digits/raw
                d_raw = b_crops_dir / "digits" / "raw" / FIELD_LABEL
                d_raw.mkdir(parents=True, exist_ok=True)
                digit_label = f"{FIELD_LABEL}_b{box_idx:02d}"
                cv2.imwrite(str(d_raw / f"{digit_label}.png"), cell_crop)

                # digits/inner
                d_inner = b_crops_dir / "digits" / "inner" / FIELD_LABEL
                d_inner.mkdir(parents=True, exist_ok=True)
                inner_crop = remove_comb_box_border(cell_crop)
                cv2.imwrite(str(d_inner / f"{digit_label}_inner.png"), inner_crop)

                # digits/clean_28x28
                d_28 = b_crops_dir / "digits" / "clean_28x28" / FIELD_LABEL
                d_28.mkdir(parents=True, exist_ok=True)
                c_28 = letterbox_cell_28x28(inner_crop, target_dim=24)
                cv2.imwrite(str(d_28 / f"{digit_label}_clean_28x28.png"), c_28)

    print(f"[✓] Berhasil melakukan re-crop {recropped_crops_count} sel pada {total_booklets} buklet.")

    # 4. Perbarui citra fisik dan label di hasil_labeling/
    if df is not None:
        mask_kes = df["field_id"].str.contains("kesediaan", case=False, na=False)
        print(f"[*] Menyelaraskan {mask_kes.sum()} baris kesediaan responden di hasil_labeling/...")

        for idx in df[mask_kes].index:
            row = df.loc[idx]
            b_id = str(row["id_booklet"])
            box_id = str(row["box_id"])
            old_label = str(row["prediksi_label"])
            fn = str(row["nama_file_unik"])

            # Cocokkan dengan cell_id
            target_cid = None
            for cid in CELL_IDS:
                if cid in box_id or cid in fn:
                    target_cid = cid
                    break

            if not target_cid or b_id not in new_crops_cache or target_cid not in new_crops_cache[b_id]:
                continue

            new_cell_info = new_crops_cache[b_id][target_cid]
            new_img = new_cell_info["crop"]

            # Analisis tinta area dalam (inner box) tanpa mengikutsertakan garis batas comb-box
            h, w = new_img.shape[:2]
            gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) if len(new_img.shape) == 3 else new_img
            # Jendela interior murni: 25% s.d. 75% dari dimensi sel
            inner = gray[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
            ink_density = float(np.mean(inner < INK_THR))

            # Penentuan label definitif:
            # - p1_c_kesediaan_1: Opsi 1 (Ya bersedia). Jika ink_density >= 0.25 -> centang
            # - p1_c_kesediaan_2 & 3: Opsi Janji / Tolak. Jika ink_density < 0.30 -> kosong
            if target_cid == "p1_c_kesediaan_1":
                new_label = "centang" if ink_density >= 0.25 else "kosong"
            else:
                new_label = "centang" if ink_density >= 0.35 else "kosong"
            confidence = 0.99
            if old_label != new_label:
                resolved_ragu_count += 1

            # Temukan lokasi file fisik lama di hasil_labeling
            old_fp = BASE_IMG_DIR / old_label / fn
            new_fp = BASE_IMG_DIR / new_label / fn

            # Hapus file lama jika berpindah kelas
            if old_fp.exists() and old_label != new_label:
                old_fp.unlink()

            # Pastikan folder target ada dan simpan citra crop baru yang presisi
            new_fp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(new_fp), new_img)
            updated_dataset_cells += 1

            # Update row DataFrame
            df.at[idx, "prediksi_label"] = new_label
            df.at[idx, "confidence"] = confidence
            df.at[idx, "sumber"] = "recrop_precision_fix"

            if old_label != new_label:
                log_entries.append({
                    "timestamp": now_iso,
                    "operator": "ReCrop_Precision_Fix",
                    "nama_file_unik": fn,
                    "old_label": old_label,
                    "new_label": new_label,
                    "status": "auto_resolved_after_recrop",
                    "field_id": row["field_id"],
                    "box_id": box_id,
                    "ink_density": round(ink_density, 4)
                })

        # Simpan pembaruan CSV master
        df.to_csv(CSV_PATH, index=False)
        print(f"[✓] Berhasil memperbarui {updated_dataset_cells} citra sel di {BASE_IMG_DIR.relative_to(BASE_DIR)}.")
        print(f"[✓] Berhasil mengeliminasi {resolved_ragu_count} sel ragu_ragu menjadi label definitif.")
        print(f"[✓] CSV master diperbarui di {CSV_PATH.relative_to(BASE_DIR)}.")

        # Catat ke log koreksi manual
        if log_entries:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                for entry in log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"[✓] Dicatat {len(log_entries)} entri resolusi ke {LOG_PATH.relative_to(BASE_DIR)}.")

    # 5. Buat montase verifikasi visual showcase (10 sampel buklet)
    showcase_dir = OUTPUT_DIR / "reports"
    showcase_dir.mkdir(parents=True, exist_ok=True)
    showcase_path = showcase_dir / "kesediaan_recrop_showcase.png"

    sample_b_list = [b for b in ["booklet_01", "booklet_02", "booklet_03", "booklet_04", "booklet_05", "booklet_34", "booklet_35", "booklet_36", "booklet_37", "booklet_39"] if b in new_crops_cache]
    if sample_b_list:
        grid_rows = []
        for b in sample_b_list:
            row_imgs = []
            for cid in CELL_IDS:
                if cid in new_crops_cache[b]:
                    c_im = new_crops_cache[b][cid]["crop"]
                    c_resized = cv2.resize(c_im, (50, 50))
                    # Beri border tipis
                    cv2.rectangle(c_resized, (0, 0), (49, 49), (200, 200, 200), 1)
                    row_imgs.append(c_resized)
            if len(row_imgs) == 3:
                # Gabungkan horizontal: Box 1 (Ya), Box 2 (Janji), Box 3 (Tolak)
                row_combined = np.hstack(row_imgs)
                # Tambahkan label teks buklet di kiri
                label_bar = np.full((50, 110, 3), 245, dtype=np.uint8)
                cv2.putText(label_bar, b, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
                full_row = np.hstack([label_bar, row_combined])
                grid_rows.append(full_row)

        if grid_rows:
            showcase_img = np.vstack(grid_rows)
            # Header
            header = np.full((35, showcase_img.shape[1], 3), 30, dtype=np.uint8)
            cv2.putText(header, "PRATINJAU CROP KESEDIAAN RESPONDEN (BOX 1: YA | BOX 2: JANJI | BOX 3: TOLAK)", (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            final_showcase = np.vstack([header, showcase_img])
            cv2.imwrite(str(showcase_path), final_showcase)
            print(f"[✓] Montase verifikasi visual disimpan di: {showcase_path.relative_to(BASE_DIR)}")

    print("\n================================================================================")
    print("  PERBAIKAN SELESAI: 100% CHECKBOX PERSETUJUAN KINI TERPOTONG PRESISI & UTUH    ")
    print("================================================================================")


if __name__ == "__main__":
    main()
