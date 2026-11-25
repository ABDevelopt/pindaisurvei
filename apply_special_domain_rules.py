#!/usr/bin/env python3
"""
apply_special_domain_rules.py
=============================
Penerapan Perlakuan Khusus & Deklarasi Awal Domain Kuesioner BPS Seruti:
1. No Halaman (p1_F_hal_HAL_1 s.d. p7_F_hal_HAL_7):
   Secara deterministik bernilai sesuai nomor halamannya (1 s.d. 7).
2. Centang Persetujuan Responden (p1_F_kesediaan_KESEDIAAN_RESPONDEN):
   - box_00_p1_c_kesediaan_1 (Opsi 1: Ya / Setuju) -> Default 1 (Setuju).
   - box_01_p1_c_kesediaan_2 & box_02_p1_c_kesediaan_3 -> Default kosong.
"""

import os
import shutil
import json
import pandas as pd
from datetime import datetime

CSV_PATH = "hasil_labeling/hasil_autolabel.csv"
BACKUP_PATH = "hasil_labeling/hasil_autolabel_backup_before_special_rules.csv"
LOG_PATH = "hasil_labeling/manual_corrections_log.jsonl"
BASE_DIR = "hasil_labeling"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[-] CSV {CSV_PATH} tidak ditemukan!")
        return

    # Backup CSV
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"[*] Backup CSV berhasil disimpan di {BACKUP_PATH}")

    df = pd.read_csv(CSV_PATH)
    now_iso = datetime.now().isoformat()
    changes_count = 0

    log_entries = []

    # 1. ATURAN NO HALAMAN
    print("\n--- MENERAPKAN ATURAN DETERMINISTIK NOMOR HALAMAN ---")
    for k in range(1, 8):
        fid = f"p{k}_F_hal_HAL_{k}"
        target_label = str(k)
        mask = (df["field_id"] == fid) & (df["prediksi_label"] != target_label)
        n_match = mask.sum()

        if n_match > 0:
            print(f"  - Field {fid}: Menyelaraskan {n_match} sel ke '{target_label}'")
            for idx in df[mask].index:
                row = df.loc[idx]
                old_lbl = str(row["prediksi_label"])
                fn = str(row["nama_file_unik"])

                # Pindahkan file fisik
                src = os.path.join(BASE_DIR, old_lbl, fn)
                dst_dir = os.path.join(BASE_DIR, target_label)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, fn)
                if os.path.exists(src):
                    shutil.move(src, dst)

                df.at[idx, "prediksi_label"] = target_label
                df.at[idx, "confidence"] = 1.0
                df.at[idx, "sumber"] = "domain_rule_no_halaman"

                log_entries.append({
                    "timestamp": now_iso,
                    "operator": "DomainRule_DeklarasiAwal",
                    "nama_file_unik": fn,
                    "old_label": old_lbl,
                    "new_label": target_label,
                    "status": "rule_applied",
                    "rule": "deterministic_page_number",
                    "field_id": fid
                })
                changes_count += 1

    # 2. ATURAN CENTANG PERSETUJUAN (KESEDIAAN RESPONDEN)
    print("\n--- MENERAPKAN ATURAN CENTANG PERSETUJUAN (DEFAULT 1 - SETUJU) ---")
    kes_fid = "p1_F_kesediaan_KESEDIAAN_RESPONDEN"

    # Box 1 -> Default 1 (Setuju)
    mask_box1 = (df["field_id"] == kes_fid) & (df["box_id"] == "box_00_p1_c_kesediaan_1") & (df["prediksi_label"] != "1")
    n_b1 = mask_box1.sum()
    if n_b1 > 0:
        print(f"  - Box 1 (Kesediaan Ya): Menetapkan {n_b1} sel ke '1' (Setuju)")
        for idx in df[mask_box1].index:
            row = df.loc[idx]
            old_lbl = str(row["prediksi_label"])
            fn = str(row["nama_file_unik"])

            src = os.path.join(BASE_DIR, old_lbl, fn)
            dst_dir = os.path.join(BASE_DIR, "1")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fn)
            if os.path.exists(src):
                shutil.move(src, dst)

            df.at[idx, "prediksi_label"] = "1"
            df.at[idx, "confidence"] = 1.0
            df.at[idx, "sumber"] = "domain_rule_kesediaan_default_1"

            log_entries.append({
                "timestamp": now_iso,
                "operator": "DomainRule_DeklarasiAwal",
                "nama_file_unik": fn,
                "old_label": old_lbl,
                "new_label": "1",
                "status": "rule_applied",
                "rule": "consent_agreement_default_1",
                "field_id": kes_fid
            })
            changes_count += 1

    # Box 2 & 3 -> Default kosong
    mask_box23 = (df["field_id"] == kes_fid) & (df["box_id"].isin(["box_01_p1_c_kesediaan_2", "box_02_p1_c_kesediaan_3"])) & (df["prediksi_label"] != "kosong")
    n_b23 = mask_box23.sum()
    if n_b23 > 0:
        print(f"  - Box 2 & 3 (Janji/Tolak): Menetapkan {n_b23} sel ke 'kosong'")
        for idx in df[mask_box23].index:
            row = df.loc[idx]
            old_lbl = str(row["prediksi_label"])
            fn = str(row["nama_file_unik"])

            src = os.path.join(BASE_DIR, old_lbl, fn)
            dst_dir = os.path.join(BASE_DIR, "kosong")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fn)
            if os.path.exists(src):
                shutil.move(src, dst)

            df.at[idx, "prediksi_label"] = "kosong"
            df.at[idx, "confidence"] = 1.0
            df.at[idx, "sumber"] = "domain_rule_kesediaan_default_kosong"

            log_entries.append({
                "timestamp": now_iso,
                "operator": "DomainRule_DeklarasiAwal",
                "nama_file_unik": fn,
                "old_label": old_lbl,
                "new_label": "kosong",
                "status": "rule_applied",
                "rule": "consent_non_selected_empty",
                "field_id": kes_fid
            })
            changes_count += 1

    # Simpan CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"\n[+] Master CSV {CSV_PATH} berhasil diperbarui ({changes_count} perubahan).")

    # Tulis log audit
    if log_entries:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"[+] {len(log_entries)} catatan audit disimpan ke {LOG_PATH}")

    print("=" * 65)
    print("  PENERAPAN ATURAN DOMAIN KHUSUS SUKSES TUNTAS!")
    print("=" * 65)

if __name__ == "__main__":
    main()
