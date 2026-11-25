#!/usr/bin/env python3
"""
prepare_hitl_queue.py
=====================
Skrip generator antrean koreksi anotasi Human-in-the-Loop (HITL).
Mengekstrak sel berisiko tinggi (Tier 1: Ambigu & Tier 2: Low Confidence <= 0.90)
untuk diperiksa dan dikoreksi secara cepat oleh operator manusia.
"""

import os
import json
import pandas as pd

def main():
    csv_path = "hasil_labeling/hasil_autolabel.csv"
    if not os.path.exists(csv_path):
        print(f"[-] Berkas {csv_path} tidak ditemukan!")
        return

    df = pd.read_csv(csv_path)
    total_populasi = len(df)
    
    # Tier 1: Ambigu Fisik (ragu_ragu)
    tier1 = df[df["prediksi_label"] == "ragu_ragu"].copy()
    tier1["tier"] = 1
    tier1["alasan_review"] = "AI Flagged as Ambiguous (Group A)"
    
    # Tier 2: Low-Confidence Digits (<= 0.90)
    tier2 = df[(df["prediksi_label"] != "ragu_ragu") & 
               (df["prediksi_label"] != "kosong") & 
               (df["confidence"] <= 0.90)].copy()
    tier2["tier"] = 2
    tier2["alasan_review"] = "Borderline Confidence (<= 0.90)"

    queue_df = pd.concat([tier1, tier2]).reset_index(drop=True)
    
    output_json = "hasil_labeling/hitl_triage_queue.json"
    queue_records = queue_df.to_dict(orient="records")
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(queue_records, f, indent=2)
        
    print("=" * 65)
    print("  HASIL EKSTRAKSI ANTREAN HITL TRIAGE (BPS SERUTI)")
    print("=" * 65)
    print(f"Total Populasi Sel         : {total_populasi:,} citra")
    print(f"Total Antrean Prioritas    : {len(queue_df):,} citra ({(len(queue_df)/total_populasi)*100:.2f}%)")
    print(f"  - Tier 1 (Ambigu Fisik)  : {len(tier1):,} citra")
    print(f"  - Tier 2 (Conf <= 0.90)  : {len(tier2):,} citra")
    print(f"Estimasi Waktu Operator    : ~{(len(queue_df)*2)/60:.1f} menit")
    print(f"Berkas Antrean Disimpan di : {output_json}")
    print("=" * 65)

if __name__ == "__main__":
    main()
