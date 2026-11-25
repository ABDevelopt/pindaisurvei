import os, cv2, re
import pandas as pd
import numpy as np
from tqdm import tqdm

print("================================================================================")
print("  AUDIT CITRA CROP SERUTI: PRESERVASI ASIMETRI NOMOR HALAMAN & TRIWULAN")
print("================================================================================")

df = pd.read_csv("hasil_labeling/hasil_autolabel.csv")
print(f"[*] Total baris master dataset: {len(df):,}")

SUBKOLOM_HAL5 = {'p5_F014_none', 'p5_F017_none', 'p5_F028_none', 'p5_F031_none', 'p5_F042_none'}

records = []

# Scan through all rows
for idx, r in tqdm(df.iterrows(), total=len(df), desc="Memindai dimensi & asimetri"):
    lbl = str(r["prediksi_label"])
    fn = str(r["nama_file_unik"])
    fid = str(r["field_id"])
    
    p = os.path.join("hasil_labeling", lbl, fn)
    if not os.path.exists(p):
        continue
    
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        records.append({
            "nama_file_unik": fn,
            "id_booklet": r["id_booklet"],
            "halaman": r["halaman"],
            "field_id": fid,
            "box_id": r["box_id"],
            "label": lbl,
            "confidence": r["confidence"],
            "w": 0, "h": 0, "ar": 0, "mean": 0, "dark_ratio": 0,
            "is_defect": True,
            "kategori_desain": "Corrupt Image",
            "kategori_anomali": "corrupt_image",
            "anomaly_types": "corrupt_image",
            "keterangan_status": "Citra tidak dapat dibaca (Corrupt)"
        })
        continue

    h, w = img.shape
    ar = round(w / h, 2) if h > 0 else 0
    mean_val = round(float(np.mean(img)), 1)
    dark_ratio = round(float(np.sum(img < 140) / img.size), 3) if img.size > 0 else 0

    # 1. Identifikasi Asimetri Desain Resmi Kuesioner BPS
    is_official_asymmetry = False
    asymmetry_kind = None
    
    if "_hal_" in fid or "HAL_" in fid:
        is_official_asymmetry = True
        asymmetry_kind = "Nomor Halaman Resmi (Asimetris)"
    elif "triwulan" in fid.lower() or "tw" in fid.lower():
        is_official_asymmetry = True
        asymmetry_kind = "Triwulan Romawi (Asimetris)"
    elif "kesediaan" in fid.lower():
        is_official_asymmetry = True
        asymmetry_kind = "Centang Persetujuan (Asimetris)"
    elif "p7_f043" in fid.lower():
        is_official_asymmetry = True
        asymmetry_kind = "Banner Blok Catatan Hal 7 (Asimetris)"
    elif fid in SUBKOLOM_HAL5:
        is_official_asymmetry = True
        asymmetry_kind = "Sub-Kolom Vertikal Hal 5 (Asimetris)"

    # 2. Evaluasi Defek & Kategori Anomali
    anomaly_types = []
    is_defect = False

    # Deteksi defek fisik/optik absolut
    if h < 5 or w < 5:
        is_defect = True
        anomaly_types.append("degenerate_collapse")
    elif mean_val < 60:
        is_defect = True
        anomaly_types.append("severe_dark_smudge")
    elif dark_ratio > 0.85:
        is_defect = True
        anomaly_types.append("heavy_black_coverage")

    # Evaluasi aspek rasio & ukuran
    if is_official_asymmetry:
        # Untuk bidang asimetri resmi: asimetri adalah fitur desain, BUKAN defek!
        anomaly_types.append("official_layout_asymmetry")
        kategori_desain = asymmetry_kind
        keterangan = f"Valid - {asymmetry_kind} Dipertahankan (AR: {ar})"
    else:
        # Untuk comb-box angka standar (0..9)
        kategori_desain = "Comb-Box Standar"
        keterangan = "Comb-Box Standar"
        if h < 25 or w < 25:
            is_defect = True
            anomaly_types.append("too_small")
        if h > 110 or w > 110:
            is_defect = True
            anomaly_types.append("too_large")
        if ar < 0.40 or ar > 2.20:
            is_defect = True
            anomaly_types.append("extreme_aspect_ratio")

    # Catat jika tergolong asimetri resmi ATAU memiliki defek fisik/geometris
    if is_official_asymmetry or is_defect:
        records.append({
            "nama_file_unik": fn,
            "id_booklet": r["id_booklet"],
            "halaman": r["halaman"],
            "field_id": fid,
            "box_id": r["box_id"],
            "label": lbl,
            "confidence": r["confidence"],
            "w": w,
            "h": h,
            "ar": ar,
            "mean": mean_val,
            "dark_ratio": dark_ratio,
            "is_defect": is_defect,
            "is_official_asymmetry": is_official_asymmetry,
            "kategori_desain": kategori_desain,
            "anomaly_types": ";".join(anomaly_types),
            "keterangan_status": keterangan
        })

anom_df = pd.DataFrame(records)
print(f"\n[DONE] Selesai audit {len(df):,} citra.")
print(f"Total entitas tercatat: {len(anom_df):,} entitas ({len(anom_df)/len(df)*100:.2f}%)")

# Simpan ke CSV report
anom_df.to_csv("hasil_labeling/anomalous_crops_report.csv", index=False)
print("[OK] Laporan audit berhasil disimpan ke: hasil_labeling/anomalous_crops_report.csv")

# Ringkasan Statistik
print("\n--- Breakdown Kategori Desain ---")
print(anom_df["kategori_desain"].value_counts())

print("\n--- Breakdown Defek Sejati vs Asimetri Resmi ---")
print(f"Asimetri Resmi Dipertahankan : {(anom_df['is_official_asymmetry']).sum():,} sel")
print(f"Defek Fisik Nyata (Smudge/Shadow): {(anom_df['is_defect']).sum():,} sel")
print(f"Degenerate Pipih (H < 5 px)  : {(anom_df['h'] < 5).sum():,} sel (Lengkap Tereliminasi)")

print("\n--- Breakdown Defek Fisik Nyata per Booklet (Top 5) ---")
defects_only = anom_df[anom_df['is_defect']]
if not defects_only.empty:
    print(defects_only["id_booklet"].value_counts().head(5))
