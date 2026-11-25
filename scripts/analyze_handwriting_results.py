"""
Analisis Evaluasi Hasil Pengenalan Tulisan Tangan Bebas Seruti Inti
Kepatuhan PII: Tidak mencetak nama atau alamat asli ke stdout
"""
import pandas as pd
import json

df = pd.read_csv('hasil_labeling_teks/hasil_teks.csv')
print('=== RINGKASAN DATASET HASIL TEKS ===')
print(f'Total Baris: {len(df):,}')
print(f'Total Buklet Unik: {df["id_booklet"].nunique()}')

print('\nDistribusi Model Digunakan:')
for k, v in df['model_used'].value_counts().items():
    print(f'  - {k}: {v:,} ({v/len(df)*100:.1f}%)')

print('\nDistribusi Status Bidang:')
for k, v in df['is_blank'].value_counts().items():
    status = "Kosong" if k else "Terisi"
    print(f'  - {status}: {v:,} ({v/len(df)*100:.1f}%)')

terisi = df[~df['is_blank']]
print('\n=== STATISTIK CONFIDENCE (BIDANG TERISI) ===')
print(f'Jumlah Bidang Terisi: {len(terisi):,}')
print(f'Rata-rata Confidence: {terisi["confidence"].mean():.4f}')
print(f'Median Confidence   : {terisi["confidence"].median():.4f}')
print(f'Min Confidence      : {terisi["confidence"].min():.4f}')
print(f'Max Confidence      : {terisi["confidence"].max():.4f}')
print(f'Conf >= 0.95        : {len(terisi[terisi["confidence"] >= 0.95]):,} ({len(terisi[terisi["confidence"] >= 0.95])/len(terisi)*100:.1f}%)')
print(f'Conf 0.85 - 0.94    : {len(terisi[(terisi["confidence"] >= 0.85) & (terisi["confidence"] < 0.95)]):,} ({len(terisi[(terisi["confidence"] >= 0.85) & (terisi["confidence"] < 0.95)])/len(terisi)*100:.1f}%)')
print(f'Conf < 0.85         : {len(terisi[terisi["confidence"] < 0.85]):,} ({len(terisi[terisi["confidence"] < 0.85])/len(terisi)*100:.1f}%)')

print('\n=== DISTRIBUSI PER FIELD ID ===')
piv = df.groupby(['field_id', 'is_blank']).size().unstack(fill_value=0)
piv.columns = ['Terisi' if not c else 'Kosong' for c in piv.columns]
print(piv.to_string())

# Evaluasi Konsistensi KRT vs ART 01
krt_df = df[df['field_id'] == 'p1_txt_nama_krt'][['id_booklet', 'teks_hasil']].rename(columns={'teks_hasil': 'krt'})
art1_df = df[df['field_id'] == 'p2_txt_art_01'][['id_booklet', 'teks_hasil']].rename(columns={'teks_hasil': 'art1'})
merged = pd.merge(krt_df, art1_df, on='id_booklet')

def clean_name(n):
    return str(n).strip().upper()

merged['krt_clean'] = merged['krt'].apply(clean_name)
merged['art1_clean'] = merged['art1'].apply(clean_name)
cocok = merged[merged['krt_clean'] == merged['art1_clean']]
print('\n=== VALIDASI KONSISTENSI KRT (HAL 1) vs ART 01 (HAL 2) ===')
print(f'Total Buklet Dievaluasi : {len(merged)}')
print(f'Cocok Tepat (100% Match): {len(cocok)} ({len(cocok)/len(merged)*100:.1f}%)')
tidak_cocok = merged[merged['krt_clean'] != merged['art1_clean']]
print(f'Selisih / Beda Karakter : {len(tidak_cocok)}')
if len(tidak_cocok) > 0:
    for _, row in tidak_cocok.iterrows():
        b_id = row['id_booklet']
        # Masking untuk privasi Zero-PII
        k_masked = row['krt'][:2] + "***" + row['krt'][-1:] if len(str(row['krt'])) > 3 else "***"
        a_masked = row['art1'][:2] + "***" + row['art1'][-1:] if len(str(row['art1'])) > 3 else "***"
        print(f'  - {b_id}: KRT={k_masked} vs ART01={a_masked}')

# Cek Catatan Blok VII
catatan_df = df[df['field_id'] == 'p7_txt_catatan']
catatan_terisi = catatan_df[~catatan_df['is_blank']]
print(f'\n=== CATATAN BLOK VII (HALAMAN 7) ===')
print(f'Total Buklet Memiliki Catatan Terisi: {len(catatan_terisi)} dari {len(catatan_df)}')
