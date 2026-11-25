import os
import pandas as pd

df = pd.read_csv('hasil_labeling/hasil_autolabel.csv')
print('=== MASTER CSV SUMMARY ===')
print(f'Total rows: {len(df):,}')
counts = df['prediksi_label'].value_counts()
print(counts)
sisa_rr = (df['prediksi_label'] == 'ragu_ragu').sum()
print(f"ragu_ragu in CSV: {sisa_rr}")

print('\n=== PHYSICAL FILES IN DIRECTORIES ===')
base_dir = 'hasil_labeling'
folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
total_files = 0
for d in sorted(folders):
    n = len(os.listdir(os.path.join(base_dir, d)))
    total_files += n
    csv_cnt = counts.get(d, 0)
    print(f"  {d:<12}: {n:>7,} files (CSV: {csv_cnt:>7,}) | diff: {n - csv_cnt}")

print(f"\nTotal physical files: {total_files:,}")
print(f"Parity match: {total_files == len(df)}")
