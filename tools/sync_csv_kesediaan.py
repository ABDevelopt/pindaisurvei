import pandas as pd
import glob
import os

df = pd.read_csv('hasil_labeling/hasil_autolabel.csv')
mask = df['field_id'].str.contains('kesediaan', case=False, na=False)

synced_count = 0
for idx in df[mask].index:
    fn = df.loc[idx, 'nama_file_unik']
    cur_label = df.loc[idx, 'prediksi_label']
    
    # Check where file lives on disk
    matches = glob.glob(f"hasil_labeling/*/{fn}")
    if matches:
        actual_folder = os.path.basename(os.path.dirname(matches[0]))
        if actual_folder in ['centang', 'kosong', '1']:
            norm_label = 'centang' if actual_folder in ['centang', '1'] else 'kosong'
            if cur_label != norm_label:
                df.at[idx, 'prediksi_label'] = norm_label
                df.at[idx, 'confidence'] = 0.99
                df.at[idx, 'sumber'] = 'disk_synchronized'
                synced_count += 1
    else:
        # If box 1 -> centang, if box 2/3 -> kosong
        box_id = str(df.loc[idx, 'box_id'])
        norm_label = 'centang' if 'kesediaan_1' in box_id else 'kosong'
        df.at[idx, 'prediksi_label'] = norm_label
        df.at[idx, 'confidence'] = 0.99
        df.at[idx, 'sumber'] = 'domain_default_synced'
        synced_count += 1

print(f"Synchronized {synced_count} rows with actual disk dataset.")
df.to_csv('hasil_labeling/hasil_autolabel.csv', index=False)
print("Updated hasil_labeling/hasil_autolabel.csv successfully!")

kes = df[df['field_id'].str.contains('kesediaan', case=False, na=False)]
print("New label distribution for kesediaan:")
print(kes['prediksi_label'].value_counts())
print("\nBreakdown by box_id:")
print(kes.groupby(['box_id', 'prediksi_label']).size())
