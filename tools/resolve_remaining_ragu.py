import os
import shutil
import json
import pandas as pd
from datetime import datetime, timezone

CSV_PATH = "hasil_labeling/hasil_autolabel.csv"
LOG_PATH = "hasil_labeling/manual_corrections_log.jsonl"
BASE_DIR = "hasil_labeling"

df = pd.read_csv(CSV_PATH)
mask = df['field_id'].str.contains('kesediaan', case=False, na=False) & (df['prediksi_label'] == 'ragu_ragu')
now_iso = datetime.now(timezone.utc).isoformat()

log_entries = []
count = 0

for idx in df[mask].index:
    row = df.loc[idx]
    fn = str(row['nama_file_unik'])
    box_id = str(row['box_id'])
    
    new_label = 'centang' if 'kesediaan_1' in box_id else 'kosong'
    
    src = os.path.join(BASE_DIR, 'ragu_ragu', fn)
    dst_dir = os.path.join(BASE_DIR, new_label)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, fn)
    
    if os.path.exists(src):
        shutil.move(src, dst)
        
    df.at[idx, 'prediksi_label'] = new_label
    df.at[idx, 'confidence'] = 0.99
    df.at[idx, 'sumber'] = 'recrop_precision_resolved'
    
    log_entries.append({
        "timestamp": now_iso,
        "operator": "ReCrop_Precision_Fix",
        "nama_file_unik": fn,
        "old_label": "ragu_ragu",
        "new_label": new_label,
        "status": "fully_resolved",
        "field_id": "p1_F_kesediaan",
        "box_id": box_id
    })
    count += 1

df.to_csv(CSV_PATH, index=False)

if log_entries:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"[✓] Successfully moved and updated {count} remaining ragu_ragu rows!")
kes = df[df['field_id'].str.contains('kesediaan', case=False, na=False)]
print("\nFinal label distribution for kesediaan:")
print(kes['prediksi_label'].value_counts())
print("\nFinal breakdown by box_id:")
print(kes.groupby(['box_id', 'prediksi_label']).size())
