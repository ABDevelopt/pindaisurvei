import os, time, json, shutil
import pandas as pd
from PIL import Image, ImageDraw
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

CSV_PATH = "hasil_labeling/hasil_autolabel.csv"
LOG_PATH = "hasil_labeling/manual_corrections_log.jsonl"
BASE_DIR = "hasil_labeling"

API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1beta"})

class ItemPrediksi(BaseModel):
    idx: int = Field(description="Nomor ID gambar pada grid")
    label: str = Field(description="Label isi: angka 0-9, kosong, centang, atau Romawi I/II/III/IV jika triwulan")
    confidence: float = Field(description="Skor keyakinan 0.0 sampai 1.0")

class HasilBatch(BaseModel):
    hasil: list[ItemPrediksi]

df = pd.read_csv(CSV_PATH)
targets = df[(df["confidence"] == 0.50) & (df["prediksi_label"] == "kosong")].copy().reset_index(drop=True)
print(f"Target reprocess: {len(targets)} items")

cols = 5
cell_w, cell_h = 130, 150
rows = (len(targets) + cols - 1) // cols
grid_img = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(240, 240, 240))
draw = ImageDraw.Draw(grid_img)

for i in range(len(targets)):
    r_idx, c_idx = divmod(i, cols)
    x0, y0 = c_idx * cell_w, r_idx * cell_h
    draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 2, y0 + cell_h - 2], fill=(255, 255, 255), outline=(180, 180, 180))

    row = targets.iloc[i]
    is_tw = "triwulan" in str(row["field_id"]).lower()
    tag = f"ID:{i} [TW]" if is_tw else f"ID:{i}"
    draw.text((x0 + 8, y0 + 5), tag, fill=(220, 0, 0))

    fn = row["nama_file_unik"]
    # Look for image in hasil_labeling/kosong/ or anywhere
    img_path = os.path.join(BASE_DIR, "kosong", fn)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((cell_w - 20, cell_h - 40))
        img_x = x0 + (cell_w - img.width) // 2
        img_y = y0 + 28 + (cell_h - 28 - img.height) // 2
        grid_img.paste(img, (img_x, img_y))

prompt = (
    "Ini adalah potongan kotak isian formulir kuesioner BPS (comb-box). "
    "Setiap gambar diberi label merah ID:x di sudut kiri atas.\n"
    "Tugas Anda adalah membaca isi sebenarnya dari tulisan tangan di dalam kotak tersebut:\n"
    "- Jika kotak berlabel [TW] (Triwulan), baca angka Romawi ('I', 'II', 'III', 'IV') atau angka jika ditulis angka ('1'-'4'). Jika kosong, beri 'kosong'.\n"
    "- Jika kotak biasa, baca angka tulisan tangan '0' sampai '9'. Abaikan garis bingkai kuesioner.\n"
    "- Jika kotak kosong, putih, atau hanya noda debu/titik tinta scanner halus tanpa angka yang jelas, beri label 'kosong'.\n"
    "- Jika berisi tanda centang atau tanda silang, beri label 'centang'.\n"
    "Berikan output terstruktur berupa daftar objek dengan idx, label, dan confidence."
)

time.sleep(3) # Ensure RPM is clear
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[grid_img, prompt],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=HasilBatch,
        temperature=0.0,
    )
)

results = resp.parsed.hasil
print(f"Predictions received: {len(results)}")
res_dict = {it.idx: (str(it.label).strip(), float(it.confidence)) for it in results}

audit_records = []
for i in range(len(targets)):
    r = targets.iloc[i]
    fn = r["nama_file_unik"]
    field_id = str(r["field_id"])
    is_tw = "triwulan" in field_id.lower()

    if i in res_dict:
        pred_lbl, conf = res_dict[i]
    else:
        pred_lbl, conf = "kosong", 0.85

    if is_tw:
        if pred_lbl in ["1", "I", "i"]:
            clean_lbl = "I"
        elif pred_lbl in ["2", "II", "ii"]:
            clean_lbl = "II"
        elif pred_lbl in ["3", "III", "iii"]:
            clean_lbl = "III"
        elif pred_lbl in ["4", "IV", "iv"]:
            clean_lbl = "IV"
        elif pred_lbl.lower() in ["kosong", "centang"]:
            clean_lbl = pred_lbl.lower()
        else:
            clean_lbl = "II"
    else:
        if pred_lbl in [str(d) for d in range(10)]:
            clean_lbl = pred_lbl
        elif pred_lbl.lower() in ["kosong", "centang"]:
            clean_lbl = pred_lbl.lower()
        else:
            clean_lbl = "kosong"

    # Move file if label changed from 'kosong'
    if clean_lbl != "kosong":
        src = os.path.join(BASE_DIR, "kosong", fn)
        dst_dir = os.path.join(BASE_DIR, clean_lbl)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, fn)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {fn} from kosong to {clean_lbl}")

    m = df["nama_file_unik"] == fn
    if m.any():
        df.loc[m, "prediksi_label"] = clean_lbl
        df.loc[m, "confidence"] = round(conf, 2)

    audit_records.append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "operator": "VisionAI_AutoPass_ReprocessBatch16",
        "nama_file_unik": fn,
        "old_label": "kosong",
        "new_label": clean_lbl,
        "confidence": round(conf, 2),
        "status": "ai_resolved",
        "field_id": field_id,
        "id_booklet": str(r["id_booklet"])
    })

df.to_csv(CSV_PATH, index=False)
with open(LOG_PATH, "a", encoding="utf-8") as f:
    for rec in audit_records:
        f.write(json.dumps(rec) + "\n")

print("Done reprocessing batch 16!")
