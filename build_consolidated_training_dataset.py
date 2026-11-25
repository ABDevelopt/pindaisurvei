"""
Pembangun Dataset Latih ML Konsolidasi & Galeri Audit Visual Seruti Inti
- Berbasis 100% Citra RAW Asli (Zero Truncation / No 28x28 Mangling)
- Partisi Booklet-Level Stratified: 70% Train (54 buklet), 15% Val (11 buklet), 15% Test (11 buklet)
- Ekspor folder citra raw: dataset/cells/<split>/<class>/<nama_unik>.png
- Ekspor NumPy arrays terstandarisasi proporsional letterbox: dataset/seruti_digits_dataset.npz
- Pembuatan galeri audit visual interaktif: hasil_labeling/review_label.html
"""

import os
import sys
import glob
import json
import base64
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

CSV_PATH = "hasil_labeling/hasil_autolabel.csv"
DATASET_ROOT = "dataset"
CELLS_DIR = os.path.join(DATASET_ROOT, "cells")
NPZ_PATH = os.path.join(DATASET_ROOT, "seruti_digits_dataset.npz")
HTML_PATH = "hasil_labeling/review_label.html"

# Rasio split buklet
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Ukuran canvas NumPy array (proportional letterbox tanpa cropping)
CANVAS_SIZE = (64, 64)


def split_booklets(booklet_ids, seed=42):
    random.seed(seed)
    b_list = sorted(list(set(booklet_ids)))
    random.shuffle(b_list)

    n = len(b_list)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_set = set(b_list[:n_train])
    val_set = set(b_list[n_train : n_train + n_val])
    test_set = set(b_list[n_train + n_val :])

    return train_set, val_set, test_set


def pad_to_canvas(img, target_size=(64, 64)):
    """Menyelaraskan citra ke ukuran canvas tetap dengan white padding (tanpa potong)."""
    img_gray = img.convert("L")
    w, h = img_gray.size
    tw, th = target_size

    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img_gray.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("L", target_size, color=255)
    offset_x = (tw - nw) // 2
    offset_y = (th - nh) // 2
    canvas.paste(resized, (offset_x, offset_y))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    return 1.0 - arr  # Inversi: latar belakang 0.0, coretan 1.0


def buat_galeri_html(df, max_per_class=150):
    print("[*] Merender galeri audit visual HITL...")
    df_sorted = df.sort_values(by=["prediksi_label", "confidence"], ascending=[True, True])
    df_sample = df_sorted.groupby("prediksi_label").head(max_per_class)

    html_cards = []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="  Rendering cards", unit="card"):
        label = str(row["prediksi_label"])
        conf = float(row.get("confidence", 1.0))
        nama_unik = str(row["nama_file_unik"])
        b_id = str(row.get("id_booklet", ""))
        hal = str(row.get("halaman", ""))
        box_id = str(row.get("box_id", ""))
        model_used = str(row.get("model_used", ""))

        img_path = os.path.join("hasil_labeling", label, nama_unik)
        if not os.path.exists(img_path):
            img_path = str(row.get("raw_path", ""))

        b64_str = ""
        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass

        if label == "ragu_ragu" or conf < 0.70:
            badge_color = "#e74c3c"  # Merah (Audit HITL)
        elif label in ("kosong", "centang"):
            badge_color = "#2980b9"  # Biru
        else:
            badge_color = "#27ae60"  # Hijau (Digit pasti)

        html_cards.append(f"""
        <div class="card" data-label="{label}" data-search="{nama_unik.lower()}">
            <div class="img-wrap"><img src="data:image/png;base64,{b64_str}" loading="lazy"></div>
            <div class="badge" style="background:{badge_color};">Label: <b>{label}</b></div>
            <div class="meta">
                <b>Conf:</b> {conf:.2f} ({model_used})<br>
                <span class="booklet">{b_id} | {hal}</span><br>
                <span>{box_id}</span>
            </div>
        </div>""")

    classes = sorted(df["prediksi_label"].astype(str).unique())
    filter_buttons = [f'<button class="btn active" onclick="filterClass(\'all\', this)">Semua ({len(df):,})</button>']
    for c in classes:
        cnt = (df["prediksi_label"].astype(str) == c).sum()
        filter_buttons.append(f'<button class="btn" onclick="filterClass(\'{c}\', this)">{c} ({cnt:,})</button>')

    html_text = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<title>Galeri Audit Visual Auto-Labeling Seruti Inti</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
    .header {{ margin-bottom: 20px; }}
    .filter-bar {{ position: sticky; top: 0; background: #1e293b; padding: 12px; border-radius: 8px; margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
    .search-box {{ margin-left: auto; padding: 8px 14px; background: #0f172a; border: 1px solid #334155; color: #fff; border-radius: 6px; min-width: 250px; }}
    .btn {{ padding: 8px 14px; border: 1px solid #334155; border-radius: 6px; background: #1e293b; color: #94a3b8; cursor: pointer; font-weight: 600; font-size: 13px; }}
    .btn.active, .btn:hover {{ background: #3b82f6; color: #fff; border-color: #3b82f6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px; text-align: center; transition: transform 0.15s; }}
    .card:hover {{ transform: translateY(-2px); border-color: #60a5fa; }}
    .img-wrap {{ height: 75px; display: flex; align-items: center; justify-content: center; background: #fff; border-radius: 4px; margin-bottom: 8px; padding: 4px; }}
    .card img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    .badge {{ color: #fff; padding: 4px; border-radius: 4px; font-size: 12px; margin-bottom: 6px; }}
    .meta {{ font-size: 10px; color: #94a3b8; line-height: 1.4; }}
    .booklet {{ color: #38bdf8; font-weight: 600; }}
</style>
</head>
<body>
    <div class="header">
        <h2>🔍 Galeri Audit Visual Pelabelan Digit Seruti Inti (RAW 100%)</h2>
        <p style="color:#94a3b8;">Total Dataset: {len(df):,} entri | Sampel Visual: {len(df_sample):,} citra raw</p>
    </div>
    <div class="filter-bar">
        {''.join(filter_buttons)}
        <input type="text" id="searchInput" class="search-box" placeholder="Cari booklet_01, hal_3, field..." onkeyup="applyFilter()">
    </div>
    <div class="grid">{''.join(html_cards)}</div>
    <script>
        let currentClass = 'all';
        function filterClass(c, btn) {{
            currentClass = c;
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            applyFilter();
        }}
        function applyFilter() {{
            let q = document.getElementById('searchInput').value.toLowerCase();
            document.querySelectorAll('.card').forEach(card => {{
                let matchClass = (currentClass === 'all' || card.dataset.label === currentClass);
                let matchSearch = card.dataset.search.includes(q);
                card.style.display = (matchClass && matchSearch) ? 'block' : 'none';
            }});
        }}
    </script>
</body>
</html>"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"[+] Galeri audit disimpan ke: {HTML_PATH}")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} belum ada!")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"[*] Memuat {len(df):,} baris dari {CSV_PATH}")

    # 1. Partisi Tingkat Buklet
    booklet_ids = df["id_booklet"].unique()
    train_b, val_b, test_b = split_booklets(booklet_ids)
    print(f"[*] Stratifikasi Buklet ({len(booklet_ids)} total):")
    print(f"    - Train (70%): {len(train_b)} buklet")
    print(f"    - Val   (15%): {len(val_b)} buklet")
    print(f"    - Test  (15%): {len(test_b)} buklet")

    def assign_split(b_id):
        if b_id in train_b:
            return "train"
        elif b_id in val_b:
            return "val"
        else:
            return "test"

    df["split"] = df["id_booklet"].apply(assign_split)

    # 2. Siapkan Folder Cells
    for split in ["train", "val", "test"]:
        for c in df["prediksi_label"].unique():
            os.makedirs(os.path.join(CELLS_DIR, split, str(c)), exist_ok=True)

    # 3. Ekspor Citra RAW & Siapkan Array NumPy
    arrays = {
        "X_train": [], "y_train": [],
        "X_val": [], "y_val": [],
        "X_test": [], "y_test": []
    }

    # Label encoder: 0-9 -> int, centang -> 10, kosong -> 11, ragu_ragu -> 12
    label_map = {str(i): i for i in range(10)}
    label_map["centang"] = 10
    label_map["kosong"] = 11
    label_map["ragu_ragu"] = 12

    print("[*] Mengekspor citra RAW dan menyusun array NumPy terkompresi...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Exporting cells", unit="box"):
        label = str(row["prediksi_label"])
        split = str(row["split"])
        nama_unik = str(row["nama_file_unik"])
        raw_path = str(row.get("raw_path", ""))

        src_path = os.path.join("hasil_labeling", label, nama_unik)
        if not os.path.exists(src_path):
            src_path = raw_path

        if os.path.exists(src_path):
            # Salin citra RAW ke dataset/cells/<split>/<class>/
            dst_cell = os.path.join(CELLS_DIR, split, label, nama_unik)
            shutil.copy2(src_path, dst_cell)

            # Buat array NumPy 64x64 letterbox (hanya jika label valid)
            if label in label_map:
                try:
                    with Image.open(src_path) as im:
                        arr = pad_to_canvas(im, CANVAS_SIZE)
                        arrays[f"X_{split}"].append(arr)
                        arrays[f"y_{split}"].append(label_map[label])
                except Exception:
                    pass

    # 4. Simpan NPZ
    print("[*] Menyimpan array terkompresi ke seruti_digits_dataset.npz...")
    npz_dict = {}
    for k in ["train", "val", "test"]:
        x_key, y_key = f"X_{k}", f"y_{k}"
        npz_dict[x_key] = np.array(arrays[x_key], dtype=np.float32)
        npz_dict[y_key] = np.array(arrays[y_key], dtype=np.int64)
        print(f"    - {x_key}: {npz_dict[x_key].shape} | {y_key}: {npz_dict[y_key].shape}")

    np.savez_compressed(NPZ_PATH, **npz_dict)
    print(f"[+] Dataset NumPy tersimpan di: {NPZ_PATH}")

    # 5. Render Galeri Visual
    buat_galeri_html(df)
    print("\n✅ Ekspor Dataset ML Selesai 100%!")


if __name__ == "__main__":
    main()
