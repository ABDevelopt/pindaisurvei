import os
import json
import zipfile
import base64
import cv2
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ZIP_PATH = "raw.zip"
STATE_DIR = "state"
OUTPUT_ROOT = "hasil_labeling"
MANIFEST_PATH = os.path.join(STATE_DIR, "manifest.json")
SEED_LABELS_PATH = os.path.join(STATE_DIR, "seed_labels.json")
CONFIDENCE_THRESHOLD = 0.65  # Jika di bawah 65%, masukkan ke folder 'ragu_ragu'

# Inisialisasi HOG Descriptor (Kompatibel dengan OpenCV 4 & OpenCV 5+)
if hasattr(cv2, "HOGDescriptor"):
    _hog_desc = cv2.HOGDescriptor(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    def compute_hog(img_32x32):
        return _hog_desc.compute(img_32x32).flatten()
else:
    def compute_hog(img_32x32):
        """Fallback HOG deterministik NumPy/OpenCV untuk lingkungan OpenCV 5.x."""
        gx = cv2.Sobel(img_32x32, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img_32x32, cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        ang = ang % 180
        
        cell_size, nbins = 8, 9
        bin_width = 180 / nbins
        cells_r = img_32x32.shape[0] // cell_size
        cells_c = img_32x32.shape[1] // cell_size
        hist_tensor = np.zeros((cells_r, cells_c, nbins), dtype=np.float32)
        
        for r in range(cells_r):
            for c in range(cells_c):
                c_mag = mag[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
                c_ang = ang[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
                bin_idx = np.floor(c_ang / bin_width).astype(int) % nbins
                for b in range(nbins):
                    hist_tensor[r, c, b] = c_mag[bin_idx == b].sum()
                    
        blocks = []
        for br in range(cells_r - 1):
            for bc in range(cells_c - 1):
                block = hist_tensor[br:br+2, bc:bc+2, :].flatten()
                norm = np.linalg.norm(block) + 1e-6
                blocks.append(block / norm)
        return np.concatenate(blocks)

def praproses_citra(img_bytes, margin=0.12):
    """Memotong tepi kotak kuesioner dan menormalisasi ukuran ke 32x32 piksel."""
    arr = np.frombuffer(img_bytes, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    h, w = gray.shape
    dy, dx = int(h * margin), int(w * margin)
    crop = gray[dy:h-dy, dx:w-dx]
    if crop.size == 0:
        crop = gray
    # Binarisasi adaptif + resize ke 32x32
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(bw, (32, 32), interpolation=cv2.INTER_AREA)
    return resized

def ekstrak_fitur(img_32x32):
    """Menggabungkan fitur gradien bentuk (HOG) dan intensitas piksel."""
    fitur_hog = compute_hog(img_32x32)
    fitur_pixel = img_32x32.flatten() / 255.0
    return np.concatenate([fitur_hog, fitur_pixel])

def augmentasi_citra(img_32x32):
    """Memperbanyak data latih (geser & rotasi tipis) agar model lokal lebih kebal tulisan miring."""
    variasi = [img_32x32]
    # Geser 2 piksel ke kiri, kanan, atas, bawah
    for tx, ty in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        variasi.append(cv2.warpAffine(img_32x32, M_trans, (32, 32)))
    # Rotasi -8 dan +8 derajat
    for angle in [-8, 8]:
        M_rot = cv2.getRotationMatrix2D((16, 16), angle, 1.0)
        variasi.append(cv2.warpAffine(img_32x32, M_rot, (32, 32)))
    return variasi

def buat_nama_unik(zpath):
    folder = os.path.basename(os.path.dirname(zpath))
    fname = os.path.basename(zpath)
    return folder, fname, (f"{folder}__{fname}" if folder else fname)

def jalankan_training_dan_prediksi():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    with open(SEED_LABELS_PATH, "r") as f:
        seed_labels = json.load(f)

    is_zip = manifest.get("is_zip", False)
    zf = zipfile.ZipFile(ZIP_PATH, "r") if (is_zip and os.path.exists(ZIP_PATH)) else None

    def baca_bytes(p):
        if zf is not None:
            return zf.read(p)
        else:
            with open(p, "rb") as f_in:
                return f_in.read()

    # 1. Pasangkan hasil pembacaan Agent (seed_labels) dengan path gambar aslinya
    peta_seed = {}
    for grid_name, label_list in seed_labels.items():
        if grid_name in manifest["grids"]:
            paths = manifest["grids"][grid_name]
            for zpath, lbl in zip(paths, label_list):
                peta_seed[zpath] = str(lbl).strip().lower()

    # Cari gambar di manifest yang belum dibaca oleh Agent (untuk diprediksi model lokal)
    semua_berisi = [p for paths in manifest["grids"].values() for p in paths]
    belum_dilabeli = [p for p in semua_berisi if p not in peta_seed]

    print(f"Data Seed dari Agent: {len(peta_seed)} citra | Sisa yang akan diprediksi model lokal: {len(belum_dilabeli)} citra")

    hasil_akhir = []

    # A. Masukkan semua yang terdeteksi kosong otomatis oleh OpenCV
    for zpath in manifest["kosong_otomatis"]:
        folder, fname, unik = buat_nama_unik(zpath)
        img_bytes = baca_bytes(zpath)
        hasil_akhir.append({
            "path_di_zip": zpath, "folder_field": folder, "nama_file_asli": fname,
            "nama_file_unik": unik, "prediksi_label": "kosong",
            "confidence": 1.0, "sumber": "opencv_auto", "bytes": img_bytes
        })

    # B. Kumpulkan fitur untuk Data Latih (Seed dari Agent)
    X_train, y_train = [], []
    for zpath, lbl in peta_seed.items():
        img_bytes = baca_bytes(zpath)
        folder, fname, unik = buat_nama_unik(zpath)
        hasil_akhir.append({
            "path_di_zip": zpath, "folder_field": folder, "nama_file_asli": fname,
            "nama_file_unik": unik, "prediksi_label": lbl,
            "confidence": 1.0, "sumber": "agent_vision", "bytes": img_bytes
        })

        # Hanya gunakan label valid (bukan ragu_ragu) untuk melatih model
        if lbl != "ragu_ragu":
            img_proc = praproses_citra(img_bytes)
            if img_proc is not None:
                for aug_img in augmentasi_citra(img_proc):
                    X_train.append(ekstrak_fitur(aug_img))
                    y_train.append(lbl)

    # C. Jika masih ada ribuan citra yang belum dilabeli, latih model & prediksi!
    if len(belum_dilabeli) > 0 and len(set(y_train)) > 1:
        print(f"Melatih model lokal dengan {len(X_train)} sampel teraugmentasi...")
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
        )
        clf.fit(np.array(X_train), np.array(y_train))
        kelas_model = clf.classes_

        print(f"Memprediksi {len(belum_dilabeli)} citra sisanya secara lokal...")
        for zpath in belum_dilabeli:
            img_bytes = baca_bytes(zpath)
            folder, fname, unik = buat_nama_unik(zpath)
            img_proc = praproses_citra(img_bytes)

            if img_proc is None:
                lbl_pred, conf = "ragu_ragu", 0.0
            else:
                fitur = ekstrak_fitur(img_proc).reshape(1, -1)
                probs = clf.predict_proba(fitur)[0]
                max_idx = np.argmax(probs)
                conf = float(probs[max_idx])
                lbl_pred = str(kelas_model[max_idx]) if conf >= CONFIDENCE_THRESHOLD else "ragu_ragu"

            hasil_akhir.append({
                "path_di_zip": zpath, "folder_field": folder, "nama_file_asli": fname,
                "nama_file_unik": unik, "prediksi_label": lbl_pred,
                "confidence": round(conf, 4), "sumber": "local_model", "bytes": img_bytes
            })

    if zf is not None:
        zf.close()

    # 2. Simpan gambar ke folder kelas & ekspor CSV
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    rows_csv = []
    for item in hasil_akhir:
        lbl = item["prediksi_label"]
        target_dir = os.path.join(OUTPUT_ROOT, lbl)
        os.makedirs(target_dir, exist_ok=True)

        with open(os.path.join(target_dir, item["nama_file_unik"]), "wb") as f_out:
            f_out.write(item["bytes"])

        rows_csv.append({k: v for k, v in item.items() if k != "bytes"})

    df = pd.DataFrame(rows_csv)
    csv_path = os.path.join(OUTPUT_ROOT, "hasil_autolabel.csv")
    df.to_csv(csv_path, index=False)

    # 3. Buat file review_label.html otomatis
    buat_galeri_html(df)
    print(f"\nSelesai! Total {len(df)} citra berhasil dilabeli.")
    print(df["prediksi_label"].value_counts())

def buat_galeri_html(df, maks_per_kelas=200):
    """Membuat dashboard HTML untuk verifikasi visual cepat."""
    df_sorted = df.sort_values(by=["prediksi_label", "confidence"], ascending=[True, True])
    cards = []
    for _, row in df_sorted.groupby("prediksi_label").head(maks_per_kelas).iterrows():
        lbl = str(row["prediksi_label"])
        img_p = os.path.join(OUTPUT_ROOT, lbl, row["nama_file_unik"])
        if not os.path.exists(img_p):
            continue
        with open(img_p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        warna = "#e74c3c" if (lbl == "ragu_ragu" or row["confidence"] < 0.7) else "#2ecc71"
        cards.append(f"""
        <div class="card" data-label="{lbl}">
            <img src="data:image/png;base64,{b64}">
            <div class="badge" style="background:{warna}"><b>{lbl}</b> ({row['confidence']:.2f})</div>
            <div class="meta">{row['sumber']}<br>{row['folder_field']}</div>
        </div>""")

    kelas_unik = sorted(df["prediksi_label"].astype(str).unique())
    btns = f'<button onclick="flt(\'all\')">Semua ({len(df)})</button>'
    for k in kelas_unik:
        jml = len(df[df["prediksi_label"].astype(str) == k])
        btns += f'<button onclick="flt(\'{k}\')">{k} ({jml})</button>'

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Review Auto-Label</title>
    <style>
        body {{ font-family: sans-serif; background: #f4f6f9; margin: 20px; }}
        .bar {{ position: sticky; top: 0; background: #fff; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 16px; }}
        button {{ margin: 3px; padding: 6px 12px; cursor: pointer; border-radius: 4px; border: 1px solid #ccc; font-weight: bold; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 10px; }}
        .card {{ background: #fff; padding: 6px; border-radius: 6px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .card img {{ width: 64px; height: 64px; object-fit: contain; border: 1px solid #eee; }}
        .badge {{ color: #fff; font-size: 12px; padding: 3px; border-radius: 3px; margin: 4px 0; }}
        .meta {{ font-size: 9px; color: #666; word-break: break-all; }}
    </style></head><body>
    <div class="bar">{btns}</div><div class="grid">{''.join(cards)}</div>
    <script>function flt(k){{document.querySelectorAll('.card').forEach(c=>{{c.style.display=(k==='all'||c.dataset.label===k)?'block':'none'}})}}</script>
    </body></html>"""

    with open("review_label.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Dashboard 'review_label.html' siap dibuka di browser!")

if __name__ == "__main__":
    jalankan_training_dan_prediksi()
