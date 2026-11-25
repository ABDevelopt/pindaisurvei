import os
import sys
import io
import re
import glob
import time
import base64
import shutil
import zipfile
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Konfigurasi API Key & Path
API_KEY = os.getenv("GEMINI_API_KEY", "")
os.environ["GEMINI_API_KEY"] = API_KEY
client = genai.Client(api_key=API_KEY)

# Model prioritas (gemini-3.8-flash sebagai prioritas utama sesuai permintaan pengguna)
ACTIVE_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

if len(sys.argv) > 1:
    INPUT_PATH = sys.argv[1]
elif os.path.exists("output/crops_raw_zips"):
    INPUT_PATH = "output/crops_raw_zips"
else:
    INPUT_PATH = "."

output_root = "hasil_labeling"
csv_output_path = os.path.join(output_root, "hasil_autolabel.csv")
BATCH_SIZE = 25  # 25 gambar (grid 5x5) per 1 request API

kelas_valid = [str(i) for i in range(10)] + ["kosong", "centang", "ragu_ragu"]
for k in kelas_valid:
    os.makedirs(os.path.join(output_root, k), exist_ok=True)


def is_cell_empty(img_bytes, threshold_variance=350.0, dark_threshold=0.025):
    """
    Deteksi cerdas sel kosong menggunakan OpenCV multi-kriteria:
    1. Memotong margin 20% ke dalam untuk mengabaikan border hitam kotak comb-box fisik kuesioner.
    2. Menghitung rasio piksel tinta/gelap (< 160). Jika rasio < 2.5%, sel dipastikan kosong.
    3. Jika varians < 350 dan rata-rata intensitas > 230 (kertas putih bersih), sel kosong.
    4. Mencegah false-positive sel kosong terdeteksi sebagai angka '0' (seperti kasus b09).
    Mengeliminasi ~81% kotak kuesioner tanpa isian secara lokal sebelum memanggil Gemini API.
    """
    arr = np.frombuffer(img_bytes, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.size == 0:
        return True

    h, w = gray.shape
    if h < 10 or w < 10:
        return True

    dy, dx = int(h * 0.20), int(w * 0.20)
    inner = gray[dy:h-dy, dx:w-dx]
    if inner.size == 0:
        inner = gray

    dark_ratio = float(np.sum(inner < 160) / inner.size)
    var = float(np.var(inner))
    mean = float(np.mean(inner))

    if dark_ratio < dark_threshold:
        return True
    if var < threshold_variance and mean > 230:
        return True
    if var < 100:
        return True

    return False


# 2. Skema Output Terstruktur (Pydantic)
class ItemPrediksi(BaseModel):
    idx: int = Field(description="Nomor indeks gambar pada grid (mulai dari 0)")
    label: str = Field(
        description="Isi kotak: angka '0'-'9', 'kosong' jika tidak diisi, 'centang' jika berisi tanda centang/silang, atau 'ragu_ragu' jika coretan tidak terbaca"
    )
    confidence: float = Field(description="Skor keyakinan prediksi antara 0.0 hingga 1.0")


class HasilBatch(BaseModel):
    hasil: list[ItemPrediksi]


def parse_struktur_booklet(zip_filename, path_dalam_zip):
    base_zip = os.path.splitext(os.path.basename(zip_filename))[0]
    id_booklet = re.sub(r"_raw$", "", base_zip, flags=re.IGNORECASE)

    norm_path = path_dalam_zip.replace("\\", "/").strip("/")
    parts = norm_path.split("/")

    try:
        idx_mikro = parts.index("mikro")
        if parts[idx_mikro + 1] == "raw" and idx_mikro >= 1 and len(parts) >= idx_mikro + 4:
            halaman = parts[idx_mikro - 1]
            field_id = parts[idx_mikro + 2]
            box_filename = parts[idx_mikro + 3]
        else:
            return None
    except (ValueError, IndexError):
        if len(parts) >= 2:
            halaman = "hal_1"
            field_id = parts[-2]
            box_filename = parts[-1]
        else:
            return None

    box_id = os.path.splitext(box_filename)[0]
    nama_unik = f"{id_booklet}__{halaman}__{field_id}__{box_filename}"
    nama_unik = re.sub(r'[\\/*?:"<>|]', "_", nama_unik)

    return {
        "id_booklet": id_booklet,
        "halaman": halaman,
        "field_id": field_id,
        "box_id": box_id,
        "nama_file_asli": box_filename,
        "nama_file_unik": nama_unik,
    }


def buat_citra_grid(batch_items, cols=5, cell_w=110, cell_h=130):
    rows = (len(batch_items) + cols - 1) // cols
    grid_img = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid_img)

    for idx, item in enumerate(batch_items):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h

        draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 2, y0 + cell_h - 2], fill=(255, 255, 255), outline=(180, 180, 180))
        draw.text((x0 + 8, y0 + 5), f"ID:{idx}", fill=(220, 0, 0))

        img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        img.thumbnail((cell_w - 16, cell_h - 32))
        img_x = x0 + (cell_w - img.width) // 2
        img_y = y0 + 24 + (cell_h - 24 - img.height) // 2
        grid_img.paste(img, (img_x, img_y))

    return grid_img


def buat_file_review_html(dataframe, folder_output, nama_file_html="review_label.html", maks_per_kelas=250):
    if dataframe.empty:
        return None
    df_sorted = dataframe.sort_values(by=["prediksi_label", "confidence"], ascending=[True, True])
    df_sampel = df_sorted.groupby("prediksi_label").head(maks_per_kelas)

    html_cards = []
    for _, row in tqdm(df_sampel.iterrows(), total=len(df_sampel), desc="Membuat Galeri HTML", unit="img"):
        label = str(row["prediksi_label"])
        nama_unik = str(row["nama_file_unik"])
        conf = float(row["confidence"])
        booklet = str(row["id_booklet"])
        halaman = str(row["halaman"])
        field_id = str(row["field_id"])
        box_id = str(row["box_id"])

        img_path = os.path.join(folder_output, label, nama_unik)
        if not os.path.exists(img_path):
            continue

        with open(img_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        warna_badge = "#e74c3c" if (label == "ragu_ragu" or conf < 0.65) else ("#2980b9" if label in ("kosong", "centang") else "#27ae60")

        html_cards.append(f"""
        <div class="card" data-label="{label}" data-search="{nama_unik.lower()}">
            <div class="img-wrap"><img src="data:image/png;base64,{b64_img}" loading="lazy"></div>
            <div class="label-box" style="background:{warna_badge};">Label: <b>{label}</b></div>
            <div class="meta">
                <b>Conf:</b> {conf:.2f}<br>
                <span class="booklet">{booklet} | {halaman}</span><br>
                <span class="field" title="{field_id}">{field_id}</span><br>
                <span title="{box_id}">{box_id}</span>
            </div>
        </div>""")

    daftar_kelas = sorted(dataframe["prediksi_label"].astype(str).unique())
    tombol_filter = f'<button class="btn active" onclick="filterKelas(\'all\', this)">Semua ({len(dataframe)})</button>'
    for k in daftar_kelas:
        jml = len(dataframe[dataframe["prediksi_label"].astype(str) == k])
        tombol_filter += f'<button class="btn" onclick="filterKelas(\'{k}\', this)">{k} ({jml})</button>'

    html_content = f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8"><title>Review Auto-Labeling Booklet</title>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }}
    .filter-bar {{ position: sticky; top: 0; background: #fff; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); z-index: 100; margin-bottom: 20px; border-radius: 8px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
    .search-box {{ margin-left: auto; padding: 7px 12px; border: 1px solid #ccc; border-radius: 5px; min-width: 250px; }}
    .btn {{ padding: 7px 13px; border: none; border-radius: 5px; background: #e2e8f0; cursor: pointer; font-weight: 600; }}
    .btn.active, .btn:hover {{ background: #2563eb; color: #fff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(145px, 1fr)); gap: 12px; }}
    .card {{ background: #fff; border-radius: 8px; padding: 8px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    .img-wrap {{ height: 75px; display: flex; align-items: center; justify-content: center; background: #fafafa; border: 1px solid #eee; margin-bottom: 6px; }}
    .card img {{ max-width: 70px; max-height: 70px; object-fit: contain; }}
    .label-box {{ color: #fff; padding: 4px; border-radius: 4px; font-size: 13px; margin-bottom: 5px; }}
    .meta {{ font-size: 10px; color: #555; line-height: 1.35; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .booklet {{ color: #b91c1c; font-weight: bold; }}
    .field {{ color: #1e40af; font-weight: 600; }}
</style></head><body>
    <h3 style="margin-top:0;">Dashboard Review Auto-Labeling (Booklet & Multi-Halaman hal_1 s.d. hal_7)</h3>
    <div class="filter-bar">
        {tombol_filter}
        <input type="text" id="searchInput" class="search-box" placeholder="Cari booklet_01 / hal_3 / field_id..." onkeyup="terapkanFilter()">
    </div>
    <div class="grid">{''.join(html_cards)}</div>
    <script>
        let kelasAktif = 'all';
        function filterKelas(k, btn) {{
            kelasAktif = k;
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            terapkanFilter();
        }}
        function terapkanFilter() {{
            let q = document.getElementById('searchInput').value.toLowerCase();
            document.querySelectorAll('.card').forEach(c => {{
                let okKelas = (kelasAktif === 'all' || c.dataset.label === kelasAktif);
                let okCari = c.dataset.search.includes(q);
                c.style.display = (okKelas && okCari) ? 'block' : 'none';
            }});
        }}
    </script>
</body></html>"""

    html_path = os.path.join(folder_output, nama_file_html)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


def panggil_gemini_batch(grid_image, prompt_instruksi):
    """Memanggil Gemini API dengan fallback berjenjang antar model yang aktif."""
    global ACTIVE_MODELS
    for model_name in list(ACTIVE_MODELS):
        # 1 percobaan untuk flagship jika server sedang spike (503), 2 percobaan untuk model stabil
        max_percobaan = 2 if "lite" in model_name else 1
        for percobaan in range(max_percobaan):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[grid_image, prompt_instruksi],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=HasilBatch,
                        temperature=0.0,
                    ),
                )
                if response and response.parsed and response.parsed.hasil:
                    return response.parsed, model_name
            except Exception as e:
                err_str = str(e)
                # Jika kuota habis atau server 503 overload pada model flagship
                if "GenerateRequestsPerDay" in err_str or "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    if model_name in ACTIVE_MODELS and len(ACTIVE_MODELS) > 1 and "lite" not in model_name:
                        print(f"\n[INFO] {model_name} tidak responsif (503/429), beralih ke model berikutnya...")
                        ACTIVE_MODELS.remove(model_name)
                        break
                # Jika 429 RPM pada model lite, tunggu backoff
                if "429" in err_str:
                    time.sleep(5 * (percobaan + 1))
                else:
                    time.sleep(2)
    return None, "FAILED"


def main():
    print("=" * 80)
    print("  AUTO-LABELING GEMINI DENGAN PRE-FILTER CITRA KOSONG OPENCV (PER-BOOKLET)")
    print(f"  Model Prioritas: {', '.join(ACTIVE_MODELS)} | Input: {INPUT_PATH}")
    print("=" * 80)

    data_csv = []
    sudah_diproses = set()
    booklet_selesai = set()
    if os.path.exists(csv_output_path):
        try:
            df_lama = pd.read_csv(csv_output_path)
            data_csv = df_lama.to_dict("records")
            sudah_diproses = set(df_lama["nama_file_unik"].astype(str))
            # Cek booklet yang sudah lengkap terproses (> 1500 citra)
            counts = df_lama["id_booklet"].value_counts()
            for b_id, cnt in counts.items():
                if cnt >= 1800:
                    booklet_selesai.add(str(b_id))
            print(f"[*] Checkpoint ditemukan: {len(sudah_diproses):,} citra sudah terproses dari {len(counts)} booklet.")
            print(f"[*] Booklet yang sudah tuntas dilewati: {len(booklet_selesai)}")
        except Exception:
            pass

    if os.path.isdir(INPUT_PATH):
        daftar_zip = sorted(glob.glob(os.path.join(INPUT_PATH, "booklet_*_raw.zip")))
        if not daftar_zip:
            daftar_zip = [
                z for z in sorted(glob.glob(os.path.join(INPUT_PATH, "*.zip")))
                if "hasil_labeling" not in os.path.basename(z)
            ]
    else:
        daftar_zip = [INPUT_PATH]

    print(f"[*] Ditemukan {len(daftar_zip)} file ZIP booklet di: {INPUT_PATH}")
    if not daftar_zip:
        print(f"[ERROR] Tidak ada berkas ZIP yang ditemukan di {INPUT_PATH}!")
        return

    prompt_instruksi = (
        "Gambar ini adalah grid berisi potongan kotak isian formulir kuesioner. "
        "Setiap sel memiliki nomor 'ID:x' berwarna merah di atasnya. "
        "Untuk setiap ID, tentukan label isinya dengan aturan:\n"
        "- Jika berisi angka tulisan tangan (0 sampai 9), berikan angkanya ('0' s.d. '9'). Abaikan garis tepi kotak formulir.\n"
        "- Jika kotak hanya berisi titik-titik panduan atau kosong tanpa tulisan, berikan 'kosong'.\n"
        "- Jika berisi tanda centang (v) atau silang (x), berikan 'centang'.\n"
        "- Jika coretan rusak atau tidak bisa dikenali, berikan 'ragu_ragu'."
    )

    total_booklet = len(daftar_zip)
    for idx_b, zip_path in enumerate(daftar_zip, 1):
        nama_zip = os.path.basename(zip_path)
        id_b = re.sub(r"_raw$", "", os.path.splitext(nama_zip)[0], flags=re.IGNORECASE)

        if id_b in booklet_selesai:
            continue

        print(f"\n" + "-" * 70)
        print(f"[{idx_b}/{total_booklet}] MEMPROSES BOOKLET: {nama_zip}")
        print("-" * 70)

        item_berisi_b = []
        kosong_b = 0
        total_b = 0

        with zipfile.ZipFile(zip_path, "r") as zf:
            daftar_file = [
                f for f in zf.namelist()
                if not f.endswith("/") and f.lower().endswith((".png", ".jpg", ".jpeg")) and "__MACOSX" not in f
            ]
            for path_zip in sorted(daftar_file):
                total_b += 1
                meta = parse_struktur_booklet(nama_zip, path_zip)
                if meta is None or meta["nama_file_unik"] in sudah_diproses:
                    continue

                img_bytes = zf.read(path_zip)

                # Filter citra kosong OpenCV
                if is_cell_empty(img_bytes):
                    kosong_b += 1
                    target_path = os.path.join(output_root, "kosong", meta["nama_file_unik"])
                    with open(target_path, "wb") as f_out:
                        f_out.write(img_bytes)

                    data_csv.append({
                        "id_booklet": meta["id_booklet"],
                        "halaman": meta["halaman"],
                        "field_id": meta["field_id"],
                        "box_id": meta["box_id"],
                        "prediksi_label": "kosong",
                        "confidence": 1.0,
                        "sumber": "opencv_auto",
                        "nama_file_unik": meta["nama_file_unik"],
                        "path_di_zip": path_zip,
                        "file_zip_asal": nama_zip
                    })
                    sudah_diproses.add(meta["nama_file_unik"])
                else:
                    item_berisi_b.append({
                        **meta,
                        "file_zip_asal": nama_zip,
                        "path_di_zip": path_zip,
                        "bytes": img_bytes
                    })

        total_ai_b = len(item_berisi_b)
        batches_b = (total_ai_b + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  -> Total Sel di ZIP         : {total_b:,} citra")
        print(f"  -> Sel Kosong Tereliminasi  : {kosong_b:,} citra ({kosong_b/max(1,total_b)*100:.1f}%)")
        print(f"  -> Sel Berisi Masuk AI      : {total_ai_b:,} citra ({batches_b} batch)")

        if total_ai_b > 0:
            with tqdm(total=total_ai_b, desc=f"  Inferensi AI {id_b}", unit="img", dynamic_ncols=True) as pbar:
                for i in range(0, total_ai_b, BATCH_SIZE):
                    batch = item_berisi_b[i : i + BATCH_SIZE]
                    batch_num = (i // BATCH_SIZE) + 1

                    grid_image = buat_citra_grid(batch)
                    hasil_batch, used_model = panggil_gemini_batch(grid_image, prompt_instruksi)

                    map_pred = {it.idx: it for it in hasil_batch.hasil} if (hasil_batch and hasil_batch.hasil) else {}

                    for idx, item in enumerate(batch):
                        pred = map_pred.get(idx)
                        label = pred.label.strip().lower() if pred else "ragu_ragu"
                        conf = round(float(pred.confidence), 4) if pred else 0.0

                        if label not in kelas_valid:
                            label = "ragu_ragu"

                        target_path = os.path.join(output_root, label, item["nama_file_unik"])
                        with open(target_path, "wb") as f_out:
                            f_out.write(item["bytes"])

                        data_csv.append({
                            "id_booklet": item["id_booklet"],
                            "halaman": item["halaman"],
                            "field_id": item["field_id"],
                            "box_id": item["box_id"],
                            "prediksi_label": label,
                            "confidence": conf,
                            "sumber": f"gemini_{used_model}",
                            "nama_file_unik": item["nama_file_unik"],
                            "path_di_zip": item["path_di_zip"],
                            "file_zip_asal": item["file_zip_asal"]
                        })
                        sudah_diproses.add(item["nama_file_unik"])
                        item["bytes"] = None

                    pbar.update(len(batch))
                    time.sleep(3.0)  # Pacing 3 detik untuk menjaga kepatuhan rate limit

        # Simpan CSV checkpoint setiap booklet selesai
        df = pd.DataFrame(data_csv)
        df.to_csv(csv_output_path, index=False)
        print(f"  [v] Selesai {id_b}! Total database saat ini: {len(df):,} baris.")

    df = pd.DataFrame(data_csv)
    df.to_csv(csv_output_path, index=False)
    html_path = buat_file_review_html(df, output_root)
    zip_hasil_path = shutil.make_archive("hasil_labeling", "zip", ".", output_root)

    print("\n" + "=" * 80)
    print(f"  SEMUA BOOKLET SELESAI DIPROSES!")
    print(f"  Total Citra di CSV       : {len(df):,}")
    print(f"  File HTML Review         : {html_path}")
    print(f"  File ZIP Hasil           : {zip_hasil_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
