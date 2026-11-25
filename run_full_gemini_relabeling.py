"""
Orkestrator Pelabelan Ulang Penuh (Full Re-Labeling) Seruti Inti Digit Crops
- Menggunakan citra RAW 100% asli (mencegah digit terpotong)
- Model OCR Google: Multi-Model Circuit Breaker (gemini-3.5-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-3.8-flash, gemini-3.7-flash)
- Deteksi ketat kelas 'ragu_ragu' untuk kasus ambigu, cacat crop, dan confidence < 0.70
- Pre-filtering lokal OpenCV untuk kotak kosong (~76.5% zero-cost)
- Batch composite 50-in-1 grid (5 kolom x 10 baris) untuk efisiensi kuota 2x lipat
"""

import os
import sys
import io
import re
import glob
import time
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Muat variabel lingkungan dari .env (project atau user home) dengan dukungan UTF-8 BOM Windows
for _env_path in [".env", os.path.expanduser("~/.env")]:
    if os.path.exists(_env_path):
        try:
            with open(_env_path, "r", encoding="utf-8-sig") as _f:
                for _line in _f.read().splitlines():
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        _k = _k.strip().lstrip("\ufeff")
                        _v = _v.strip().strip("'\"")
                        if _k and _v:
                            os.environ[_k] = _v
        except Exception:
            pass

# 1. Konfigurasi Kredensial & Model Hierarchy
API_KEY = os.getenv("GEMINI_API_KEY", "")
os.environ["GEMINI_API_KEY"] = API_KEY

OUTPUT_ROOT = "hasil_labeling"
CSV_PATH = os.path.join(OUTPUT_ROOT, "hasil_autolabel.csv")
BATCH_SIZE = 50  # Grid 5x10 (50 kotak per panggilan API untuk efisiensi kuota)

VALID_CLASSES = [str(i) for i in range(10)] + ["kosong", "centang", "ragu_ragu"]
for c in VALID_CLASSES:
    os.makedirs(os.path.join(OUTPUT_ROOT, c), exist_ok=True)

# Status Circuit Breaker Model (Prioritas Model Paid Tier Terbaik)
CANDIDATE_MODELS = [
    "gemini-3.8-flash",       # Prioritas 1: Flagship Vision Terbaik Google
    "gemini-3.7-flash",       # Prioritas 2: Next-Gen Flash Flagship
    "gemini-3.5-flash",       # Prioritas 3: Workhorse Cepat & Presisi
    "gemini-3.5-flash-lite",  # Prioritas 4: Ultra-Fast Lite
    "gemini-3.1-flash-lite"   # Prioritas 5: Fallback Lite
]

MODEL_COOLDOWN = {m: 0.0 for m in CANDIDATE_MODELS}
COOLDOWN_DURATION = 60.0  # 1 menit cooldown jika terjadi spike transient


# 2. Skema Output Terstruktur Pydantic
class ItemPrediksi(BaseModel):
    idx: int = Field(description="Nomor indeks gambar pada grid (mulai dari 0)")
    label: str = Field(
        description=(
            "Isi kotak: '0'-'9' jika angka tunggal jelas dan terbaca pasti; "
            "'kosong' jika sel kosong atau hanya pola titik panduan; "
            "'centang' jika tanda centang (v) atau silang (x); "
            "'ragu_ragu' jika ambigu, angka terpotong/ketidakteraturan crop, "
            "coretan tertimpa revisi, multi-digit, garis tabel membelah angka, atau tidak dapat dipastikan."
        )
    )
    confidence: float = Field(description="Skor keyakinan prediksi 0.0 - 1.0. Jika ragu_ragu, berikan skor < 0.70")


class HasilBatch(BaseModel):
    hasil: list[ItemPrediksi]


PROMPT_INSTRUKSI = (
    "Anda adalah pakar verifikasi OCR kuesioner sensus/survei BPS. "
    "Gambar ini adalah grid berisi potongan kotak digit formulir kuesioner dalam format RAW ASLI "
    "(potongan utuh menjaga seluruh coretan tulisan tangan).\n"
    "Setiap sel memiliki label merah 'ID:x' di sudut kiri atas.\n\n"
    "Tugas: Tentukan label isi untuk setiap ID dari 0 sampai ID terakhir dengan aturan klasifikasi ketat:\n"
    "1. '0' s.d. '9': Hanya jika kotak berisi satu digit angka tulisan tangan yang JELAS dan TERBACA PASTI. "
    "Abaikan garis batas tepi kotak kuesioner jika ada di pinggir.\n"
    "2. 'kosong': Jika kotak kosong tanpa tulisan atau hanya memuat pola titik-titik panduan.\n"
    "3. 'centang': Jika kotak berisi tanda centang (✓, v) atau tanda silang (x).\n"
    "4. 'ragu_ragu': WAJIB diberikan untuk SEMUA kondisi berikut:\n"
    "   - KETIDAKTERATURAN CROP: Angka terpotong sebagian di pinggir (partial digit), pergeseran kotak yang memuat pecahan angka dari kotak sebelah, atau garis tabel kotor memotong angka.\n"
    "   - KASUS AMBIGU: Bentuk angka tidak jelas atau mirip dengan digit lain (misal ragu antara '1' vs '7', '0' vs '6', '3' vs '5', '4' vs '9').\n"
    "   - CORETAN TIMPA / BEKAS HAPUS: Angka yang dicoret pembetulan atau ditimpa tulisan lain.\n"
    "   - NOISE: Noda kertas, coretan acak non-digit.\n\n"
    "PRINSIP UTAMA: Lebih baik memberi label 'ragu_ragu' daripada menebak salah satu angka pada citra yang cacat crop atau ambigu!"
)


def get_gemini_client():
    return genai.Client(
        api_key=API_KEY,
        http_options={"api_version": "v1beta"}
    )


def buat_citra_grid_raw(batch_items, cols=5, cell_w=120, cell_h=110):
    """
    Membuat citra grid komposit menggunakan citra RAW utuh,
    menjaga rasio aspek proporsional tanpa crop terpotong.
    """
    rows = (len(batch_items) + cols - 1) // cols
    grid_img = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid_img)

    for idx, item in enumerate(batch_items):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h

        draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 2, y0 + cell_h - 2], fill=(255, 255, 255), outline=(180, 180, 180))
        draw.text((x0 + 8, y0 + 5), f"ID:{idx}", fill=(220, 0, 0))

        # Gunakan raw_file 100% asli
        src_path = item.get("raw_file")
        if not src_path or not os.path.exists(src_path):
            src_path = item.get("inner_file", "")

        if os.path.exists(src_path):
            try:
                img = Image.open(src_path).convert("RGB")
                # Scale proporsional masuk ke ruang sel tanpa distorsi
                max_w = cell_w - 16
                max_h = cell_h - 32
                img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

                img_x = x0 + (cell_w - img.width) // 2
                img_y = y0 + 24 + (cell_h - 24 - img.height) // 2
                grid_img.paste(img, (img_x, img_y))
            except Exception:
                pass

    return grid_img


def panggil_gemini_batch_circuit(client, grid_image, prompt_instruksi):
    """
    Memanggil Gemini API dengan circuit breaker otomatis:
    Mencoba model prioritas (gemini-3.5-flash dsb), jika overload/quota beralih instan ke fallback berikutnya.
    """
    global MODEL_COOLDOWN
    now = time.time()

    for model_name in CANDIDATE_MODELS:
        if now < MODEL_COOLDOWN.get(model_name, 0.0):
            # Model sedang dalam masa cooldown
            continue

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
            if any(tok in err_str for tok in ["503", "504", "DEADLINE_EXCEEDED", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                MODEL_COOLDOWN[model_name] = time.time() + COOLDOWN_DURATION
                print(f"\n[INFO] {model_name} rate limit / overload ({err_str[:60]}...), aktifkan cooldown 5m. Beralih ke fallback...")
            else:
                time.sleep(1)

    return None, "FAILED"


def save_checkpoint(records, csv_path):
    if not records:
        return
    df = pd.DataFrame(records)
    cols = [
        "id_booklet", "halaman", "field_id", "box_id", "digit_label", "role",
        "prediksi_label", "confidence", "sumber", "model_used",
        "nama_file_unik", "raw_path"
    ]
    existing_cols = [c for c in cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]
    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Full Re-Labeling Seruti Inti Digit Crops (RAW Format)")
    parser.add_argument("--all", action="store_true", help="Proses seluruh 46 buklet sah")
    parser.add_argument("--limit-booklets", type=int, default=None, help="Batas jumlah buklet untuk diproses")
    parser.add_argument("--limit-grids", type=int, default=None, help="Batas jumlah grid Gemini per buklet")
    args = parser.parse_args()

    print("=" * 80)
    print("  🚀 FULL RE-LABELING SERUTI INTI DENGAN GOOGLE GEMINI VISION (FORMAT RAW)")
    print(f"  Format Citra    : 100% RAW Asli (Coretan Penuh Tanpa Terpotong)")
    print(f"  Batch Composite : 50-in-1 Grid (Efisiensi 2x Lipat)")
    print(f"  Model Pool      : {', '.join(CANDIDATE_MODELS)}")
    print(f"  Kelas Ragu-Ragu : KETAT (ambigu, cacat crop, split/bleed, conf < 0.70)")
    print("=" * 80)

    client = get_gemini_client()

    # Load existing checkpoint jika ada (Hanya yang valid, buang gemini_FAILED lama)
    records = []
    processed_names = set()
    if os.path.exists(CSV_PATH):
        try:
            df_old = pd.read_csv(CSV_PATH)
            df_valid = df_old[df_old["sumber"] != "gemini_FAILED"]
            records = df_valid.to_dict("records")
            processed_names = set(df_valid["nama_file_unik"].astype(str))
            print(f"[*] Checkpoint valid ditemukan: {len(processed_names):,} digit telah tersimpan.")
        except Exception as e:
            print(f"[!] Gagal membaca checkpoint: {e}")

    # Temukan seluruh direktori buklet di output/crops (Filter Duplikat & Cacat Pindaian)
    from seruti_pipeline.stage_gate_validator import DUPLICATE_BOOKLET_IDS, REJECTED_BOOKLET_IDS

    booklet_dirs = sorted([
        d for d in glob.glob("output/crops/booklet_*")
        if os.path.basename(d) not in DUPLICATE_BOOKLET_IDS
        and os.path.basename(d) not in REJECTED_BOOKLET_IDS
    ])
    if not booklet_dirs:
        print("[ERROR] Tidak ditemukan folder booklet_* yang valid di output/crops/!")
        return

    if args.limit_booklets:
        booklet_dirs = booklet_dirs[:args.limit_booklets]

    print(f"[*] Memproses {len(booklet_dirs)} buklet sah...")

    total_blank_count = 0
    total_ai_count = 0
    quota_exhausted_global = False

    for idx_b, b_dir in enumerate(booklet_dirs, 1):
        b_id = os.path.basename(b_dir)
        print(f"\n[{idx_b}/{len(booklet_dirs)}] Memproses: {b_id}")

        manifest_files = sorted(glob.glob(os.path.join(b_dir, "hal_*", "crop_manifest.json")))
        if not manifest_files:
            print(f"  [WARN] Tidak ada crop_manifest.json di {b_id}")
            continue

        booklet_items_to_ai = []

        for m_file in manifest_files:
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
            except Exception as e:
                print(f"  [ERROR] Gagal memuat {m_file}: {e}")
                continue

            page_num = m_data.get("page_num", 1)
            digits = m_data.get("tiers", {}).get("digits", [])

            for d in digits:
                cell_id = d.get("cell_id", "c000")
                digit_label = d.get("digit_label", "digit_00")
                field_id = d.get("field_id", "field")
                role = d.get("role", "questionnaire_digit")

                nama_unik = f"{b_id}__hal_{page_num}__{cell_id}__{field_id}_{digit_label}.png"

                if nama_unik in processed_names:
                    continue

                is_blank = d.get("is_blank", False)
                raw_path = d.get("raw_file", "")
                inner_path = d.get("inner_file", "")

                # 1. Kotak Kosong -> Otomatis beri label 'kosong' secara lokal (Zero Cost)
                if is_blank:
                    pred_label = "kosong"
                    conf = 1.0
                    sumber = "opencv_auto"
                    used_model = "opencv_prefilter"

                    # Simpan salinan citra RAW ke folder hasil_labeling/kosong/
                    dst_img = os.path.join(OUTPUT_ROOT, "kosong", nama_unik)
                    src_copy = raw_path if os.path.exists(raw_path) else inner_path
                    if os.path.exists(src_copy):
                        shutil.copy2(src_copy, dst_img)

                    records.append({
                        "id_booklet": b_id,
                        "halaman": f"hal_{page_num}",
                        "field_id": field_id,
                        "box_id": cell_id,
                        "digit_label": digit_label,
                        "role": role,
                        "prediksi_label": pred_label,
                        "confidence": conf,
                        "sumber": sumber,
                        "model_used": used_model,
                        "nama_file_unik": nama_unik,
                        "raw_path": raw_path
                    })
                    processed_names.add(nama_unik)
                    total_blank_count += 1
                else:
                    # Kotak Terisi -> Antrekan untuk pemanggilan AI
                    booklet_items_to_ai.append({
                        "b_id": b_id,
                        "page_num": page_num,
                        "field_id": field_id,
                        "cell_id": cell_id,
                        "digit_label": digit_label,
                        "role": role,
                        "nama_unik": nama_unik,
                        "raw_file": raw_path,
                        "inner_file": inner_path
                    })

        # Simpan checkpoint otomatis untuk kotak kosong yang baru terlabeli
        save_checkpoint(records, CSV_PATH)

        total_ai_b = len(booklet_items_to_ai)
        num_grids = (total_ai_b + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  -> Kotak Kosong Terlabeli Otomatis : {total_blank_count:,} (kumulatif)")
        print(f"  -> Kotak Terisi Masuk AI ({b_id})   : {total_ai_b:,} kotak ({num_grids} grid 50-in-1)")

        if total_ai_b == 0 or quota_exhausted_global:
            if quota_exhausted_global:
                print(f"  -> [INFO] Melewati antrean AI {b_id} karena kuota harian pool model sedang habis.")
            continue

        if args.limit_grids:
            booklet_items_to_ai = booklet_items_to_ai[:args.limit_grids * BATCH_SIZE]
            total_ai_b = len(booklet_items_to_ai)
            num_grids = (total_ai_b + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  -> [TEST MODE] Dibatasi menjadi {num_grids} grid ({total_ai_b} kotak).")

        # 2. Proses Grid Batch AI (50-in-1)
        with tqdm(total=total_ai_b, desc=f"  Gemini Vision {b_id}", unit="box") as pbar:
            for g_idx in range(0, total_ai_b, BATCH_SIZE):
                batch_items = booklet_items_to_ai[g_idx : g_idx + BATCH_SIZE]
                grid_image = buat_citra_grid_raw(batch_items)

                hasil_batch, used_model = panggil_gemini_batch_circuit(client, grid_image, PROMPT_INSTRUKSI)

                if used_model == "FAILED" or not hasil_batch or not hasil_batch.hasil:
                    print(f"\n[QUOTA EXHAUSTED] Seluruh model dalam pool Gemini mencapai limit kuota / cooldown.")
                    print(f"  Menyimpan checkpoint progress ({len(records):,} digit tersimpan aman).")
                    quota_exhausted_global = True
                    save_checkpoint(records, CSV_PATH)
                    break

                map_pred = {it.idx: it for it in hasil_batch.hasil}

                for idx, item in enumerate(batch_items):
                    pred_item = map_pred.get(idx)
                    if pred_item:
                        pred_label = str(pred_item.label).strip().lower()
                        conf = float(pred_item.confidence)
                        sumber = "gemini_ai"

                        # Normalisasi label
                        if pred_label in [str(i) for i in range(10)]:
                            pass
                        elif pred_label in ["centang", "check", "v", "x", "silang"]:
                            pred_label = "centang"
                        elif pred_label in ["kosong", "blank", "empty"]:
                            pred_label = "kosong"
                        else:
                            pred_label = "ragu_ragu"

                        # SAFETY NET: Ambang keyakinan < 0.70 atau cacat crop -> ragu_ragu
                        if conf < 0.70 and pred_label != "kosong":
                            pred_label = "ragu_ragu"
                    else:
                        pred_label = "ragu_ragu"
                        conf = 0.50
                        sumber = "gemini_unindexed"

                    # Simpan salinan citra RAW ke folder kelas yang sesuai
                    dst_folder = os.path.join(OUTPUT_ROOT, pred_label)
                    os.makedirs(dst_folder, exist_ok=True)
                    dst_img = os.path.join(dst_folder, item["nama_unik"])
                    src_copy = item["raw_file"] if os.path.exists(item["raw_file"]) else item["inner_file"]
                    if os.path.exists(src_copy):
                        shutil.copy2(src_copy, dst_img)

                    records.append({
                        "id_booklet": item["b_id"],
                        "halaman": f"hal_{item['page_num']}",
                        "field_id": item["field_id"],
                        "box_id": item["cell_id"],
                        "digit_label": item["digit_label"],
                        "role": item["role"],
                        "prediksi_label": pred_label,
                        "confidence": conf,
                        "sumber": sumber,
                        "model_used": used_model,
                        "nama_file_unik": item["nama_unik"],
                        "raw_path": item["raw_file"]
                    })
                    processed_names.add(item["nama_unik"])
                    total_ai_count += 1
                    pbar.update(1)

                # Simpan checkpoint berkala setiap 1 grid batch selesai
                save_checkpoint(records, CSV_PATH)
                time.sleep(0.3)

    # Simpan hasil akhir
    save_checkpoint(records, CSV_PATH)
    print("\n" + "=" * 80)
    print("  ✅ SELESAI PEMROSESAN BATCH")
    print(f"  Total Digit Terekam     : {len(records):,} entri")
    print(f"  Total Kotak Kosong      : {total_blank_count:,}")
    print(f"  Total Kotak Terisi (AI) : {total_ai_count:,}")
    print(f"  File Checkpoint         : {CSV_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
