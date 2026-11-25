import os
import io
import time
import shutil
import base64
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Konfigurasi Multi-Key Pool (dari environment)
raw_keys = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
if not KEYS:
    KEYS = ["YOUR_GEMINI_API_KEY"]
clients = [genai.Client(api_key=k) for k in KEYS]

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]

OUTPUT_ROOT = "hasil_labeling"
CSV_PATH = os.path.join(OUTPUT_ROOT, "hasil_autolabel.csv")
BATCH_SIZE = 25

kelas_valid = [str(i) for i in range(10)] + ["kosong", "centang", "ragu_ragu"]
for k in kelas_valid:
    os.makedirs(os.path.join(OUTPUT_ROOT, k), exist_ok=True)

# 2. Skema Output Terstruktur Pydantic
class ItemPrediksi(BaseModel):
    idx: int = Field(description="Nomor indeks gambar pada grid (mulai dari 0)")
    label: str = Field(description="Isi kotak: angka '0'-'9', 'kosong', 'centang', atau 'ragu_ragu'")
    confidence: float = Field(description="Skor keyakinan prediksi antara 0.0 hingga 1.0")

class HasilBatch(BaseModel):
    hasil: list[ItemPrediksi]

def buat_citra_grid(batch_items, cols=5, cell_w=110, cell_h=130):
    rows = (len(batch_items) + cols - 1) // cols
    grid_img = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid_img)

    for idx, item in enumerate(batch_items):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h

        draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 2, y0 + cell_h - 2], fill=(255, 255, 255), outline=(180, 180, 180))
        draw.text((x0 + 8, y0 + 5), f"ID:{idx}", fill=(220, 0, 0))

        img = Image.open(item["path"]).convert("RGB")
        img.thumbnail((cell_w - 16, cell_h - 32))
        img_x = x0 + (cell_w - img.width) // 2
        img_y = y0 + 24 + (cell_h - 24 - img.height) // 2
        grid_img.paste(img, (img_x, img_y))

    return grid_img

def panggil_gemini_batch(client, grid_image, prompt_instruksi):
    for model_name in MODELS:
        for percobaan in range(2):
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
                err = str(e)
                if "429" in err:
                    time.sleep(3 * (percobaan + 1))
                elif "503" in err or "UNAVAILABLE" in err:
                    break
                else:
                    time.sleep(1)
    return None, "FAILED"

def main():
    print("=" * 80)
    print("  PASS KEDUA: RE-PROSES CITRA GAGAL KUOTA (gemini_FAILED) DENGAN DUAL-KEY")
    print(f"  Jumlah Klien Aktif: {len(clients)} | Model: {', '.join(MODELS)}")
    print("=" * 80)

    if not os.path.exists(CSV_PATH):
        print("[ERROR] CSV tidak ditemukan!")
        return

    df = pd.read_csv(CSV_PATH)
    mask_failed = df["sumber"] == "gemini_FAILED"
    idx_failed = df[mask_failed].index.tolist()

    total_target = len(idx_failed)
    print(f"[*] Ditemukan {total_target:,} citra bertanda 'gemini_FAILED' yang perlu diproses ulang.")

    if total_target == 0:
        print("[*] Tidak ada citra yang memerlukan re-proses. Selesai.")
        return

    items_to_process = []
    for i in idx_failed:
        row = df.loc[i]
        fname = str(row["nama_file_unik"])
        fpath = os.path.join(OUTPUT_ROOT, "ragu_ragu", fname)
        if os.path.exists(fpath):
            items_to_process.append({
                "df_index": i,
                "nama_file_unik": fname,
                "path": fpath,
                "id_booklet": row["id_booklet"]
            })

    print(f"[*] Terverifikasi {len(items_to_process):,} berkas fisik tersedia di folder ragu_ragu.")

    prompt_instruksi = (
        "Gambar ini adalah grid potongan kotak isian formulir kuesioner. "
        "Setiap kotak memiliki nomor 'ID:x' merah di kiri atas. "
        "Tentukan label isinya:\n"
        "- Angka tulisan tangan: berikan angka '0' s.d. '9' (abaikan border kotak).\n"
        "- Jika kotak kosong / titik panduan: 'kosong'.\n"
        "- Tanda centang / silang: 'centang'.\n"
        "- Coretan rusak tidak terbaca: 'ragu_ragu'."
    )

    total_batch = (len(items_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    client_idx = 0

    print(f"\n[*] Memulai inferensi {len(items_to_process):,} citra dalam {total_batch:,} batch...")

    with tqdm(total=len(items_to_process), desc="Re-Proses Dual-Key", unit="img", dynamic_ncols=True) as pbar:
        for b_start in range(0, len(items_to_process), BATCH_SIZE):
            batch = items_to_process[b_start : b_start + BATCH_SIZE]
            batch_num = (b_start // BATCH_SIZE) + 1

            grid_img = buat_citra_grid(batch)
            active_client = clients[client_idx % len(clients)]
            client_idx += 1

            pbar.set_postfix({
                "Batch": f"{batch_num}/{total_batch}",
                "Key": f"#{(client_idx-1)%len(clients)+1}"
            })

            hasil_batch, used_model = panggil_gemini_batch(active_client, grid_img, prompt_instruksi)

            map_pred = {it.idx: it for it in hasil_batch.hasil} if (hasil_batch and hasil_batch.hasil) else {}

            for idx, item in enumerate(batch):
                pred = map_pred.get(idx)
                label = pred.label.strip().lower() if pred else "ragu_ragu"
                conf = round(float(pred.confidence), 4) if pred else 0.0

                if label not in kelas_valid:
                    label = "ragu_ragu"

                # Update DataFrame
                row_idx = item["df_index"]
                df.at[row_idx, "prediksi_label"] = label
                df.at[row_idx, "confidence"] = conf
                df.at[row_idx, "sumber"] = f"gemini_{used_model}"

                # Pindahkan file fisik jika labelnya bukan ragu_ragu
                if label != "ragu_ragu":
                    target_file = os.path.join(OUTPUT_ROOT, label, item["nama_file_unik"])
                    try:
                        shutil.move(item["path"], target_file)
                    except Exception:
                        pass

            pbar.update(len(batch))

            # Auto-save CSV setiap 10 batch (250 citra)
            if batch_num % 10 == 0 or (b_start + BATCH_SIZE) >= len(items_to_process):
                df.to_csv(CSV_PATH, index=False)

            time.sleep(1.5)  # Pacing 1.5 detik dengan 3 kunci rotasi (~13.3 RPM per kunci)

    # Simpan hasil akhir
    df.to_csv(CSV_PATH, index=False)
    print("\n[OK] CSV master berhasil diperbarui!")

    print("\n[*] Memperbarui galeri visual HTML...")
    try:
        from run_gemini_autolabel import buat_file_review_html
        buat_file_review_html(df, OUTPUT_ROOT)
        print("[OK] review_label.html berhasil diperbarui!")
    except Exception as e:
        print("[WARN] Gagal update review_label.html:", e)

    try:
        import subprocess
        subprocess.run(["python", "inspect_ragu_ragu.py"], check=False)
        print("[OK] pemeriksaan_ragu_ragu.html berhasil diperbarui!")
    except Exception as e:
        print("[WARN] Gagal update pemeriksaan_ragu_ragu.html:", e)

    print("\n[*] Mengompres ulang folder hasil_labeling menjadi ZIP...")
    shutil.make_archive("hasil_labeling", "zip", ".", OUTPUT_ROOT)
    print("[OK] hasil_labeling.zip berhasil diperbarui!")

    print("\n" + "=" * 80)
    print("  SELESAI RE-PROSES SEMUA CITRA TERTUNDA!")
    print(f"  Total Data di CSV: {len(df):,}")
    print("  Distribusi Label Baru:")
    print(df["prediksi_label"].value_counts())
    print("=" * 80)

if __name__ == "__main__":
    main()
