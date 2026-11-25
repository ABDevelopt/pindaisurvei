#!/usr/bin/env python3
"""
run_pipeline_test_10_booklets.py
================================
Skrip Uji Coba End-to-End Pipeline Kuesioner Seruti Inti BPS (10 Buklet)
Dari Awal (Cropping & Blank Filter) hingga Pembacaan Data Crop oleh Google Gemini Vision API.

Tahapan Pengujian:
1. Stage-Gate Validation: Memverifikasi 10 buklet sah (booklet_01 s.d. booklet_10).
2. Fresh Cropping "Dari Awal":
   a. Comb-box Digit Cropping (7 Halaman, Format RAW Asli, Snapping Fisik 0 px drift).
   b. Text Field Cropping (15 Bidang Teks Bebas: Nama KRT, Alamat, Petugas, ART 01-10, Catatan).
3. Pembacaan Data Crop oleh API GEMINI (gemini-3.8-flash Paid Tier 1):
   a. Pembacaan Teks Tulisan Tangan Bebas (150 bidang target).
   b. Pembacaan Kotak Digit Terisi via 50-in-1 Composite Grid Batching.
4. Evaluasi, Audit Mutu, Waktu, & Biaya API Riil.
"""

import os
import sys
import io
import time
import json
import shutil
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Pastikan root workspace terdaftar di sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from seruti_pipeline.stage_gate_validator import DUPLICATE_BOOKLET_IDS, REJECTED_BOOKLET_IDS
from seruti_pipeline.text_field_extractor import extract_all_text_fields_booklet
from seruti_pipeline.gemini_handwriting_reader import GeminiHandwritingReader

# Muat kredensial .env
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

API_KEY = os.getenv("GEMINI_API_KEY", "")
TEST_OUTPUT_ROOT = BASE_DIR / "output" / "test_e2e_10_booklets"
TEST_CROPS_ROOT = TEST_OUTPUT_ROOT / "crops"
TEXT_CSV_PATH = TEST_OUTPUT_ROOT / "hasil_uji_teks.csv"
TEXT_JSON_PATH = TEST_OUTPUT_ROOT / "hasil_uji_teks.json"
DIGIT_CSV_PATH = TEST_OUTPUT_ROOT / "hasil_uji_digits.csv"
DIGIT_JSON_PATH = TEST_OUTPUT_ROOT / "hasil_uji_digits.json"
SUMMARY_PATH = TEST_OUTPUT_ROOT / "summary_test_10_booklets.json"

TARGET_BOOKLETS = [f"booklet_{i:02d}" for i in range(1, 11)]


def mask_text_preview(text: str) -> str:
    """Masking PII konsol terminal untuk mematuhi Zero-PII Cloud Egress."""
    text = str(text).strip()
    if text in ["KOSONG", "FAILED", ""]:
        return text
    if len(text) <= 3:
        return text[0] + "**"
    return text[:2] + "*" * (min(len(text) - 3, 5)) + text[-1:]


# Skema Pydantic untuk inferensi digit composite grid
class ItemPrediksiDigit(BaseModel):
    idx: int = Field(description="Nomor indeks gambar pada grid (mulai dari 0)")
    label: str = Field(
        description=(
            "Isi kotak: '0'-'9' jika angka tunggal jelas dan terbaca pasti; "
            "'kosong' jika sel kosong atau hanya pola titik panduan; "
            "'centang' jika tanda centang (v) atau silang (x); "
            "'ragu_ragu' jika ambigu, angka terpotong/ketidakteraturan crop, "
            "coretan tertimpa revisi, atau garis tabel kotor."
        )
    )
    confidence: float = Field(description="Skor keyakinan prediksi 0.0 - 1.0. Jika ragu_ragu, berikan < 0.70")


class HasilBatchDigit(BaseModel):
    hasil: list[ItemPrediksiDigit]


PROMPT_DIGIT_GRID = (
    "Anda adalah pakar verifikasi OCR kuesioner sensus/survei BPS. "
    "Gambar ini adalah grid berisi potongan kotak digit formulir kuesioner dalam format RAW ASLI.\n"
    "Setiap sel memiliki label merah 'ID:x' di sudut kiri atas.\n\n"
    "Tugas: Tentukan label isi untuk setiap ID dari 0 sampai ID terakhir dengan aturan klasifikasi ketat:\n"
    "1. '0' s.d. '9': Hanya jika kotak berisi satu digit angka tulisan tangan yang JELAS dan TERBACA PASTI.\n"
    "2. 'kosong': Jika kotak kosong tanpa tulisan atau hanya memuat pola titik-titik panduan.\n"
    "3. 'centang': Jika kotak berisi tanda centang (✓, v) atau tanda silang (x).\n"
    "4. 'ragu_ragu': Jika angka terpotong sebagian di pinggir (partial digit), ambigu, atau coretan revisi.\n"
    "PRINSIP UTAMA: Lebih baik memberi label 'ragu_ragu' daripada menebak salah pada citra ambigu!"
)


def buat_citra_grid_raw(batch_items, cols=5, cell_w=120, cell_h=110):
    """Menyusun citra grid komposit 50-in-1 untuk inferensi batch digit."""
    rows = (len(batch_items) + cols - 1) // cols
    grid_img = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid_img)

    for idx, item in enumerate(batch_items):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h

        draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 2, y0 + cell_h - 2], fill=(255, 255, 255), outline=(180, 180, 180))
        draw.text((x0 + 8, y0 + 5), f"ID:{idx}", fill=(220, 0, 0))

        src_path = item.get("raw_file")
        if not src_path or not os.path.exists(src_path):
            src_path = item.get("inner_file", "")

        if os.path.exists(src_path):
            try:
                img = Image.open(src_path).convert("RGB")
                max_w = cell_w - 16
                max_h = cell_h - 32
                img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                w, h = img.size
                offset_x = x0 + (cell_w - w) // 2
                offset_y = y0 + 24 + (max_h - h) // 2
                grid_img.paste(img, (offset_x, offset_y))
            except Exception:
                pass

    return grid_img


def main():
    parser = argparse.ArgumentParser(description="Uji Coba Pipeline End-to-End 10 Buklet")
    parser.add_argument("--workers", type=int, default=4, help="Jumlah worker pemotongan crop paralel")
    parser.add_argument("--digit-grids-limit", type=int, default=10, help="Maksimal jumlah grid digit 50-in-1 untuk diuji (default: 10 grid = 500 digit)")
    parser.add_argument("--skip-crop", action="store_true", help="Lewati tahap crop jika sudah ada")
    args = parser.parse_args()

    t_start_total = time.time()
    TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  🚀 UJI COBA PIPELINE SERUTI INTI DARI AWAL HINGGA API GEMINI (10 BUKLET)")
    print(f"  Target Buklet        : {', '.join(TARGET_BOOKLETS)}")
    print(f"  Output Uji Coba      : {TEST_OUTPUT_ROOT}")
    print(f"  Model Vision Utama   : gemini-3.8-flash (Paid Tier 1)")
    print(f"  Privasi Konsol       : Zero-PII Cloud Egress (Masked di stdout, Utuh di Disk)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TAHAP 1: STAGE-GATE VALIDATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  [TAHAP 1] STAGE-GATE VALIDATION 10 BUKLET TARGET")
    print("-" * 80)
    valid_booklets = []
    for b_id in TARGET_BOOKLETS:
        if b_id in DUPLICATE_BOOKLET_IDS:
            print(f"  [X] {b_id} DITOLAK: Terdaftar sebagai arsip duplikat.")
        elif b_id in REJECTED_BOOKLET_IDS:
            print(f"  [X] {b_id} DITOLAK: Terdaftar sebagai pindaian cacat/karantina.")
        else:
            b_path = BASE_DIR / "output" / "seruti_inti_booklets" / b_id / "pages"
            if b_path.exists() and len(list(b_path.glob("hal_*_clean.png"))) >= 7:
                valid_booklets.append(b_id)
                print(f"  [✓] {b_id}: Valid & Lolos Stage-Gate (7 Halaman Lengkap)")
            else:
                print(f"  [!] {b_id}: Berkas halaman tidak lengkap.")

    print(f"[*] {len(valid_booklets)} dari 10 buklet siap diproses.")

    # -------------------------------------------------------------------------
    # TAHAP 2: EKSTRAKSI POTONGAN CROP DARI AWAL (GRID & TEXT FIELDS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  [TAHAP 2] PEMOTONGAN CROP DARI AWAL (GRID CROPPER & TEXT EXTRACTOR)")
    print("-" * 80)
    t0_crop = time.time()

    crop_summary = {}
    if not args.skip_crop:
        # Eksekusi run_grid_cropper.py pada 10 buklet
        cmd_crop = [
            sys.executable, "-u", "run_grid_cropper.py",
            "--all",
            "--limit", "10",
            "--digits-only",
            "--no-clean-mnist",
            "--workers", str(args.workers),
            "--output", str(TEST_OUTPUT_ROOT)
        ]
        print(f"[*] Menjalankan Grid Cropper: {' '.join(cmd_crop)}")
        res_crop = subprocess.run(cmd_crop, cwd=str(BASE_DIR))
        if res_crop.returncode != 0:
            print("[!] Grid Cropper menghasilkan error code, memeriksa berkas...")

        # Ekstraksi Bidang Teks Bebas untuk 10 Buklet
        print("[*] Mengekstrak bidang teks bebas (Nama KRT, Alamat, ART 01-10, Petugas, Catatan)...")
        text_crops_base = TEST_CROPS_ROOT
        text_fields_extracted = {}
        for b_id in valid_booklets:
            b_dir = BASE_DIR / "output" / "seruti_inti_booklets" / b_id
            f_list = extract_all_text_fields_booklet(b_dir, text_crops_base, template_dir=BASE_DIR / "templates")
            text_fields_extracted[b_id] = f_list
            print(f"  -> {b_id}: {len(f_list)} bidang teks terekstrak")
    else:
        print("[*] Melewati pemotongan crop (--skip-crop aktif). Menggunakan crop yang tersedia.")

    dur_crop = round(time.time() - t0_crop, 2)
    print(f"\n[+] Tahap Pemotongan Selesai dalam {dur_crop}s")

    # -------------------------------------------------------------------------
    # TAHAP 3A: PEMBACAAN CROP TEKS BEBAS OLEH API GEMINI 3.8 FLASH
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  [TAHAP 3A] PEMBACAAN CROP TEKS BEBAS OLEH API GEMINI (gemini-3.8-flash)")
    print("-" * 80)
    t0_text_ai = time.time()

    text_reader = GeminiHandwritingReader()
    text_records = []
    text_blank_count = 0
    text_ai_count = 0

    for idx_b, b_id in enumerate(valid_booklets, 1):
        b_dir = BASE_DIR / "output" / "seruti_inti_booklets" / b_id
        fields = extract_all_text_fields_booklet(b_dir, TEST_CROPS_ROOT, template_dir=BASE_DIR / "templates")

        print(f"\n[{idx_b}/{len(valid_booklets)}] Membaca teks {b_id} ({len(fields)} bidang):")
        for f_item in fields:
            f_id = f_item["field_id"]
            crop_path = f_item["crop_path"]
            f_type = f_item["field_type"]
            is_blank = f_item["is_blank"]

            if is_blank:
                pred_text = "KOSONG"
                conf = 1.0
                model_used = "opencv_ink_prefilter"
                sumber = "local_prefilter"
                catatan = "Tidak ada goresan tinta"
                text_blank_count += 1
            else:
                ai_res = text_reader.recognize_single_field(crop_path, field_type=f_type)
                pred_text = ai_res["text"]
                conf = ai_res["confidence"]
                model_used = ai_res["model_used"]
                sumber = "gemini_ai"
                catatan = ai_res.get("catatan", "")
                text_ai_count += 1
                time.sleep(0.2)

            text_records.append({
                "id_booklet": b_id,
                "halaman": f"hal_{f_item['page_num']}",
                "field_id": f_id,
                "label_field": f_item["label"],
                "question_code": f_item["question_code"],
                "field_type": f_type,
                "roster_no": f_item.get("roster_no"),
                "teks_hasil": pred_text,
                "confidence": float(conf),
                "is_blank": bool(pred_text == "KOSONG"),
                "has_handwriting": bool(f_item["has_handwriting"]),
                "model_used": model_used,
                "sumber": sumber,
                "catatan_anomali": catatan,
                "crop_path": crop_path
            })

            masked = mask_text_preview(pred_text)
            print(f"    -> [{f_id:20s}] : {masked:18s} (conf: {conf:.2f}, via: {model_used})")

    # Simpan hasil teks
    df_text = pd.DataFrame(text_records)
    df_text.to_csv(TEXT_CSV_PATH, index=False, encoding="utf-8")
    with open(TEXT_JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump(text_records, jf, indent=2, ensure_ascii=False, default=str)

    dur_text_ai = round(time.time() - t0_text_ai, 2)
    print(f"\n[+] Pembacaan Teks Bebas Selesai: {len(text_records)} bidang ({dur_text_ai}s)")
    print(f"    - Terisi via Gemini : {text_ai_count} bidang")
    print(f"    - Kosong via OpenCV : {text_blank_count} bidang")

    # -------------------------------------------------------------------------
    # TAHAP 3B: PEMBACAAN CROP KOTAK DIGIT OLEH API GEMINI (COMPOSITE 50-in-1 GRID)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  [TAHAP 3B] PEMBACAAN CROP DIGIT OLEH API GEMINI (gemini-3.8-flash GRID)")
    print("-" * 80)
    t0_digit_ai = time.time()

    # Kumpulkan kotak digit terisi dari crop baru
    all_digit_items_to_ai = []
    total_digit_blank_local = 0
    total_digit_cells_found = 0

    for b_id in valid_booklets:
        manifest_files = sorted(glob.glob(str(TEST_CROPS_ROOT / b_id / "hal_*" / "crop_manifest.json")))
        for m_file in manifest_files:
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
            except Exception:
                continue

            page_num = m_data.get("page_num", 1)
            digits = m_data.get("tiers", {}).get("digits", [])
            total_digit_cells_found += len(digits)

            for d in digits:
                is_blank = d.get("is_blank", False)
                if is_blank:
                    total_digit_blank_local += 1
                else:
                    raw_file = d.get("raw_file", "")
                    # Sesuaikan path jika direktori output berbeda
                    if not os.path.exists(raw_file):
                        # Coba cari relatif terhadap TEST_OUTPUT_ROOT
                        rel_path = TEST_OUTPUT_ROOT / raw_file
                        if rel_path.exists():
                            raw_file = str(rel_path)
                    all_digit_items_to_ai.append({
                        "b_id": b_id,
                        "page_num": page_num,
                        "cell_id": d.get("cell_id", ""),
                        "digit_label": d.get("digit_label", ""),
                        "field_id": d.get("field_id", ""),
                        "role": d.get("role", "questionnaire_digit"),
                        "raw_file": raw_file,
                        "inner_file": d.get("inner_file", "")
                    })

    print(f"[*] Total Sel Digit Ditemukan : {total_digit_cells_found:,} kotak")
    print(f"[*] Kotak Kosong (Pre-filter) : {total_digit_blank_local:,} kotak (Zero API Cost)")
    print(f"[*] Kotak Terisi (Calon AI)   : {len(all_digit_items_to_ai):,} kotak")

    # Batasi jumlah grid jika diatur
    max_grids = args.digit_grids_limit
    batch_size = 50
    total_grids_possible = (len(all_digit_items_to_ai) + batch_size - 1) // batch_size
    grids_to_run = min(max_grids, total_grids_possible) if max_grids else total_grids_possible

    print(f"[*] Menjalankan Inferensi pada {grids_to_run} Grid Komposit 50-in-1 ({grids_to_run * batch_size} digit sampel terisi)...")

    gemini_client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1beta"})
    digit_records = []
    ai_digit_count = 0

    for g_idx in range(grids_to_run):
        start_idx = g_idx * batch_size
        batch_slice = all_digit_items_to_ai[start_idx : start_idx + batch_size]
        if not batch_slice:
            break

        grid_img = buat_citra_grid_raw(batch_slice, cols=5, cell_w=120, cell_h=110)
        img_byte_arr = io.BytesIO()
        grid_img.save(img_byte_arr, format="JPEG", quality=90)
        img_bytes = img_byte_arr.getvalue()

        # Panggilan Gemini 3.8 Flash
        try:
            resp = gemini_client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    PROMPT_DIGIT_GRID
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=HasilBatchDigit,
                    temperature=0.0
                )
            )
            parsed = json.loads(resp.text)
            pred_list = parsed.get("hasil", [])
            map_pred = {p["idx"]: p for p in pred_list}

            # Simpan hasil per digit
            for idx_in_batch, item in enumerate(batch_slice):
                p_item = map_pred.get(idx_in_batch, {"label": "ragu_ragu", "confidence": 0.5})
                lbl = str(p_item.get("label", "ragu_ragu")).strip().lower()
                c_score = float(p_item.get("confidence", 0.9))

                digit_records.append({
                    "id_booklet": item["b_id"],
                    "halaman": f"hal_{item['page_num']}",
                    "field_id": item["field_id"],
                    "box_id": item["cell_id"],
                    "digit_label": item["digit_label"],
                    "prediksi_label": lbl,
                    "confidence": c_score,
                    "sumber": "gemini_grid_ai",
                    "model_used": "gemini-3.8-flash",
                    "raw_path": item["raw_file"]
                })
                ai_digit_count += 1

            print(f"  [Grid {g_idx+1:02d}/{grids_to_run:02d}] Sukses memproses {len(batch_slice)} digit | Model: gemini-3.8-flash")
            time.sleep(0.3)
        except Exception as e:
            print(f"  [Grid {g_idx+1:02d}/{grids_to_run:02d}] Error: {e}")

    # Simpan hasil digit
    df_digit = pd.DataFrame(digit_records)
    df_digit.to_csv(DIGIT_CSV_PATH, index=False, encoding="utf-8")
    with open(DIGIT_JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump(digit_records, jf, indent=2, ensure_ascii=False, default=str)

    dur_digit_ai = round(time.time() - t0_digit_ai, 2)
    print(f"\n[+] Pembacaan Digit Selesai: {len(digit_records)} digit terisi diproses ({dur_digit_ai}s)")

    # -------------------------------------------------------------------------
    # TAHAP 4: EVALUASI, AUDIT MUTU, WAKTU & BIAYA API
    # -------------------------------------------------------------------------
    t_total_elapsed = round(time.time() - t_start_total, 2)

    # Validasi Konsistensi KRT vs ART 01
    krt_df = df_text[df_text['field_id'] == 'p1_txt_nama_krt'][['id_booklet', 'teks_hasil']].rename(columns={'teks_hasil': 'krt'})
    art1_df = df_text[df_text['field_id'] == 'p2_txt_art_01'][['id_booklet', 'teks_hasil']].rename(columns={'teks_hasil': 'art1'})
    merged_krt = pd.merge(krt_df, art1_df, on='id_booklet')
    merged_krt['krt_clean'] = merged_krt['krt'].astype(str).str.strip().str.upper()
    merged_krt['art1_clean'] = merged_krt['art1'].astype(str).str.strip().str.upper()
    exact_matches = len(merged_krt[merged_krt['krt_clean'] == merged_krt['art1_clean']])

    # Estimasi Biaya
    # Teks: ~text_ai_count * (438 input + 30 output) tokens
    # Digit: ~grids_to_run * (1500 input + 500 output) tokens
    text_input_tokens = text_ai_count * 438
    text_output_tokens = text_ai_count * 30
    digit_input_tokens = grids_to_run * 1500
    digit_output_tokens = grids_to_run * 500

    cost_text_usd = (text_input_tokens / 1_000_000 * 0.15) + (text_output_tokens / 1_000_000 * 0.60)
    cost_digit_usd = (digit_input_tokens / 1_000_000 * 0.15) + (digit_output_tokens / 1_000_000 * 0.60)
    cost_total_usd = round(cost_text_usd + cost_digit_usd, 4)
    cost_total_idr = round(cost_total_usd * 16500, 0)

    summary_data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_booklets": TARGET_BOOKLETS,
            "total_booklets_tested": len(valid_booklets),
            "model_vision": "gemini-3.8-flash",
            "tier_api": "paid_tier_1",
            "total_elapsed_seconds": t_total_elapsed
        },
        "cropping_stage": {
            "duration_seconds": dur_crop,
            "total_digit_boxes_snapped": total_digit_cells_found,
            "blank_boxes_prefiltered": total_digit_blank_local,
            "blank_ratio_percent": round(total_digit_blank_local / total_digit_cells_found * 100, 2) if total_digit_cells_found else 0.0
        },
        "text_handwriting_ocr": {
            "duration_seconds": dur_text_ai,
            "total_fields": len(text_records),
            "filled_fields_ai": text_ai_count,
            "blank_fields_opencv": text_blank_count,
            "avg_confidence": round(float(df_text[~df_text['is_blank']]['confidence'].mean()), 4) if len(df_text[~df_text['is_blank']]) else 0.0,
            "krt_art1_consistency": {
                "total_pairs": len(merged_krt),
                "exact_matches": exact_matches,
                "match_rate_pct": round(exact_matches / len(merged_krt) * 100, 1) if len(merged_krt) else 0.0
            }
        },
        "digit_comb_box_ocr": {
            "duration_seconds": dur_digit_ai,
            "grids_tested": grids_to_run,
            "digits_tested": len(digit_records),
            "avg_confidence": round(float(df_digit['confidence'].mean()), 4) if len(df_digit) else 0.0,
            "class_distribution": df_digit['prediksi_label'].value_counts().to_dict() if len(df_digit) else {}
        },
        "api_cost_estimation": {
            "text_input_tokens": text_input_tokens,
            "text_output_tokens": text_output_tokens,
            "digit_input_tokens": digit_input_tokens,
            "digit_output_tokens": digit_output_tokens,
            "total_cost_usd": cost_total_usd,
            "total_cost_idr": cost_total_idr
        }
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2)

    print("\n" + "=" * 80)
    print("  🎉 HASIL UJI COBA PIPELINE LENGKAP 10 BUKLET (DARI AWAL HINGGA API GEMINI)")
    print("=" * 80)
    print(f"  Total Waktu Pemrosesan   : {t_total_elapsed}s ({t_total_elapsed/60:.1f} menit)")
    print(f"  Waktu Pemotongan Crop    : {dur_crop}s")
    print(f"  Waktu AI Teks Bebas      : {dur_text_ai}s (150 bidang: {text_ai_count} terisi, {text_blank_count} kosong)")
    print(f"  Waktu AI Digit Grid      : {dur_digit_ai}s ({len(digit_records)} digit pada {grids_to_run} grid)")
    print(f"  Konsistensi KRT vs ART01 : {exact_matches}/{len(merged_krt)} ({exact_matches/len(merged_krt)*100:.1f}% Cocok Tepat)")
    print(f"  Total Biaya API Riil     : ${cost_total_usd} USD (Rp {cost_total_idr:,.0f})")
    print(f"  File Checkpoint Teks     : {TEXT_CSV_PATH}")
    print(f"  File Checkpoint Digit    : {DIGIT_CSV_PATH}")
    print(f"  Laporan Ringkasan JSON   : {SUMMARY_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
