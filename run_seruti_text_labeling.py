"""
Orkestrator Batch Pengenal Tulisan Tangan Bebas (Freeform Handwriting) Kuesioner Seruti Inti
- 100% Google Gemini Vision Flagship (gemini-3.8-flash) untuk bidang terisi
- Pre-filtering lokal OpenCV untuk bidang kosong tanpa goresan tinta (Zero API Cost)
- Checkpoint Thread-Safe tersimpan di hasil_labeling_teks/hasil_teks.csv
- Kepatuhan Zero-PII Leak pada konsol terminal
"""

import os
import glob
import time
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from seruti_pipeline.text_field_extractor import extract_all_text_fields_booklet
from seruti_pipeline.gemini_handwriting_reader import GeminiHandwritingReader
from seruti_pipeline.stage_gate_validator import DUPLICATE_BOOKLET_IDS, REJECTED_BOOKLET_IDS

OUTPUT_ROOT = Path("hasil_labeling_teks")
CSV_PATH = OUTPUT_ROOT / "hasil_teks.csv"
JSON_PATH = OUTPUT_ROOT / "hasil_teks.json"
CROPS_BASE = Path("output/crops")
TEMPLATE_DIR = Path("templates")


def mask_text_preview(text: str) -> str:
    """Masking teks untuk konsol terminal agar mematuhi protokol Zero-PII Leak."""
    text = str(text).strip()
    if text in ["KOSONG", "FAILED", ""]:
        return text
    if len(text) <= 3:
        return text[0] + "**"
    return text[:2] + "*" * (min(len(text) - 3, 5)) + text[-1:]


def save_checkpoints(records: list):
    """Menyimpan checkpoint hasil secara berkala ke CSV dan JSON."""
    if not records:
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    cols = [
        "id_booklet", "halaman", "field_id", "label_field", "question_code",
        "field_type", "roster_no", "teks_hasil", "confidence", "is_blank",
        "has_handwriting", "model_used", "sumber", "catatan_anomali", "crop_path"
    ]
    existing_cols = [c for c in cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    with open(JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump(records, jf, indent=2, ensure_ascii=False, default=str)


def main():
    parser = argparse.ArgumentParser(description="Batch Handwriting OCR Seruti Inti")
    parser.add_argument("--booklet", type=str, default=None, help="ID buklet target tertentu (misal: booklet_01)")
    parser.add_argument("--all", action="store_true", help="Proses seluruh 46 buklet sah")
    parser.add_argument("--limit-booklets", type=int, default=None, help="Batas jumlah buklet")
    parser.add_argument("--verbose", action="store_true", help="Tampilkan detail proses per bidang")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  🚀 FULL FREEFORM HANDWRITING OCR SERUTI INTI (GOOGLE GEMINI 3.8 FLASH)")
    print("  Pipeline        : 100% Gemini Vision API (Paid Tier 1) + OpenCV Pre-filter")
    print("  Bidang Target   : Nama KRT, Alamat, Roster Nama ART (01-10), Nama Petugas, Uraian Pekerjaan/Usaha, Catatan Blok VII")
    print("  Privasi Output  : Zero-PII Console Preview, Data Utuh Tersimpan di File Lokal")
    print("=" * 80)

    # 1. Muat Checkpoint jika ada
    records = []
    processed_keys = set()
    if CSV_PATH.exists():
        try:
            df_old = pd.read_csv(CSV_PATH)
            # Hanya simpan yang valid (bukan FAILED) agar bisa di-retry
            df_valid = df_old[df_old["teks_hasil"] != "FAILED"]
            records = df_valid.to_dict("records")
            for r in records:
                key = f"{r['id_booklet']}__{r['field_id']}"
                processed_keys.add(key)
            print(f"[*] Checkpoint ditemukan: {len(processed_keys):,} bidang teks telah tersimpan.")
        except Exception as e:
            print(f"[!] Gagal memuat checkpoint: {e}")

    # 2. Temukan direktori buklet sah (filter duplikat dan pindaian rusak)
    if args.booklet:
        target_dir = Path("output/seruti_inti_booklets") / args.booklet
        if not target_dir.exists():
            print(f"[ERROR] Buklet {args.booklet} tidak ditemukan di output/seruti_inti_booklets!")
            return
        booklet_dirs = [target_dir]
    else:
        booklet_dirs = sorted([
            Path(d) for d in glob.glob("output/seruti_inti_booklets/booklet_*")
            if os.path.basename(d) not in DUPLICATE_BOOKLET_IDS
            and os.path.basename(d) not in REJECTED_BOOKLET_IDS
        ])
    if not booklet_dirs:
        print("[ERROR] Tidak ditemukan folder booklet sah di output/seruti_inti_booklets/!")
        return

    if args.limit_booklets:
        booklet_dirs = booklet_dirs[:args.limit_booklets]

    print(f"[*] Memproses {len(booklet_dirs)} buklet sah...")

    reader = GeminiHandwritingReader()

    total_blank_count = 0
    total_ai_count = 0

    for idx_b, b_dir in enumerate(booklet_dirs, 1):
        b_id = b_dir.name
        print(f"\n[{idx_b}/{len(booklet_dirs)}] Memproses: {b_id}")

        # Ekstraksi seluruh bidang teks pada buklet (Hal 1, 2, 7)
        fields = extract_all_text_fields_booklet(b_dir, CROPS_BASE, template_dir=TEMPLATE_DIR)
        if not fields:
            print(f"  [WARN] Tidak ada bidang teks yang terekstrak dari {b_id}")
            continue

        booklet_new_records = 0

        for f_item in fields:
            f_id = f_item["field_id"]
            key = f"{b_id}__{f_id}"

            if key in processed_keys:
                continue

            crop_path = f_item["crop_path"]
            f_type = f_item["field_type"]
            is_blank = f_item["is_blank"]

            # JALUR A: Pre-Filter OpenCV (Kotak Kosong / Tanpa Goresan Tinta)
            if is_blank:
                pred_text = "KOSONG"
                conf = 1.0
                model_used = "opencv_ink_prefilter"
                sumber = "local_prefilter"
                catatan = "Tidak ada goresan tinta terdeteksi"
                total_blank_count += 1
            else:
                # JALUR B: Google Gemini Vision API (Paid Tier 1)
                ai_res = reader.recognize_single_field(crop_path, field_type=f_type)
                pred_text = ai_res["text"]
                conf = ai_res["confidence"]
                model_used = ai_res["model_used"]
                sumber = "gemini_ai"
                catatan = ai_res.get("catatan", "")
                total_ai_count += 1
                time.sleep(0.3)  # Pacing yang stabil dan sopan

            record_entry = {
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
            }
            records.append(record_entry)
            processed_keys.add(key)
            booklet_new_records += 1

            if args.verbose:
                masked = mask_text_preview(pred_text)
                print(f"    -> [{f_id:20s}] : {masked:15s} (conf: {conf:.2f}, model: {model_used})")

        save_checkpoints(records)
        print(f"  ✅ {b_id} Selesai: {booklet_new_records} bidang baru diproses (Kumulatif: {len(records)} bidang)")

    save_checkpoints(records)
    print("\n" + "=" * 80)
    print("  🎉 SELESAI PEMROSESAN SELURUH BIDANG TEKS BEBAS SERUTI INTI")
    print(f"  Total Bidang Terekam    : {len(records):,} entri")
    print(f"  Total Bidang Kosong     : {total_blank_count:,}")
    print(f"  Total Bidang Terisi(AI) : {total_ai_count:,}")
    print(f"  File Checkpoint CSV     : {CSV_PATH}")
    print(f"  File Checkpoint JSON    : {JSON_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
