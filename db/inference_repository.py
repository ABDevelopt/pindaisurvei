"""
db/inference_repository.py
===========================
Repositori penyimpanan dan sinkronisasi otomatis hasil inferensi AI kuesioner BPS
langsung ke basis data SQLite (db/si_periksa.db) terikat ke ID buklet (booklet_id).

Mendukung:
1. Penyimpanan riwayat inferensi penuh (JSON pages, validation, summary).
2. Penyimpanan granular nilai per-variabel kuesioner (booklet_field_values).
3. Pembaruan otomatis status OCR dan metadata buklet (NKS, NUS, Nama KRT).
4. Kepatuhan Zero-PII: Seluruh data diproses dan disimpan 100% lokal di mesin BPS.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from db.db_init import get_connection

BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTION_DIR = BASE_DIR / "output" / "extractions"


def init_inference_tables(conn: Optional[sqlite3.Connection] = None):
    """Membuat tabel booklet_inferences dan booklet_field_values jika belum ada."""
    own_conn = False
    if conn is None:
        conn = get_connection()
        own_conn = True

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS booklet_inferences (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                booklet_id       TEXT NOT NULL REFERENCES booklets(id),
                model_name       TEXT DEFAULT 'gemini_38_flash',
                total_digits     INTEGER DEFAULT 0,
                blank_digits     INTEGER DEFAULT 0,
                filled_digits    INTEGER DEFAULT 0,
                auto_accepted    INTEGER DEFAULT 0,
                needs_review     INTEGER DEFAULT 0,
                avg_confidence   REAL DEFAULT 0.0,
                extraction_json  TEXT DEFAULT '{}',
                validation_json  TEXT DEFAULT '{}',
                created_at       TEXT DEFAULT (datetime('now','localtime')),
                updated_at       TEXT DEFAULT (datetime('now','localtime')),
                is_latest        INTEGER DEFAULT 1
            );
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_inferences_booklet ON booklet_inferences(booklet_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inferences_latest ON booklet_inferences(booklet_id, is_latest);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS booklet_field_values (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                booklet_id   TEXT NOT NULL REFERENCES booklets(id),
                page         INTEGER,
                block_name   TEXT,
                field_id     TEXT NOT NULL,
                var_name     TEXT NOT NULL,
                var_label    TEXT,
                data_type    TEXT,
                raw_string   TEXT,
                value        TEXT,
                confidence   REAL DEFAULT 1.0,
                needs_review INTEGER DEFAULT 0,
                created_at   TEXT DEFAULT (datetime('now','localtime')),
                updated_at   TEXT DEFAULT (datetime('now','localtime')),
                UNIQUE(booklet_id, var_name)
            );
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_field_values_booklet ON booklet_field_values(booklet_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_field_values_var ON booklet_field_values(var_name);")
        conn.commit()
    finally:
        if own_conn:
            conn.close()


def save_booklet_inference_to_db(
    booklet_id: str,
    extraction_dict: Dict[str, Any],
    model_name: str = "gemini_38_flash",
    conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Menyimpan otomatis hasil inferensi AI kuesioner ke SQLite sesuai id booklet.
    - Menyimpan rekap statistik inferensi dan payload JSON lengkap.
    - Menyimpan nilai per-variabel kuesioner ke booklet_field_values.
    - Memperbarui status buklet di tabel booklets menjadi ocr_status = 'COMPLETED'.
    """
    init_inference_tables(conn)
    own_conn = False
    if conn is None:
        conn = get_connection()
        own_conn = True

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        pages = extraction_dict.get("pages", {})
        summary = extraction_dict.get("summary", {})
        validations = extraction_dict.get("validations", {})

        # Hitung metrik jika summary belum lengkap
        total_digits = 0
        blank_digits = 0
        filled_digits = 0
        auto_accepted = 0
        needs_review = 0
        conf_list: List[float] = []

        field_records = []

        # Ekstraksi field per halaman
        for p_key, p_data in pages.items():
            # Tentukan nomor halaman (misal 'hal_1' -> 1)
            p_num = 1
            try:
                p_num = int("".join(filter(str.isdigit, p_key)) or 1)
            except Exception:
                pass

            seen_page_vars = set()
            flds = p_data.get("fields", {})
            for var_key, f_val in flds.items():
                fid = f_val.get("field_id") or var_key
                v_name = f_val.get("var_name") or var_key
                if v_name in seen_page_vars:
                    continue
                seen_page_vars.add(v_name)
                v_label = f_val.get("label") or f_val.get("var_label") or v_name
                block = f_val.get("block") or f"Blok Hal {p_num}"
                dt = f_val.get("data_type") or "integer"
                raw_s = str(f_val.get("raw_string") or "")
                val_s = str(f_val.get("value") if f_val.get("value") is not None else "")

                # ATURAN KHUSUS TRIWULAN: Simpan sebagai Romawi (I, II, III, IV)
                if "triwulan" in fid.lower() or "triwulan" in v_name.lower():
                    roman_map = {"1": "I", "01": "I", "2": "II", "02": "II", "3": "III", "03": "III", "4": "IV", "04": "IV"}
                    clean_v = val_s.strip()
                    if clean_v in roman_map:
                        val_s = roman_map[clean_v]
                        raw_s = roman_map[clean_v]
                    elif clean_v.upper() in ["I", "II", "III", "IV"]:
                        val_s = clean_v.upper()
                        raw_s = clean_v.upper()

                mean_c = float(f_val.get("mean_confidence") or 1.0)
                n_rev = 1 if f_val.get("needs_review") else 0

                # Evaluasi digit box
                b_boxes = f_val.get("box_details", [])
                for b in b_boxes:
                    total_digits += 1
                    is_b = b.get("is_blank", False)
                    c = float(b.get("conf") or 1.0)
                    if is_b:
                        blank_digits += 1
                    else:
                        filled_digits += 1
                        conf_list.append(c)
                        if c >= 0.80 and not b.get("needs_review"):
                            auto_accepted += 1
                        elif c < 0.80 or b.get("needs_review"):
                            needs_review += 1

                field_records.append({
                    "booklet_id": booklet_id,
                    "page": p_num,
                    "block_name": block,
                    "field_id": fid,
                    "var_name": v_name,
                    "var_label": v_label,
                    "data_type": dt,
                    "raw_string": raw_s,
                    "value": val_s,
                    "confidence": mean_c,
                    "needs_review": n_rev
                })

            # Evaluasi Bidang Teks Bebas (Text Fields Gemini Vision)
            t_flds = p_data.get("text_fields", {})
            for t_key, t_val in t_flds.items():
                t_fid = t_val.get("field_id") or t_key
                t_vname = t_val.get("var_name") or t_key
                t_label = t_val.get("label") or t_val.get("var_label") or t_fid
                t_raw = str(t_val.get("raw_string") or t_val.get("value") or "")
                t_val_s = str(t_val.get("value") if t_val.get("value") is not None else "")
                t_conf = float(t_val.get("confidence") or 1.0)
                t_is_b = bool(t_val.get("is_blank", False))
                t_block = t_val.get("block") or f"Blok Hal {p_num} (Teks)"

                field_records.append({
                    "booklet_id": booklet_id,
                    "page": p_num,
                    "block_name": t_block,
                    "field_id": t_fid,
                    "var_name": t_vname,
                    "var_label": t_label,
                    "data_type": "text",
                    "raw_string": t_raw,
                    "value": "" if t_is_b else t_val_s,
                    "confidence": t_conf,
                    "needs_review": 1 if t_conf < 0.85 and not t_is_b else 0
                })

        # Gunakan summary bawaan jika ada
        if summary.get("total_digits"):
            total_digits = summary.get("total_digits", total_digits)
        if summary.get("blank_digits_opencv"):
            blank_digits = summary.get("blank_digits_opencv", blank_digits)
        if summary.get("auto_accepted"):
            auto_accepted = summary.get("auto_accepted", auto_accepted)
        if summary.get("needs_review") is not None:
            needs_review = summary.get("needs_review", needs_review)

        avg_conf = round((sum(conf_list) / max(len(conf_list), 1)) * 100, 2) if conf_list else 98.5

        # 1. Non-aktifkan flag is_latest pada entri lama
        conn.execute("UPDATE booklet_inferences SET is_latest = 0 WHERE booklet_id = ?", (booklet_id,))

        # 2. Simpan ke booklet_inferences
        ext_json_str = json.dumps(extraction_dict, ensure_ascii=False)
        val_json_str = json.dumps(validations, ensure_ascii=False)

        conn.execute("""
            INSERT INTO booklet_inferences
            (booklet_id, model_name, total_digits, blank_digits, filled_digits,
             auto_accepted, needs_review, avg_confidence, extraction_json, validation_json,
             created_at, updated_at, is_latest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            booklet_id, model_name, total_digits, blank_digits, filled_digits,
            auto_accepted, needs_review, avg_conf, ext_json_str, val_json_str,
            now, now
        ))

        # 3. Bersihkan entri lama buklet ini agar variabel usang tidak tertinggal
        conn.execute("DELETE FROM booklet_field_values WHERE booklet_id = ?", (booklet_id,))

        # Simpan nilai per-variabel ke booklet_field_values (UPSERT)
        for fr in field_records:
            conn.execute("""
                INSERT OR REPLACE INTO booklet_field_values
                (booklet_id, page, block_name, field_id, var_name, var_label, data_type, raw_string, value, confidence, needs_review, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fr["booklet_id"], fr["page"], fr["block_name"], fr["field_id"], fr["var_name"],
                fr["var_label"], fr["data_type"], fr["raw_string"], fr["value"], fr["confidence"],
                fr["needs_review"], now
            ))

        # 4. Sinkronisasi metadata otomatis ke tabel booklets
        # Ekstrak NKS (p1_F006 / b1_r106_nobs) & NUS (p1_F008 / b1_r108_nus) & Nama KRT jika ada
        extracted_nks = None
        extracted_nus = None
        extracted_krt = None

        for fr in field_records:
            v_low = fr["var_name"].lower()
            f_low = fr["field_id"].lower()
            if "r106" in v_low or "f006" in f_low:
                extracted_nks = fr["value"] or fr["raw_string"]
            elif "r108" in v_low or "f008" in f_low:
                extracted_nus = fr["value"] or fr["raw_string"]
            elif "nama_krt" in v_low or "r110" in v_low:
                extracted_krt = fr["value"] or fr["raw_string"]

        update_sql = "UPDATE booklets SET ocr_status = 'COMPLETED', updated_at = ?"
        update_params = [now]

        if extracted_nks:
            update_sql += ", nks = COALESCE(nks, ?)"
            update_params.append(extracted_nks)
        if extracted_nus:
            update_sql += ", nus = COALESCE(nus, ?)"
            update_params.append(extracted_nus)
        if extracted_krt:
            update_sql += ", nama_krt = COALESCE(nama_krt, ?)"
            update_params.append(extracted_krt)

        update_sql += " WHERE id = ?"
        update_params.append(booklet_id)

        conn.execute(update_sql, update_params)
        conn.commit()

        return {
            "success": True,
            "booklet_id": booklet_id,
            "model_name": model_name,
            "total_fields_saved": len(field_records),
            "total_digits": total_digits,
            "blank_digits": blank_digits,
            "filled_digits": filled_digits,
            "auto_accepted": auto_accepted,
            "needs_review": needs_review,
            "avg_confidence": avg_conf,
            "timestamp": now
        }

    except Exception as e:
        if own_conn:
            conn.rollback()
        raise e
    finally:
        if own_conn:
            conn.close()


def sync_all_existing_extractions_to_db() -> Dict[str, Any]:
    """
    Memindai seluruh berkas output/extractions/{booklet_id}/{booklet_id}_extracted.json
    yang telah ada dan menyinkronkannya langsung ke SQLite.
    """
    init_inference_tables()
    conn = get_connection()
    synced = []
    errors = []

    try:
        cand_files = sorted(EXTRACTION_DIR.glob("*/*_extracted.json"))
        for f in cand_files:
            booklet_id = f.parent.name
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                res = save_booklet_inference_to_db(booklet_id, data, model_name="seruti_multi_model", conn=conn)
                synced.append({
                    "booklet_id": booklet_id,
                    "fields": res["total_fields_saved"],
                    "avg_conf": res["avg_confidence"]
                })
            except Exception as e:
                errors.append({"booklet_id": booklet_id, "error": str(e)})

        conn.commit()
    finally:
        conn.close()

    return {
        "total_files_scanned": len(cand_files),
        "total_synced": len(synced),
        "total_errors": len(errors),
        "synced_booklets": synced,
        "errors": errors
    }


def get_booklet_field_values_from_db(
    booklet_id: str,
    page: Optional[int] = None,
    block_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Mengambil nilai per-variabel kuesioner dari database SQLite."""
    conn = get_connection()
    try:
        sql = "SELECT * FROM booklet_field_values WHERE booklet_id = ?"
        params: List[Any] = [booklet_id]

        if page is not None:
            sql += " AND page = ?"
            params.append(page)
        if block_name:
            sql += " AND block_name = ?"
            params.append(block_name)

        sql += " ORDER BY page ASC, field_id ASC"
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
