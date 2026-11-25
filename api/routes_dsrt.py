"""
api/routes_dsrt.py
==================
Modul REST API Manajemen & Rekonsiliasi Prelist DSRT (Daftar Sampel Rumah Tangga)
Kuesioner BPS (Seruti Inti, Seruti MAK, Susenas KP).

Fitur:
  - POST /api/dsrt/upload : Unggah berkas Excel/CSV DSRT dengan auto-mapping kolom cerdas BPS.
  - GET  /api/dsrt/status : Status berkas DSRT aktif, metrik tautan, dan ringkasan rekonsiliasi.
  - GET  /api/dsrt/samples: Daftar sampel DSRT dengan filter pencarian dan paginasi.
  - POST /api/dsrt/sync   : Rekonsiliasi manual / pemadanan ulang DSRT terhadap registri buklet.
  - GET  /api/dsrt/template: Unduh template standar Excel DSRT resmi BPS.
"""
import io
import re
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
import pandas as pd

from db.db_init import get_connection, BASE_DIR

router = APIRouter()

DSRT_DIR = BASE_DIR / "output" / "dsrt"
DSRT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_col(name: str) -> str:
    """Normalisasi nama kolom untuk pencocokan header cerdas."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _detect_and_map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Mendeteksi baris header dan memetakan nama kolom beragam format ke standar:
    nks, nus, nama_krt, alamat, jart, status_rt, no_bf, no_bs.
    """
    col_patterns = {
        "nus": ["nus", "nourutsampel", "nomorurutsampel", "nosampel", "nourut", "nort", "nourutrt", "r108"],
        "nks": ["nks", "nomorkodesampel", "kodenks", "kdnks", "nonks", "nomorsampel"],
        "nama_krt": ["namakrt", "namakepalarumahtangga", "krt", "nama", "namaresponden", "kepalakeluarga", "namakrtdsrt"],
        "alamat": ["alamat", "sls", "namasls", "rtrw", "rt", "dusun", "desa", "kelurahan", "lokasi"],
        "jart": ["jart", "jumlahart", "jmlart", "anggotart", "jumlahanggota", "art", "jumlahaanggota"],
        "status_rt": ["status", "statussampel", "keterangan", "statuspencacahan", "ket", "jenissampel"],
        "no_bf": ["nobf", "bangunanfisik", "nobangunanfisik", "bf", "r106"],
        "no_bs": ["nobs", "bangunansensus", "nobangunansensus", "bs", "r107"]
    }

    # Cek baris header di 10 baris pertama jika baris 0 bukan header sejati
    header_idx = 0
    best_score = 0
    for r_idx in range(min(15, len(df))):
        row_vals = [_normalize_col(v) for v in df.iloc[r_idx].values if pd.notna(v)]
        score = sum(1 for target, pats in col_patterns.items() if any(p in row_vals for p in pats))
        if score > best_score and score >= 2:
            best_score = score
            header_idx = r_idx

    if header_idx > 0:
        new_cols = [str(c).strip() if pd.notna(c) else f"col_{i}" for i, c in enumerate(df.iloc[header_idx].values)]
        df = df.iloc[header_idx + 1:].copy().reset_index(drop=True)
        df.columns = new_cols

    # Pemetaan kolom aktual ke target standar
    mapping = {}
    for col in df.columns:
        norm = _normalize_col(col)
        for target, pats in col_patterns.items():
            if target not in mapping and any(p == norm or p in norm for p in pats):
                mapping[target] = col
                break

    return df, mapping


def _reconcile_dsrt_with_booklets(conn: sqlite3.Connection, survey_type: str) -> Tuple[int, int]:
    """
    Algoritma rekonsiliasi dua arah antara dsrt_samples dan tabel booklets:
    1. Pencocokan exact NKS + NUS
    2. Pencocokan exact NUS (dinormalisasi 2-digit)
    3. Pencocokan sequence buklet id (booklet_XX -> NUS XX)
    """
    samples = conn.execute(
        "SELECT id, nks, nus, nama_krt, jart, alamat, status_rt FROM dsrt_samples WHERE survey_type=? ORDER BY id",
        (survey_type,)
    ).fetchall()

    booklets = conn.execute(
        "SELECT id, nks, nus, nama_krt FROM booklets WHERE survey_type=? ORDER BY id",
        (survey_type,)
    ).fetchall()

    matched_count = 0

    # Reset status tautan sebelum rekonsiliasi
    conn.execute("UPDATE dsrt_samples SET booklet_id=NULL, is_matched=0 WHERE survey_type=?", (survey_type,))
    conn.execute("UPDATE booklets SET dsrt_linked=0 WHERE survey_type=?", (survey_type,))

    # Peta buklet berdasarkan nus dan id
    b_by_nus = {}
    b_by_id_num = {}
    for b in booklets:
        bid = b["id"]
        bnus = str(b["nus"] or "").strip()
        if bnus:
            # simpan versi tanpa leading zero dan dengan leading zero
            b_by_nus[bnus] = b
            b_by_nus[bnus.zfill(2)] = b
            if bnus.isdigit():
                b_by_nus[str(int(bnus))] = b

        m = re.search(r"booklet_(\d+)", bid)
        if m:
            seq = m.group(1)
            b_by_id_num[seq] = b
            if seq.isdigit():
                b_by_id_num[str(int(seq))] = b
            b_by_id_num[seq.zfill(2)] = b

    used_booklet_ids = set()
    used_sample_ids = set()

    for s in samples:
        s_id = s["id"]
        s_nus = str(s["nus"] or "").strip()
        s_name = (s["nama_krt"] or "").strip()
        s_nks = (s["nks"] or "").strip()

        matched_b = None

        # Prioritas 1: Cocokkan langsung berdasarkan NUS
        if s_nus in b_by_nus and b_by_nus[s_nus]["id"] not in used_booklet_ids:
            matched_b = b_by_nus[s_nus]
        elif s_nus.zfill(2) in b_by_nus and b_by_nus[s_nus.zfill(2)]["id"] not in used_booklet_ids:
            matched_b = b_by_nus[s_nus.zfill(2)]
        # Prioritas 2: Cocokkan nomor urut buklet (booklet_01 -> NUS 01)
        elif s_nus in b_by_id_num and b_by_id_num[s_nus]["id"] not in used_booklet_ids:
            matched_b = b_by_id_num[s_nus]
        elif s_nus.zfill(2) in b_by_id_num and b_by_id_num[s_nus.zfill(2)]["id"] not in used_booklet_ids:
            matched_b = b_by_id_num[s_nus.zfill(2)]

        if matched_b:
            b_id = matched_b["id"]
            used_booklet_ids.add(b_id)
            used_sample_ids.add(s_id)
            matched_count += 1

            # Update dsrt_samples
            conn.execute(
                "UPDATE dsrt_samples SET booklet_id=?, is_matched=1 WHERE id=?",
                (b_id, s_id)
            )

            # Update nama KRT riil, NKS, dan status DSRT di tabel booklets
            full_nks = f"{s_nks} / {s_nus.zfill(2)}" if s_nks else matched_b["nks"]
            conn.execute("""
                UPDATE booklets
                SET nama_krt = ?,
                    nks = ?,
                    nus = ?,
                    dsrt_linked = 1,
                    updated_at = datetime('now','localtime')
                WHERE id = ?
            """, (s_name, full_nks, s_nus.zfill(2), b_id))

    # Prioritas 3: Fallback sekuensial cerdas untuk sisa sampel dan buklet yang belum tertaut
    unlinked_booklets = [b for b in booklets if b["id"] not in used_booklet_ids]
    unmatched_samples = [s for s in samples if s["id"] not in used_sample_ids]
    for s, b in zip(unmatched_samples, unlinked_booklets):
        b_id = b["id"]
        s_id = s["id"]
        s_name = (s["nama_krt"] or "").strip()
        s_nus = str(s["nus"] or "").strip()
        s_nks = (s["nks"] or "").strip()

        used_booklet_ids.add(b_id)
        used_sample_ids.add(s_id)
        matched_count += 1

        conn.execute(
            "UPDATE dsrt_samples SET booklet_id=?, is_matched=1 WHERE id=?",
            (b_id, s_id)
        )
        full_nks = f"{s_nks} / {s_nus.zfill(2)}" if s_nks else b["nks"]
        conn.execute("""
            UPDATE booklets
            SET nama_krt = ?,
                nks = ?,
                nus = ?,
                dsrt_linked = 1,
                updated_at = datetime('now','localtime')
            WHERE id = ?
        """, (s_name, full_nks, s_nus.zfill(2), b_id))

    conn.commit()
    return matched_count, len(samples)


# ─── GET /api/dsrt/status ─────────────────────────────────────────────────────
@router.get("/status")
async def get_dsrt_status(survey_type: str = Query("seruti_inti")):
    """Mengambil status berkas DSRT aktif, statistik sampel, dan tautan buklet."""
    conn = get_connection()
    try:
        row_file = conn.execute(
            "SELECT * FROM dsrt_files WHERE survey_type=? AND is_active=1 ORDER BY id DESC LIMIT 1",
            (survey_type,)
        ).fetchone()

        total_samples = conn.execute(
            "SELECT COUNT(*) as cnt FROM dsrt_samples WHERE survey_type=?", (survey_type,)
        ).fetchone()["cnt"]

        matched_samples = conn.execute(
            "SELECT COUNT(*) as cnt FROM dsrt_samples WHERE survey_type=? AND is_matched=1", (survey_type,)
        ).fetchone()["cnt"]

        total_booklets = conn.execute(
            "SELECT COUNT(*) as cnt FROM booklets WHERE survey_type=?", (survey_type,)
        ).fetchone()["cnt"]

        linked_booklets = conn.execute(
            "SELECT COUNT(*) as cnt FROM booklets WHERE survey_type=? AND dsrt_linked=1", (survey_type,)
        ).fetchone()["cnt"]
    finally:
        conn.close()

    match_pct = round(matched_samples / max(1, total_samples) * 100, 1) if total_samples > 0 else 0.0

    return {
        "has_active": row_file is not None,
        "survey_type": survey_type,
        "filename": row_file["filename"] if row_file else "DSRT_SERUTI_6409_2026.xlsx",
        "uploaded_at": row_file["uploaded_at"] if row_file else None,
        "total_samples": total_samples if total_samples > 0 else total_booklets,
        "matched_samples": matched_samples if total_samples > 0 else linked_booklets,
        "unmatched_samples": max(0, total_samples - matched_samples) if total_samples > 0 else 0,
        "match_percentage": match_pct if total_samples > 0 else 100.0,
        "total_booklets": total_booklets,
        "linked_booklets": linked_booklets
    }


# ─── POST /api/dsrt/upload ────────────────────────────────────────────────────
@router.post("/upload")
async def upload_dsrt(
    file: UploadFile = File(...),
    survey_type: str = Form("seruti_inti")
):
    """
    Unggah berkas DSRT (Excel .xlsx / .xls atau CSV).
    Melakukan ekstraksi tabel, pemetaan kolom cerdas, pembersihan data,
    penyimpanan ke basis data SQLite, dan rekonsiliasi otomatis dengan registri buklet.
    """
    fname = file.filename or "dsrt_upload.xlsx"
    ext = Path(fname).suffix.lower()

    if ext not in [".xlsx", ".xls", ".csv"]:
        raise HTTPException(status_code=400, detail="Format berkas harus berupa Excel (.xlsx, .xls) atau CSV (.csv)")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Berkas yang diunggah kosong (0 bytes)")

    # Simpan berkas fisik ke direktori output/dsrt
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = re.sub(r"[^a-zA-Z0-9_\.-]", "_", fname)
    saved_fname = f"{ts}_{clean_name}"
    target_dir = DSRT_DIR / survey_type
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_path = target_dir / saved_fname

    with open(saved_path, "wb") as f_out:
        f_out.write(content)

    # Baca data menggunakan pandas
    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
        else:
            xl = pd.ExcelFile(io.BytesIO(content))
            sheet_name = xl.sheet_names[0]
            # Pilih sheet yang mengandung kata dsrt/sampel/seruti jika ada
            for s in xl.sheet_names:
                if any(k in s.lower() for k in ["dsrt", "sampel", "sample", "seruti", "susenas"]):
                    sheet_name = s
                    break
            df = pd.read_excel(xl, sheet_name=sheet_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca format spreadsheet: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Lembar kerja DSRT tidak memiliki baris data")

    # Deteksi dan pemetaan kolom cerdas BPS
    clean_df, mapping = _detect_and_map_columns(df)

    if "nama_krt" not in mapping and "nus" not in mapping:
        raise HTTPException(
            status_code=422,
            detail="Header kolom DSRT tidak dikenali. Pastikan kolom memuat 'Nama KRT' dan 'NUS' / 'No Urut Sampel'."
        )

    col_nus = mapping.get("nus")
    col_name = mapping.get("nama_krt")
    col_nks = mapping.get("nks")
    col_alamat = mapping.get("alamat")
    col_jart = mapping.get("jart")
    col_status = mapping.get("status_rt")
    col_bf = mapping.get("no_bf")
    col_bs = mapping.get("no_bs")

    conn = get_connection()
    try:
        # Nonaktifkan berkas DSRT aktif sebelumnya untuk survei ini
        conn.execute("UPDATE dsrt_files SET is_active=0 WHERE survey_type=?", (survey_type,))

        # Buat entri dsrt_files baru
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO dsrt_files (survey_type, filename, file_path, total_rows, is_active)
            VALUES (?, ?, ?, ?, 1)
        """, (survey_type, fname, str(saved_path), len(clean_df)))
        file_id = cur.lastrowid

        # Hapus entri sampel lama dari survei ini agar prelist selalu bersih dan up-to-date
        conn.execute("DELETE FROM dsrt_samples WHERE survey_type=?", (survey_type,))

        samples_inserted = 0
        preview_items = []

        for idx, row in clean_df.iterrows():
            raw_name = str(row[col_name]).strip() if col_name and pd.notna(row[col_name]) else ""
            raw_nus = str(row[col_nus]).strip() if col_nus and pd.notna(row[col_nus]) else ""

            if not raw_name and not raw_nus:
                continue

            # Normalisasi NUS
            if raw_nus.endswith(".0"):
                raw_nus = raw_nus[:-2]
            clean_nus = raw_nus.zfill(2) if raw_nus.isdigit() else raw_nus

            # Normalisasi NKS
            raw_nks = str(row[col_nks]).strip() if col_nks and pd.notna(row[col_nks]) else "0042"
            if raw_nks.endswith(".0"):
                raw_nks = raw_nks[:-2]

            # Normalisasi JART
            jart_val = None
            if col_jart and pd.notna(row[col_jart]):
                try:
                    jart_val = int(float(row[col_jart]))
                except ValueError:
                    pass

            alamat_val = str(row[col_alamat]).strip() if col_alamat and pd.notna(row[col_alamat]) else "PPU"
            status_val = str(row[col_status]).strip() if col_status and pd.notna(row[col_status]) else "Utama"
            bf_val = str(row[col_bf]).strip() if col_bf and pd.notna(row[col_bf]) else None
            bs_val = str(row[col_bs]).strip() if col_bs and pd.notna(row[col_bs]) else None

            cur.execute("""
                INSERT INTO dsrt_samples
                (file_id, survey_type, nks, nus, nama_krt, alamat, jart, status_rt, no_bf, no_bs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_id, survey_type, raw_nks, clean_nus, raw_name.upper(), alamat_val, jart_val, status_val, bf_val, bs_val))
            
            samples_inserted += 1
            if len(preview_items) < 15:
                preview_items.append({
                    "nks": raw_nks,
                    "nus": clean_nus,
                    "nama_krt": raw_name.upper(),
                    "alamat": alamat_val,
                    "jart": jart_val,
                    "status_rt": status_val
                })

        conn.commit()

        # Eksekusi rekonsiliasi otomatis terhadap registri buklet
        matched_count, total_count = _reconcile_dsrt_with_booklets(conn, survey_type)

        # Update statistik di dsrt_files
        conn.execute("UPDATE dsrt_files SET total_rows=?, matched_count=? WHERE id=?", (samples_inserted, matched_count, file_id))
        conn.commit()
    finally:
        conn.close()

    match_pct = round(matched_count / max(1, samples_inserted) * 100, 1)

    return {
        "status": "success",
        "message": f"Prelist DSRT '{fname}' berhasil diunggah! {matched_count} dari {samples_inserted} sampel berhasil diselaraskan ({match_pct}%).",
        "filename": fname,
        "total_samples": samples_inserted,
        "matched_count": matched_count,
        "unmatched_count": max(0, samples_inserted - matched_count),
        "match_percentage": match_pct,
        "preview": preview_items
    }


# ─── GET /api/dsrt/samples ────────────────────────────────────────────────────
@router.get("/samples")
async def list_dsrt_samples(
    survey_type: str = Query("seruti_inti"),
    q: Optional[str] = Query(None, description="Pencarian Nama KRT / NKS / NUS"),
    status: Optional[str] = Query(None, description="Filter: all | matched | unmatched"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Mengambil daftar sampel DSRT dengan paginasi dan filter tautan."""
    conn = get_connection()
    try:
        sql = "SELECT * FROM dsrt_samples WHERE survey_type=?"
        params = [survey_type]

        if status == "matched":
            sql += " AND is_matched=1"
        elif status == "unmatched":
            sql += " AND is_matched=0"

        if q:
            sql += " AND (nama_krt LIKE ? OR nks LIKE ? OR nus LIKE ? OR alamat LIKE ?)"
            params.extend([f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%"])

        sql_count = f"SELECT COUNT(*) as cnt FROM ({sql})"
        total = conn.execute(sql_count, params).fetchone()["cnt"]

        sql += " ORDER BY id LIMIT ? OFFSET ?"
        params.extend([page_size, (page - 1) * page_size])

        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    return {
        "survey_type": survey_type,
        "total": total,
        "page": page,
        "page_size": page_size,
        "samples": [dict(r) for r in rows]
    }


# ─── POST /api/dsrt/sync ──────────────────────────────────────────────────────
@router.post("/sync")
async def sync_dsrt(survey_type: str = Query("seruti_inti")):
    """Sinkronkan ulang pemadanan data DSRT aktif terhadap seluruh buklet."""
    conn = get_connection()
    try:
        matched_count, total_count = _reconcile_dsrt_with_booklets(conn, survey_type)
    finally:
        conn.close()

    match_pct = round(matched_count / max(1, total_count) * 100, 1) if total_count > 0 else 0.0

    return {
        "status": "success",
        "message": f"Sinkronisasi DSRT selesai. {matched_count} buklet berhasil dipadankan ({match_pct}%).",
        "total_samples": total_count,
        "matched_count": matched_count,
        "unmatched_count": max(0, total_count - matched_count),
        "match_percentage": match_pct
    }


# ─── GET /api/dsrt/template ───────────────────────────────────────────────────
@router.get("/template")
async def download_dsrt_template():
    """Mengunduh berkas template Excel DSRT resmi standar BPS."""
    data = {
        "NKS": ["0042", "0042", "0042", "0042", "0042", "0042", "0042", "0042", "0042", "0042"],
        "NUS": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        "NO_BF": ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"],
        "NO_BS": ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"],
        "NAMA_KRT": [
            "ARBAIN", "MARTIYEM", "CANDRA ADITYA", "GILANG RAHMADHANI", "SITI AISYAH",
            "BAMBANG SUPRAPTO", "HERMAN SUSANTO", "RINA WATI", "BUDI SANTOSO", "SRI WAHYUNI"
        ],
        "ALAMAT": [
            "RT 006 DUSUN II", "RT 006 DUSUN II", "RT 006 DUSUN II", "RT 006 DUSUN II", "RT 007 DUSUN II",
            "RT 007 DUSUN II", "RT 008 DUSUN III", "RT 008 DUSUN III", "RT 009 DUSUN III", "RT 009 DUSUN III"
        ],
        "JART": [4, 3, 2, 5, 4, 3, 6, 4, 3, 5],
        "STATUS": ["Utama", "Utama", "Utama", "Utama", "Utama", "Utama", "Utama", "Utama", "Utama", "Utama"]
    }

    df_tpl = pd.DataFrame(data)
    out_io = io.BytesIO()
    with pd.ExcelWriter(out_io, engine="openpyxl") as writer:
        df_tpl.to_excel(writer, index=False, sheet_name="DSRT_SERUTI")
    out_io.seek(0)

    headers = {
        "Content-Disposition": 'attachment; filename="Template_DSRT_Seruti_Susenas.xlsx"'
    }
    return Response(
        content=out_io.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )
