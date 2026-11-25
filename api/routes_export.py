"""
api/routes_export.py — Ekspor data hasil pembacaan survei (CSV, JSON, FormGear ZIP)
"""
import io
import json
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd

from db.db_init import get_connection, EXTRACTION_DIR

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent


# ─── GET /api/export/batch/csv (Must be BEFORE /{booklet_id} to avoid collision) ─
@router.get("/batch/csv")
async def export_batch_csv():
    """
    Kompilasi ringkasan seluruh buklet yang telah selesai diekstraksi ke 1 file CSV.
    """
    conn = get_connection()
    booklets = conn.execute("SELECT id, nks, nus, nama_krt, gate_status, ocr_status, val_status FROM booklets").fetchall()
    conn.close()

    rows = []
    for b in booklets:
        bid = b["id"]
        val_json = EXTRACTION_DIR / bid / f"{bid}_validation.json"

        val_rate = 0.0
        avg_conf = 0.0
        if val_json.exists():
            try:
                vdata = json.loads(val_json.read_text(encoding="utf-8"))
                val_rate = vdata.get("summary", {}).get("semantic_validation", {}).get("pass_rate_pct", 0.0)
                avg_conf = vdata.get("summary", {}).get("average_confidence", 0.0)
            except Exception:
                pass

        rows.append({
            "booklet_id": bid,
            "nks": b["nks"],
            "nus": b["nus"],
            "nama_krt": b["nama_krt"],
            "gate_status": b["gate_status"],
            "ocr_status": b["ocr_status"],
            "val_status": b["val_status"],
            "average_confidence": avg_conf,
            "validation_pass_rate": val_rate
        })

    df = pd.DataFrame(rows)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=si_periksa_batch_rekap_51_buklet_unik.csv"}
    )


# ─── GET /api/export/{booklet_id}/csv ─────────────────────────────────────────
@router.get("/{booklet_id}/csv")
async def export_csv(booklet_id: str):
    """
    Ekspor hasil ekstraksi buklet ke format CSV tabular standar BPS.
    """
    ext_dir = EXTRACTION_DIR / booklet_id
    csv_candidates = [
        ext_dir / f"{booklet_id}_summary.csv",
        ext_dir / "summary.csv",
        ext_dir / f"{booklet_id}.csv"
    ]

    for p in csv_candidates:
        if p.exists():
            return FileResponse(
                str(p),
                media_type="text/csv",
                filename=f"{booklet_id}_ekstraksi_bps.csv"
            )

    json_path = ext_dir / f"{booklet_id}_extracted.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            fields = data.get("fields", [])
            df = pd.DataFrame(fields)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            return StreamingResponse(
                io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={booklet_id}_ekstraksi_bps.csv"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal generate CSV: {e}")

    raise HTTPException(status_code=404, detail=f"Data ekstraksi untuk {booklet_id} tidak ditemukan")


# ─── GET /api/export/{booklet_id}/json ────────────────────────────────────────
@router.get("/{booklet_id}/json")
async def export_json(booklet_id: str):
    """
    Ekspor data mikro format JSON untuk integrasi sistem/API.
    """
    ext_dir = EXTRACTION_DIR / booklet_id
    json_path = ext_dir / f"{booklet_id}_extracted.json"
    if not json_path.exists():
        json_path = ext_dir / "extraction.json"

    if json_path.exists():
        return FileResponse(
            str(json_path),
            media_type="application/json",
            filename=f"{booklet_id}_microdata.json"
        )

    raise HTTPException(status_code=404, detail=f"File JSON untuk {booklet_id} tidak ditemukan")


# ─── GET /api/export/{booklet_id}/zip ────────────────────────────────────────
@router.get("/{booklet_id}/zip")
async def export_formgear_zip(booklet_id: str):
    """
    Ekspor paket FormGear-compatible ZIP:
    - data.json: nilai variabel untuk CAPI/CAWI FASIH-SM
    - comment.json: field notes / audit trail
    """
    ext_dir = EXTRACTION_DIR / booklet_id
    json_path = ext_dir / f"{booklet_id}_extracted.json"

    data_payload = {}
    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            for field in raw.get("fields", []):
                var = field.get("var_name") or field.get("variable")
                val = field.get("value")
                if var:
                    data_payload[var] = val
        except Exception:
            data_payload = {"status": "partial_data"}

    conn = get_connection()
    logs = conn.execute(
        "SELECT variable, old_digit, new_digit, operator, corrected_at FROM correction_log WHERE booklet_id = ?",
        (booklet_id,)
    ).fetchall()
    conn.close()

    comment_payload = {
        "booklet_id": booklet_id,
        "source": "Si-Periksa Seruti Scanner v2.0",
        "hitl_corrections": [dict(l) for l in logs]
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("data.json", json.dumps(data_payload, indent=2, ensure_ascii=False))
        zip_file.writestr("comment.json", json.dumps(comment_payload, indent=2, ensure_ascii=False))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={booklet_id}_formgear_package.zip"}
    )
