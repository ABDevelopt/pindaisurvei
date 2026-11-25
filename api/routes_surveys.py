"""
api/routes_surveys.py — Manajemen & pemilihan multi-survei (Seruti Inti, Seruti MAK, Susenas KP)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.db_init import get_connection

router = APIRouter()

SURVEY_CATALOG = {
    "seruti_inti": {
        "id": "seruti_inti",
        "code": "VSERUTI26.INTI",
        "title": "Seruti Inti",
        "description": "Survei Ekonomi Rumah Tangga Triwulanan — Kuesioner Inti (8 Halaman)",
        "format": "4 Spread A3 (300 DPI)",
        "pages_count": 8,
    },
    "seruti_mak": {
        "id": "seruti_mak",
        "code": "VSERUTI26.MAK",
        "title": "Seruti MAK",
        "description": "Survei Ekonomi Rumah Tangga — Modul Konsumsi Bahan Makanan & Minuman",
        "format": "6 Spread A3 (300 DPI)",
        "pages_count": 12,
    },
    "susenas_kp": {
        "id": "susenas_kp",
        "code": "VSEN26.KP",
        "title": "Susenas KP",
        "description": "Survei Sosial Ekonomi Nasional — Modul Konsumsi & Pengeluaran",
        "format": "Kuesioner Rumah Tangga",
        "pages_count": 18,
    }
}


class SurveySelectRequest(BaseModel):
    survey_id: str


# ─── GET /api/surveys ────────────────────────────────────────────────────────
@router.get("")
async def list_surveys():
    """Daftar instrumen survei yang didukung beserta jumlah buklet dan status aktif."""
    conn = get_connection()

    # Ambil survei aktif dari system_config
    row_active = conn.execute("SELECT value FROM system_config WHERE key='active_survey'").fetchone()
    active_id = row_active["value"] if row_active else "seruti_inti"

    # Hitung jumlah buklet riil per survei
    counts_rows = conn.execute(
        "SELECT survey_type, COUNT(*) as cnt FROM booklets GROUP BY survey_type"
    ).fetchall()
    counts = {r["survey_type"]: r["cnt"] for r in counts_rows}
    conn.close()

    result = []
    for sid, info in SURVEY_CATALOG.items():
        item = dict(info)
        item["total_booklets"] = counts.get(sid, 0)
        item["active"] = (sid == active_id)
        result.append(item)

    return {
        "active_survey": active_id,
        "active_info": SURVEY_CATALOG.get(active_id, SURVEY_CATALOG["seruti_inti"]),
        "surveys": result
    }


# ─── POST /api/surveys/select ────────────────────────────────────────────────
@router.post("/select")
async def select_survey(req: SurveySelectRequest):
    """Ganti instrumen survei yang sedang aktif di sistem."""
    if req.survey_id not in SURVEY_CATALOG:
        raise HTTPException(
            status_code=400,
            detail=f"Survei '{req.survey_id}' tidak valid. Pilihan: {list(SURVEY_CATALOG.keys())}"
        )

    conn = get_connection()
    conn.execute(
        "INSERT INTO system_config (key, value) VALUES ('active_survey', ?) ON CONFLICT(key) DO UPDATE SET value=?",
        (req.survey_id, req.survey_id)
    )
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "active_survey": req.survey_id,
        "survey_info": SURVEY_CATALOG[req.survey_id],
        "message": f"Instrumen survei dialihkan ke {SURVEY_CATALOG[req.survey_id]['code']}."
    }
