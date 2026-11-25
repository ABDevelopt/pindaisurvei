"""
api/routes_validation.py — Validasi konsistensi aritmetika & aturan relasi Seruti
ZERO DUMMY DATA: Menggunakan data riil hasil ekstraksi atau relasi resmi templates/semantic_relations.json
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from db.db_init import get_connection, EXTRACTION_DIR

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent
SEMANTIC_RELATIONS_FILE = BASE_DIR / "templates" / "semantic_relations.json"


class ReconcileRequest(BaseModel):
    booklet_id: str
    auto_apply: bool = False


# ─── GET /api/validation/{booklet_id} ─────────────────────────────────────────
@router.get("/{booklet_id}")
async def get_validation_report(booklet_id: str):
    """
    Mengambil hasil validasi aturan konsistensi (R01–R11 / REL_*) untuk buklet tertentu.
    Jika booklet_XX_validation.json sudah ada -> kembalikan hasil riil.
    Jika belum ada tetapi extracted.json ada -> jalankan SerutiArithmeticValidator secara langsung (live).
    Jika belum diekstraksi -> kembalikan relasi resmi dari templates/semantic_relations.json
    dengan status PENDING (tanpa data fiktif / dummy).
    """
    ext_dir = EXTRACTION_DIR / booklet_id

    # 1. Cek file laporan validasi riil yang sudah pernah digenerate
    candidates = []
    if ext_dir.exists():
        candidates.extend([
            ext_dir / f"{booklet_id}_validation.json",
            ext_dir / "validation_report.json",
            ext_dir / "validation.json"
        ])
    # Cek juga di folder laporan validasi global
    candidates.append(BASE_DIR / "output" / "reports" / "validation" / f"{booklet_id}_validation.json")

    for p in candidates:
        if p.exists():
            try:
                report_data = json.loads(p.read_text(encoding="utf-8"))
                return _format_validation_response(booklet_id, report_data)
            except Exception:
                continue

    # 2. Jika laporan belum ada tapi file extracted.json ada, jalankan validator riil
    extracted_candidates = [
        ext_dir / f"{booklet_id}_extracted.json",
        ext_dir / "extracted.json"
    ] if ext_dir.exists() else []

    for ext_path in extracted_candidates:
        if ext_path.exists():
            try:
                extracted_data = json.loads(ext_path.read_text(encoding="utf-8"))
                from seruti_pipeline.validator import SerutiArithmeticValidator
                validator = SerutiArithmeticValidator()
                report = validator.validate_booklet(extracted_data)
                report_dict = report.to_dict()

                # Simpan agar pemanggilan berikutnya instan
                try:
                    save_path = ext_dir / f"{booklet_id}_validation.json"
                    save_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

                return _format_validation_response(booklet_id, report_dict)
            except Exception as e:
                print(f"[validation] Error running live validator for {booklet_id}: {e}")
                break

    # 3. Jika belum diekstraksi, kembalikan relasi resmi dari template dengan status PENDING
    return _generate_pending_relations_response(booklet_id)


def _format_validation_response(booklet_id: str, report_data: dict) -> dict:
    """Format hasil validasi riil dari mesin SerutiArithmeticValidator."""
    summary = report_data.get("summary", {})
    raw_validations = report_data.get("validations", report_data.get("rules", []))

    rules = []
    for item in raw_validations:
        status_raw = str(item.get("status", "")).upper()
        if status_raw in ["MATCH", "PASS", "SUCCESS", "VALID"]:
            norm_status = "PASS"
        elif status_raw in ["MISMATCH", "FAIL", "ERROR", "INVALID"]:
            norm_status = "FAIL"
        else:
            norm_status = "WARN"

        rules.append({
            "id": item.get("rule_id", "R_UNSPECIFIED"),
            "name": item.get("description", item.get("rule_name", item.get("rule_id", ""))),
            "type": item.get("type", "arithmetic"),
            "target_field": item.get("target_field", ""),
            "target_value": item.get("target_value"),
            "expected_value": item.get("expected_value"),
            "diff": item.get("difference", 0),
            "status": norm_status,
            "raw_status": status_raw
        })

    passed_count = sum(1 for r in rules if r["status"] == "PASS")
    total_count = len(rules)

    return {
        "booklet_id": booklet_id,
        "is_pending": False,
        "summary": {
            "total_rules": total_count,
            "rules_passed": passed_count,
            "rules_mismatched": total_count - passed_count,
            "pass_rate_pct": round((passed_count / max(total_count, 1)) * 100, 2),
            "all_passed": passed_count == total_count and total_count > 0,
            "average_confidence": summary.get("average_confidence", 0),
            "total_fields": summary.get("total_fields", 0),
        },
        "rules": rules
    }


def _generate_pending_relations_response(booklet_id: str) -> dict:
    """
    Mengambil relasi resmi dari templates/semantic_relations.json dengan status PENDING.
    TIDAK ADA DATA DUMMY / ANGKA FIKTIF.
    """
    rules = []
    if SEMANTIC_RELATIONS_FILE.exists():
        try:
            sem_data = json.loads(SEMANTIC_RELATIONS_FILE.read_text(encoding="utf-8"))
            for rel in sem_data.get("relations", []):
                rules.append({
                    "id": rel.get("relation_id", ""),
                    "name": rel.get("description", ""),
                    "type": rel.get("rule_type", "arithmetic"),
                    "target_field": rel.get("target_field", ""),
                    "target_value": None,
                    "expected_value": None,
                    "diff": 0,
                    "status": "PENDING",
                    "raw_status": "PENDING"
                })
        except Exception as e:
            print(f"[validation] Error reading semantic_relations.json: {e}")

    return {
        "booklet_id": booklet_id,
        "is_pending": True,
        "message": f"Kuesioner {booklet_id} belum melalui tahap ekstraksi OCR. Status relasi PENDING.",
        "summary": {
            "total_rules": len(rules),
            "rules_passed": 0,
            "rules_mismatched": 0,
            "pass_rate_pct": 0.0,
            "all_passed": False
        },
        "rules": rules
    }


# ─── POST /api/validation/reconcile/{booklet_id} ──────────────────────────────
@router.post("/reconcile/{booklet_id}")
async def reconcile_booklet(booklet_id: str, background_tasks: BackgroundTasks):
    """
    Menjalankan script seruti validator dengan opsi auto-reconcile
    """
    ext_file = EXTRACTION_DIR / booklet_id / f"{booklet_id}_extracted.json"
    if not ext_file.exists():
        raise HTTPException(status_code=404, detail="File extracted.json tidak ditemukan untuk divalidasi")

    cmd = [
        sys.executable,
        str(BASE_DIR / "run_seruti_validator.py"),
        "--booklet", booklet_id,
        "--auto-reconcile"
    ]

    def _run_val():
        try:
            subprocess.run(cmd, cwd=str(BASE_DIR), check=True, capture_output=True)
            conn = get_connection()
            conn.execute("UPDATE booklets SET val_status='PASS', updated_at=datetime('now','localtime') WHERE id=?", (booklet_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[validator] Error running reconcile: {e}")

    background_tasks.add_task(_run_val)

    return {
        "status": "reconciling",
        "booklet_id": booklet_id,
        "message": f"Auto-rekonsiliasi untuk {booklet_id} sedang diproses di background."
    }
