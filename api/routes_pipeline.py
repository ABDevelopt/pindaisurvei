"""
api/routes_pipeline.py — Pipeline runner + SSE log streaming
"""
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db.db_init import get_connection

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent

# State pipeline global (satu proses pipeline per waktu)
_pipeline_proc: Optional[subprocess.Popen] = None
_pipeline_log: list = []
_pipeline_status = {
    "running": False,
    "started_at": None,
    "stage": "idle",
    "booklets_total": 0,
    "booklets_done": 0,
    "current_booklet": None,
    "progress_pct": 0,
}


class PipelineRunRequest(BaseModel):
    booklet_ids: Optional[list] = None   # None = semua
    stages: Optional[list] = ["gate", "crop", "ocr", "validate", "sync_db", "auto_accept"]
    model: Optional[str] = "gemini_38_flash"
    workers: Optional[int] = 4


# ─── GET /api/pipeline/stages ────────────────────────────────────────────────
@router.get("/stages")
async def get_pipeline_stages():
    """Mengembalikan katalog seluruh tahapan pipeline yang dapat dipilih pengguna."""
    return {
        "stages": [
            {
                "id": "gate",
                "name": "Stage-Gate Audit",
                "description": "Evaluasi mutu 4 gerbang: deviasi affine alignment, invarian 2.073 sel digit, dan keputusan operasional.",
                "default": True,
                "badge": "Gate 0-4"
            },
            {
                "id": "crop",
                "name": "Grid Cropping & Filter",
                "description": "Pemotongan 7 tingkatan hierarki dan deteksi kotak kosong instan OpenCV (Zero API Cost).",
                "default": True,
                "badge": "Hierarki"
            },
            {
                "id": "ocr",
                "name": "OCR AI Inference",
                "description": "Pengenalan karakter (Angka Arab, Centang Persetujuan, Romawi Triwulan) hanya pada kotak terisi.",
                "default": True,
                "badge": "Zero-Loss"
            },
            {
                "id": "validate",
                "name": "Validasi Semantik",
                "description": "Pengecekan 21 aturan relasi resmi Seruti dan Smart Arithmetic Auto-Reconciliation.",
                "default": True,
                "badge": "21 Relasi"
            },
            {
                "id": "sync_db",
                "name": "Auto-Persistence SQLite",
                "description": "Penyimpanan otomatis seluruh hasil inferensi ke tabel booklet_field_values & inferences.",
                "default": True,
                "badge": "SQLite"
            },
            {
                "id": "dsrt_sync",
                "name": "Pemadanan DSRT",
                "description": "Rekonsiliasi data KOP dengan prelist sampel rumah tangga DSRT aktif.",
                "default": False,
                "badge": "Prelist"
            },
            {
                "id": "auto_accept",
                "name": "Smart Auto-Accept",
                "description": "Otomatis meloloskan digit berkeyakinan >= 80% dan template tetap (pangkas 79.2% beban HITL).",
                "default": True,
                "badge": "Aman 94%"
            }
        ]
    }





# ─── GET /api/pipeline/stage-gate/summary ───────────────────────────────────
@router.get("/stage-gate/summary")
def get_stage_gate_summary():
    """Mengambil ringkasan keputusan Stage-Gate dari seluruh buklet di basis data."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT COALESCE(gate_status, 'UNKNOWN') as status, COUNT(*) as cnt
            FROM booklets
            GROUP BY status
        """).fetchall()
        dist = {r["status"]: r["cnt"] for r in rows}
        total = sum(dist.values())
        proceed_cnt = dist.get("PROCEED_TO_OCR", 0) + dist.get("PROCEED", 0)
        return {
            "total_booklets": total,
            "distribution": dist,
            "pass_rate_pct": round(proceed_cnt / max(1, total) * 100, 1) if total > 0 else 0.0
        }
    finally:
        conn.close()


# ─── GET /api/pipeline/stage-gate/{booklet_id} ───────────────────────────────
@router.get("/stage-gate/{booklet_id}")
async def get_stage_gate_report(booklet_id: str):
    """Mengambil rincian metrik Stage-Gate untuk 1 buklet."""
    from seruti_pipeline.stage_gate_validator import StageGateValidator
    validator = StageGateValidator(workspace_root=BASE_DIR)
    report = validator.validate_booklet(booklet_id, check_snapping=False)
    return report


# ─── POST /api/pipeline/stage-gate/run ───────────────────────────────────────
class StageGateRunRequest(BaseModel):
    booklet_id: Optional[str] = None
    booklet_ids: Optional[List[str]] = None
    deep_snapping: bool = False

@router.post("/stage-gate/run")
def run_stage_gate_audit(req: StageGateRunRequest):
    """Memicu evaluasi Stage-Gate secara on-demand dan memperbarui basis data SQLite."""
    from seruti_pipeline.stage_gate_validator import StageGateValidator
    validator = StageGateValidator(workspace_root=BASE_DIR)
    conn = get_connection()
    try:
        target_id = req.booklet_id
        if not target_id and req.booklet_ids and len(req.booklet_ids) == 1:
            target_id = req.booklet_ids[0]

        if target_id:
            res = validator.validate_booklet(target_id, check_snapping=req.deep_snapping)
            dec = res.get("decision", "PROCEED_TO_OCR")
            conn.execute("UPDATE booklets SET gate_status = ?, updated_at = datetime('now','localtime') WHERE id = ?", (dec, target_id))
            conn.commit()
            return {"status": "success", "mode": "single", "booklet_id": target_id, "report": res}
        elif req.booklet_ids and len(req.booklet_ids) > 1:
            results = []
            for b_id in req.booklet_ids:
                r = validator.validate_booklet(b_id, check_snapping=req.deep_snapping)
                dec = r.get("decision", "PROCEED_TO_OCR")
                conn.execute("UPDATE booklets SET gate_status = ?, updated_at = datetime('now','localtime') WHERE id = ?", (dec, b_id))
                results.append({"id": b_id, "decision": dec})
            conn.commit()
            return {"status": "success", "mode": "batch_partial", "count": len(results), "results": results}
        else:
            summary = validator.validate_all_booklets(check_snapping_deep=req.deep_snapping)
            for b in summary.get("booklets", []):
                conn.execute("UPDATE booklets SET gate_status = ?, updated_at = datetime('now','localtime') WHERE id = ?", (b["decision"], b["booklet_id"]))
            conn.commit()
            return {"status": "success", "mode": "all", "summary": summary}
    finally:
        conn.close()





# ─── POST /api/pipeline/run ──────────────────────────────────────────────────
@router.post("/run")
async def run_pipeline(req: PipelineRunRequest):
    """Jalankan run_master_pipeline.py sebagai subprocess."""
    global _pipeline_proc, _pipeline_log, _pipeline_status

    if _pipeline_status["running"]:
        return {"status": "already_running", "message": "Pipeline sedang berjalan."}

    _pipeline_log.clear()
    _pipeline_status.update({
        "running": True,
        "started_at": datetime.now().isoformat(),
        "stage": "starting",
        "booklets_done": 0,
        "progress_pct": 0,
    })

    # Hitung total booklet
    if req.booklet_ids:
        total = len(req.booklet_ids)
    else:
        conn = get_connection()
        total = conn.execute("SELECT COUNT(*) FROM booklets").fetchone()[0]
        conn.close()
    _pipeline_status["booklets_total"] = total

    # Simpan ke DB
    conn = get_connection()
    run_id = conn.execute(
        "INSERT INTO pipeline_runs (booklets_total, status) VALUES (?, 'RUNNING')",
        (total,)
    ).lastrowid
    conn.commit()
    conn.close()

    # Build command
    cmd = [sys.executable, str(BASE_DIR / "run_master_pipeline.py")]
    if req.booklet_ids:
        cmd += ["--booklets"] + req.booklet_ids
    else:
        cmd += ["--all"]  # BUG FIX: tanpa booklet_ids = proses semua buklet

    if req.stages:
        cmd += ["--stages"] + req.stages
    if req.model:
        cmd += ["--model", req.model]
    if req.workers:
        cmd += ["--workers", str(req.workers)]

    # Jalankan subprocess secara non-blocking
    asyncio.create_task(_run_subprocess(cmd, run_id))

    return {
        "status": "started",
        "run_id": run_id,
        "message": f"Pipeline dimulai untuk {total} buklet dengan {len(req.stages or [])} tahapan.",
        "log_stream": "/api/pipeline/log"
    }


async def _run_subprocess(cmd: list, run_id: int):
    """Jalankan subprocess pipeline dan capture output."""
    global _pipeline_proc, _pipeline_log, _pipeline_status
    try:
        _pipeline_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )

        async for line in _pipeline_proc.stdout:
            line_str = line.decode("utf-8", errors="replace").rstrip()
            _pipeline_log.append(line_str)
            # Keep last 1000 lines
            if len(_pipeline_log) > 1000:
                _pipeline_log.pop(0)
            # Parse progress hints
            _parse_progress_line(line_str)

        await _pipeline_proc.wait()
        success = _pipeline_proc.returncode == 0

    except Exception as e:
        _pipeline_log.append(f"[ERROR] {e}")
        success = False
    finally:
        _pipeline_status.update({
            "running": False,
            "stage": "done" if success else "error",
            "progress_pct": 100 if success else _pipeline_status["progress_pct"],
        })
        # Update DB
        try:
            conn = get_connection()
            conn.execute(
                "UPDATE pipeline_runs SET status=?, finished_at=datetime('now','localtime') WHERE id=?",
                ("DONE" if success else "ERROR", run_id)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass


def _parse_progress_line(line: str):
    """Parse baris log pipeline untuk update progress."""
    global _pipeline_status
    line_lower = line.lower()

    # Contoh: "Processing booklet_01 (1/76)"
    if "booklet_" in line_lower and "/" in line:
        parts = line.split("/")
        if len(parts) >= 2:
            try:
                done = int(''.join(filter(str.isdigit, parts[0].split()[-1])))
                total = int(''.join(filter(str.isdigit, parts[1].split()[0])))
                _pipeline_status["booklets_done"] = done
                _pipeline_status["progress_pct"] = round(done / max(total, 1) * 100)
            except (ValueError, IndexError):
                pass

    # Stage detection
    if "stage 0" in line_lower or "stage-gate" in line_lower:
        _pipeline_status["stage"] = "Stage 0: Stage-Gate Quality Audit (Gate 0-4)"
    elif "stage 1" in line_lower or "cropping" in line_lower or "blank box" in line_lower:
        _pipeline_status["stage"] = "Stage 1: Grid Cropping & Filter Kotak Kosong OpenCV"
    elif "stage 2" in line_lower or "ocr" in line_lower:
        _pipeline_status["stage"] = "Stage 2: OCR Inference (Kotak Terisi) & Variabel BPS"
    elif "stage 3" in line_lower or "validat" in line_lower:
        _pipeline_status["stage"] = "Stage 3: Validasi Konsistensi Semantik BPS"
    elif "stage 4" in line_lower or "auto-persistence" in line_lower or "sync_db" in line_lower:
        _pipeline_status["stage"] = "Stage 4: Auto-Persistence SQLite (Zero PII)"
    elif "stage 5" in line_lower or "dsrt" in line_lower:
        _pipeline_status["stage"] = "Stage 5: DSRT Sample Matching & Two-Way Linking"
    elif "stage 6" in line_lower or "auto-accept" in line_lower:
        _pipeline_status["stage"] = "Stage 6: Smart Auto-Accept Multi-Tier"


# ─── GET /api/pipeline/status ────────────────────────────────────────────────
@router.get("/status")
async def pipeline_status():
    """Status pipeline saat ini + per-booklet dari DB."""
    conn = get_connection()
    booklet_rows = conn.execute(
        "SELECT id, nama_krt, ocr_status, hitl_count, hitl_done, val_status, gate_status FROM booklets ORDER BY id"
    ).fetchall()
    last_run = conn.execute(
        "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    # Muat ringkasan crop per-booklet jika tersedia
    crop_stats_map = {}
    crop_sum_p = BASE_DIR / "output" / "crops" / "batch_crop_summary.json"
    if crop_sum_p.exists():
        try:
            cdata = json.loads(crop_sum_p.read_text(encoding="utf-8"))
            for b in cdata.get("booklets", []):
                bid = b.get("booklet_id")
                cnts = b.get("counts", {})
                if bid and cnts:
                    tot = cnts.get("digits", 2073)
                    fill = cnts.get("digits_filled", 515)
                    crop_stats_map[bid] = {
                        "total_digits": tot,
                        "filled_boxes": fill,
                        "blank_boxes": max(0, tot - fill)
                    }
        except Exception:
            pass

    jobs = []
    for r in booklet_rows:
        status_map = {
            "DONE": "done",
            "RUNNING": "active",
            "ERROR": "error",
            "PENDING": "queued"
        }
        ocr_st = r["ocr_status"]
        hitl_pending = (r["hitl_count"] or 0) - (r["hitl_done"] or 0)
        c_stat = crop_stats_map.get(r["id"], {"total_digits": 2073, "blank_boxes": 1558, "filled_boxes": 515})
        jobs.append({
            "id": r["id"],
            "krt": r["nama_krt"],
            "stage": _stage_label(ocr_st, hitl_pending),
            "status": "hitl" if hitl_pending > 0 else status_map.get(ocr_st, "queued"),
            "progress": _estimate_progress(ocr_st),
            "ambigu": hitl_pending,
            "gate_status": r["gate_status"] or "UNKNOWN",
            "val_status": r["val_status"] or "PENDING",
            "blank_boxes": c_stat["blank_boxes"],
            "filled_boxes": c_stat["filled_boxes"],
            "total_digits": c_stat["total_digits"],
        })

    return {
        "pipeline": _pipeline_status,
        "last_run": dict(last_run) if last_run else None,
        "jobs": jobs
    }


def _stage_label(ocr_status: str, hitl_pending: int) -> str:
    if hitl_pending > 0:
        return "Butuh HITL"
    stage_labels = {
        "DONE": "Selesai",
        "RUNNING": "Ekstraksi CNN OCR",
        "PENDING": "Menunggu",
        "ERROR": "Error"
    }
    return stage_labels.get(ocr_status, "Menunggu")


def _estimate_progress(ocr_status: str) -> int:
    return {"DONE": 100, "RUNNING": 60, "PENDING": 0, "ERROR": 30}.get(ocr_status, 0)


# ─── GET /api/pipeline/log (SSE) ─────────────────────────────────────────────
@router.get("/log")
async def stream_log():
    """Server-Sent Events: stream log pipeline real-time."""
    async def event_generator():
        last_idx = 0
        # Kirim heartbeat/initial frame agar koneksi SSE langsung terjalin 200 OK
        yield f": ping\n\n"
        if not _pipeline_log:
            status_line = "[Pipeline idle — belum ada proses aktif]" if not _pipeline_status["running"] else "[Pipeline aktif — menunggu log awal...]"
            yield f"data: {json.dumps({'line': status_line, 'ts': datetime.now().isoformat()})}\n\n"

        while True:
            new_lines = _pipeline_log[last_idx:]
            for line in new_lines:
                yield f"data: {json.dumps({'line': line, 'ts': datetime.now().isoformat()})}\n\n"
            last_idx = len(_pipeline_log)

            if not _pipeline_status["running"] and last_idx > 0:
                yield f"data: {json.dumps({'line': '[Pipeline selesai]', 'done': True})}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ─── POST /api/pipeline/stop ─────────────────────────────────────────────────
@router.post("/stop")
async def stop_pipeline():
    """Hentikan subprocess pipeline."""
    global _pipeline_proc, _pipeline_status
    if _pipeline_proc and _pipeline_status["running"]:
        _pipeline_proc.terminate()
        _pipeline_status["running"] = False
        _pipeline_status["stage"] = "stopped"
        return {"status": "stopped", "message": "Pipeline dihentikan."}
    return {"status": "not_running", "message": "Pipeline tidak sedang berjalan."}
