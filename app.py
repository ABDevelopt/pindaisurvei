"""
app.py — PindaiSurvei v2.0 FastAPI entrypoint
Mendukung multi-survei: Seruti Inti (VSERUTI26.INTI), Seruti MAK (VSERUTI26.MAK), Susenas (VSEN26.KP)
Jalankan: python app.py [--lan] [--port 8000]
"""
import sys
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, HTMLResponse
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

# Import routers
from api.routes_surveys    import router as surveys_router
from api.routes_booklets   import router as booklets_router
from api.routes_pipeline   import router as pipeline_router
from api.routes_hitl       import router as hitl_router
from api.routes_validation import router as validation_router
from api.routes_export     import router as export_router
from api.routes_system     import router as system_router
from api.routes_gemini     import router as gemini_router
from api.routes_dsrt       import router as dsrt_router

# Init DB saat startup
from db.db_init import init_db


# ─── Lifespan Context Manager (Modern FastAPI) ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[PindaiSurvei] Inisialisasi basis data multi-survei...")
    init_db()
    print("[PindaiSurvei] Server siap melayani.")
    yield
    print("[PindaiSurvei] Server berhenti.")


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PindaiSurvei v2.0",
    description="Sistem Pembacaan & Digitalisasi Dokumen Survei Multi-Instrumen (Seruti Inti, MAK, Susenas) — BPS Kab. PPU",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
)

# ─── GZip Middleware (memangkas bandwidth LAN hingga 70% pada respons JSON besar) ───
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ─── Static files ─────────────────────────────────────────────────────────────
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

OUTPUT_DIR = BASE_DIR / "output"
if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

SERUTI_INTI_DIR = BASE_DIR / "seruti_inti"
if SERUTI_INTI_DIR.exists():
    app.mount("/scans", StaticFiles(directory=str(SERUTI_INTI_DIR)), name="scans")

DOKUMEN_PETUGAS_DIR = BASE_DIR / "dokumen_petugas"
if DOKUMEN_PETUGAS_DIR.exists():
    app.mount("/dokumen_petugas", StaticFiles(directory=str(DOKUMEN_PETUGAS_DIR)), name="dokumen_petugas")

# ─── API Routers ──────────────────────────────────────────────────────────────
app.include_router(surveys_router,    prefix="/api/surveys",    tags=["Survei"])
app.include_router(booklets_router,   prefix="/api/booklets",   tags=["Booklets"])
app.include_router(pipeline_router,   prefix="/api/pipeline",   tags=["Pipeline"])
app.include_router(hitl_router,       prefix="/api/hitl",       tags=["HITL"])
app.include_router(validation_router, prefix="/api/validation", tags=["Validasi"])
app.include_router(export_router,     prefix="/api/export",     tags=["Ekspor"])
app.include_router(system_router,     prefix="/api/system",     tags=["Sistem"])
app.include_router(gemini_router,     prefix="/api/inference/gemini", tags=["Gemini Vision"])
app.include_router(dsrt_router,       prefix="/api/dsrt",       tags=["DSRT"])

# ─── Full-Scale Application Clean Page Routes (HTML5 SPA) ───────────────────
def serve_spa():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h3>Si-Periksa: static/index.html tidak ditemukan</h3>", status_code=500)
    return FileResponse(index_file, media_type="text/html")

@app.get("/", include_in_schema=False)
@app.get("/input", include_in_schema=False)
async def page_input():
    return serve_spa()

@app.get("/booklets", include_in_schema=False)
@app.get("/booklets/{booklet_id}", include_in_schema=False)
@app.get("/buku", include_in_schema=False)
@app.get("/preproc", include_in_schema=False)
async def page_booklets(booklet_id: str = None):
    return serve_spa()

@app.get("/pipeline", include_in_schema=False)
async def page_pipeline():
    return serve_spa()

@app.get("/grid", include_in_schema=False)
async def page_grid():
    return serve_spa()

@app.get("/crop", include_in_schema=False)
async def page_crop():
    return serve_spa()

@app.get("/inference", include_in_schema=False)
@app.get("/inferensi", include_in_schema=False)
async def page_inference():
    return serve_spa()

@app.get("/kuesioner", include_in_schema=False)
@app.get("/kuesioner/{booklet_id}", include_in_schema=False)
async def page_kuesioner(booklet_id: str = None):
    return serve_spa()

@app.get("/kuesioner-digital", include_in_schema=False)
@app.get("/kuesioner-digital/{booklet_id}", include_in_schema=False)
async def page_kuesioner_digital_direct(booklet_id: str = "booklet_01"):
    kuesioner_file = STATIC_DIR / "kuesioner_digital.html"
    if kuesioner_file.exists():
        return FileResponse(kuesioner_file, media_type="text/html")
    return serve_spa()

@app.get("/hitl", include_in_schema=False)
@app.get("/verifikasi", include_in_schema=False)
async def page_hitl():
    return serve_spa()

@app.get("/output", include_in_schema=False)
@app.get("/digital", include_in_schema=False)
@app.get("/ekspor", include_in_schema=False)
async def page_output():
    return serve_spa()

@app.get("/training", include_in_schema=False)
@app.get("/pelatihan", include_in_schema=False)
async def page_training():
    return serve_spa()

@app.get("/admin", include_in_schema=False)
@app.get("/system", include_in_schema=False)
@app.get("/settings", include_in_schema=False)
async def page_admin():
    return serve_spa()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Kompatibilitas jika ada tab browser lama yang masih memanggil port 8000
@app.get("/api/stats", include_in_schema=False)
async def stats_alias():
    from api.routes_hitl import hitl_stats
    return await hitl_stats()

@app.get("/api/kbli-options", include_in_schema=False)
@app.get("/api/kbli-2d-options", include_in_schema=False)
async def kbli_compat():
    return []

# ─── SPA Fallback Catch-All Route ─────────────────────────────────────────────
@app.get("/{full_path:path}", include_in_schema=False)
async def catch_all_spa(full_path: str):
    for prefix in ["api/", "static/", "output/", "scans/", "dokumen_petugas/"]:
        if full_path.startswith(prefix) or full_path == prefix.rstrip("/"):
            raise HTTPException(status_code=404, detail="Not Found")
    return serve_spa()

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Si-Periksa v2.0 Server")
    parser.add_argument("--lan",  action="store_true", help="Bind ke 0.0.0.0 (akses LAN)")
    parser.add_argument("--port", type=int, default=8000, help="Port server (default: 8000)")
    args = parser.parse_args()

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    mode = "LAN (0.0.0.0)" if args.lan else "Lokal (127.0.0.1)"

    print(f"""
╔══════════════════════════════════════════════════════╗
║     PindaiSurvei v2.0 — Multi-Survei Engine BPS      ║
║   Seruti Inti · Seruti MAK · Susenas Konsumsi (KP)   ║
║          BPS Kabupaten Penajam Paser Utara           ║
╠══════════════════════════════════════════════════════╣
║  Mode    : {mode:<42}║
║  Alamat  : http://{host}:{args.port:<31}║
║  API     : http://{host}:{args.port}/api/docs{' ' * 24}║
║  Zero-PII: Semua data diproses lokal offline        ║
╚══════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "app:app",
        host=host,
        port=args.port,
        reload=False,
        log_level="info",
    )
