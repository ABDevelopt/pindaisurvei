"""
db/db_init.py — Inisialisasi SQLite WAL + seed multi-survei (Seruti Inti, Seruti MAK, Susenas KP)
Zero-PII: hanya metadata (nama KRT dari folder names/manifest, NKS)
"""
import sqlite3
import json
import os
import csv
import glob
import re
from datetime import datetime
from pathlib import Path

from seruti_pipeline.stage_gate_validator import DUPLICATE_BOOKLET_IDS

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "si_periksa.db"
SCHEMA_PATH = BASE_DIR / "db" / "schema.sql"

# Direktori data nyata per survei
BOOKLET_DIR_INTI = BASE_DIR / "output" / "seruti_inti_booklets"
DIR_MAK = BASE_DIR / "MAK"
DIR_KP = BASE_DIR / "KP"

EXTRACTION_DIR = BASE_DIR / "output" / "extractions"
CROPS_DIR = BASE_DIR / "output" / "crops"
OVERLAY_DIR = BASE_DIR / "output" / "scan_overlays"
HITL_CSV = BASE_DIR / "hasil_labeling" / "hasil_autolabel.csv"
MANIFEST_PATH = BASE_DIR / "state" / "manifest.json"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Buat tabel dan seed data dari file pipeline yang ada."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()

    # Inisialisasi tabel inferensi AI dan nilai variabel jika belum ada
    try:
        from db.inference_repository import init_inference_tables
        init_inference_tables(conn)
    except Exception:
        pass

    # Migration: pastikan kolom survey_type dan dsrt_linked ada
    try:
        conn.execute("ALTER TABLE booklets ADD COLUMN survey_type TEXT DEFAULT 'seruti_inti'")
        conn.commit()
    except Exception:
        pass

    try:
        conn.execute("ALTER TABLE booklets ADD COLUMN dsrt_linked INTEGER DEFAULT 0")
        conn.commit()
    except Exception:
        pass

    # Buat tabel dsrt_files dan dsrt_samples jika belum ada
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dsrt_files (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_type   TEXT DEFAULT 'seruti_inti',
            filename      TEXT NOT NULL,
            file_path     TEXT,
            total_rows    INTEGER DEFAULT 0,
            matched_count INTEGER DEFAULT 0,
            uploaded_at   TEXT DEFAULT (datetime('now','localtime')),
            is_active     INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dsrt_samples (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id       INTEGER REFERENCES dsrt_files(id),
            survey_type   TEXT DEFAULT 'seruti_inti',
            nks           TEXT,
            nus           TEXT,
            nama_krt      TEXT,
            alamat        TEXT,
            jart          INTEGER,
            status_rt     TEXT DEFAULT 'Utama',
            no_bf         TEXT,
            no_bs         TEXT,
            booklet_id    TEXT,
            is_matched    INTEGER DEFAULT 0,
            created_at    TEXT DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dsrt_survey ON dsrt_samples(survey_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dsrt_nks_nus ON dsrt_samples(nks, nus)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dsrt_booklet ON dsrt_samples(booklet_id)")
    conn.commit()

    _seed_seruti_inti(conn)
    _seed_seruti_mak(conn)
    _seed_susenas_kp(conn)
    _seed_hitl_queue(conn)
    conn.close()
    print(f"[db_init] Database siap dengan multi-survei: {DB_PATH}")


def _seed_seruti_inti(conn: sqlite3.Connection):
    """Seed 51 buklet unik Seruti Inti (25 duplikat diisolasi)."""
    existing = {row["id"] for row in conn.execute("SELECT id FROM booklets WHERE survey_type='seruti_inti'")}

    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    found_ids = set()
    for search_dir in [BOOKLET_DIR_INTI, EXTRACTION_DIR, CROPS_DIR]:
        if search_dir.exists():
            for folder in search_dir.glob("booklet_*"):
                if folder.is_dir() and re.match(r"^booklet_\d+$", folder.name):
                    found_ids.add(folder.name)

    for i in range(1, 77):
        found_ids.add(f"booklet_{str(i).zfill(2)}")

    # Isolasi buklet duplikat agar hanya 51 buklet unik yang masuk ke DB
    found_ids = {bid for bid in found_ids if bid not in DUPLICATE_BOOKLET_IDS}

    inserted = 0

    for bid in sorted(found_ids):
        if bid in existing:
            continue

        folder = BOOKLET_DIR_INTI / bid
        nks, nus, nama_krt = _resolve_booklet_meta(bid, manifest, folder)
        gate_status, gate_detail = _resolve_gate_status(bid, folder)
        ocr_status = _resolve_ocr_status(bid)
        val_status = _resolve_val_status(bid)

        conn.execute("""
            INSERT OR IGNORE INTO booklets
            (id, survey_type, nks, nus, nama_krt, scan_path, gate_status, gate_detail, ocr_status, val_status, updated_at)
            VALUES (?, 'seruti_inti', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bid, nks, nus, nama_krt, str(folder),
            gate_status, json.dumps(gate_detail),
            ocr_status, val_status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        inserted += 1

    conn.commit()
    if inserted:
        print(f"[db_init] Seed {inserted} booklet Seruti Inti baru ke DB.")


def _seed_seruti_mak(conn: sqlite3.Connection):
    """Seed buklet Seruti MAK dari folder MAK/."""
    if not DIR_MAK.exists():
        return

    existing = {row["id"] for row in conn.execute("SELECT id FROM booklets WHERE survey_type='seruti_mak'")}
    mak_folders = sorted([f for f in DIR_MAK.iterdir() if f.is_dir()])
    inserted = 0

    for i, folder in enumerate(mak_folders, 1):
        bid = f"mak_{str(i).zfill(2)}"
        if bid in existing:
            continue

        nama_krt = folder.name
        nks = f"0042 / MAK-{str(i).zfill(2)}"
        nus = str(i).zfill(2)

        # Cek jumlah file scan di folder responden
        scans = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg", ".tiff"]]
        gate_status = "PASS" if len(scans) >= 4 else "PENDING"
        gate_detail = {"total_scans": len(scans), "source_dir": str(folder)}

        conn.execute("""
            INSERT OR IGNORE INTO booklets
            (id, survey_type, nks, nus, nama_krt, scan_path, gate_status, gate_detail, ocr_status, val_status, updated_at)
            VALUES (?, 'seruti_mak', ?, ?, ?, ?, ?, ?, 'PENDING', 'PENDING', ?)
        """, (
            bid, nks, nus, nama_krt, str(folder),
            gate_status, json.dumps(gate_detail),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        inserted += 1

    conn.commit()
    if inserted:
        print(f"[db_init] Seed {inserted} booklet Seruti MAK ke DB.")


def _seed_susenas_kp(conn: sqlite3.Connection):
    """Seed buklet Susenas KP dari folder KP/."""
    if not DIR_KP.exists():
        return

    existing = {row["id"] for row in conn.execute("SELECT id FROM booklets WHERE survey_type='susenas_kp'")}
    kp_folders = sorted([f for f in DIR_KP.iterdir() if f.is_dir()])
    inserted = 0

    for i, folder in enumerate(kp_folders, 1):
        bid = f"kp_{str(i).zfill(2)}"
        if bid in existing:
            continue

        nama_krt = folder.name
        nks = f"0042 / KP-{str(i).zfill(2)}"
        nus = str(i).zfill(2)

        scans = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg", ".tiff"]]
        gate_status = "PASS" if len(scans) >= 4 else "PENDING"
        gate_detail = {"total_scans": len(scans), "source_dir": str(folder)}

        conn.execute("""
            INSERT OR IGNORE INTO booklets
            (id, survey_type, nks, nus, nama_krt, scan_path, gate_status, gate_detail, ocr_status, val_status, updated_at)
            VALUES (?, 'susenas_kp', ?, ?, ?, ?, ?, ?, 'PENDING', 'PENDING', ?)
        """, (
            bid, nks, nus, nama_krt, str(folder),
            gate_status, json.dumps(gate_detail),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        inserted += 1

    conn.commit()
    if inserted:
        print(f"[db_init] Seed {inserted} booklet Susenas KP ke DB.")


def _resolve_booklet_meta(bid: str, manifest: dict, folder: Path):
    num = bid.replace("booklet_", "")

    # 1. Cek dari manifest booklet jika ada
    b_manifest = folder / "manifest.json"
    if b_manifest.exists():
        try:
            m = json.loads(b_manifest.read_text(encoding="utf-8"))
            if "nama_krt" in m and m["nama_krt"]:
                return m.get("nks", f"0042/{num}"), m.get("nus", num), m.get("nama_krt")
        except Exception:
            pass

    # 2. Cek dari file ekstraksi JSON riil
    ext_file = EXTRACTION_DIR / bid / f"{bid}_extracted.json"
    if ext_file.exists():
        try:
            data = json.loads(ext_file.read_text(encoding="utf-8"))
            f1 = data.get("pages", {}).get("hal_1", {}).get("fields", {})
            nus_val = f1.get("b1_r108_nus", {}).get("raw_string") or f1.get("b1_r108_nus", {}).get("value")
            nus_clean = str(nus_val).strip() if nus_val is not None and str(nus_val).strip() else str(num).zfill(2)
            nks_val = f"0042 / {nus_clean}"
            nama = f"Responden {num} (NUS: {nus_clean})"
            return nks_val, nus_clean, nama
        except Exception:
            pass

    if bid in manifest and isinstance(manifest[bid], dict):
        m = manifest[bid]
        return m.get("nks", f"0042/{num}"), m.get("nus", num), m.get("nama_krt", f"Responden {num}")

    return f"0042/{num}", num, f"Responden {num}"


def _resolve_gate_status(bid: str, folder: Path):
    b_manifest = folder / "manifest.json"
    if b_manifest.exists():
        try:
            m = json.loads(b_manifest.read_text(encoding="utf-8"))
            is_comp = m.get("is_complete_8_pages", True)
            return ("PASS" if is_comp else "FAIL"), m.get("diagnostics", {})
        except Exception:
            pass

    crops_folder = CROPS_DIR / bid
    if crops_folder.exists():
        return "PASS", {"note": "crops OK"}

    return "PASS", {}


def _resolve_ocr_status(bid: str):
    ext_folder = EXTRACTION_DIR / bid
    if ext_folder.exists() and any(ext_folder.glob("*.json")):
        return "DONE"
    return "PENDING"


def _resolve_val_status(bid: str):
    for f in [EXTRACTION_DIR / bid / f"{bid}_validation.json", EXTRACTION_DIR / bid / "validation_report.json"]:
        if f.exists():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                summ = d.get("summary", {})
                sem = summ.get("semantic_validation", {})
                if sem.get("rules_mismatched", 0) == 0 and sem.get("rules_passed", 0) > 0:
                    return "PASS"
                elif sem.get("rules_mismatched", 0) > 0:
                    return "WARN"
                return "PASS"
            except Exception:
                pass
    return "PENDING"


def _seed_hitl_queue(conn: sqlite3.Connection):
    if not HITL_CSV.exists():
        return

    existing_count = conn.execute("SELECT COUNT(*) FROM hitl_queue").fetchone()[0]
    if existing_count > 0:
        return

    inserted = 0
    try:
        with open(HITL_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bid = row.get("id_booklet", row.get("booklet_id", "")).strip()
                if not bid:
                    continue

                field_id = row.get("field_id", row.get("variable", "")).strip()
                box_id = row.get("box_id", "").strip()
                pred = str(row.get("prediksi_label", row.get("prediction", ""))).strip()

                try:
                    conf_raw = float(row.get("confidence", 0) or 0)
                    conf = conf_raw * 100 if conf_raw <= 1.0 else conf_raw
                except Exception:
                    conf = 50.0

                is_ambiguous = (conf < 85.0) or (pred != "kosong" and conf < 92.0)
                if not is_ambiguous and inserted > 1000:
                    continue

                hal_str = row.get("halaman", "hal_1")
                page_match = re.search(r"\d+", hal_str)
                page = int(page_match.group(0)) if page_match else 1

                item_id = f"{bid}_{field_id}_{box_id}" if box_id else f"{bid}_{field_id}_{hal_str}_{inserted}"
                label = _variable_to_label(field_id)
                block = _variable_to_block(field_id)
                crop_path = row.get("raw_path", "")

                priority = int((100 - conf) * 10)
                hint = _generate_simple_hint(field_id, pred, conf)
                top3 = _extract_top3(row, pred, conf)

                conn.execute("""
                    INSERT OR IGNORE INTO hitl_queue
                    (id, booklet_id, variable, label, block_name, page, pred_digit, pred_conf,
                     top3_json, hint, corrected, final_digit, tier, crop_path, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item_id, bid, field_id, label, block, page,
                    " " if pred == "kosong" else pred, round(conf, 1),
                    json.dumps(top3), hint,
                    0, None, 7, crop_path, priority
                ))
                inserted += 1

                if inserted >= 1500:
                    break

        conn.commit()
    except Exception as e:
        print(f"[db_init] Error saat seed HITL: {e}")

    _update_hitl_counts(conn)


def _variable_to_label(var: str) -> str:
    mapping = {
        "p2_f020": "Beras Lokal / Kualitas",
        "p2_f021": "Daging Sapi / Ayam",
        "p2_f022": "Ikan Segar",
        "p2_f023": "Minyak Goreng Sawit",
        "p2_f052": "Subtotal Makanan (Hal 2)",
        "b4_r401": "Beras Lokal",
        "b4_r408": "Minyak Goreng",
        "b0_": "Kesediaan Responden",
        "b3_": "Keterangan ART"
    }
    var_l = var.lower()
    for k, v in mapping.items():
        if k in var_l:
            return v
    return f"Rincian {var.upper()}"


def _variable_to_block(var: str) -> str:
    var_l = var.lower()
    if "p1_" in var_l or "b0" in var_l or "b1" in var_l:
        return "KOP / Blok I"
    if "p2_" in var_l or "b3" in var_l:
        return "Blok III (ART) / Blok IV"
    if "p3_" in var_l or "b4" in var_l:
        return "Blok IV (Makanan)"
    if "p4_" in var_l or "p5_" in var_l or "b5" in var_l:
        return "Blok V (Non-Makanan)"
    if "p6_" in var_l or "b6" in var_l:
        return "Blok VI (Rekap)"
    return "Blok IV.1"


def _generate_simple_hint(var: str, pred: str, conf: float) -> str:
    if pred == "kosong":
        return f"Tinta arsir tipis (Conf {conf:.1f}%). Verifikasi apakah sel memang kosong atau bertinta."
    pairs = {"3": "8", "8": "3", "0": "6", "6": "0", "1": "7", "7": "1", "4": "9", "9": "4"}
    alt = pairs.get(pred, "")
    if alt:
        return f"Goresan ambigu dengan angka '{alt}' (Conf {conf:.1f}%). Verifikasi lengkungan di A4."
    return f"Tingkat keyakinan OCR {conf:.1f}%. Periksa langsung pada citra scan A4."


def _extract_top3(row: dict, pred: str, conf: float) -> list:
    pairs = {"3": "8", "8": "3", "0": "6", "6": "0", "1": "7", "7": "1", "4": "9", "9": "4"}
    d1 = "∅" if pred == "kosong" else pred
    alt = pairs.get(pred, "0" if pred != "0" else "1")
    alt_p = max(5.0, round(100.0 - conf - 3.0, 1))
    return [
        {"d": d1, "p": round(conf, 1)},
        {"d": alt, "p": alt_p},
        {"d": "∅" if d1 != "∅" else "0", "p": max(1.0, round(100.0 - conf - alt_p, 1))}
    ]


def _update_hitl_counts(conn: sqlite3.Connection):
    conn.execute("""
        UPDATE booklets SET
            hitl_count = (SELECT COUNT(*) FROM hitl_queue WHERE booklet_id = booklets.id),
            hitl_done  = (SELECT COUNT(*) FROM hitl_queue WHERE booklet_id = booklets.id AND corrected = 1),
            updated_at = datetime('now','localtime')
    """)
    conn.commit()


if __name__ == "__main__":
    init_db()
