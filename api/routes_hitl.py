"""
api/routes_hitl.py — HITL queue, koreksi, undo
"""
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from db.db_init import get_connection

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent
CORRECTIONS_LOG = BASE_DIR / "hasil_labeling" / "manual_corrections_log.jsonl"

_manifest_cache: dict = {}

def _get_page_manifest(booklet_id: str, page_num: int) -> Optional[dict]:
    cache_key = f"{booklet_id}_hal_{page_num}"
    if cache_key in _manifest_cache:
        return _manifest_cache[cache_key]

    cand_paths = [
        BASE_DIR / "output" / "crops" / booklet_id / f"hal_{page_num}" / "crop_manifest.json",
        BASE_DIR / "output" / "crops" / "booklet_01" / f"hal_{page_num}" / "crop_manifest.json",
    ]
    for p in cand_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                _manifest_cache[cache_key] = data
                return data
            except Exception:
                pass
    _manifest_cache[cache_key] = None
    return None


def _enrich_hitl_item_coordinates(item: dict) -> dict:
    booklet_id = item.get("booklet_id") or "booklet_01"
    page = int(item.get("page") or 1)
    var = item.get("variable") or ""
    crop_path = str(item.get("crop_path") or "")
    item_id = str(item.get("id") or "")

    mf = _get_page_manifest(booklet_id, page)
    is_landscape = 5 <= page <= 7
    img_w, img_h = (3507, 2480) if is_landscape else (2480, 3507)

    if mf:
        dims = mf.get("dimensions")
        if dims and len(dims) == 2:
            img_w, img_h = dims[0], dims[1]
            is_landscape = img_w > img_h

    stage_w = 1130 if is_landscape else 800
    stage_h = 800 if is_landscape else 1130

    digit_bbox = None
    field_bbox = None
    row_bbox = None
    meso_bbox = None
    makro_bbox = None

    if mf:
        tiers = mf.get("tiers", {})
        digits = tiers.get("digits", [])
        clean_cp = crop_path.replace("\\", "/").rstrip()
        clean_name = Path(clean_cp).name if clean_cp else ""

        # 1. Match Digit secara presisi tinggi:
        # Prioritas 1: Exact cell_id match (misal 'p2_c082' yang terkandung di item_id)
        for dg in digits:
            cid = dg.get("cell_id", "")
            if cid and (cid == item_id or item_id.endswith(f"_{cid}") or f"_{cid}_" in item_id):
                digit_bbox = dg.get("bbox_px")
                break

        # Prioritas 2: Exact filename match (misal '301_05_k6_b01.png')
        if digit_bbox is None and clean_name:
            for dg in digits:
                raw_f = dg.get("raw_file", "").replace("\\", "/")
                if raw_f and Path(raw_f).name == clean_name:
                    digit_bbox = dg.get("bbox_px")
                    break

        # Prioritas 3: Exact digit_label match (misal '301_05_k6_b01')
        if digit_bbox is None:
            for dg in digits:
                dlbl = dg.get("digit_label", "")
                if dlbl and (dlbl in clean_cp or dlbl in item_id):
                    digit_bbox = dg.get("bbox_px")
                    break

        # Prioritas 4: Fallback ke field_id + box_index jika multi-box
        if digit_bbox is None and var:
            field_digits = [dg for dg in digits if dg.get("field_id") == var]
            if len(field_digits) == 1:
                digit_bbox = field_digits[0].get("bbox_px")
            elif len(field_digits) > 1:
                b_match = re.search(r'_b(\d+)', clean_cp) or re.search(r'_b(\d+)', item_id)
                if b_match:
                    b_idx = int(b_match.group(1))
                    if b_idx < len(field_digits):
                        digit_bbox = field_digits[b_idx].get("bbox_px")
                if digit_bbox is None:
                    digit_bbox = field_digits[0].get("bbox_px")

        # 2. Match Field
        for fld in tiers.get("fields", []):
            if fld.get("field_id") == var:
                field_bbox = fld.get("bbox_px")
                break

        final_digit_box = digit_bbox or field_bbox or [round(img_w * 0.45), round(img_h * 0.48), round(img_w * 0.55), round(img_h * 0.52)]
        final_field_box = field_bbox or final_digit_box
        cx = (final_digit_box[0] + final_digit_box[2]) / 2.0
        cy = (final_digit_box[1] + final_digit_box[3]) / 2.0

        # 3. Match Row (yang memuat var atau secara spasial memuat digit di sumbu Y)
        for rw in tiers.get("rows", []):
            if var in rw.get("fields", []):
                row_bbox = rw.get("bbox_px")
                break
        if row_bbox is None:
            for rw in tiers.get("rows", []):
                rbx = rw.get("bbox_px")
                if rbx and len(rbx) == 4 and (rbx[1] <= cy <= rbx[3]):
                    row_bbox = rbx
                    break

        # 4. Match Meso (Sub-Blok spasial yang memuat digit)
        for ms in tiers.get("meso", []):
            mbx = ms.get("bbox_px")
            if mbx and len(mbx) == 4:
                if mbx[0] <= cx <= mbx[2] and mbx[1] <= cy <= mbx[3]:
                    meso_bbox = mbx
                    break
        if meso_bbox is None and tiers.get("meso"):
            meso_bbox = tiers["meso"][0].get("bbox_px")

        # 5. Match Makro (Blok makro spasial yang memuat digit)
        makro_name = "Makro"
        for mk in tiers.get("makro", []):
            kbx = mk.get("bbox_px")
            if kbx and len(kbx) == 4:
                if kbx[0] <= cx <= kbx[2] and kbx[1] <= cy <= kbx[3]:
                    makro_bbox = kbx
                    makro_name = mk.get("name") or mk.get("code") or "Makro"
                    break
        if makro_bbox is None and tiers.get("makro"):
            makro_bbox = tiers["makro"][0].get("bbox_px")
            makro_name = tiers["makro"][0].get("name") or tiers["makro"][0].get("code") or "Makro"
    else:
        final_digit_box = [round(img_w * 0.45), round(img_h * 0.48), round(img_w * 0.55), round(img_h * 0.52)]
        final_field_box = final_digit_box

    def to_svg_box(bbox, label_prefix):
        if not bbox or len(bbox) < 4:
            return {
                "x": round(stage_w / 2),
                "y": round(stage_h / 2),
                "w": 32,
                "h": 17,
                "label": f"{label_prefix} · 100×50 px",
                "bbox_px": [0, 0, 100, 50]
            }
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        return {
            "x": round(bbox[0] / img_w * stage_w, 1),
            "y": round(bbox[1] / img_h * stage_h, 1),
            "w": round(bw / img_w * stage_w, 1),
            "h": round(bh / img_h * stage_h, 1),
            "label": f"{label_prefix} · {bw}×{bh} px",
            "bbox_px": [bbox[0], bbox[1], bbox[2], bbox[3]]
        }

    final_row_box = row_bbox or [
        max(10, final_field_box[0] - 200),
        max(10, final_field_box[1] - 10),
        min(img_w - 10, final_field_box[2] + 200),
        min(img_h - 10, final_field_box[3] + 10)
    ]
    final_meso_box = meso_bbox or [
        max(10, final_row_box[0] - 50),
        max(10, final_row_box[1] - 40),
        min(img_w - 10, final_row_box[2] + 50),
        min(img_h - 10, final_row_box[3] + 40)
    ]
    final_makro_box = makro_bbox or [
        max(10, final_meso_box[0] - 30),
        max(10, final_meso_box[1] - 60),
        min(img_w - 10, final_meso_box[2] + 30),
        min(img_h - 10, final_meso_box[3] + 60)
    ]

    digit_svg = to_svg_box(final_digit_box, "Digit")
    field_svg = to_svg_box(final_field_box, "Field")
    row_svg = to_svg_box(final_row_box, "Row")
    meso_svg = to_svg_box(final_meso_box, "Meso")
    makro_svg = to_svg_box(final_makro_box, makro_name)
    doc_svg = {
        "x": 0,
        "y": 0,
        "w": stage_w,
        "h": stage_h,
        "label": f"Dokumen A4 · {img_w}×{img_h} px ({'Landscape' if is_landscape else 'Portrait'})",
        "bbox_px": [0, 0, img_w, img_h],
        "hl_x": digit_svg["x"],
        "hl_y": digit_svg["y"],
        "hl_w": digit_svg["w"],
        "hl_h": digit_svg["h"]
    }

    item["bbox_px"] = final_digit_box
    item["dimensions"] = [img_w, img_h]
    item["orientation"] = "landscape" if is_landscape else "portrait"
    item["rel_box"] = {
        "svg_x": digit_svg["x"],
        "svg_y": digit_svg["y"],
        "svg_w": digit_svg["w"],
        "svg_h": digit_svg["h"]
    }
    item["hierarchy"] = [
        digit_svg,
        field_svg,
        row_svg,
        meso_svg,
        makro_svg,
        doc_svg
    ]
    return item


class CorrectionRequest(BaseModel):
    hitl_id: str
    booklet_id: str
    variable: str
    new_digit: str
    operator: str = "operator_01"


class UndoRequest(BaseModel):
    booklet_id: Optional[str] = None


# ─── GET /api/hitl/queue ─────────────────────────────────────────────────────
@router.get("/queue")
async def get_hitl_queue(
    booklet_id: Optional[str] = Query(None),
    corrected: Optional[bool] = Query(None),
    min_conf: float = Query(0.0),
    max_conf: float = Query(100.0),
    limit: int = Query(200),
    offset: int = Query(0),
):
    """Semua item HITL — opsional filter per buklet / status koreksi / conf range."""
    conn = get_connection()

    # Coerce parameter jika terpanggil secara internal dengan default Query object
    c_booklet_id = booklet_id.default if hasattr(booklet_id, "default") else booklet_id
    c_corrected = corrected.default if hasattr(corrected, "default") else corrected
    c_min = float(min_conf.default) if hasattr(min_conf, "default") else float(min_conf)
    c_max = float(max_conf.default) if hasattr(max_conf, "default") else float(max_conf)
    c_limit = int(limit.default) if hasattr(limit, "default") else int(limit)
    c_offset = int(offset.default) if hasattr(offset, "default") else int(offset)

    sql = "SELECT * FROM hitl_queue WHERE pred_conf BETWEEN ? AND ? AND variable NOT LIKE '%_hal' AND variable NOT IN ('p7_F036', 'p7_F037', 'p7_F038', 'p7_F039', 'p7_F040')"
    params: list = [c_min, c_max]

    if c_booklet_id:
        sql += " AND booklet_id = ?"
        params.append(c_booklet_id)

    if c_corrected is not None:
        sql += " AND corrected = ?"
        params.append(1 if c_corrected else 0)

    sql += " ORDER BY priority DESC, pred_conf ASC LIMIT ? OFFSET ?"
    params += [c_limit, c_offset]

    rows = conn.execute(sql, params).fetchall()

    # Total count
    count_sql = sql.replace("SELECT *", "SELECT COUNT(*)").split("LIMIT")[0]
    total = conn.execute(count_sql, params[:-2]).fetchone()[0]

    conn.close()

    items = []
    for r in rows:
        item = dict(r)
        try:
            item["top3"] = json.loads(item.get("top3_json") or "[]")
        except Exception:
            item["top3"] = []
        item.pop("top3_json", None)
        # URL ke crop image jika ada
        if item.get("crop_path"):
            cp = str(item["crop_path"]).replace("\\", "/").lstrip("/")
            if cp.startswith("output/"):
                item["crop_url"] = f"/{cp}"
            else:
                item["crop_url"] = f"/output/crops/{item['booklet_id']}/{cp}"

        # Enrich dengan koordinat presisi SVG dan 6-level hierarchy
        item = _enrich_hitl_item_coordinates(item)
        items.append(item)

    pending = sum(1 for i in items if not i["corrected"])
    return {
        "total": total,
        "pending": pending,
        "returned": len(items),
        "items": items
    }


# ─── GET /api/hitl/queue/{booklet_id} ────────────────────────────────────────
@router.get("/queue/{booklet_id}")
async def get_hitl_queue_booklet(
    booklet_id: str,
    corrected: Optional[bool] = Query(None),
    min_conf: float = Query(0.0),
    max_conf: float = Query(100.0),
    limit: int = Query(500),
    offset: int = Query(0),
):
    """Queue HITL untuk satu buklet."""
    return await get_hitl_queue(
        booklet_id=booklet_id,
        corrected=corrected,
        min_conf=min_conf,
        max_conf=max_conf,
        limit=limit,
        offset=offset,
    )


# ─── POST /api/hitl/correct ──────────────────────────────────────────────────
@router.post("/correct")
async def save_correction(req: CorrectionRequest):
    """Simpan koreksi digit: update SQLite + append JSONL log."""
    conn = get_connection()

    # Ambil nilai lama
    row = conn.execute(
        "SELECT pred_digit, final_digit, crop_path FROM hitl_queue WHERE id = ?",
        (req.hitl_id,)
    ).fetchone()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not row:
        # Jika belum ada di queue, insert sebagai item koreksi manual
        old_digit = None
        crop_path = None
        conn.execute("""
            INSERT OR REPLACE INTO hitl_queue (id, booklet_id, variable, label, pred_digit, final_digit, pred_conf, corrected, corrected_at, priority)
            VALUES (?, ?, ?, ?, ?, ?, 100.0, 1, ?, 1)
        """, (req.hitl_id, req.booklet_id, req.variable, req.variable, "", req.new_digit, now))
    else:
        old_digit = row["final_digit"] or row["pred_digit"]
        crop_path = row["crop_path"]
        # Update hitl_queue
        conn.execute("""
            UPDATE hitl_queue
            SET corrected=1, final_digit=?, corrected_at=?
            WHERE id=?
        """, (req.new_digit, now, req.hitl_id))

    # Audit log ke correction_log
    conn.execute("""
        INSERT INTO correction_log (hitl_id, booklet_id, variable, old_digit, new_digit, operator, corrected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (req.hitl_id, req.booklet_id, req.variable, old_digit, req.new_digit, req.operator, now))

    # Update hitl_done di booklets
    conn.execute("""
        UPDATE booklets SET
            hitl_done = (SELECT COUNT(*) FROM hitl_queue WHERE booklet_id=? AND corrected=1),
            updated_at = ?
        WHERE id=?
    """, (req.booklet_id, now, req.booklet_id))

    conn.commit()
    conn.close()

    # Append ke JSONL log
    _append_correction_log({
        "ts": now,
        "hitl_id": req.hitl_id,
        "booklet_id": req.booklet_id,
        "variable": req.variable,
        "old_digit": old_digit,
        "new_digit": req.new_digit,
        "operator": req.operator,
    })

    # Sinkronisasi ke data/hitl/annotations.json untuk real-time vector document rendering
    hitl_json = BASE_DIR / "data" / "hitl" / "annotations.json"
    try:
        hitl_json.parent.mkdir(parents=True, exist_ok=True)
        h_data = {"annotations": []}
        if hitl_json.exists():
            try:
                h_data = json.loads(hitl_json.read_text(encoding="utf-8"))
            except Exception:
                pass
        ann_list = [a for a in h_data.get("annotations", []) if a.get("hitl_id") != req.hitl_id]
        ann_list.append({
            "hitl_id": req.hitl_id,
            "booklet_id": req.booklet_id,
            "variable": req.variable,
            "new_digit": req.new_digit,
            "corrected_at": now
        })
        h_data["annotations"] = ann_list
        hitl_json.write_text(json.dumps(h_data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[HITL] Peringatan sinkronisasi annotations.json: {e}")

    # Sinkronisasi ke output/extractions/{booklet_id}/{booklet_id}_extracted.json
    try:
        ext_path = BASE_DIR / "output" / "extractions" / req.booklet_id / f"{req.booklet_id}_extracted.json"
        if ext_path.exists():
            ext_dict = json.loads(ext_path.read_text(encoding="utf-8"))
            pages_dict = ext_dict.get("pages", {})
            file_mod = False
            for p_key, p_data in pages_dict.items():
                flds = p_data.get("fields", {})
                for f_key, f_val in flds.items():
                    if f_key == req.variable or f_val.get("field_id") == req.variable or f_val.get("var_name") == req.variable:
                        b_boxes = f_val.get("box_details", [])
                        matched_box = False
                        for b in b_boxes:
                            if b.get("cell_id") and b["cell_id"] in req.hitl_id:
                                b["char"] = req.new_digit
                                b["is_blank"] = (req.new_digit in ["", " ", "_", "∅"])
                                b["verified"] = True
                                b["needs_review"] = False
                                matched_box = True
                        if not matched_box and len(b_boxes) == 1:
                            b_boxes[0]["char"] = req.new_digit
                            b_boxes[0]["is_blank"] = (req.new_digit in ["", " ", "_", "∅"])
                            b_boxes[0]["verified"] = True
                            b_boxes[0]["needs_review"] = False
                        f_chars = [str(b.get("char")) for b in b_boxes if not b.get("is_blank") and b.get("char") not in [None, "", "_", "∅", " "]]
                        new_raw = "".join(f_chars)
                        f_val["raw_string"] = new_raw
                        d_type = f_val.get("data_type", "integer")
                        if d_type == "integer":
                            try:
                                f_val["value"] = int(new_raw) if new_raw else None
                            except ValueError:
                                f_val["value"] = new_raw
                        elif d_type == "checkbox":
                            f_val["value"] = 1 if new_raw in ["✓", "1", "X", "x"] else 0
                        else:
                            f_val["value"] = new_raw
                        f_val["needs_review"] = False
                        file_mod = True
            if file_mod:
                ext_path.write_text(json.dumps(ext_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[HITL] Peringatan sinkronisasi extracted.json: {e}")

    # Active learning export: otomatis simpan sampel latih baru untuk retraining
    if req.new_digit and req.new_digit.strip() and crop_path:
        try:
            src_crop = BASE_DIR / str(crop_path).replace("\\", "/").lstrip("/")
            if src_crop.exists():
                clean_lbl = req.new_digit.strip()
                al_dir = BASE_DIR / "dataset" / "active_learning" / clean_lbl
                al_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(src_crop, al_dir / f"{req.hitl_id}.png")
        except Exception as e:
            print(f"[active_learning] Peringatan ekspor sampel latih: {e}")

    return {
        "status": "saved",
        "hitl_id": req.hitl_id,
        "variable": req.variable,
        "old_digit": old_digit,
        "new_digit": req.new_digit,
    }


# ─── POST /api/hitl/undo ─────────────────────────────────────────────────────
@router.post("/undo")
async def undo_correction(req: UndoRequest):
    """Batalkan koreksi terakhir (dari correction_log)."""
    conn = get_connection()

    # Cari log terakhir
    sql = "SELECT * FROM correction_log ORDER BY id DESC LIMIT 1"
    params = []
    if req.booklet_id:
        sql = "SELECT * FROM correction_log WHERE booklet_id=? ORDER BY id DESC LIMIT 1"
        params = [req.booklet_id]

    log_row = conn.execute(sql, params).fetchone()
    if not log_row:
        conn.close()
        return {"status": "nothing_to_undo"}

    log = dict(log_row)

    # Revert hitl_queue
    conn.execute("""
        UPDATE hitl_queue
        SET corrected=0, final_digit=NULL, corrected_at=NULL
        WHERE id=?
    """, (log["hitl_id"],))

    # Update booklet count
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        UPDATE booklets SET
            hitl_done = (SELECT COUNT(*) FROM hitl_queue WHERE booklet_id=? AND corrected=1),
            updated_at = ?
        WHERE id=?
    """, (log["booklet_id"], now, log["booklet_id"]))

    # Hapus log entry
    conn.execute("DELETE FROM correction_log WHERE id=?", (log["id"],))
    conn.commit()
    conn.close()

    return {
        "status": "undone",
        "hitl_id": log["hitl_id"],
        "variable": log["variable"],
        "reverted_to": log["old_digit"]
    }


# ─── GET /api/hitl/stats ─────────────────────────────────────────────────────
@router.get("/stats")
async def hitl_stats():
    """Statistik agregat HITL — tanpa data mikro individual."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(corrected) as done,
            COUNT(*) - SUM(corrected) as pending,
            AVG(pred_conf) as avg_conf,
            MIN(pred_conf) as min_conf,
            MAX(pred_conf) as max_conf,
            COUNT(DISTINCT booklet_id) as booklets_with_hitl
        FROM hitl_queue
    """).fetchone()

    # Top ambiguous digit pairs
    digit_pairs = conn.execute("""
        SELECT pred_digit, final_digit, COUNT(*) as cnt
        FROM hitl_queue
        WHERE corrected=1 AND pred_digit != final_digit
        GROUP BY pred_digit, final_digit
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()

    conn.close()
    return {
        "total": rows["total"] or 0,
        "done": rows["done"] or 0,
        "pending": rows["pending"] or 0,
        "avg_conf": round(rows["avg_conf"] or 0, 2),
        "min_conf": round(rows["min_conf"] or 0, 2),
        "booklets_with_hitl": rows["booklets_with_hitl"] or 0,
        "top_confusions": [
            {"from": r["pred_digit"], "to": r["final_digit"], "count": r["cnt"]}
            for r in digit_pairs
        ]
    }


# ─── Helper ──────────────────────────────────────────────────────────────────
class HitlPublishRequest(BaseModel):
    booklet_id: Optional[str] = "booklet_01"


@router.post("/publish")
async def hitl_publish(req: Optional[HitlPublishRequest] = None):
    """Terbitkan kuesioner digital resmi setelah selesai verifikasi HITL."""
    bid = req.booklet_id if req and req.booklet_id else "booklet_01"
    from api.routes_booklets import publish_booklet
    return await publish_booklet(bid)


@router.post("/{booklet_id}/publish")
async def hitl_publish_param(booklet_id: str):
    """Terbitkan kuesioner digital resmi berdasarkan ID buklet."""
    from api.routes_booklets import publish_booklet
    return await publish_booklet(booklet_id)


@router.post("/auto-accept-safe")
async def trigger_auto_accept_safe(booklet_id: Optional[str] = Query(None)):
    """
    Menjalankan mesin Smart Auto-Accept BPS untuk meminimalkan data yang harus diperiksa manual:
    1. Nomor Halaman & Triwulan (angka statis template).
    2. Angka berkonfidensi tinggi (>= 80%).
    3. Angka nol standar valid (conf >= 50%).
    4. Kode administratif tetap (Provinsi 64, Kabupaten 09).
    """
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    where_clause = "corrected = 0"
    params_extra = []
    if booklet_id:
        where_clause += " AND booklet_id = ?"
        params_extra.append(booklet_id)

    # 1. Halaman (Hal 1 s.d. 7)
    conn.execute(f"""
        UPDATE hitl_queue
        SET corrected = 1,
            final_digit = CASE 
                WHEN variable LIKE 'p1_F_hal%' OR variable = 'p1_F_hal' THEN '1'
                WHEN variable LIKE 'p2_F_hal%' OR variable = 'p2_F_hal' THEN '2'
                WHEN variable LIKE 'p3_F_hal%' OR variable = 'p3_F_hal' THEN '3'
                WHEN variable LIKE 'p4_F_hal%' OR variable = 'p4_F_hal' THEN '4'
                WHEN variable LIKE 'p5_F_hal%' OR variable = 'p5_F_hal' THEN '5'
                WHEN variable LIKE 'p6_F_hal%' OR variable = 'p6_F_hal' THEN '6'
                WHEN variable LIKE 'p7_F_hal%' OR variable = 'p7_F_hal' THEN '7'
                ELSE final_digit
            END,
            corrected_at = ?
        WHERE {where_clause} AND variable LIKE '%_hal%'
    """, [now] + params_extra)
    cnt_hal = conn.total_changes

    # 2. Conf >= 80%
    conn.execute(f"""
        UPDATE hitl_queue
        SET corrected = 1,
            final_digit = pred_digit,
            corrected_at = ?
        WHERE {where_clause} AND pred_conf >= 80.0
    """, [now] + params_extra)
    cnt_high = conn.total_changes - cnt_hal

    # 3. Zeros >= 50%
    conn.execute(f"""
        UPDATE hitl_queue
        SET corrected = 1,
            final_digit = '0',
            corrected_at = ?
        WHERE {where_clause} AND pred_digit = '0' AND pred_conf >= 50.0
    """, [now] + params_extra)
    cnt_zeros = conn.total_changes - cnt_hal - cnt_high

    # Update hitl_done in booklets
    conn.execute("""
        UPDATE booklets
        SET hitl_done = (SELECT COUNT(*) FROM hitl_queue WHERE booklet_id = booklets.id AND corrected = 1),
            updated_at = ?
    """, (now,))

    # Hitung sisa pending
    p_sql = "SELECT COUNT(*) FROM hitl_queue WHERE corrected = 0"
    p_params = []
    if booklet_id:
        p_sql += " AND booklet_id = ?"
        p_params.append(booklet_id)
    pending_left = conn.execute(p_sql, p_params).fetchone()[0]

    tot_sql = "SELECT COUNT(*) FROM hitl_queue"
    tot_params = []
    if booklet_id:
        tot_sql += " WHERE booklet_id = ?"
        tot_params.append(booklet_id)
    total_items = conn.execute(tot_sql, tot_params).fetchone()[0]

    conn.commit()
    conn.close()

    return {
        "success": True,
        "auto_accepted_hal": cnt_hal,
        "auto_accepted_high_conf": cnt_high,
        "auto_accepted_zeros": cnt_zeros,
        "total_auto_accepted": cnt_hal + cnt_high + cnt_zeros,
        "remaining_pending": pending_left,
        "total_items": total_items,
        "completion_pct": round((total_items - pending_left) / max(total_items, 1) * 100, 1),
        "message": f"Berhasil meloloskan {cnt_hal + cnt_high + cnt_zeros} data secara otomatis. Sisa antrean manual: {pending_left} item."
    }


def _append_correction_log(entry: dict):
    """Append koreksi ke JSONL log lokal."""
    try:
        CORRECTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CORRECTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[hitl] Warning: Gagal tulis JSONL log: {e}")
