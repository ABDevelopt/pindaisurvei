"""
api/routes_gemini.py — REST API Router untuk Integrasi Model Berbayar Google Gemini API
Mendukung:
- Pengujian & Validasi Kunci API Gemini
- Inferensi Vision Tunggal (Per-Digit Crop)
- Re-evaluasi Ambigu Batch (Membantu Verifikasi HITL)
- Zero-PII Compliance (Hanya citra kotak digit comb-box anonim yang dikirim)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from db.db_init import get_connection

from api.gemini_service import (
    get_gemini_status,
    set_gemini_api_key,
    test_gemini_connection,
    classify_digit_crop,
    classify_crop_grid_batch,
    get_gemini_api_key,
    AVAILABLE_MODELS
)

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent


def _normalize_gemini_model(model: Optional[str]) -> str:
    """Normalisasi alias UI ke ID model resmi Google GenAI API."""
    if not model:
        return "gemini-3.8-flash"
    m_map = {
        "gemini_38_flash": "gemini-3.8-flash",
        "gemini_38_high": "gemini-3.8-flash-high",
        "gemini_38_med": "gemini-3.8-flash-medium",
        "gemini_37_flash": "gemini-3.7-flash",
        "gemini_35_flash": "gemini-3.5-flash",
        "gemini_31_pro": "gemini-3.1-pro-preview",
        "gemini_3_flash": "gemini-3-flash-preview",
        "gemini_flash": "gemini-3.8-flash",
        "gemini_pro": "gemini-3.1-pro-preview",
        "gemini_15_flash": "gemini-1.5-flash",
        "gemini_15_pro": "gemini-1.5-pro"
    }
    return m_map.get(model, model)


class GeminiKeyPayload(BaseModel):
    api_key: str = Field(..., description="Google Gemini API Key (format: AIzaSy...)")


class GeminiTestPayload(BaseModel):
    api_key: Optional[str] = Field(None, description="API Key opsional untuk diuji sebelum disimpan")
    model: Optional[str] = Field("gemini-3.8-flash", description="ID Model Gemini yang diuji")


class GeminiClassifyDigitPayload(BaseModel):
    crop_path: Optional[str] = Field(None, description="Path relatif atau absolut ke berkas crop citra digit")
    booklet_id: Optional[str] = Field(None, description="ID buklet (contoh: booklet_01)")
    page: Optional[int] = Field(None, description="Nomor halaman (1-8)")
    digit_id: Optional[str] = Field(None, description="ID digit (contoh: p1_c000 atau TRIWULAN_b00)")
    model: Optional[str] = Field("gemini-3.8-flash", description="Model Gemini: gemini-3.8-flash")
    force_eval: Optional[bool] = Field(False, description="Paksa evaluasi AI meskipun terdeteksi kosong oleh OpenCV")


class GeminiAmbiguousPayload(BaseModel):
    booklet_id: str = Field("booklet_01", description="ID Buklet")
    page: Optional[int] = Field(None, description="Filter halaman")
    limit: Optional[int] = Field(20, description="Batas jumlah digit ambigu yang dievaluasi per batch")
    model: Optional[str] = Field("gemini-3.8-flash", description="Model Gemini yang digunakan")


@router.get("/status")
async def get_status():
    """Mengambil status konfigurasi Google Gemini API, model tersedia, dan cache."""
    return get_gemini_status()


@router.post("/key")
async def save_api_key(payload: GeminiKeyPayload):
    """Menyimpan atau memperbarui Google Gemini API Key."""
    key = payload.api_key.strip()
    if not key or len(key) < 10:
        raise HTTPException(status_code=400, detail="Format API Key tidak valid.")
    
    success = set_gemini_api_key(key)
    if not success:
        raise HTTPException(status_code=500, detail="Gagal menyimpan API Key.")
    
    # Uji coba koneksi ringan
    test_res = test_gemini_connection(key, model="gemini-3.8-flash")
    
    return {
        "success": True,
        "message": "API Key berhasil disimpan.",
        "test_result": test_res,
        "status": get_gemini_status()
    }


@router.post("/test")
async def test_connection(payload: GeminiTestPayload):
    """Menguji koneksi ke Google Gemini API dan mengukur latensi."""
    res = test_gemini_connection(payload.api_key, model=payload.model or "gemini-3.8-flash")
    return res


@router.post("/classify-digit")
async def classify_digit(payload: GeminiClassifyDigitPayload):
    """
    Melakukan klasifikasi satu citra crop digit menggunakan Google Gemini Vision API.
    Zero-PII Compliance: Hanya potongan kotak comb-box anonim yang dikirimkan.
    """
    crop_path = payload.crop_path
    
    # Jika crop_path belum ada, cari berdasarkan booklet_id, page, dan digit_id
    if not crop_path and payload.booklet_id and payload.page and payload.digit_id:
        b_dir = BASE_DIR / "output" / "crops" / payload.booklet_id / f"hal_{payload.page}"
        # Cari di digits/raw/
        raw_dir = b_dir / "digits" / "raw"
        if raw_dir.exists():
            matched = list(raw_dir.glob(f"**/*{payload.digit_id}*.png"))
            if matched:
                crop_path = str(matched[0])
        
        # Fallback ke inner/
        if not crop_path:
            inner_dir = b_dir / "digits" / "inner"
            if inner_dir.exists():
                matched = list(inner_dir.glob(f"**/*{payload.digit_id}*.png"))
                if matched:
                    crop_path = str(matched[0])

    if not crop_path:
        raise HTTPException(status_code=400, detail="Jalur citra crop digit (crop_path) tidak dapat ditemukan.")

    model_name = _normalize_gemini_model(payload.model)
    result = classify_digit_crop(crop_path, model=model_name, force_eval=bool(payload.force_eval))
    
    if "error" in result and result.get("digit") is None:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return {
        "success": True,
        "crop_path": crop_path,
        "digit_id": payload.digit_id,
        "prediction": result
    }


class GeminiBatchGridPayload(BaseModel):
    booklet_id: str = Field("booklet_01", description="ID Buklet")
    page: Optional[int] = Field(None, description="Nomor halaman (1-8), atau null untuk seluruh buklet")
    scope: Optional[str] = Field("all", description="Scope: all (semua digit terisi) atau review_only")
    batch_size: Optional[int] = Field(50, description="Jumlah crop per kisi komposit (default: 50 crop per grid)")
    model: Optional[str] = Field("gemini-3.8-flash", description="Model Gemini default: gemini-3.8-flash")
    include_blank: Optional[bool] = Field(False, description="Apakah menyertakan kotak kosong (default False: kotak kosong OpenCV dilewati)")


@router.post("/batch-grid")
async def batch_grid_inference(payload: GeminiBatchGridPayload):
    """
    Menjalankan proses inferensi batch dengan menyatukan 50 crop digit menjadi satu grid komposit
    sebelum dikirim ke Gemini 3.8 Flash Vision API. Menghemat hingga ~96% token dan panggilan API.
    
    Zero-PII Safe: Hanya potongan comb-box anonim yang ditempel pada grid komposit.
    """
    import json
    booklet_id = payload.booklet_id or "booklet_01"
    b_dir = BASE_DIR / "output" / "crops" / booklet_id
    if not b_dir.exists():
        raise HTTPException(status_code=404, detail=f"Direktori crop untuk {booklet_id} belum tersedia")

    ext_file = BASE_DIR / "output" / "extractions" / booklet_id / f"{booklet_id}_extracted.json"
    if not ext_file.exists():
        raise HTTPException(status_code=404, detail=f"Berkas ekstraksi untuk {booklet_id} belum tersedia")

    ext_data = {}
    if ext_file.exists():
        try:
            ext_data = json.loads(ext_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    pages_ext = ext_data.get("pages", {})
    page_nums = [payload.page] if payload.page else range(1, 8)

    crops_to_evaluate = []

    for p_num in page_nums:
        m_path = b_dir / f"hal_{p_num}" / "crop_manifest.json"
        if not m_path.exists():
            continue
        try:
            m_data = json.loads(m_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        tiers = m_data.get("tiers", {})
        digits = tiers.get("digits") or tiers.get("mikro", [])

        page_ext_fields = pages_ext.get(f"hal_{p_num}", {}).get("fields", {})
        ocr_boxes = {}
        for var_name, fld_info in page_ext_fields.items():
            fid = fld_info.get("field_id")
            for b_idx, b_det in enumerate(fld_info.get("box_details", [])):
                ocr_boxes[f"{fid}_b{b_idx:02d}"] = b_det

        for d in digits:
            cid = d.get("cell_id")
            fid = d.get("field_id")
            b_idx = d.get("box_index", 0)
            d_lbl = d.get("digit_label") or f"{fid}_b{b_idx:02d}"

            fpath = d.get("raw_file") or d.get("file") or d.get("inner_file")
            if not fpath:
                continue

            b_det = ocr_boxes.get(d_lbl) or ocr_boxes.get(f"{fid}_b{b_idx:02d}") or {}
            conf = b_det.get("conf", 0.98 if d.get("is_filled") else 1.0)
            is_blank = b_det.get("is_blank", not d.get("is_filled", True))
            # Kalibrasi Smart Auto-Accept BPS:
            is_fixed_template = any(k in (fid or "").lower() for k in ["_hal", "triwulan", "f001", "f002"])
            if is_blank or is_fixed_template:
                needs_review = False
            elif conf >= 0.80:
                needs_review = False
            else:
                needs_review = (conf < 0.80) or (b_det.get("needs_review", False))

            # ATURAN MUTLAK: Inferensi hanya dilakukan untuk selain kotak kosong yang sudah difilter OpenCV
            if is_blank and not payload.include_blank:
                continue

            if payload.scope == "review_only" and not needs_review:
                continue

            crops_to_evaluate.append({
                "id": cid or d_lbl,
                "label": d_lbl,
                "cell_id": cid,
                "field_id": fid,
                "box_index": b_idx,
                "box_label": f"{fid}_b{b_idx:02d}",
                "page": p_num,
                "crop_path": fpath,
                "conf": conf,
                "needs_review": needs_review,
                "is_blank": is_blank
            })

    if not crops_to_evaluate:
        return {
            "success": True,
            "message": "Tidak ada digit yang memenuhi kriteria filter batch.",
            "total_digits": 0,
            "total_batches": 0,
            "results": {},
            "summary": {"auto_accepted": 0, "needs_review": 0, "avg_confidence": 0.0}
        }

    model_name = _normalize_gemini_model(payload.model)
    batch_size = payload.batch_size or 50
    batch_res = classify_crop_grid_batch(
        crops_to_evaluate,
        model=model_name,
        batch_size=batch_size
    )

    # Sinkronisasi hasil batch grid ke extracted JSON lokal & perbarui nilai variabel kuesioner digital
    if ext_file.exists() and batch_res.get("results"):
        results = batch_res["results"]
        modified = False
        for p_num in page_nums:
            p_fields = pages_ext.get(f"hal_{p_num}", {}).get("fields", {})
            for var_name, fld_info in p_fields.items():
                fid = fld_info.get("field_id")
                box_details = fld_info.get("box_details", [])
                field_modified = False
                for b_idx, b_det in enumerate(box_details):
                    box_label = f"{fid}_b{b_idx:02d}"
                    for item in crops_to_evaluate:
                        is_match = (
                            (item.get("field_id") == fid and item.get("box_index") == b_idx and item.get("page") == p_num)
                            or item.get("box_label") == box_label
                            or item.get("label") == box_label
                            or (item.get("id") and item.get("id") == b_det.get("cell_id"))
                        )
                        if is_match:
                            res_item = results.get(item.get("id")) or results.get(item.get("label")) or results.get(item.get("box_label"))
                            if res_item:
                                p_digit = res_item.get("pred_digit")
                                p_char = res_item.get("char")
                                is_b = res_item.get("is_blank", False)

                                # ATURAN MUTLAK USER: Untuk label ragu-ragu, tetap ditulis label prediksi terkuatnya
                                if not is_b and (p_digit in ["ragu_ragu", "ragu", "?", None, ""]):
                                    p_digit = res_item.get("alt_digit") or res_item.get("alt_char") or "0"
                                if not is_b and (p_char in ["ragu_ragu", "ragu", "?", None, ""]):
                                    p_char = p_digit

                                b_det["cell_id"] = item.get("cell_id") or b_det.get("cell_id")
                                b_det["pred_digit"] = p_digit
                                b_det["char"] = p_char if p_char is not None else p_digit
                                b_det["conf"] = res_item.get("conf")
                                b_det["conf_pct"] = res_item.get("conf_pct")
                                b_det["is_blank"] = is_b
                                b_det["is_doubtful"] = res_item.get("is_doubtful", False) or (float(res_item.get("conf") or 1.0) < 0.85)
                                b_det["model_engine"] = model_name
                                field_modified = True
                                modified = True
                if field_modified:
                    field_chars = []
                    for b in box_details:
                        if not b.get("is_blank") and b.get("char") not in [None, "", "_", "∅"]:
                            field_chars.append(str(b.get("char")))
                    raw_str = "".join(field_chars)
                    fld_info["raw_string"] = raw_str
                    data_type = fld_info.get("data_type", "integer")

                    # ATURAN KHUSUS TRIWULAN: Wajib huruf Romawi (I, II, III, IV)
                    fid = fld_info.get("field_id", "")
                    vname = fld_info.get("var_name", "")
                    is_triwulan = "triwulan" in fid.lower() or "triwulan" in vname.lower() or fid == "p1_F_triwulan" or vname == "kop_triwulan"
                    if is_triwulan:
                        roman_map = {"1": "I", "01": "I", "2": "II", "02": "II", "3": "III", "03": "III", "4": "IV", "04": "IV"}
                        clean_s = raw_str.strip()
                        if clean_s in roman_map:
                            fld_info["raw_string"] = roman_map[clean_s]
                            fld_info["value"] = roman_map[clean_s]
                        elif clean_s.upper() in ["I", "II", "III", "IV"]:
                            fld_info["raw_string"] = clean_s.upper()
                            fld_info["value"] = clean_s.upper()
                        else:
                            fld_info["value"] = raw_str
                    elif data_type == "integer":
                        try:
                            fld_info["value"] = int(raw_str) if raw_str else None
                        except ValueError:
                            fld_info["value"] = raw_str
                    elif data_type == "checkbox":
                        fld_info["value"] = 1 if raw_str in ["✓", "1", "X", "x"] else 0
                    else:
                        fld_info["value"] = raw_str
        if modified:
            try:
                ext_file.write_text(json.dumps(ext_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            # SIMPAN OTOMATIS KE DATABASE SQLite SESUAI ID BOOKLET
            try:
                from db.inference_repository import save_booklet_inference_to_db
                save_booklet_inference_to_db(booklet_id, ext_data, model_name=payload.model)
            except Exception as e:
                print(f"[Gemini] Peringatan: Gagal simpan otomatis inferensi ke DB: {e}")

    # Sinkronisasi ke hitl_queue SQLite untuk digit ambigu / confidence < 85%
    if batch_res.get("results"):
        try:
            conn = get_connection()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results = batch_res["results"]
            for item in crops_to_evaluate:
                res_item = results.get(item.get("id")) or results.get(item.get("label"))
                if not res_item:
                    continue
                p_digit = res_item.get("pred_digit")
                c_conf = float(res_item.get("conf") or 0.0)
                is_b = res_item.get("is_blank", False)
                is_ambig = (res_item.get("is_doubtful")) or (p_digit in ["ragu_ragu", "?", "ragu", None]) or (c_conf < 0.85 and not is_b)

                if is_ambig:
                    # ATURAN MUTLAK USER: Untuk label ragu-ragu, tetap ditulis label prediksi terkuatnya
                    strongest_pred = str(p_digit) if p_digit not in ["ragu_ragu", "?", "ragu", None, ""] else str(res_item.get("alt_digit") or res_item.get("alt_char") or "0")
                    alt_pred = str(res_item.get("alt_digit") or ("1" if strongest_pred != "1" else "7"))
                    hitl_id = f"hitl_{booklet_id}_{item.get('id')}"
                    top3 = [
                        {"digit": strongest_pred, "conf": round(c_conf * 100, 1)},
                        {"digit": alt_pred, "conf": round(max(1.0, (1.0 - c_conf) * 100), 1)},
                        {"digit": "3" if strongest_pred not in ["3", alt_pred] else "8", "conf": 0.5}
                    ]
                    conn.execute("""
                        INSERT OR REPLACE INTO hitl_queue
                        (id, booklet_id, variable, label, block_name, page, pred_digit, pred_conf,
                         top3_json, hint, corrected, final_digit, tier, crop_path, priority)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 7, ?, ?)
                    """, (
                        hitl_id, booklet_id, item.get("field_id"), item.get("box_label") or item.get("label"),
                        f"Hal {item.get('page')}", item.get("page"),
                        strongest_pred, round(c_conf * 100, 1),
                        json.dumps(top3), res_item.get("reasoning", f"Keyakinan {round(c_conf*100, 1)}% (Prediksi terkuat: {strongest_pred}). Periksa langsung pada citra scan A4."),
                        item.get("crop_path"), 2 if c_conf < 0.70 else 1
                    ))
            conn.execute("""
                UPDATE booklets SET
                    hitl_count = (SELECT COUNT(*) FROM hitl_queue WHERE booklet_id = booklets.id),
                    hitl_done  = (SELECT COUNT(*) FROM hitl_queue WHERE booklet_id = booklets.id AND corrected = 1),
                    updated_at = ?
                WHERE id = ?
            """, (now_str, booklet_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[routes_gemini] Warning update hitl_queue: {e}")

    return batch_res

