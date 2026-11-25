"""
Modul Ekstraktor Potongan Bidang Teks Bebas (Freeform Handwriting Text Fields)
Kuesioner Seruti Inti BPS.

Mengekstrak potongan citra resolusi penuh untuk:
- Halaman 1: Nama KRT (R110), Alamat (R111), Nama Pencacah (R201), Nama Pengawas (R202)
- Halaman 2: Roster Nama ART 01 s.d. 10 (Blok III Kolom 2)
- Halaman 7: Blok VII Catatan Lapangan (R700)
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from seruti_pipeline.grid_template_aligner import align_template_to_image

BASE_DIR = Path(__file__).resolve().parent.parent


def detect_text_ink_activity(crop_bgr: np.ndarray, is_dotted_field: bool = True) -> Dict[str, Any]:
    """
    Menganalisis densitas tinta tulisan tangan pada potongan bidang teks,
    membedakan tulisan tangan nyata dengan area kosong / hanya garis titik-titik.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {"is_blank": True, "ink_ratio": 0.0, "ink_pixels": 0, "has_handwriting": False}

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # 1. Binarisasi adaptif untuk mendeteksi tinta
    # Latar kertas kuesioner biasanya > 220, goresan pensil/pulpen < 185
    _, binary = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY_INV)

    # 2. Jika bidang memiliki garis titik-titik, lakukan filter morfologi horizontal
    # Garis titik biasanya setinggi 1-2 px dengan jarak renggang
    if is_dotted_field:
        # Hapus titik-titik kecil terisolasi yang lebarnya < 3 px dan tingginya < 3 px
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        filtered = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    else:
        filtered = binary

    # 3. Hitung Connected Components untuk memverifikasi goresan huruf
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered, connectivity=8)

    # Cari komponen yang memiliki ukuran karakter wajar (tinggi > 8 px, area > 25 px)
    valid_strokes = 0
    ink_pixels = 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # Karakter tulisan tangan biasanya memiliki tinggi minimal 7 px dan area minimal 20 px
        if h >= 7 and area >= 20 and w < 0.95 * W:
            valid_strokes += 1
            ink_pixels += area

    total_pixels = H * W
    ink_ratio = (ink_pixels / total_pixels) if total_pixels > 0 else 0.0

    # Bidang dianggap memiliki tulisan tangan jika ada goresan karakter valid
    has_handwriting = valid_strokes >= 1 and ink_pixels >= 45

    return {
        "is_blank": not has_handwriting,
        "has_handwriting": has_handwriting,
        "ink_ratio": round(float(ink_ratio), 4),
        "ink_pixels": int(ink_pixels),
        "stroke_count": int(valid_strokes)
    }


def extract_text_fields_page(
    booklet_dir: Path,
    page_num: int,
    output_crops_base: Path,
    template_dir: Path = Path("templates"),
    pad_px: int = 4
) -> List[Dict[str, Any]]:
    """
    Mengekstrak seluruh bidang teks bebas pada satu halaman buklet.
    Menyimpan citra potongan ke output/crops/{booklet_id}/text_fields/{field_id}.png.
    """
    booklet_id = booklet_dir.name
    t_file = template_dir / f"template_p{page_num}.json"
    if not t_file.exists():
        return []

    with open(t_file, "r", encoding="utf-8") as f:
        t_data = json.load(f)

    text_fields_def = t_data.get("text_fields", [])
    if not text_fields_def:
        return []

    # Muat citra halaman
    pages_dir = booklet_dir / "pages"
    img_path = pages_dir / f"hal_{page_num}_clean.png"
    if not img_path.exists():
        img_path = pages_dir / f"hal_{page_num}_raw.jpg"
        if not img_path.exists():
            return []

    img = cv2.imread(str(img_path))
    if img is None:
        return []

    H_img, W_img = img.shape[:2]

    # Selaraskan koordinat menggunakan aligner
    aligner = align_template_to_image(img, page_num=page_num, template_dir=template_dir)
    # Aktifkan physical cell snapping adaptif terhadap garis kisi tabel fisik pindaian
    aligner.enable_physical_cell_snapping(img)

    # Direktori output potongan teks
    out_dir = output_crops_base / booklet_id / f"hal_{page_num}" / "text_fields"
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted_items = []

    for f_def in text_fields_def:
        f_id = f_def["id"]
        cid = f_def.get("cell_id")
        bbox_pt = f_def.get("bbox_pt")
        if not bbox_pt and not cid:
            continue

        # Prioritas 1: Ambil bounding box hasil physical cell snapping adaptif
        b_px = aligner.get_cell_bbox_px(cid) if cid else None
        if b_px is not None:
            x0, y0, x1, y1 = b_px
            if pad_px > 0:
                x0 -= pad_px
                y0 -= pad_px
                x1 += pad_px
                y1 += pad_px
        else:
            # Prioritas 2: Fallback ke affine transformation pt_to_px
            x0, y0, x1, y1 = aligner.bbox_pt_to_px(bbox_pt, pad=pad_px)

        # Clamping ke batas citra
        x0 = max(0, min(W_img - 1, x0))
        y0 = max(0, min(H_img - 1, y0))
        x1 = max(0, min(W_img, x1))
        y1 = max(0, min(H_img, y1))

        if x1 <= x0 or y1 <= y0:
            continue

        crop = img[y0:y1, x0:x1].copy()

        # Deteksi aktivitas tinta
        is_dotted = "alamat" in f_id or "nama" in f_id
        ink_stats = detect_text_ink_activity(crop, is_dotted_field=is_dotted)

        # Simpan potongan citra
        crop_filename = f"{f_id}.png"
        crop_save_path = out_dir / crop_filename
        cv2.imwrite(str(crop_save_path), crop)

        rel_crop = str(crop_save_path.resolve().relative_to(BASE_DIR.resolve())).replace("\\", "/") if crop_save_path.resolve().is_relative_to(BASE_DIR.resolve()) else str(crop_save_path).replace("\\", "/")

        extracted_items.append({
            "booklet_id": booklet_id,
            "page_num": page_num,
            "page_id": f"hal_{page_num}",
            "field_id": f_id,
            "name": f_def.get("label", f_id),
            "label": f_def.get("label", f_id),
            "question_code": f_def.get("question_code", ""),
            "field_type": f_def.get("type", "text"),
            "roster_no": f_def.get("roster_no"),
            "bbox_px": [int(x0), int(y0), int(x1), int(y1)],
            "dimensions": [int(x1 - x0), int(y1 - y0)],
            "file": rel_crop,
            "crop_path": rel_crop,
            "is_blank": bool(ink_stats["is_blank"]),
            "has_handwriting": bool(ink_stats["has_handwriting"]),
            "ink_ratio": float(ink_stats["ink_ratio"]),
            "ink_pixels": int(ink_stats["ink_pixels"]),
            "stroke_count": int(ink_stats["stroke_count"])
        })

    return extracted_items


def extract_all_text_fields_booklet(
    booklet_dir: Path,
    output_crops_base: Path,
    template_dir: Path = Path("templates")
) -> List[Dict[str, Any]]:
    """Mengekstrak seluruh bidang teks bebas pada buklet (Hal 1, 2, 5, dan 7)."""
    all_fields = []
    for p in [1, 2, 5, 7]:
        fields = extract_text_fields_page(booklet_dir, p, output_crops_base, template_dir=template_dir)
        all_fields.extend(fields)
    return all_fields
