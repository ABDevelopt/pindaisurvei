#!/usr/bin/env python3
"""
run_grid_cropper.py
===================
Hierarchical Multi-Tier Grid Alignment & Crop Extractor
Kuesioner BPS Susenas Seruti Inti (VSERUTI26.INTI).

Mengimplementasikan Tingkatan Crop Hierarkis (7 Tingkat):
1. Makro (`makro/`): Blok utama kuesioner (Blok I s.d. VII, Header, Kesediaan, dsb).
   - Target: Pemahaman konteks makro dokumen oleh Vision-Language Models (VLM).
2. Meso (`meso/`): Sub-blok modul pertanyaan (R101-R111, R301-R307, R501, R502, dsb).
   - Target: Human-in-the-Loop (HITL) review dan validasi per seksi pertanyaan.
3. Rows (`rows/`): Baris roster / data tabel horizontal utuh (Zero-Cut Invariant, snapped ke garis horizontal).
   - Target: Peninjauan baris roster terintegrasi (nama ART, rincian komoditas makanan/non-makanan, konsistensi antar-kolom).
4. Fields (`fields/`): Area field isian utuh (seluruh kotak nominal/rincian sebaris).
   - Target: Verifikasi field-level, OCR multi-digit sebaris, dan audit trail per rincian.
5. Precise Grid (`precise_grid/`): Perimeter amplop kisi tabel fisik utuh 100% segi empat
   terpotong PERSIS PADA GARIS BATAS LUAR TABEL (Physical Line-Snapped, Zero-Cut Invariant)
   beserta visualisasi overlay kisi sel interior (lattice preview).
   - Catatan Penghematan Disk: Sel kisi individual terkecil (`precise_grid/cells/`) hanya
     dibuat jika flag `--crop-grid-cells` diaktifkan secara eksplisit, guna menghemat
     ruang penyimpanan disk drive lokal.
6. Mikro (`mikro/`): Kotak comb-box digit individual (51x66 px) & OMR checkbox (legacy).
   - Varian Raw (`mikro/raw/`): Citra sel asli hasil snapping fisik 0 px drift.
   - Varian Clean 28x28 (`mikro/clean_28x28/`): Citra biner terstandarisasi untuk klasifikasi digit OCR/ICR BPS.
7. Kotak Digit (`digits/`): Comb-box digit individual terstandarisasi per field berlabel seragam.
   - Varian Raw (`digits/raw/{field_label}/{digit_label}.png`): Citra sel asli berbingkai fisik.
   - Varian Inner (`digits/inner/{field_label}/{digit_label}_inner.png`): Citra isi digit tanpa border comb-box (goresan pensil utuh).
   - Varian Clean 28x28 (`digits/clean_28x28/{field_label}/{digit_label}_clean_28x28.png`): Citra biner 28x28 proporsional letterbox untuk CNN / ONNX.

Mendukung eksekusi batch multi-processing berkecepatan tinggi dengan alokasi worker aman,
penanganan error graceful per buklet & per halaman, serta ringkasan konsolidasi
`output/crops/batch_crop_summary.json` dengan kepatuhan Zero PII Cloud Egress.
"""

import sys
import os
import re
import time
import argparse
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Pastikan root workspace terdaftar di sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
from PIL import Image

from seruti_pipeline.grid_template_aligner import align_template_to_image, DEFAULT_TEMPLATE_DIR
from seruti_pipeline.semantic_relations import get_relations_for_page
from seruti_pipeline.advanced_filter import remove_comb_box_border, letterbox_cell_28x28, is_blank_cell

INK_THR = 170        # Piksel < 170 dianggap goresan tinta pensil grafit
BLANK_RATIO = 0.012  # Rasio tinta minimum agar sel comb-box dianggap terisi
PNG_COMPRESSION_LEVEL = 6  # Kompresi PNG standar (lossless dan optimal)


def save_crop_image(path: Path, img: np.ndarray, compression_level: int = PNG_COMPRESSION_LEVEL) -> bool:
    """
    Simpan citra crop dengan optimasi kompresi lossless untuk menghemat kapasitas disk:
    1. Jika citra 3-channel memiliki kanal identik (R==G==B, dokumen monokrom),
       konversi ke 1-channel grayscale (8-bit) menghemat 50-60% ukuran disk.
    2. Jika citra berwarna bergaris kisi (seperti lattice overlay), simpan dengan
       palet adaptif 8-bit (P mode PNG) menghemat >55% ukuran disk.
    3. Gunakan kompresi PNG level 6.
    """
    if img is None or img.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(img.shape) == 3:
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        if np.array_equal(b, g) and np.array_equal(b, r):
            return cv2.imwrite(str(path), b, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        else:
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=64)
                pil_im.save(str(path), optimize=True)
                return True
            except Exception:
                return cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    else:
        return cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])


def clean_cell_mnist(
    cell: np.ndarray,
    ink_thr: int = INK_THR,
    blank_ratio: float = BLANK_RATIO,
    border_margin_pct: float = 0.04,
    pad_digit_px: int = 2,
    inner_target_size: float = 24.0
) -> Optional[np.ndarray]:
    """
    Pembersihan sel comb-box kuesioner BPS menjadi kanvas 28x28 dengan letterboxing proporsional
    dan multi-factor blank cell detection (Zero Phantom Digits):
    1. Hapus garis kisi comb-box di ruang koordinat asli via remove_comb_box_border().
    2. Deteksi sel kosong secara presisi via is_blank_cell(). Mengembalikan None jika sel kosong.
    3. Letterbox proporsional menjaga rasio aspek asli tanpa distorsi pergeseran massa.
    """
    if cell is None or cell.size == 0:
        return None
    cl = remove_comb_box_border(cell)
    if is_blank_cell(cl):
        return None
    return letterbox_cell_28x28(cl, target_dim=int(inner_target_size))


def clean_cell_mnist_legacy(
    cell: np.ndarray,
    ink_thr: int = INK_THR,
    blank_ratio: float = BLANK_RATIO,
    border_margin_pct: float = 0.04,
    pad_digit_px: int = 2,
    inner_target_size: float = 20.0
) -> Optional[np.ndarray]:
    """
    Pembersihan sel comb-box kuesioner BPS menjadi kanvas 28x28 gaya MNIST dengan preservasi keutuhan digit:
    1. Konversi ke grayscale jika citra berwarna.
    2. Pangkas margin tepi minimal (default 4% atau 1-2 px) agar coretan digit tetap utuh tanpa terpotong.
    3. Ekstraksi biner tinta pensil (piksel < ink_thr).
    4. Eliminasi cerdas garis batas kotak comb fisik via connectedComponentsWithStats pada perimeter luar.
    5. Tambahkan margin pengaman (pad_digit_px) di sekeliling bounding box digit agar lengkungan pensil tidak terpangkas.
    6. Pemusatan massa (Center of Mass via cv2.moments) dengan boundary clamping agar digit tidak keluar kanvas 28x28.
    """
    if cell is None or cell.size == 0:
        return None
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    h, w = cell.shape
    # Margin tepi minimal (default 4%, min 1 px) untuk memaksimalkan area tangkapan angka
    m = max(1, int(round(border_margin_pct * min(h, w))))
    if h <= 2 * m or w <= 2 * m:
        return None
    c = cell[m:h - m, m:w - m]
    ink = (c < ink_thr).astype(np.uint8) * 255
    H, W = ink.shape
    if H < 5 or W < 5:
        return None

    # Buang sisa garis kotak yang menempel di tepi (komponen tipis di pinggir)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    valid_labels = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        edge_h = (y <= 3 or y + bh >= H - 3) and bw > 0.65 * W and bh < 0.30 * H
        edge_v = (x <= 3 or x + bw >= W - 3) and bh > 0.65 * H and bw < 0.30 * W
        edge_sliver = (x <= 2 or x + bw >= W - 2) and bw <= 4
        if edge_h or edge_v or edge_sliver or a < 5:
            ink[lab == i] = 0
        else:
            valid_labels.append(i)

    if not valid_labels or (ink.mean() / 255.0 < blank_ratio):
        return None

    # Isolasi komponen dominan: buang stroke bocor dari kotak tetangga yang terpisah jauh
    main_lbl = max(valid_labels, key=lambda i: st[i][4])  # max area
    mx, my, mw, mh, _ = st[main_lbl]
    for i in valid_labels:
        if i == main_lbl:
            continue
        cx, cy, cw, ch, _ = st[i]
        dist_x = max(0, max(cx - (mx + mw), mx - (cx + cw)))
        if dist_x > 6:
            ink[lab == i] = 0

    ys, xs = np.where(ink > 0)
    if len(ys) == 0 or len(xs) == 0:
        return None

    # Tambahkan margin padding ekstra di sekeliling digit sebelum resize agar goresan pensil utuh
    y_min = max(0, int(ys.min()) - pad_digit_px)
    y_max = min(H - 1, int(ys.max()) + pad_digit_px)
    x_min = max(0, int(xs.min()) - pad_digit_px)
    x_max = min(W - 1, int(xs.max()) + pad_digit_px)
    crop = ink[y_min:y_max + 1, x_min:x_max + 1]

    if max(crop.shape) < 0.20 * max(H, W):   # noktah kecil / debu, bukan digit
        return None

    s = inner_target_size / max(crop.shape)
    new_w = max(1, int(round(crop.shape[1] * s)))
    new_h = max(1, int(round(crop.shape[0] * s)))
    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    crop = cv2.dilate(crop, np.ones((2, 2), np.uint8))   # tebalkan ~ MNIST

    canvas = np.zeros((28, 28), np.uint8)
    y0, x0 = (28 - crop.shape[0]) // 2, (28 - crop.shape[1]) // 2
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop

    # Pemusatan massa (Center of Mass) dengan boundary-safe clamping agar digit tidak terpotong di batas kanvas
    M = cv2.moments(canvas)
    if M["m00"] > 0:
        dx = 14.0 - M["m10"] / M["m00"]
        dy = 14.0 - M["m01"] / M["m00"]
        max_shift_x = max(0.0, float(x0 - 1))
        max_shift_y = max(0.0, float(y0 - 1))
        dx = max(-max_shift_x, min(max_shift_x, dx))
        dy = max(-max_shift_y, min(max_shift_y, dy))
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))

    return canvas


def sanitize_filename(name: str) -> str:
    """Sanitasi nama string agar aman untuk sistem berkas Windows."""
    clean = re.sub(r'[<>:"/\\|?* ]+', '_', str(name))
    return clean.strip('_')


def process_booklet_page(
    booklet_dir: Path,
    page_num: int,
    output_base: Path,
    do_preview: bool = False,
    do_crop_makro: bool = True,
    do_crop_meso: bool = True,
    do_crop_rows: bool = True,
    do_crop_fields: bool = True,
    do_crop_mikro: bool = True,
    do_crop_digits: bool = True,
    do_crop_grid: bool = True,
    do_crop_grid_cells: bool = False,
    all_rows: bool = False,
    clean_mnist: bool = False,
    use_vlm: bool = False,
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
    verbose: bool = False,
    box_pad: int = 0,
    border_margin_pct: float = 0.04,
    pad_digit_px: int = 2
) -> Dict[str, Any]:
    """
    Memproses satu halaman buklet kuesioner secara komprehensif:
    1. Registrasi & Penyelarasan Grid Vektor (Physical Cell Snapping).
    2. Tingkat 1: Ekstraksi Makro (Blok Utama).
    3. Tingkat 2: Ekstraksi Meso (Sub-Blok Pertanyaan).
    4. Tingkat 3: Ekstraksi Rows (Baris Roster / Data Tabel Horizontal Utuh).
    5. Tingkat 4: Ekstraksi Fields (Field Isian Utuh).
    6. Tingkat 5: Ekstraksi Precise Grid (Perimeter Amplop Tabel Persis Pada Garis Fisik).
    7. Tingkat 6: Ekstraksi Mikro (Comb-Box Digit Raw & Clean 28x28).
    8. Penulisan Hierarchical Crop Manifest JSON (dengan preservasi tier lama).
    """
    booklet_dir = Path(booklet_dir).resolve()
    output_base = Path(output_base).resolve()
    template_dir = Path(template_dir).resolve()

    pages_dir = booklet_dir / "pages"
    img_name = f"hal_{page_num}_vlm.jpg" if use_vlm else f"hal_{page_num}_clean.png"
    img_path = pages_dir / img_name

    if not img_path.exists():
        img_name = f"hal_{page_num}_clean.png"
        img_path = pages_dir / img_name
        if not img_path.exists():
            img_name = f"hal_{page_num}_raw.jpg"
            img_path = pages_dir / img_name
            if not img_path.exists():
                if verbose:
                    print(f"  [PERINGATAN] Berkas citra tidak ditemukan: {img_path}")
                return {}

    img = cv2.imread(str(img_path))
    if img is None:
        if verbose:
            print(f"  [ERROR] Gagal memuat citra: {img_path}")
        return {}

    booklet_id = booklet_dir.name
    page_crops_dir = output_base / "crops" / booklet_id / f"hal_{page_num}"
    page_crops_dir.mkdir(parents=True, exist_ok=True)

    # Bersihkan direktori legacy lama (boxes, sub_blocks) jika ada agar direktori bersih
    for legacy_name in ["boxes", "sub_blocks"]:
        legacy_path = page_crops_dir / legacy_name
        if legacy_path.exists():
            shutil.rmtree(legacy_path, ignore_errors=True)

    # 1. Selaraskan template & aktifkan physical cell snapping
    alignment = align_template_to_image(img, page_num=page_num, template_dir=template_dir)
    snapped_digits = alignment.enable_physical_cell_snapping(img)

    if verbose:
        print(f"  [Hal {page_num}] {booklet_id} ({img.shape[1]}x{img.shape[0]} px) | "
              f"Affine: sx={alignment.scale_x:.4f}, sy={alignment.scale_y:.4f}, "
              f"tx={alignment.trans_x:.1f}, ty={alignment.trans_y:.1f} | "
              f"Snapped: {snapped_digits} digit")

    manifest_path = page_crops_dir / "crop_manifest.json"
    existing_manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                existing_manifest = json.load(mf)
        except Exception:
            existing_manifest = {}

    existing_tiers = existing_manifest.get("tiers", {})
    existing_summary = existing_manifest.get("summary", {})

    manifest: Dict[str, Any] = {
        "booklet_id": booklet_id,
        "page_num": page_num,
        "image_file": str(img_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "dimensions": [img.shape[1], img.shape[0]],
        "scale": [round(alignment.scale_x, 4), round(alignment.scale_y, 4)],
        "translation": [round(alignment.trans_x, 2), round(alignment.trans_y, 2)],
        "snapped_digits": snapped_digits,
        "summary": dict(existing_summary),
        "tiers": {
            "makro": existing_tiers.get("makro", []),
            "meso": existing_tiers.get("meso", []),
            "rows": existing_tiers.get("rows", []),
            "fields": existing_tiers.get("fields", []),
            "text_fields": existing_tiers.get("text_fields", []),
            "precise_grid": existing_tiers.get("precise_grid", []),
            "mikro": existing_tiers.get("mikro", [])
        },
        "semantic_relations": get_relations_for_page(page_num)
    }

    # =========================================================================
    # TINGKAT 1: MAKRO (Blok Utama Kuesioner)
    # =========================================================================
    makro_records = []
    if do_crop_makro:
        makro_dir = page_crops_dir / "makro"
        makro_dir.mkdir(parents=True, exist_ok=True)
        blocks = alignment.crop_all_blocks(img)
        for blk in blocks:
            b_id = blk["block_id"]
            code_safe = sanitize_filename(blk.get("code") or b_id)
            out_name = f"{b_id}_{code_safe}.png"
            out_file = makro_dir / out_name
            save_crop_image(out_file, blk["crop"])
            makro_records.append({
                "block_id": b_id,
                "code": blk.get("code"),
                "name": blk.get("name"),
                "bbox_px": blk["bbox_px"],
                "bbox_pt": blk["bbox_pt"],
                "dimensions": [blk["crop"].shape[1], blk["crop"].shape[0]],
                "file": str(out_file.relative_to(BASE_DIR)).replace("\\", "/")
            })
        manifest["tiers"]["makro"] = makro_records
        manifest["summary"]["makro_count"] = len(makro_records)
        if verbose:
            print(f"    -> [Makro]  {len(makro_records)} Blok Utama dipotong ke: {makro_dir.relative_to(BASE_DIR).as_posix()}")

    # =========================================================================
    # TINGKAT 2: MESO (Sub-Blok Modul Pertanyaan)
    # =========================================================================
    meso_records = []
    if do_crop_meso:
        meso_dir = page_crops_dir / "meso"
        meso_dir.mkdir(parents=True, exist_ok=True)
        sub_blocks = alignment.crop_all_sub_blocks(img)
        for sb in sub_blocks:
            sb_id = sb["sub_block_id"]
            sb_code_safe = sanitize_filename(sb.get("code") or sb_id)
            out_name = f"{sb_id}_{sb_code_safe}.png"
            out_file = meso_dir / out_name
            save_crop_image(out_file, sb["crop"])
            meso_records.append({
                "sub_block_id": sb_id,
                "code": sb.get("code"),
                "name": sb.get("name"),
                "type": sb.get("type"),
                "bbox_px": sb["bbox_px"],
                "bbox_pt": sb["bbox_pt"],
                "dimensions": [sb["crop"].shape[1], sb["crop"].shape[0]],
                "cells_count": sb.get("cells_count", 0),
                "fields_count": sb.get("fields_count", 0),
                "digits_count": sb.get("digits_count", 0),
                "file": str(out_file.relative_to(BASE_DIR)).replace("\\", "/")
            })
        manifest["tiers"]["meso"] = meso_records
        manifest["summary"]["meso_count"] = len(meso_records)
        if verbose:
            print(f"    -> [Meso]   {len(meso_records)} Sub-Blok dipotong ke: {meso_dir.relative_to(BASE_DIR).as_posix()}")

    # =========================================================================
    # TINGKAT ROWS: Baris Roster / Data Mengacu pada Sub-Blok (Line-Snapped)
    # =========================================================================
    row_records = []
    if do_crop_rows:
        rows_dir = page_crops_dir / "rows"
        if rows_dir.exists():
            shutil.rmtree(rows_dir, ignore_errors=True)
        rows_dir.mkdir(parents=True, exist_ok=True)
        subblock_rows = alignment.crop_all_table_rows(img, pad=0, snap_to_lines=True, data_only=not all_rows)
        for r_info in subblock_rows:
            r_id = r_info["row_id"]
            sb_id = r_info.get("sub_block_id") or "sub_general"
            sb_code = r_info.get("sub_block_code") or "GENERAL"
            sb_name = r_info.get("sub_block_name") or ""
            t_idx = r_info.get("table_index")
            r_idx = r_info["row_index"]
            lbl_safe = sanitize_filename(r_info.get("row_label") or f"r{r_idx}")

            # Organisasi folder baris mengacu pada sub-blok induk
            sb_code_safe = sanitize_filename(sb_code)
            sb_row_dir = rows_dir / sb_code_safe
            sb_row_dir.mkdir(parents=True, exist_ok=True)

            r_fname = f"row_{r_idx:02d}_{lbl_safe}.png"
            r_path = sb_row_dir / r_fname
            save_crop_image(r_path, r_info["crop"])

            row_records.append({
                "row_id": r_id,
                "sub_block_id": sb_id,
                "sub_block_code": sb_code,
                "sub_block_name": sb_name,
                "table_index": t_idx,
                "row_index": r_idx,
                "row_type": r_info["row_type"],
                "row_label": r_info["row_label"],
                "description": r_info.get("description", ""),
                "bbox_px": r_info["bbox_px"],
                "bbox_pt": r_info["bbox_pt"],
                "dimensions": r_info["dimensions"],
                "fields_count": r_info["fields_count"],
                "fields": r_info["fields"],
                "field_details": r_info["field_details"],
                "cells_count": r_info.get("cells_count", 0),
                "file": str(r_path.relative_to(BASE_DIR)).replace("\\", "/")
            })
        manifest["tiers"]["rows"] = row_records
        manifest["summary"]["rows_count"] = len(row_records)
        if verbose:
            print(f"    -> [Rows]   {len(row_records)} Baris Roster (Sub-Blok) dipotong ke: {rows_dir.relative_to(BASE_DIR).as_posix()}")

    # =========================================================================
    # TINGKAT PRECISE GRID (Amplop Perimeter Kisi Tabel Fisik Utuh Snapped ke Garis)
    # =========================================================================
    grid_records = []
    cell_records = []
    if do_crop_grid:
        grid_dir = page_crops_dir / "precise_grid"
        grid_dir.mkdir(parents=True, exist_ok=True)
        table_envelopes = alignment.crop_all_table_envelopes(img)
        for tbl in table_envelopes:
            t_idx = tbl["table_index"]
            env_fname = f"table_{t_idx:02d}_envelope.png"
            env_file = grid_dir / env_fname
            save_crop_image(env_file, tbl["crop"])

            lat_fname = f"table_{t_idx:02d}_lattice.png"
            lat_file = grid_dir / lat_fname
            save_crop_image(lat_file, tbl["lattice_preview"])

            grid_records.append({
                "table_index": t_idx,
                "envelope_status": "PHYSICAL_LINE_SNAPPED",
                "bbox_px": tbl["bbox_px"],
                "bbox_pt": tbl["bbox_pt"],
                "width": tbl["width"],
                "height": tbl["height"],
                "cell_count": tbl["cell_count"],
                "envelope_file": str(env_file.relative_to(BASE_DIR)).replace("\\", "/"),
                "lattice_file": str(lat_file.relative_to(BASE_DIR)).replace("\\", "/")
            })
        manifest["tiers"]["precise_grid"] = grid_records
        manifest["summary"]["precise_grid_count"] = len(grid_records)
        if verbose:
            print(f"    -> [Grid]   {len(grid_records)} Amplop Tabel Fisik dipotong ke: {grid_dir.relative_to(BASE_DIR).as_posix()}")

        # Pemotongan Sel Kisi Fisik Terkecil (Atomic Grid Cells) jika diaktifkan secara eksplisit
        cells_base_dir = grid_dir / "cells"
        if do_crop_grid_cells:
            cells_base_dir.mkdir(parents=True, exist_ok=True)
            atomic_cells = alignment.crop_all_grid_cells(img)
            for ac in atomic_cells:
                t_idx = ac["table_index"]
                tbl_cells_dir = cells_base_dir / f"table_{t_idx:02d}"
                tbl_cells_dir.mkdir(parents=True, exist_ok=True)
                cid = ac["cell_id"]
                r = ac.get("row")
                c = ac.get("col")
                txt_part = sanitize_filename(ac.get("text", "")[:12]) if ac.get("text") else ac.get("kind", "cell")
                if r is not None and c is not None:
                    cfname = f"r{r:02d}_c{c:02d}_{cid}_{txt_part}.png"
                else:
                    cfname = f"{cid}_{txt_part}.png"
                cpath = tbl_cells_dir / cfname
                save_crop_image(cpath, ac["crop"])
                cell_records.append({
                    "cell_id": cid,
                    "table_index": t_idx,
                    "row": r,
                    "col": c,
                    "kind": ac["kind"],
                    "fill": ac["fill"],
                    "text": ac["text"],
                    "bbox_px": ac["bbox_px"],
                    "dimensions": [ac["crop"].shape[1], ac["crop"].shape[0]],
                    "file": str(cpath.relative_to(BASE_DIR)).replace("\\", "/")
                })
            manifest["tiers"]["grid_cells"] = cell_records
            manifest["summary"]["grid_cells_count"] = len(cell_records)
            if verbose:
                print(f"    -> [Cells]  {len(cell_records)} Sel Kisi Terkecil (Atomic) dipotong ke: {cells_base_dir.relative_to(BASE_DIR).as_posix()}")
        else:
            # Jika sel kisi dinonaktifkan, hapus cells/ lama jika ada untuk menghemat ruang disk
            if cells_base_dir.exists():
                shutil.rmtree(cells_base_dir, ignore_errors=True)

    # =========================================================================
    # TINGKAT FIELDS: Field Isian Kuesioner Utuh
    # =========================================================================
    all_fields = alignment.crop_all_fields(img, include_boxes=do_crop_mikro, pad=box_pad)
    field_records = []
    if do_crop_fields:
        fields_dir = page_crops_dir / "fields"
        fields_dir.mkdir(parents=True, exist_ok=True)
        for fld in all_fields:
            fid = fld["field_id"]
            lbl = fld.get("row_label") or "none"
            safe_lbl = sanitize_filename(lbl)
            f_fname = f"{fid}_{safe_lbl}.png"
            f_path = fields_dir / f_fname
            save_crop_image(f_path, fld["crop"])
            field_records.append({
                "field_id": fid,
                "row_label": lbl,
                "parent_text": fld.get("parent_text"),
                "n_boxes": fld.get("n_boxes", 1),
                "bbox_px": fld["bbox_px"],
                "dimensions": [fld["crop"].shape[1], fld["crop"].shape[0]],
                "file": str(f_path.relative_to(BASE_DIR)).replace("\\", "/")
            })
        manifest["tiers"]["fields"] = field_records
        manifest["summary"]["fields_count"] = len(field_records)
        if verbose:
            print(f"    -> [Fields] {len(field_records)} Field Isian dipotong ke: {fields_dir.relative_to(BASE_DIR).as_posix()}")

    # =========================================================================
    # TINGKAT 5: MIKRO (Kotak Comb-Box Digit Individual & OMR Checkbox)
    # =========================================================================
    clean_count = 0
    mikro_records = []
    if do_crop_mikro:
        mikro_dir = page_crops_dir / "mikro"
        mikro_raw_dir = mikro_dir / "raw"
        mikro_clean_native_dir = mikro_dir / "clean_native"
        mikro_clean_28_dir = mikro_dir / "clean_28x28" if clean_mnist else None
        
        mikro_raw_dir.mkdir(parents=True, exist_ok=True)
        mikro_clean_native_dir.mkdir(parents=True, exist_ok=True)
        if mikro_clean_28_dir:
            mikro_clean_28_dir.mkdir(parents=True, exist_ok=True)

        for fld in all_fields:
            fid = fld["field_id"]
            lbl = fld.get("row_label") or "none"
            safe_lbl = sanitize_filename(lbl)
            folder_tag = f"{fid}_{safe_lbl}"

            fld_raw_dir = mikro_raw_dir / folder_tag
            fld_raw_dir.mkdir(parents=True, exist_ok=True)
            fld_clean_native_dir = mikro_clean_native_dir / folder_tag
            fld_clean_native_dir.mkdir(parents=True, exist_ok=True)

            if mikro_clean_28_dir:
                fld_clean_28_dir = mikro_clean_28_dir / folder_tag
                fld_clean_28_dir.mkdir(parents=True, exist_ok=True)
            else:
                fld_clean_28_dir = None

            for b_info in fld.get("boxes", []):
                b_idx = b_info["box_index"]
                cid = b_info["cell_id"]
                raw_crop = b_info["crop"]
                b_px = b_info["bbox_px"]

                raw_name = f"box_{b_idx:02d}_{cid}.png"
                raw_path = fld_raw_dir / raw_name

                # GUARDRAIL 3: Assert minimum crop dimensions (Zero Degenerate Crop)
                if raw_crop.shape[0] < 20 or raw_crop.shape[1] < 20:
                    if verbose:
                        print(f"    [WARN] Degenerate crop terdeteksi: {raw_name} ({raw_crop.shape[1]}x{raw_crop.shape[0]} px)")

                save_crop_image(raw_path, raw_crop)

                # 1. In-Situ Native Comb-Box Removal (Dimensi Asli ~66x49)
                clean_native = remove_comb_box_border(raw_crop)
                clean_native_name = f"box_{b_idx:02d}_{cid}_clean.png"
                clean_native_path = fld_clean_native_dir / clean_native_name
                save_crop_image(clean_native_path, clean_native)
                clean_native_rel = str(clean_native_path.relative_to(BASE_DIR)).replace("\\", "/")

                # 2. Multi-Factor Blank Box Detection pada Citra Native (Zero Phantom Digits)
                is_blank = is_blank_cell(clean_native)
                if not is_blank:
                    clean_count += 1

                # 3. Varian 28x28 Proportional Letterboxed (Zero Distortion & Zero Border Bleed)
                clean_28_rel = None
                if fld_clean_28_dir:
                    c_28 = letterbox_cell_28x28(clean_native, target_dim=24)
                    c_28_name = f"box_{b_idx:02d}_{cid}_clean.png"
                    c_28_path = fld_clean_28_dir / c_28_name
                    save_crop_image(c_28_path, c_28)
                    clean_28_rel = str(c_28_path.relative_to(BASE_DIR)).replace("\\", "/")

                is_snapped = True if (alignment.snapped_cells_px and cid in alignment.snapped_cells_px) else False

                mikro_records.append({
                    "cell_id": cid,
                    "field_id": fid,
                    "row_label": lbl,
                    "box_index": b_idx,
                    "kind": "digit",
                    "bbox_px": b_px,
                    "is_snapped": is_snapped,
                    "is_blank": is_blank,
                    "raw_file": str(raw_path.relative_to(BASE_DIR)).replace("\\", "/"),
                    "clean_native_file": clean_native_rel,
                    "clean_file": clean_native_rel,
                    "clean_28x28_file": clean_28_rel
                })

        manifest["tiers"]["mikro"] = mikro_records
        manifest["summary"]["mikro_boxes_count"] = len(mikro_records)
        manifest["summary"]["mikro_clean_count"] = clean_count
        if verbose:
            print(f"    -> [Mikro]  {len(mikro_records)} Kotak Sel dipotong ({clean_count} terisi) ke: {mikro_dir.relative_to(BASE_DIR).as_posix()}")

    # =========================================================================
    # TINGKAT 7: KOTAK DIGIT KHUSUS (digits/) — Comb-Box Digit Terstandarisasi
    # =========================================================================
    digits_records = []
    if do_crop_digits:
        digits_dir = page_crops_dir / "digits"
        digits_raw_dir = digits_dir / "raw"
        digits_inner_dir = digits_dir / "inner"
        digits_mnist_dir = digits_dir / "clean_28x28" if clean_mnist else None

        digits_raw_dir.mkdir(parents=True, exist_ok=True)
        digits_inner_dir.mkdir(parents=True, exist_ok=True)
        if digits_mnist_dir:
            digits_mnist_dir.mkdir(parents=True, exist_ok=True)

        digit_boxes = alignment.crop_all_digit_boxes(img, pad=box_pad)
        digits_filled_count = 0

        for dbox in digit_boxes:
            cid = dbox["cell_id"]
            fid = dbox["field_id"] or "unassigned"
            f_lbl = dbox["field_label"] or fid
            d_lbl = dbox["digit_label"]
            b_idx = dbox["box_index"]
            total_b = dbox["total_boxes"]
            raw_crop = dbox["crop"]
            b_px = dbox["bbox_px"]

            safe_f_lbl = sanitize_filename(f_lbl)
            safe_d_lbl = sanitize_filename(d_lbl)

            # 1. Simpan Raw Crop (Kotak Asli Berbingkai Fisik)
            fld_raw_folder = digits_raw_dir / safe_f_lbl
            fld_raw_folder.mkdir(parents=True, exist_ok=True)
            raw_fname = f"{safe_d_lbl}.png"
            raw_path = fld_raw_folder / raw_fname
            save_crop_image(raw_path, raw_crop)

            # 2. Simpan Inner Crop (Border Garis Comb-Box Dihilangkan)
            fld_inner_folder = digits_inner_dir / safe_f_lbl
            fld_inner_folder.mkdir(parents=True, exist_ok=True)
            inner_crop = remove_comb_box_border(raw_crop)
            inner_fname = f"{safe_d_lbl}_inner.png"
            inner_path = fld_inner_folder / inner_fname
            save_crop_image(inner_path, inner_crop)

            # Deteksi Blank Box (Zero Phantom Digits)
            is_blank = is_blank_cell(inner_crop)
            if not is_blank:
                digits_filled_count += 1

            # 3. Simpan MNIST 28x28 Format (Letterbox Proporsional untuk CNN / ONNX)
            mnist_rel = None
            if digits_mnist_dir:
                fld_mnist_folder = digits_mnist_dir / safe_f_lbl
                fld_mnist_folder.mkdir(parents=True, exist_ok=True)
                c_28 = letterbox_cell_28x28(inner_crop, target_dim=24)
                mnist_fname = f"{safe_d_lbl}_clean_28x28.png"
                mnist_path = fld_mnist_folder / mnist_fname
                save_crop_image(mnist_path, c_28)
                mnist_rel = str(mnist_path.relative_to(BASE_DIR)).replace("\\", "/")

            is_snapped = True if (alignment.snapped_cells_px and cid in alignment.snapped_cells_px) else False

            digits_records.append({
                "cell_id": cid,
                "field_id": fid,
                "field_label": f_lbl,
                "digit_label": d_lbl,
                "box_index": b_idx,
                "total_boxes": total_b,
                "bbox_px": b_px,
                "dimensions": [raw_crop.shape[1], raw_crop.shape[0]],
                "is_snapped": is_snapped,
                "is_blank": is_blank,
                "is_filled": not is_blank,
                "raw_file": str(raw_path.relative_to(BASE_DIR)).replace("\\", "/"),
                "inner_file": str(inner_path.relative_to(BASE_DIR)).replace("\\", "/"),
                "clean_28x28_file": mnist_rel,
            })

        manifest["tiers"]["digits"] = digits_records
        manifest["summary"]["digits_count"] = len(digits_records)
        manifest["summary"]["digits_filled_count"] = digits_filled_count
        manifest["summary"]["digits_blank_count"] = len(digits_records) - digits_filled_count
        if verbose:
            print(f"    -> [Digits] {len(digits_records)} Kotak Digit dipotong ({digits_filled_count} terisi) ke: {digits_dir.relative_to(BASE_DIR).as_posix()}")

    # 6. Gambar Overlay Pratinjau Aligned Preview (Opsional)
    if do_preview:
        preview_dir = output_base / "grid_previews" / booklet_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        overlay_img = alignment.draw_alignment_overlay(
            img, draw_tables=True, draw_fields=True, draw_labels=True, draw_sub_blocks=True
        )
        out_prev_path = preview_dir / f"hal_{page_num}_grid_overlay.png"
        save_crop_image(out_prev_path, overlay_img)
        manifest["preview_file"] = str(out_prev_path.relative_to(BASE_DIR)).replace("\\", "/")
        if verbose:
            print(f"    -> Pratinjau grid disimpan ke: {out_prev_path.relative_to(BASE_DIR).as_posix()}")
    else:
        manifest["preview_file"] = None

    # 6b. Ekstraksi Bidang Teks Bebas (Text Fields / Handwriting)
    if page_num in [1, 2, 7]:
        try:
            from seruti_pipeline.text_field_extractor import extract_text_fields_page
            tf_extracted = extract_text_fields_page(
                booklet_dir=booklet_dir,
                page_num=page_num,
                output_crops_base=output_base / "crops",
                template_dir=template_dir
            )
            if tf_extracted:
                manifest["tiers"]["text_fields"] = tf_extracted
                manifest["summary"]["text_fields_count"] = len(tf_extracted)
                if verbose:
                    print(f"    -> [Text Fields] {len(tf_extracted)} Bidang Teks Bebas diekstrak")
        except Exception as e:
            if verbose:
                print(f"    -> [Warning] Gagal ekstrak text fields: {e}")

    # 7. Simpan Berkas Manifest Hierarkis Halaman
    manifest_path = page_crops_dir / "crop_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"    -> Manifest hierarkis disimpan ke: {manifest_path.relative_to(BASE_DIR).as_posix()}")

    return manifest


def process_single_booklet_task(task_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fungsi worker multiprocessing untuk memproses satu buklet secara mandiri dan aman.
    Dilengkapi try...except per halaman dan per buklet untuk ketahanan proses (fault tolerance).
    """
    booklet_dir = Path(task_kwargs["booklet_dir"])
    booklet_id = booklet_dir.name
    output_base = Path(task_kwargs["output_base"])
    templates_base = Path(task_kwargs["templates_base"])
    pages_to_process = task_kwargs.get("pages", list(range(1, 8)))
    verbose = task_kwargs.get("verbose", False)
    do_preview = task_kwargs.get("do_preview", False)
    do_crop_makro = task_kwargs.get("do_crop_makro", True)
    do_crop_meso = task_kwargs.get("do_crop_meso", True)
    do_crop_rows = task_kwargs.get("do_crop_rows", True)
    do_crop_fields = task_kwargs.get("do_crop_fields", True)
    do_crop_mikro = task_kwargs.get("do_crop_mikro", True)
    do_crop_digits = task_kwargs.get("do_crop_digits", True)
    do_crop_grid = task_kwargs.get("do_crop_grid", True)
    do_crop_grid_cells = task_kwargs.get("do_crop_grid_cells", False)
    all_rows = task_kwargs.get("all_rows", False)
    clean_mnist = task_kwargs.get("clean_mnist", False)
    use_vlm = task_kwargs.get("use_vlm", False)
    box_pad = task_kwargs.get("box_pad", 0)
    border_margin_pct = task_kwargs.get("border_margin_pct", 0.04)
    pad_digit_px = task_kwargs.get("pad_digit_px", 2)

    start_t = time.time()
    page_summaries = []
    page_errors = []
    total_snapped = 0
    total_makro = 0
    total_meso = 0
    total_rows = 0
    total_fields = 0
    total_grid = 0
    total_mikro = 0
    total_clean = 0
    total_digits = 0
    total_digits_filled = 0

    try:
        if verbose:
            print(f"\n[*] Memproses buklet: {booklet_id}")

        for pno in pages_to_process:
            try:
                rec = process_booklet_page(
                    booklet_dir=booklet_dir,
                    page_num=pno,
                    output_base=output_base,
                    do_preview=do_preview,
                    do_crop_makro=do_crop_makro,
                    do_crop_meso=do_crop_meso,
                    do_crop_rows=do_crop_rows,
                    do_crop_fields=do_crop_fields,
                    do_crop_mikro=do_crop_mikro,
                    do_crop_digits=do_crop_digits,
                    do_crop_grid=do_crop_grid,
                    do_crop_grid_cells=do_crop_grid_cells,
                    all_rows=all_rows,
                    clean_mnist=clean_mnist,
                    use_vlm=use_vlm,
                    template_dir=templates_base,
                    verbose=verbose,
                    box_pad=box_pad,
                    border_margin_pct=border_margin_pct,
                    pad_digit_px=pad_digit_px
                )
                if rec and "summary" in rec:
                    s = rec["summary"]
                    snapped = rec.get("snapped_digits", 0)
                    total_snapped += snapped
                    total_makro += s.get("makro_count", 0)
                    total_meso += s.get("meso_count", 0)
                    total_rows += s.get("rows_count", 0)
                    total_fields += s.get("fields_count", 0)
                    total_grid += s.get("precise_grid_count", 0)
                    total_mikro += s.get("mikro_boxes_count", 0)
                    total_clean += s.get("mikro_clean_count", 0)
                    total_digits += s.get("digits_count", 0)
                    total_digits_filled += s.get("digits_filled_count", 0)
                    page_summaries.append({
                        "page_num": pno,
                        "snapped_digits": snapped,
                        "summary": s
                    })
                else:
                    page_errors.append({"page_num": pno, "error": f"Citra halaman {pno} kosong atau tidak ada"})
            except Exception as page_exc:
                page_errors.append({
                    "page_num": pno,
                    "error": str(page_exc),
                    "traceback": traceback.format_exc()
                })

        # Bebaskan memori secara eksplisit setelah setiap buklet
        import gc
        gc.collect()

        elapsed = time.time() - start_t
        is_success = (len(page_errors) == 0 and len(page_summaries) == len(pages_to_process))
        status = "SUCCESS" if is_success else ("PARTIAL" if len(page_summaries) > 0 else "FAILED")

        return {
            "booklet_id": booklet_id,
            "status": status,
            "pages_processed": len(page_summaries),
            "target_pages": len(pages_to_process),
            "elapsed_seconds": round(elapsed, 2),
            "snapped_digits": total_snapped,
            "counts": {
                "makro": total_makro,
                "meso": total_meso,
                "rows": total_rows,
                "fields": total_fields,
                "precise_grid": total_grid,
                "mikro_boxes": total_mikro,
                "mikro_clean": total_clean,
                "digits": total_digits,
                "digits_filled": total_digits_filled
            },
            "page_summaries": page_summaries,
            "page_errors": page_errors,
            "error": None if not page_errors else f"{len(page_errors)} halaman mengalami kendala"
        }
    except Exception as exc:
        import gc
        gc.collect()
        elapsed = time.time() - start_t
        return {
            "booklet_id": booklet_id,
            "status": "FAILED",
            "pages_processed": len(page_summaries),
            "target_pages": len(pages_to_process),
            "elapsed_seconds": round(elapsed, 2),
            "snapped_digits": total_snapped,
            "counts": {},
            "page_summaries": page_summaries,
            "page_errors": [{"page_num": -1, "error": str(exc), "traceback": traceback.format_exc()}],
            "error": str(exc)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Multi-Tier Grid Alignment & Crop Extractor — Seruti Inti."
    )
    parser.add_argument(
        "--booklet",
        type=str,
        default="booklet_01",
        help="ID buklet yang diproses (default: booklet_01, atau 'all' untuk memproses seluruh buklet)."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Proses seluruh buklet yang ditemukan di direktori input."
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        default=None,
        help="Nomor halaman yang diproses (1 s.d. 7, default: seluruh halaman 1 s.d. 7)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Jumlah worker multiprocessing paralel (default: 3 untuk keamanan RAM)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batas maksimal jumlah buklet yang diproses (berguna untuk dry-run / pengujian)."
    )
    parser.add_argument(
        "--booklets-dir",
        type=str,
        default="output/seruti_inti_booklets",
        help="Direktori berisi buklet hasil assembly (default: output/seruti_inti_booklets)."
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates",
        help="Direktori berisi template_p{X}.json (default: templates)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Direktori root output untuk preview dan crop (default: output)."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Buat visualisasi grid overlay (default: False untuk menghemat disk)."
    )
    parser.add_argument(
        "--all-tiers",
        action="store_true",
        help="Eksekusi sekaligus seluruh tingkatan crop: Makro, Meso, Rows, Fields, Mikro, dan Precise Grid."
    )
    parser.add_argument(
        "--crop-makro",
        action="store_true",
        help="Tingkat 1: Potong seluruh Blok Utama kuesioner (Blok I s.d. VII)."
    )
    parser.add_argument(
        "--crop-meso",
        action="store_true",
        help="Tingkat 2: Potong Sub-Blok modul pertanyaan (R101-R111, R301-R307, R501, R502, dsb)."
    )
    parser.add_argument(
        "--crop-rows",
        action="store_true",
        help="Tingkat 3: Potong seluruh baris roster / data tabel horizontal utuh (Zero-Cut Invariant)."
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Sertakan baris header tabel pada pemotongan baris (default: hanya baris data & total)."
    )
    parser.add_argument(
        "--crop-fields",
        action="store_true",
        help="Potong seluruh area field isian kuesioner utuh (pertahankan crop fields)."
    )
    parser.add_argument(
        "--crop-mikro",
        action="store_true",
        help="Tingkat 6: Potong sel comb-box digit individual & OMR checkbox (raw & clean 28x28)."
    )
    parser.add_argument(
        "--crop-digits",
        action="store_true",
        help="Tingkat 7: Potong seluruh kotak comb-box digit individual berlabel seragam ke direktori 'digits/' (raw, inner, clean 28x28)."
    )
    parser.add_argument(
        "--digits-only",
        action="store_true",
        help="Mode Cepat: Hanya potong kotak digit saja (lewati makro, meso, rows, fields, grid, mikro)."
    )
    parser.add_argument(
        "--crop-grid",
        action="store_true",
        help="Tingkat 5: Potong amplop perimeter kisi fisik tabel utuh segi empat (Precise Grid)."
    )
    parser.add_argument(
        "--crop-grid-cells",
        action="store_true",
        help="Tingkat Kisi Terkecil: Potong seluruh sel fisik individual terkecil (atomic grid cells: header, label, NA, comb-box)."
    )
    parser.add_argument(
        "--no-clean-mnist",
        action="store_true",
        help="Nonaktifkan pembuatan citra biner 28x28 gaya MNIST pada pemotongan Mikro dan Digits."
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Gunakan citra _vlm.jpg (RGB) daripada _clean.png (Monokrom)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Aktifkan pencatatan detail per-halaman ke konsol."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Heningkan seluruh pesan detail, hanya tampilkan progres agregat."
    )
    parser.add_argument(
        "--box-pad",
        type=int,
        default=0,
        help="Margin padding ekstra (px) saat memotong kotak sel digit (default: 0 px untuk mencegah terpotongnya garis sekat kisi)."
    )
    parser.add_argument(
        "--border-margin-pct",
        type=float,
        default=0.04,
        help="Persentase pemotongan margin tepi sel comb-box (default: 0.04 / 4%% agar goresan pensil utuh)."
    )
    parser.add_argument(
        "--pad-digit-px",
        type=int,
        default=2,
        help="Padding margin ekstra di sekeliling bounding box digit sebelum resize (default: 2 px)."
    )

    # Argumen backward compatibility
    parser.add_argument("--crop-sub-blocks", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--crop-boxes", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Evaluasi tingkatan crop yang aktif
    has_specific_crop = (args.crop_makro or args.crop_meso or args.crop_rows or args.crop_fields or
                         args.crop_mikro or args.crop_grid or args.crop_grid_cells or
                         args.crop_digits or args.digits_only or
                         args.crop_sub_blocks or args.crop_boxes)
    if args.digits_only:
        do_crop_makro = False
        do_crop_meso = False
        do_crop_rows = False
        do_crop_fields = False
        do_crop_mikro = False
        do_crop_grid = False
        do_crop_grid_cells = False
        do_crop_digits = True
    elif args.all_tiers or not has_specific_crop:
        do_crop_makro = True
        do_crop_meso = True
        do_crop_rows = True
        do_crop_fields = True
        do_crop_mikro = True
        do_crop_grid = True
        do_crop_grid_cells = args.crop_grid_cells
        do_crop_digits = True
    else:
        do_crop_makro = args.crop_makro
        do_crop_meso = args.crop_meso or args.crop_sub_blocks
        do_crop_rows = args.crop_rows
        do_crop_fields = args.crop_fields
        do_crop_mikro = args.crop_mikro or args.crop_boxes
        do_crop_grid = args.crop_grid or args.crop_grid_cells
        do_crop_grid_cells = args.crop_grid_cells
        do_crop_digits = args.crop_digits

    do_preview = args.preview
    # Format 28x28 dinonaktifkan secara permanen sesuai arahan user (preservasi keutuhan digit raw 100%)
    clean_mnist = False

    booklets_base = Path(args.booklets_dir).resolve()
    templates_base = Path(args.templates_dir).resolve()
    output_base = Path(args.output).resolve()

    pages_to_process = args.pages if args.pages else list(range(1, 8))

    # Identifikasi buklet sasaran
    from seruti_pipeline.stage_gate_validator import DUPLICATE_BOOKLET_IDS, REJECTED_BOOKLET_IDS

    is_batch_all = args.all or (args.booklet.lower() == "all")
    if is_batch_all:
        booklet_dirs = sorted([
            d for d in booklets_base.iterdir()
            if d.is_dir() and d.name.startswith("booklet_")
            and d.name not in DUPLICATE_BOOKLET_IDS
            and d.name not in REJECTED_BOOKLET_IDS
        ])
    else:
        target_dir = booklets_base / args.booklet
        if not target_dir.exists():
            print(f"[ERROR] Direktori buklet tidak ditemukan: {target_dir}")
            sys.exit(1)
        booklet_dirs = [target_dir]

    if args.limit and args.limit > 0:
        booklet_dirs = booklet_dirs[:args.limit]

    total_booklets = len(booklet_dirs)
    if total_booklets == 0:
        print("[PERINGATAN] Tidak ada direktori buklet yang ditemukan untuk diproses.")
        sys.exit(0)

    # Verbose mode
    is_single_manual = (total_booklets == 1 and not is_batch_all and not args.quiet)
    verbose_workers = args.verbose or is_single_manual

    print("=" * 80)
    print("  HIERARCHICAL MULTI-TIER CROP EXTRACTOR — SERUTI INTI")
    print(f"  Target Buklet : {total_booklets} buklet ({'ALL' if is_batch_all else booklet_dirs[0].name})")
    print(f"  Target Halaman: {pages_to_process}")
    print(f"  Worker Pool   : {args.workers} worker(s)")
    print(f"  Template Dir  : {templates_base}")
    print(f"  Output Dir    : {output_base}")
    print("  Tingkatan Crop Aktif (7 Tingkat):")
    print(f"    - Tingkat 1 (MAKRO)        : {'AKTIF' if do_crop_makro else 'NONAKTIF'} (Blok I s.d. VII)")
    print(f"    - Tingkat 2 (MESO)         : {'AKTIF' if do_crop_meso else 'NONAKTIF'} (Sub-Blok Pertanyaan)")
    print(f"    - Tingkat 3 (ROWS)         : {'AKTIF' if do_crop_rows else 'NONAKTIF'} (Baris Roster / Data)")
    print(f"    - Tingkat 4 (FIELDS)       : {'AKTIF' if do_crop_fields else 'NONAKTIF'} (Area Field Utuh)")
    print(f"    - Tingkat 5 (PRECISE GRID) : {'AKTIF' if do_crop_grid else 'NONAKTIF'} (Amplop Snapped ke Garis Fisik)")
    print(f"    - Sel Kisi Terkecil (CELLS): {'AKTIF' if do_crop_grid_cells else 'NONAKTIF'} (Atomic Grid Cells - Disk Guard)")
    print(f"    - Tingkat 6 (MIKRO)        : {'AKTIF' if do_crop_mikro else 'NONAKTIF'} (Comb-Box Raw & Clean 28x28)")
    print(f"    - Tingkat 7 (KOTAK DIGIT)  : {'AKTIF' if do_crop_digits else 'NONAKTIF'} (Digits Raw, Inner/Borderless, MNIST 28x28)")
    print(f"    - Grid Overlay Preview     : {'AKTIF' if do_preview else 'NONAKTIF'}")
    print("=" * 80)

    start_all = time.time()
    results: List[Dict[str, Any]] = []
    completed_count = 0

    task_payloads = []
    for b_dir in booklet_dirs:
        task_kw = {
            "booklet_dir": str(b_dir),
            "output_base": str(output_base),
            "templates_base": str(templates_base),
            "pages": pages_to_process,
            "verbose": verbose_workers,
            "do_preview": do_preview,
            "do_crop_makro": do_crop_makro,
            "do_crop_meso": do_crop_meso,
            "do_crop_rows": do_crop_rows,
            "do_crop_fields": do_crop_fields,
            "do_crop_mikro": do_crop_mikro,
            "do_crop_digits": do_crop_digits,
            "do_crop_grid": do_crop_grid,
            "do_crop_grid_cells": do_crop_grid_cells,
            "all_rows": args.all_rows,
            "clean_mnist": clean_mnist,
            "use_vlm": args.vlm,
            "box_pad": args.box_pad,
            "border_margin_pct": args.border_margin_pct,
            "pad_digit_px": args.pad_digit_px
        }
        task_payloads.append(task_kw)

    if args.workers <= 1 or total_booklets == 1:
        # Eksekusi sekuensial (1 worker)
        for t_idx, t_kw in enumerate(task_payloads, 1):
            res = process_single_booklet_task(t_kw)
            results.append(res)
            completed_count += 1
            elapsed_so_far = time.time() - start_all
            avg_per_b = elapsed_so_far / completed_count
            rem_sec = avg_per_b * (total_booklets - completed_count)
            rem_str = f"{int(rem_sec // 60)}m {int(rem_sec % 60):02d}s" if completed_count < total_booklets else "0s"
            status_tag = f"[{res['status']}]"
            print(f"[{completed_count:02d}/{total_booklets:02d}] {res['booklet_id']}: {status_tag} "
                  f"({res['pages_processed']}/{res['target_pages']} hal, {res['elapsed_seconds']}s) | "
                  f"Snapped: {res.get('snapped_digits', 0)} | "
                  f"Elapsed: {int(elapsed_so_far // 60)}m {int(elapsed_so_far % 60):02d}s | ETA: {rem_str}", flush=True)
    else:
        # Eksekusi multi-processing paralel aman
        max_workers = min(args.workers, total_booklets)
        print(f"[*] Menjalankan pool ProcessPoolExecutor dengan {max_workers} worker paralel...", flush=True)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_bid = {executor.submit(process_single_booklet_task, t): Path(t["booklet_dir"]).name for t in task_payloads}
            for future in as_completed(future_to_bid):
                bid = future_to_bid[future]
                try:
                    res = future.result()
                except Exception as exc:
                    res = {
                        "booklet_id": bid,
                        "status": "FAILED",
                        "pages_processed": 0,
                        "target_pages": len(pages_to_process),
                        "elapsed_seconds": 0.0,
                        "snapped_digits": 0,
                        "counts": {},
                        "page_summaries": [],
                        "page_errors": [{"page_num": -1, "error": str(exc)}],
                        "error": str(exc)
                    }
                results.append(res)
                completed_count += 1
                elapsed_so_far = time.time() - start_all
                avg_per_b = elapsed_so_far / completed_count
                rem_sec = avg_per_b * (total_booklets - completed_count)
                rem_str = f"{int(rem_sec // 60)}m {int(rem_sec % 60):02d}s" if completed_count < total_booklets else "0s"
                status_tag = f"[{res['status']}]"
                print(f"[{completed_count:02d}/{total_booklets:02d}] {res['booklet_id']}: {status_tag} "
                      f"({res['pages_processed']}/{res['target_pages']} hal, {res['elapsed_seconds']}s) | "
                      f"Snapped: {res.get('snapped_digits', 0)} | "
                      f"Elapsed: {int(elapsed_so_far // 60)}m {int(elapsed_so_far % 60):02d}s | ETA: {rem_str}", flush=True)

    # Urutkan hasil berdasarkan nama buklet agar konsisten
    results.sort(key=lambda r: r["booklet_id"])

    # Agregasi metrik
    total_elapsed = time.time() - start_all
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    partial_count = sum(1 for r in results if r["status"] == "PARTIAL")
    total_pages_cropped = sum(r["pages_processed"] for r in results)
    total_snapped_digits = sum(r.get("snapped_digits", 0) for r in results)

    total_makro = sum(r.get("counts", {}).get("makro", 0) for r in results)
    total_meso = sum(r.get("counts", {}).get("meso", 0) for r in results)
    total_rows = sum(r.get("counts", {}).get("rows", 0) for r in results)
    total_fields = sum(r.get("counts", {}).get("fields", 0) for r in results)
    total_grid = sum(r.get("counts", {}).get("precise_grid", 0) for r in results)
    total_mikro = sum(r.get("counts", {}).get("mikro_boxes", 0) for r in results)
    total_clean = sum(r.get("counts", {}).get("mikro_clean", 0) for r in results)
    total_digits = sum(r.get("counts", {}).get("digits", 0) for r in results)
    total_digits_filled = sum(r.get("counts", {}).get("digits_filled", 0) for r in results)

    crops_dir = output_base / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Hitung penggunaan disk output/crops (hanya booklet yang baru diproses jika subset)
    if len(results) < 10:
        crops_size_bytes = sum(
            f.stat().st_size
            for r in results
            for f in (crops_dir / r["booklet_id"]).rglob("*")
            if (crops_dir / r["booklet_id"]).exists() and f.is_file()
        )
    else:
        crops_size_bytes = sum(f.stat().st_size for f in crops_dir.rglob("*") if f.is_file())
    crops_size_mb = round(crops_size_bytes / (1024 * 1024), 2)

    batch_summary = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workers": args.workers,
            "total_booklets": total_booklets,
            "success_count": success_count,
            "failed_count": failed_count,
            "partial_count": partial_count,
            "total_pages_cropped": total_pages_cropped,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "disk_usage_mb": crops_size_mb,
            "tiers_configured": {
                "makro": do_crop_makro,
                "meso": do_crop_meso,
                "rows": do_crop_rows,
                "all_rows": args.all_rows,
                "fields": do_crop_fields,
                "precise_grid": do_crop_grid,
                "precise_grid_cells": do_crop_grid_cells,
                "mikro": do_crop_mikro,
                "digits": do_crop_digits,
                "clean_mnist": clean_mnist,
                "preview": do_preview
            }
        },
        "aggregate_counts": {
            "total_snapped_digits": total_snapped_digits,
            "makro_blocks": total_makro,
            "meso_sub_blocks": total_meso,
            "rows": total_rows,
            "fields": total_fields,
            "precise_grid_tables": total_grid,
            "mikro_boxes": total_mikro,
            "mikro_clean_mnist": total_clean,
            "digits_count": total_digits,
            "digits_filled_count": total_digits_filled
        },
        "booklets": [
            {
                "booklet_id": r["booklet_id"],
                "status": r["status"],
                "pages_processed": r["pages_processed"],
                "target_pages": r["target_pages"],
                "elapsed_seconds": r["elapsed_seconds"],
                "snapped_digits": r.get("snapped_digits", 0),
                "counts": r.get("counts", {}),
                "error": r.get("error")
            }
            for r in results
        ]
    }

    # Simpan batch_crop_summary.json di output/crops/
    batch_summary_path = crops_dir / "batch_crop_summary.json"
    with open(batch_summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)

    # Simpan rekap backward compatibility di output/hierarchical_crop_summary.json
    compat_records = []
    for r in results:
        for ps in r.get("page_summaries", []):
            compat_records.append({
                "booklet_id": r["booklet_id"],
                "page_num": ps["page_num"],
                "snapped_digits": ps["snapped_digits"],
                "summary": ps["summary"]
            })
    compat_path = output_base / "hierarchical_crop_summary.json"
    with open(compat_path, "w", encoding="utf-8") as f:
        json.dump({"total_pages_processed": len(compat_records), "records": compat_records}, f, indent=2)

    print("\n" + "=" * 80)
    print("  RINGKASAN EKSEKUSI BATCH CROP — SERUTI INTI")
    print(f"  Total Buklet Diproses : {total_booklets}")
    print(f"  Status Sukses         : {success_count} / {total_booklets} ({(success_count/total_booklets)*100:.1f}%)")
    print(f"  Status Gagal / Anomali: {failed_count + partial_count}")
    print(f"  Total Halaman Terpotong: {total_pages_cropped} halaman")
    print(f"  Total Digit Snapped   : {total_snapped_digits:,} digit")
    print(f"  Total Kotak Digit     : {total_digits:,} digit ({total_digits_filled:,} terisi)")
    print(f"  Total Waktu Pemrosesan: {int(total_elapsed // 60)}m {total_elapsed % 60:.1f}s "
          f"({total_elapsed/max(1, total_booklets):.1f}s per buklet)")
    print(f"  Ukuran Disk Crops     : {crops_size_mb} MB")
    print(f"  Berkas Ringkasan      : {batch_summary_path.relative_to(BASE_DIR).as_posix()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
