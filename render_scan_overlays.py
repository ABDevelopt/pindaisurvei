#!/usr/bin/env python3
"""
render_scan_overlays.py
=======================
Merender visualisasi overlaying scan asli kuesioner Seruti Inti 2026 (VSERUTI26.INTI)
dengan template overlay yang sudah dibuat dalam 3 tingkatan hierarki:
  1. Tingkat Makro (BLOK Overlay on Scan)
  2. Tingkat Meso (SUB-BLOK Overlay on Scan)
  3. Tingkat Mikro (FULL TEMPLATE Overlay on Scan: Grid, Comb-Box, Field, Sel Arsir)

Mendukung mode citra scan asli:
  - 'clean': Citra pindaian de-skewed monokrom beresolusi tinggi 300 DPI (paling tajam dan kontras)
  - 'raw': Citra pindaian asli 24-bit RGB langsung dari feed scanner fisik
  - 'vlm': Citra pindaian warna terkoreksi kontras (VLM-enhanced)

Penggunaan:
    python render_scan_overlays.py --booklet booklet_01 --image-type raw
    python render_scan_overlays.py --booklet booklet_01 --image-type clean
    python render_scan_overlays.py --booklet booklet_01 --pages 1 2 5 7
"""
import sys
import os
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from seruti_pipeline.grid_template_aligner import align_template_to_image, GridAlignment, DEFAULT_TEMPLATE_DIR
from render_text_scan_overlays import render_scan_text_overlay

ARTIFACT_PREVIEWS = Path(r"C:\Users\ajian\.gemini\antigravity\brain\ffb6c438-eb71-4258-a214-3baaa5a8baad\previews")
ARTIFACT_PREVIEWS.mkdir(parents=True, exist_ok=True)

# Palet warna konsisten BGR
BLOCK_COLORS = {
    "blok_header": (120, 80, 0),           # Dark Cyan / Teal
    "blok_kesediaan": (140, 30, 180),      # Ungu/Magenta
    "blok_1": (180, 50, 0),                 # Biru Tua
    "blok_2": (0, 130, 200),                # Oranye / Kuning Tua
    "blok_3": (160, 40, 0),                 # Biru Gelap
    "blok_4_1": (0, 150, 60),               # Hijau
    "blok_4_2_a_p1": (140, 20, 140),        # Ungu Magenta
    "blok_4_2_a_p2": (140, 20, 140),        # Ungu Magenta
    "blok_4_2_b": (20, 110, 180),           # Biru Langit
    "blok_4_3": (0, 140, 120),              # Teal / Hijau Laut
    "blok_5_p1": (180, 60, 0),              # Biru
    "blok_5_p2": (160, 80, 20),             # Cokelat / Oranye Tua
    "blok_6": (170, 40, 0),                 # Biru Tua
    "blok_7": (0, 130, 180),                # Oranye Tua
    "blok_waktu_selesai": (140, 30, 180),   # Ungu/Magenta
    "blok_catatan_p4": (100, 100, 100),     # Abu-abu
    "blok_catatan_p6": (100, 100, 100),     # Abu-abu
    "default": (100, 50, 150)               # Indigo/Ungu Halaman
}

SUBBLOCK_PALETTE = [
    (180, 50, 0),     # Biru Tua
    (0, 130, 200),    # Oranye
    (0, 150, 60),     # Hijau
    (140, 20, 140),   # Magenta
    (20, 110, 180),   # Biru Langit
    (0, 140, 120),    # Teal
    (100, 50, 150),   # Indigo
    (160, 80, 20),    # Cokelat
    (100, 100, 100),  # Abu-abu
]

def is_neutral_gray(rgb: List[int]) -> bool:
    r, g, b = rgb
    return (max(r, g, b) - min(r, g, b) <= 8) and (min(r, g, b) < 248)

def create_header_banner(
    width: int,
    title_text: str,
    meta_text: str,
    legend_items: List[Tuple[str, Tuple[int, int, int], bool]] = None,
    banner_height: int = 56
) -> np.ndarray:
    """Membuat banner informasi profesional di bagian atas overlay scan."""
    banner = np.full((banner_height, width, 3), 245, dtype=np.uint8)
    cv2.line(banner, (0, banner_height - 1), (width, banner_height - 1), (190, 190, 190), 2)
    
    # Judul Kiri
    cv2.putText(banner, title_text, (18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(banner, meta_text, (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 90, 90), 1, cv2.LINE_AA)
    
    # Legend Kanan
    if legend_items:
        lx = width - (len(legend_items) * 170 + 20)
        for l_txt, l_col, l_fill in legend_items:
            box_y0, box_y1 = 18, 38
            if l_fill:
                cv2.rectangle(banner, (lx, box_y0), (lx + 16, box_y1), l_col, -1)
            cv2.rectangle(banner, (lx, box_y0), (lx + 16, box_y1), l_col, 2)
            cv2.putText(banner, l_txt, (lx + 22, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
            lx += 170
            
    return banner

def render_scan_template_overlay(
    base_img: np.ndarray,
    alignment: GridAlignment,
    booklet_id: str,
    page_num: int,
    image_type: str
) -> np.ndarray:
    """
    Tingkat Mikro: Overlay kisi tabel, kotak digit comb-box, field isian,
    dan sel arsir (NA) langsung di atas citra scan fisik asli.
    """
    tpl = alignment.template_data
    cells = tpl.get("cells", [])
    fields = tpl.get("fields", [])
    tables = tpl.get("tables", [])
    
    # Kotak digit putih valid
    valid_digit_ids = set(c["id"] for c in cells if c["kind"] == "digit" and min(c["fill_rgb"]) >= 245)
    
    vis = base_img.copy() if len(base_img.shape) == 3 else cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    
    # 1. Sel struktur & kotak digit (Shading transparan)
    for c in cells:
        b_px = alignment.get_cell_bbox_px(c["id"])
        if b_px is not None:
            x0, y0, x1, y1 = b_px
        else:
            x0, y0, x1, y1 = alignment.bbox_pt_to_px(c["bbox_pt"])
        rgb = c.get("fill_rgb", [255, 255, 255])
        
        # Sel arsir (NA) tidak digambar shading-nya sesuai arahan (mengurangi visual clutter)
        if c["kind"] == "digit":
            if c["id"] in valid_digit_ids:
                # Kotak digit putih sah: Cyan/Biru transparan
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (235, 150, 30), -1)
                
    # Blend shading sel (alpha 0.35)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # 2. Gambar Tracing Kisi Tabel Fisik & Sel 2D Presisi (Menggantikan Garis Kisi Linear Kaku)
    # Catatan: Garis linear teoritis (cv2.line dari template pt_to_px) dihapus total karena
    # meleset hingga 15px pada distorsi non-linear kertas/roller scanner di bagian tengah halaman.
    table_grids = alignment.get_precise_table_grid_segments(base_img, struct_only=True)
    cell_dict = {c["id"]: c for c in cells}
    for tg in table_grids:
        tx0, ty0, tx1, ty1 = tg["envelope_px"]
        cv2.rectangle(vis, (tx0, ty0), (tx1, ty1), (0, 200, 0), 2)
        for c_info in tg["cells"]:
            cid = c_info.get("id")
            c_orig = cell_dict.get(cid)
            if c_orig:
                rgb = c_orig.get("fill_rgb", [255, 255, 255])
                # Sel arsir (NA) tidak digambar bounding box-nya agar visualisasi bersih
                if c_orig.get("fill") == "na" or is_neutral_gray(rgb):
                    continue
            cx0, cy0, cx1, cy1 = c_info["bbox_px"]
            cv2.rectangle(vis, (cx0, cy0), (cx1, cy1), (0, 180, 0), 1)

    # Kotak digit comb-box (border oranye tegas 2px untuk digit valid, abu-abu untuk non-valid)
    for c in cells:
        if c["kind"] == "digit":
            b_px = alignment.get_cell_bbox_px(c["id"]) or alignment.bbox_pt_to_px(c["bbox_pt"])
            x0, y0, x1, y1 = b_px
            if c["id"] in valid_digit_ids:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (180, 100, 0), 2)
            else:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (120, 120, 120), 1)
    
    # 3. Field isian (Bounding box oranye + label rincian)
    clean_field_count = 0
    for fld in fields:
        boxes = [b for b in fld["boxes"] if b in valid_digit_ids]
        if not boxes:
            continue
        clean_field_count += 1
        
        b_cells = [c for c in cells if c["id"] in boxes]
        b_coords = [alignment.get_cell_bbox_px(c["id"]) or alignment.bbox_pt_to_px(c["bbox_pt"]) for c in b_cells]
        x0 = min(b[0] for b in b_coords)
        y0 = min(b[1] for b in b_coords)
        x1 = max(b[2] for b in b_coords)
        y1 = max(b[3] for b in b_coords)
        
        # Bounding box field: Oranye 2px
        cv2.rectangle(vis, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0, 100, 255), 2)
        
        lbl = str(fld.get("row_label") or "")
        if lbl:
            t_size, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            tx = x0
            ty = max(y0 - 3, t_size[1] + 4)
            cv2.rectangle(vis, (tx - 1, ty - t_size[1] - 2), (tx + t_size[0] + 3, ty + 2), (255, 255, 255), -1)
            cv2.putText(vis, lbl, (tx + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 180), 1, cv2.LINE_AA)
            
    # Header Banner
    H, W = vis.shape[:2]
    title_text = f"SERUTI INTI 2026 — [{booklet_id.upper()}] HALAMAN {page_num} ({image_type.upper()} SCAN) — TEMPLATE GRID OVERLAY"
    meta_text = (f"Scan Resolusi: {W}x{H} px (300 DPI) | Affine: sx={alignment.scale_x:.4f}, sy={alignment.scale_y:.4f}, "
                 f"tx={alignment.trans_x:.1f}, ty={alignment.trans_y:.1f} | Sel: {len(cells)} | Digit: {len(valid_digit_ids)} | Field: {clean_field_count}")
                 
    legend_items = [
        ("Garis Tabel", (0, 180, 0), False),
        ("Kotak Digit", (220, 130, 20), True),
        ("Field Isian", (0, 100, 255), False),
    ]
    banner = create_header_banner(W, title_text, meta_text, legend_items)
    return np.vstack([banner, vis])


def render_scan_grid_overlay(
    base_img: np.ndarray,
    alignment: GridAlignment,
    booklet_id: str,
    page_num: int,
    image_type: str,
    physical_cells: Optional[List[Tuple[int, int, int, int]]] = None
) -> np.ndarray:
    """
    Tingkat Kisi Fisik / Grid Tabel: Overlay tracing seluruh garis kisi tabel fisik kuesioner
    menggunakan mesh hijau bersinar (Physical Lattice Mesh) langsung di atas citra pindaian asli,
    sebagaimana pada seruti_scan_hal4_full_precise_grid.
    """
    from seruti_pipeline.cell_table_extractor import PhysicalLatticeExtractor

    vis = base_img.copy() if len(base_img.shape) == 3 else cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]

    # Ambil sel fisik dari alignment atau ekstraksi langsung jika belum tersedia
    if physical_cells is None:
        if alignment.physical_cells is not None:
            physical_cells = alignment.physical_cells
        else:
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY) if len(vis.shape) == 3 else vis
            physical_cells, _ = PhysicalLatticeExtractor.extract_physical_cells(gray)

    # Ambil amplop segi empat terluar dari alignment untuk menjamin integritas segi empat 100%
    table_envelopes = [env["bbox_px"] for env in alignment.get_snapped_table_envelopes()]

    # Render mesh kisi hijau bersinar (0, 220, 0) dengan ketebalan 2 px dan perimeter terluar 3 px
    vis = PhysicalLatticeExtractor.render_precise_grid_overlay(
        vis, physical_cells, color=(0, 220, 0), thickness=2,
        table_envelopes=table_envelopes, envelope_thickness=3
    )

    # Header Banner
    title_text = f"SERUTI INTI 2026 — [{booklet_id.upper()}] HALAMAN {page_num} ({image_type.upper()} SCAN) — KISI TABEL FISIK (PRECISE GRID)"
    meta_text = (f"Scan Resolusi: {W}x{H} px (300 DPI) | Rangka Kisi Terkunci: {len(physical_cells)} Sel Fisik | "
                 f"Affine: sx={alignment.scale_x:.4f}, sy={alignment.scale_y:.4f}, tx={alignment.trans_x:.1f}, ty={alignment.trans_y:.1f}")

    legend_items = [
        ("Garis Kisi Fisik", (0, 220, 0), False),
        ("Tracing Sel 2D", (0, 180, 0), False),
    ]
    banner = create_header_banner(W, title_text, meta_text, legend_items)
    return np.vstack([banner, vis])


def render_scan_subblock_overlay(
    base_img: np.ndarray,
    alignment: GridAlignment,
    booklet_id: str,
    page_num: int,
    image_type: str
) -> np.ndarray:
    """
    Tingkat Meso: Overlay unit modular Sub-Blok (28 sub-blok) dengan shading warna transparan,
    border tebal 3px, badge nama sub-blok, dan framing blok makro di atas scan asli.
    """
    tpl = alignment.template_data
    blocks = tpl.get("blocks", [])
    sub_blocks = tpl.get("sub_blocks", [])
    
    vis = base_img.copy() if len(base_img.shape) == 3 else cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    
    # 1. Garis framing Makro Blok (Garis abu-abu gelap tebal 2px)
    for b in blocks:
        bx0, by0, bx1, by1 = alignment.bbox_pt_to_px(b["bbox_pt"])
        cv2.rectangle(vis, (bx0, by0), (bx1, by1), (70, 70, 70), 2)
        
    # Precompute all subblock bboxes for collision detection
    sb_boxes = [alignment.bbox_pt_to_px(sb["bbox_pt"]) for sb in sub_blocks]
    
    # 2. Sub-Blok (Shading + Border + Badge)
    for idx, sb in enumerate(sub_blocks):
        sx0, sy0, sx1, sy1 = sb_boxes[idx]
        color = SUBBLOCK_PALETTE[idx % len(SUBBLOCK_PALETTE)]
        
        # Shading transparan (alpha 0.12)
        cv2.rectangle(overlay, (sx0, sy0), (sx1, sy1), color, -1)
        
        # Border tebal 4px
        cv2.rectangle(vis, (sx0, sy0), (sx1, sy1), color, 4)
        
        # Badge Sub-Blok Cerdas (Anti-Tabrakan / Collision-Free)
        sb_code = sb.get("code", "")
        sb_name = sb.get("name", "")
        sb_type = sb.get("type", "")
        c_cnt = sb.get("cells_count", 0)
        f_cnt = sb.get("fields_count", 0)
        d_cnt = sb.get("digits_count", 0)
        
        if sb_type == "page_number":
            badge_text = f" [{sb_code}] {sb_name} "
            t_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bw, bh = t_size[0] + 14, t_size[1] + 10
            badge_y0 = sy1 + 4
            badge_y1 = badge_y0 + bh
            cx = (sx0 + sx1) // 2
            badge_x0 = max(10, cx - bw // 2)
            badge_x1 = min(vis.shape[1] - 10, badge_x0 + bw)
        elif sb_code == "WAKTU_SELESAI":
            badge_text = f" [{sb_code}] {sb_name} "
            t_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bw, bh = t_size[0] + 14, t_size[1] + 10
            badge_y0 = sy1 + 4
            badge_y1 = badge_y0 + bh
            badge_x0 = sx0
            badge_x1 = min(vis.shape[1] - 10, sx0 + bw)
        else:
            badge_text = f" [{sb_code}] {sb_name} | Type: {sb_type} [{c_cnt} Sel | {f_cnt} Field | {d_cnt} Digit] "
            t_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            bw, bh = t_size[0] + 16, t_size[1] + 12
            
            # Hitung clearance di atas sub-blok ini
            clearance_above = sy0
            for j, (os_x0, os_y0, os_x1, os_y1) in enumerate(sb_boxes):
                if idx != j and not (os_x1 <= sx0 or os_x0 >= sx1) and os_y1 <= sy0:
                    clearance_above = min(clearance_above, sy0 - os_y1)
                    
            if clearance_above >= bh + 6 and sy0 >= bh + 8:
                # Cukup ruang di atas sub-blok
                badge_y0 = sy0 - bh - 3
                badge_y1 = sy0 - 3
                badge_x0 = sx0
                badge_x1 = min(vis.shape[1] - 10, sx0 + bw)
            else:
                # Ruang atas sempit/menempel: letakkan di pojok kanan atas bagian dalam sub-blok
                badge_x1 = min(vis.shape[1] - 10, sx1 - 6)
                badge_x0 = max(sx0 + 6, badge_x1 - bw)
                badge_y0 = sy0 + 6
                badge_y1 = badge_y0 + bh
            
        cv2.rectangle(vis, (badge_x0, badge_y0), (badge_x1, badge_y1), color, -1)
        cv2.rectangle(vis, (badge_x0, badge_y0), (badge_x1, badge_y1), (255, 255, 255), 2)
        cv2.putText(vis, badge_text, (badge_x0 + 8, badge_y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
                    
    # Blend shading transparan
    vis = cv2.addWeighted(overlay, 0.12, vis, 0.88, 0)
    
    # Header Banner
    H, W = vis.shape[:2]
    title_text = f"SERUTI INTI 2026 — [{booklet_id.upper()}] HALAMAN {page_num} ({image_type.upper()} SCAN) — SUB-BLOK OVERLAY"
    meta_text = (f"Scan Resolusi: {W}x{H} px (300 DPI) | Total {len(sub_blocks)} Sub-Blok Fungsional | "
                 f"Affine: sx={alignment.scale_x:.4f}, sy={alignment.scale_y:.4f}, tx={alignment.trans_x:.1f}, ty={alignment.trans_y:.1f}")
                 
    banner = create_header_banner(W, title_text, meta_text, None)
    return np.vstack([banner, vis])


def render_scan_block_overlay(
    base_img: np.ndarray,
    alignment: GridAlignment,
    booklet_id: str,
    page_num: int,
    image_type: str
) -> np.ndarray:
    """
    Tingkat Makro: Overlay batas luar Blok utuh (Kesediaan, Blok I s.d. VII, Halaman)
    dengan shading transparan, border tebal 4px, dan badge header makro di atas scan asli.
    """
    tpl = alignment.template_data
    blocks = tpl.get("blocks", [])
    cells = tpl.get("cells", [])
    fields = tpl.get("fields", [])
    valid_digit_ids = set(c["id"] for c in cells if c["kind"] == "digit" and min(c["fill_rgb"]) >= 245)
    
    vis = base_img.copy() if len(base_img.shape) == 3 else cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    
    # Precompute all block bboxes for collision detection
    block_boxes = [alignment.bbox_pt_to_px(b["bbox_pt"]) for b in blocks]
    
    for i, b in enumerate(blocks):
        bid = b["id"]
        bbox_pt = b["bbox_pt"]
        bx0, by0, bx1, by1 = block_boxes[i]
        color = BLOCK_COLORS.get(bid, BLOCK_COLORS["default"])
        
        # Hitung statistik komponen dalam blok
        t_cells = [c for c in cells if bbox_pt[0] - 2 <= ((c["bbox_pt"][0]+c["bbox_pt"][2])/2) <= bbox_pt[2] + 2 and
                                      bbox_pt[1] - 2 <= ((c["bbox_pt"][1]+c["bbox_pt"][3])/2) <= bbox_pt[3] + 2]
        t_digits = [c for c in t_cells if c["id"] in valid_digit_ids]
        t_cell_ids = set(c["id"] for c in t_cells)
        t_fields = [f for f in fields if any(bx in valid_digit_ids for bx in f["boxes"]) and
                    any(bx in t_cell_ids for bx in f["boxes"])]
                    
        # 1. Shading blok tipis (alpha 0.10)
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), color, -1)
        
        # 2. Border tebal Blok (4px)
        cv2.rectangle(vis, (bx0, by0), (bx1, by1), color, 4)
        
        # 3. Header Badge Blok Cerdas (Anti-Tabrakan / Collision-Free)
        badge_code = b.get("code", "")
        badge_title = b.get("title", "")
        
        if badge_code == "HALAMAN":
            badge_text = f" {badge_code} {page_num} "
            t_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            tw, th = t_size[0] + 16, t_size[1] + 10
            badge_y0 = by1 + 4
            badge_y1 = badge_y0 + th
            cx = (bx0 + bx1) // 2
            badge_x0 = max(10, cx - tw // 2)
            badge_x1 = min(vis.shape[1] - 10, badge_x0 + tw)
        elif bid == "blok_waktu_selesai":
            badge_text = f" {badge_code}: {badge_title} "
            t_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            tw, th = t_size[0] + 16, t_size[1] + 10
            badge_y0 = by1 + 4
            badge_y1 = badge_y0 + th
            badge_x0 = bx0
            badge_x1 = min(vis.shape[1] - 10, bx0 + tw)
        else:
            badge_text = f" {badge_code}: {badge_title}  [{len(t_cells)} Sel | {len(t_fields)} Field | {len(t_digits)} Kotak] "
            t_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            tw, th = t_size[0] + 16, t_size[1] + 12
            
            # Hitung clearance di atas blok ini
            clearance_above = by0
            for j, (obx0, oby0, obx1, oby1) in enumerate(block_boxes):
                if i != j and not (obx1 <= bx0 or obx0 >= bx1) and oby1 <= by0:
                    clearance_above = min(clearance_above, by0 - oby1)
                    
            if clearance_above >= th + 6 and by0 >= th + 8:
                # Cukup ruang di atas blok
                badge_y0 = by0 - th - 3
                badge_y1 = by0 - 3
                badge_x0 = bx0
                badge_x1 = min(vis.shape[1] - 10, bx0 + tw)
            else:
                # Ruang atas sempit/menempel blok lain: letakkan di pojok kanan atas bagian dalam blok
                badge_x1 = min(vis.shape[1] - 10, bx1 - 6)
                badge_x0 = max(bx0 + 6, badge_x1 - tw)
                badge_y0 = by0 + 6
                badge_y1 = badge_y0 + th
            
        cv2.rectangle(vis, (badge_x0, badge_y0), (badge_x1, badge_y1), color, -1)
        cv2.rectangle(vis, (badge_x0, badge_y0), (badge_x1, badge_y1), (255, 255, 255), 2)
        cv2.putText(vis, badge_text, (badge_x0 + 8, badge_y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                    
    # Blend shading transparan
    vis = cv2.addWeighted(overlay, 0.10, vis, 0.90, 0)
    
    # Header Banner
    H, W = vis.shape[:2]
    title_text = f"SERUTI INTI 2026 — [{booklet_id.upper()}] HALAMAN {page_num} ({image_type.upper()} SCAN) — MAKRO BLOK OVERLAY"
    meta_text = (f"Scan Resolusi: {W}x{H} px (300 DPI) | Total {len(blocks)} Blok Kuesioner Makro | "
                 f"Affine: sx={alignment.scale_x:.4f}, sy={alignment.scale_y:.4f}, tx={alignment.trans_x:.1f}, ty={alignment.trans_y:.1f}")
                 
    banner = create_header_banner(W, title_text, meta_text, None)
    return np.vstack([banner, vis])


def process_scan_overlays(
    booklet_id: str = "booklet_01",
    pages: List[int] = None,
    image_type: str = "clean",
    booklets_base: Path = Path("output/seruti_inti_booklets"),
    output_base: Path = Path("output/scan_overlays"),
    save_previews: bool = True
) -> Dict[str, Any]:
    """Memproses pembuatan 3 visualisasi overlaying scan asli kuesioner Seruti Inti."""
    if pages is None:
        pages = list(range(1, 8))
        
    booklets_base = Path(booklets_base).resolve()
    output_base = Path(output_base).resolve()
    
    booklet_dir = booklets_base / booklet_id
    if not booklet_dir.exists():
        raise FileNotFoundError(f"Direktori buklet tidak ditemukan: {booklet_dir}")
        
    pages_dir = booklet_dir / "pages"
    out_dir = output_base / booklet_id / image_type
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"  SERUTI INTI 2026 — OVERLAY SCAN ASLI DENGAN TEMPLATE OVERLAY")
    print(f"  Target Buklet : {booklet_id}")
    print(f"  Mode Citra    : {image_type.upper()} (hal_X_{image_type}.png/jpg)")
    print(f"  Target Halaman: {pages}")
    print(f"  Output Dir    : {out_dir}")
    print(f"{'='*80}\n")
    
    summary = {
        "booklet_id": booklet_id,
        "image_type": image_type,
        "pages_processed": [],
    }
    
    for p in pages:
        # Tentukan file citra scan yang digunakan
        if image_type == "raw":
            img_file = pages_dir / f"hal_{p}_raw.jpg"
        elif image_type == "vlm":
            img_file = pages_dir / f"hal_{p}_vlm.jpg"
        else: # clean
            img_file = pages_dir / f"hal_{p}_clean.png"
            
        clean_file = pages_dir / f"hal_{p}_clean.png"
        
        if not img_file.exists():
            print(f"  [PERINGATAN] Berkas citra tidak ditemukan: {img_file}")
            continue
            
        # Muat citra scan
        scan_img = cv2.imread(str(img_file))
        if scan_img is None:
            print(f"  [ERROR] Gagal membaca citra: {img_file}")
            continue
            
        # Penyelarasan affine dilakukan menggunakan clean_file untuk presisi deteksi batas kisi tertinggi
        clean_img = cv2.imread(str(clean_file)) if clean_file.exists() else scan_img
        alignment = align_template_to_image(clean_img, page_num=p)
        snapped_digits = alignment.enable_physical_cell_snapping(clean_img)
        
        H, W = scan_img.shape[:2]
        print(f"[*] Halaman {p}: {W}x{H} px | Affine: sx={alignment.scale_x:.4f}, sy={alignment.scale_y:.4f}, tx={alignment.trans_x:.1f}, ty={alignment.trans_y:.1f} | Kisi Fisik: {len(alignment.physical_cells or [])} Sel | Snapped: {snapped_digits} Digit")
        
        # 1. Render Physical Table Grid Overlay (Kisi Tabel Fisik)
        grid_overlay = render_scan_grid_overlay(scan_img, alignment, booklet_id, p, image_type)
        grid_out_file = out_dir / f"hal_{p}_scan_grid_overlay.jpg"
        cv2.imwrite(str(grid_out_file), grid_overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        
        # 2. Render Template Grid Overlay
        full_overlay = render_scan_template_overlay(scan_img, alignment, booklet_id, p, image_type)
        full_out_file = out_dir / f"hal_{p}_scan_template_overlay.jpg"
        cv2.imwrite(str(full_out_file), full_overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        
        # 3. Render Sub-Block Overlay
        sub_overlay = render_scan_subblock_overlay(scan_img, alignment, booklet_id, p, image_type)
        sub_out_file = out_dir / f"hal_{p}_scan_subblock_overlay.jpg"
        cv2.imwrite(str(sub_out_file), sub_overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        
        # 4. Render Macro Block Overlay
        blk_overlay = render_scan_block_overlay(scan_img, alignment, booklet_id, p, image_type)
        blk_out_file = out_dir / f"hal_{p}_scan_block_overlay.jpg"
        cv2.imwrite(str(blk_out_file), blk_overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 5. Render Freeform Text Overlay (jika halaman memuat text_fields: Hal 1, 2, 7)
        txt_out_file = None
        has_text_fields = bool(alignment.template_data.get("text_fields"))
        if has_text_fields:
            txt_overlay = render_scan_text_overlay(scan_img, alignment, booklet_id, p, image_type=image_type, show_ocr=True, mask_pii=True)
            txt_out_file = out_dir / f"hal_{p}_scan_text_overlay.jpg"
            cv2.imwrite(str(txt_out_file), txt_overlay, [cv2.IMWRITE_JPEG_QUALITY, 93])
        
        # 6. Simpan pratinjau web 150 DPI (~1240 px) untuk artefak chat UI
        if save_previews:
            scale_preview = 1240.0 / float(W)
            prev_w = 1240
            prev_h = int(round((H + 56) * scale_preview))
            
            p_grid = cv2.resize(grid_overlay, (prev_w, prev_h), interpolation=cv2.INTER_AREA)
            p_full = cv2.resize(full_overlay, (prev_w, prev_h), interpolation=cv2.INTER_AREA)
            p_sub = cv2.resize(sub_overlay, (prev_w, prev_h), interpolation=cv2.INTER_AREA)
            p_blk = cv2.resize(blk_overlay, (prev_w, prev_h), interpolation=cv2.INTER_AREA)
            
            prefix = f"{image_type}_" if image_type != "clean" else ""
            p_grid_file = ARTIFACT_PREVIEWS / f"{prefix}scan_hal_{p}_grid_overlay.png"
            p_full_file = ARTIFACT_PREVIEWS / f"{prefix}scan_hal_{p}_template_overlay.png"
            p_sub_file = ARTIFACT_PREVIEWS / f"{prefix}scan_hal_{p}_subblock_overlay.png"
            p_blk_file = ARTIFACT_PREVIEWS / f"{prefix}scan_hal_{p}_block_overlay.png"
            
            cv2.imwrite(str(p_grid_file), p_grid)
            cv2.imwrite(str(p_full_file), p_full)
            cv2.imwrite(str(p_sub_file), p_sub)
            cv2.imwrite(str(p_blk_file), p_blk)

            if has_text_fields and txt_overlay is not None:
                prev_text_h = int(round(txt_overlay.shape[0] * (prev_w / float(W))))
                p_txt = cv2.resize(txt_overlay, (prev_w, prev_text_h), interpolation=cv2.INTER_AREA)
                p_txt_file = ARTIFACT_PREVIEWS / f"{prefix}scan_hal_{p}_text_overlay.png"
                cv2.imwrite(str(p_txt_file), p_txt)
            
        saved_names = [grid_out_file.name, full_out_file.name, sub_out_file.name, blk_out_file.name]
        if txt_out_file:
            saved_names.append(txt_out_file.name)
        print(f"    -> Tersimpan: {', '.join(saved_names)}")
        page_record = {
            "page_num": p,
            "dimensions": [W, H],
            "affine": {
                "sx": round(alignment.scale_x, 4),
                "sy": round(alignment.scale_y, 4),
                "tx": round(alignment.trans_x, 2),
                "ty": round(alignment.trans_y, 2)
            },
            "physical_cells_count": len(alignment.physical_cells or []),
            "snapped_digits_count": snapped_digits,
            "files": {
                "grid_overlay": str(grid_out_file.relative_to(BASE_DIR)),
                "template_overlay": str(full_out_file.relative_to(BASE_DIR)),
                "subblock_overlay": str(sub_out_file.relative_to(BASE_DIR)),
                "block_overlay": str(blk_out_file.relative_to(BASE_DIR))
            }
        }
        if txt_out_file:
            page_record["files"]["text_overlay"] = str(txt_out_file.relative_to(BASE_DIR))
        summary["pages_processed"].append(page_record)
        
    summary_file = out_dir / "scan_overlay_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Rekapitulasi overlay scan asli tersimpan ke: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Merender visualisasi overlaying scan asli kuesioner Seruti Inti 2026."
    )
    parser.add_argument(
        "--booklet",
        type=str,
        default="booklet_01",
        help="ID buklet yang diproses (default: booklet_01)."
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        default=None,
        help="Nomor halaman yang diproses (1 s.d. 7, default: semua halaman)."
    )
    parser.add_argument(
        "--image-type",
        choices=["clean", "raw", "vlm"],
        default="clean",
        help="Jenis citra scan: clean (monokrom tajam), raw (RGB fisik asli), vlm (kontras warna)"
    )
    args = parser.parse_args()
    
    process_scan_overlays(
        booklet_id=args.booklet,
        pages=args.pages,
        image_type=args.image_type
    )

if __name__ == "__main__":
    main()
