#!/usr/bin/env python3
"""
render_subblock_overlays.py - Merender visualisasi overlay komprehensif tingkat SUB-BLOK
pada kuesioner Seruti Inti 2026 PDF (halaman 1 sampai 7).
"""
import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import pymupdf as fitz

PDF_PATH = r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf"
TEMPLATE_DIR = Path("templates")
OUT_DIR = Path("output/subblock_overlays")
ARTIFACT_PREVIEWS = Path(r"C:\Users\ajian\.gemini\antigravity\brain\ffb6c438-eb71-4258-a214-3baaa5a8baad\previews")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PREVIEWS.mkdir(parents=True, exist_ok=True)

DPI = 150
K = DPI / 72.0

SUBBLOCK_COLORS = [
    (180, 50, 0),     # Biru Tua
    (0, 130, 200),    # Oranye / Kuning Tua
    (0, 150, 60),     # Hijau
    (140, 20, 140),   # Ungu / Magenta
    (20, 110, 180),   # Biru Langit
    (100, 100, 100)   # Abu-abu
]

def render_page_subblocks(pno):
    doc = fitz.open(PDF_PATH)
    page = doc[pno]
    pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy()
    
    tpl_path = TEMPLATE_DIR / f"template_p{pno+1}.json"
    with open(tpl_path, "r", encoding="utf-8") as f:
        tpl = json.load(f)
        
    blocks = tpl.get("blocks", [])
    sub_blocks = tpl.get("sub_blocks", [])
    cells = tpl.get("cells", [])
    fields = tpl.get("fields", [])
    
    valid_digit_ids = set(c["id"] for c in cells if c["kind"] == "digit" and min(c["fill_rgb"]) >= 245)
    
    overlay = img.copy()
    
    # 1. Gambar batas Makro BLOK (Garis tebal tipis abu-abu/biru gelap)
    for b in blocks:
        bx0, by0, bx1, by1 = [int(round(v * K)) for v in b["bbox_pt"]]
        cv2.rectangle(img, (bx0, by0), (bx1, by1), (70, 70, 70), 2)
        
    # 2. Gambar batas dan badge SUB-BLOK
    for idx, sb in enumerate(sub_blocks):
        sx0, sy0, sx1, sy1 = [int(round(v * K)) for v in sb["bbox_pt"]]
        color = SUBBLOCK_COLORS[idx % len(SUBBLOCK_COLORS)]
        
        # Shading transparan sub-blok (alpha = 0.08)
        cv2.rectangle(overlay, (sx0, sy0), (sx1, sy1), color, -1)
        
        # Garis border sub-blok (3px)
        cv2.rectangle(img, (sx0, sy0), (sx1, sy1), color, 3)
        
        # Badge Sub-Blok
        sb_code = sb.get("code", "")
        sb_name = sb.get("name", "")
        sb_type = sb.get("type", "")
        c_cnt = sb.get("cells_count", 0)
        f_cnt = sb.get("fields_count", 0)
        d_cnt = sb.get("digits_count", 0)
        
        badge_text = f" [{sb_code}] {sb_name} | Type: {sb_type} [{c_cnt} Sel | {f_cnt} Field | {d_cnt} Digit] "
        
        t_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        bw, bh = t_size[0] + 10, t_size[1] + 8
        
        badge_y0 = max(0, sy0 - bh - 2)
        badge_y1 = badge_y0 + bh
        if sb_type == "page_number":
            cx = (sx0 + sx1) // 2
            badge_x0 = max(2, cx - bw // 2)
            badge_x1 = min(img.shape[1] - 2, badge_x0 + bw)
        else:
            badge_x0 = sx0
            badge_x1 = min(img.shape[1] - 2, sx0 + bw)
        
        cv2.rectangle(img, (badge_x0, badge_y0), (badge_x1, badge_y1), color, -1)
        cv2.rectangle(img, (badge_x0, badge_y0), (badge_x1, badge_y1), (255, 255, 255), 1)
        cv2.putText(img, badge_text, (badge_x0 + 4, badge_y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
                    
    # Blend shading transparan
    img = cv2.addWeighted(overlay, 0.10, img, 0.90, 0)
    
    # Banner Atas
    H, W = img.shape[:2]
    banner_h = 32
    banner = np.full((banner_h, W, 3), 245, dtype=np.uint8)
    cv2.line(banner, (0, banner_h - 1), (W, banner_h - 1), (200, 200, 200), 1)
    
    title = f"SERUTI INTI 2026 - HIERARKI SUB-BLOK: HALAMAN {pno + 1}"
    cv2.putText(banner, title, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 30, 30), 2, cv2.LINE_AA)
    
    final_img = np.vstack([banner, img])
    
    out_file = OUT_DIR / f"subblock_hal_{pno+1}_overlay.png"
    cv2.imwrite(str(out_file), final_img)
    
    art_file = ARTIFACT_PREVIEWS / f"subblock_hal_{pno+1}_overlay.png"
    shutil.copy2(str(out_file), str(art_file))
    
    print(f"Halaman {pno+1}: {len(sub_blocks)} Sub-Blok dirender -> {out_file.name}")
    return str(art_file)

if __name__ == "__main__":
    print(f"Merender overlay tingkat SUB-BLOK pada {DPI} DPI...")
    for p in range(7):
        render_page_subblocks(p)
    print("Selesai! Seluruh visualisasi tingkat SUB-BLOK berhasil dibuat.")
