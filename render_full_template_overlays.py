#!/usr/bin/env python3
"""
render_full_template_overlays.py - Merender overlay grid template ke file template PDF Seruti Inti full semua halaman (1-7).
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
OUT_DIR = Path("output/template_pdf_overlays")
ARTIFACT_PREVIEWS = Path(r"C:\Users\ajian\.gemini\antigravity\brain\ffb6c438-eb71-4258-a214-3baaa5a8baad\previews")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PREVIEWS.mkdir(parents=True, exist_ok=True)

DPI = 150
K = DPI / 72.0

def is_neutral_gray(rgb):
    r, g, b = rgb
    return (max(r, g, b) - min(r, g, b) <= 8) and (min(r, g, b) < 248)

def render_page_overlay(pno):
    doc = fitz.open(PDF_PATH)
    page = doc[pno]
    pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
    base_img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy()
    
    tpl_path = TEMPLATE_DIR / f"template_p{pno+1}.json"
    with open(tpl_path, "r", encoding="utf-8") as f:
        tpl = json.load(f)
        
    cells = tpl.get("cells", [])
    fields = tpl.get("fields", [])
    
    # In Hal 5, filter out false gray boxes (min(fill_rgb) < 245)
    valid_digit_ids = set()
    for c in cells:
        if c["kind"] == "digit":
            if min(c["fill_rgb"]) >= 245:
                valid_digit_ids.add(c["id"])
        
    overlay = base_img.copy()
    
    # 1. Gambar sel struktur dan arsir
    for c in cells:
        x0, y0, x1, y1 = [int(round(v * K)) for v in c["bbox_pt"]]
        rgb = c.get("fill_rgb", [255, 255, 255])
        
        if c["kind"] == "struct":
            # Sel arsir (NA) dilewati, tidak digambar bounding box atau shading
            if c.get("fill") == "na" or is_neutral_gray(rgb):
                continue
            # Garis struktur tabel: Hijau lembut
            cv2.rectangle(base_img, (x0, y0), (x1, y1), (0, 160, 0), 1)
        elif c["kind"] == "digit":
            if c["id"] in valid_digit_ids:
                # Kotak digit putih sah: Biru/Cyan transparan
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (235, 150, 30), -1)
                cv2.rectangle(base_img, (x0, y0), (x1, y1), (180, 100, 0), 1)
            else:
                # Kotak palsu abu-abu (Halaman 5): Abu-abu / struct
                cv2.rectangle(base_img, (x0, y0), (x1, y1), (120, 120, 120), 1)
                
    # Blend overlay arsir & digit (alpha = 0.35)
    img = cv2.addWeighted(overlay, 0.35, base_img, 0.65, 0)
    
    # 2. Gambar field isian (bounding box kelompok kotak)
    clean_field_count = 0
    for fld in fields:
        boxes = [b for b in fld["boxes"] if b in valid_digit_ids]
        if not boxes:
            continue
        clean_field_count += 1
        
        # Hitung bbox dari boxes yang valid
        b_cells = [c for c in cells if c["id"] in boxes]
        x0 = int(round(min(c["bbox_pt"][0] for c in b_cells) * K))
        y0 = int(round(min(c["bbox_pt"][1] for c in b_cells) * K))
        x1 = int(round(max(c["bbox_pt"][2] for c in b_cells) * K))
        y1 = int(round(max(c["bbox_pt"][3] for c in b_cells) * K))
        
        # Bounding box field: Oranye tebal 1px
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (0, 100, 255), 1)
        
        # Label baris
        lbl = fld.get("row_label", "")
        if lbl:
            txt = str(lbl)
            t_size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            tx = x0
            ty = max(y0 - 2, t_size[1] + 2)
            # Background putih kecil di bawah teks label agar terbaca jelas
            cv2.rectangle(img, (tx - 1, ty - t_size[1] - 1), (tx + t_size[0] + 1, ty + 1), (255, 255, 255), -1)
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 180), 1, cv2.LINE_AA)
            
    # 3. Header Banner Informasi
    H, W = img.shape[:2]
    banner_h = 32
    banner = np.full((banner_h, W, 3), 245, dtype=np.uint8)
    cv2.line(banner, (0, banner_h - 1), (W, banner_h - 1), (200, 200, 200), 1)
    
    title = f"SERUTI INTI 2026 - TEMPLATE OVERLAY HALAMAN {pno + 1}"
    cv2.putText(banner, title, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 30, 30), 2, cv2.LINE_AA)
    
    # Legend
    legend_items = [
        ("Garis Tabel", (0, 140, 0), False),
        ("Kotak Digit", (220, 130, 20), True),
        ("Field Isian", (0, 100, 255), False),
    ]
    
    lx = W - 520
    for l_txt, l_col, l_fill in legend_items:
        if l_fill:
            cv2.rectangle(banner, (lx, 10), (lx + 12, 22), l_col, -1)
        cv2.rectangle(banner, (lx, 10), (lx + 12, 22), l_col, 1)
        cv2.putText(banner, l_txt, (lx + 16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1, cv2.LINE_AA)
        lx += 120
        
    final_img = np.vstack([banner, img])
    
    out_file = OUT_DIR / f"template_hal_{pno+1}_overlay.png"
    cv2.imwrite(str(out_file), final_img)
    
    # Update juga template_p{X}_preview.png di direktori templates
    tpl_preview = TEMPLATE_DIR / f"template_p{pno+1}_preview.png"
    shutil.copy2(str(out_file), str(tpl_preview))
    
    # Copy ke artifact previews jika ada
    if ARTIFACT_PREVIEWS:
        ARTIFACT_PREVIEWS.mkdir(parents=True, exist_ok=True)
        art_file = ARTIFACT_PREVIEWS / f"template_hal_{pno+1}_overlay.png"
        shutil.copy2(str(out_file), str(art_file))
    
    print(f"Halaman {pno+1}: Ukuran {W}x{H+banner_h} px | Sel: {len(cells)} | Digit: {len(valid_digit_ids)} | Field: {clean_field_count}")
    return str(art_file)

if __name__ == "__main__":
    print(f"Merender template PDF overlay untuk seluruh 7 halaman pada {DPI} DPI...")
    for p in range(7):
        render_page_overlay(p)
    print("Selesai! Seluruh 7 file overlay berhasil dibuat.")
