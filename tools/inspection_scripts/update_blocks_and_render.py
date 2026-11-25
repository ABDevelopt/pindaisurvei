#!/usr/bin/env python3
"""
update_blocks_and_render.py
Memperbarui metadata blok di templates/template_p1.json s.d. template_p7.json
dan render_block_overlays.py dengan koordinat presisi garis vektor PDF,
lalu merender ulang overlay ke direktori output dan artefak.
"""
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import pymupdf as fitz

PDF_PATH = r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf"
TEMPLATE_DIR = Path("templates")
OUT_DIR = Path("output/block_overlays")
ARTIFACT_PREVIEWS = Path(r"C:\Users\ajian\.gemini\antigravity\brain\5c3ae983-46fc-4457-9270-8e70f75a6308\previews")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PREVIEWS.mkdir(parents=True, exist_ok=True)

# Exact vector block definitions for all 7 pages
EXACT_BLOCKS = {
    0: [ # Hal 1
        {"id": "blok_1", "code": "BLOK I", "title": "KETERANGAN TEMPAT", "color": (180, 50, 0), "bbox_pt": [14.16, 251.66, 582.00, 542.47]},
        {"id": "blok_2", "code": "BLOK II", "title": "KETERANGAN PENDATAAN", "color": (0, 130, 200), "bbox_pt": [14.16, 560.23, 582.00, 809.50]},
    ],
    1: [ # Hal 2
        {"id": "blok_3", "code": "BLOK III", "title": "KETERANGAN RUMAH TANGGA", "color": (160, 40, 0), "bbox_pt": [14.16, 14.16, 580.80, 335.93]},
        {"id": "blok_4_1", "code": "BLOK IV.1", "title": "KONSUMSI BAHAN MAKANAN (R405)", "color": (0, 150, 60), "bbox_pt": [14.16, 345.17, 580.80, 406.73]},
        {"id": "blok_4_2_a_p1", "code": "BLOK IV.2.A", "title": "PENGELUARAN BUKAN MAKANAN", "color": (140, 20, 140), "bbox_pt": [14.16, 415.01, 582.60, 826.06]},
    ],
    2: [ # Hal 3
        {"id": "blok_4_2_a_p2", "code": "BLOK IV.2.A (Lanjutan)", "title": "PENGELUARAN BUKAN MAKANAN (SEBULAN & TIGA BULAN)", "color": (140, 20, 140), "bbox_pt": [14.16, 14.16, 582.60, 785.02]},
    ],
    3: [ # Hal 4
        {"id": "blok_4_2_b", "code": "BLOK IV.2.B", "title": "PENGELUARAN KESEHATAN DAN PENDIDIKAN", "color": (20, 110, 180), "bbox_pt": [14.16, 22.56, 582.60, 318.89]},
        {"id": "blok_4_3", "code": "BLOK IV.3", "title": "PERSEPSI KONDISI EKONOMI RT (JUDUL S.D. PERTANYAAN 470)", "color": (0, 140, 120), "bbox_pt": [14.16, 328.01, 582.00, 721.18]},
        {"id": "blok_catatan_p4", "code": "CATATAN", "title": "KODE KETERANGAN BLOK III", "color": (100, 100, 100), "bbox_pt": [14.16, 723.58, 582.00, 821.38]},
    ],
    4: [ # Hal 5
        {"id": "blok_5_p1", "code": "BLOK V", "title": "PENDAPATAN, PENERIMAAN, & PENGELUARAN SELAIN KONSUMSI (TABEL 501, 502, 503)", "color": (180, 60, 0), "bbox_pt": [13.44, 14.16, 822.94, 580.44]},
    ],
    5: [ # Hal 6
        {"id": "blok_5_p2", "code": "BLOK V (Lanjutan)", "title": "PENDAPATAN, PENERIMAAN, & PENGELUARAN SELAIN KONSUMSI (TABEL 504, 505)", "color": (160, 80, 20), "bbox_pt": [13.44, 14.16, 822.94, 470.86]},
        {"id": "blok_catatan_p6", "code": "CATATAN", "title": "KLASIFIKASI LAPANGAN USAHA & PEKERJAAN", "color": (100, 100, 100), "bbox_pt": [14.16, 472.66, 823.18, 579.36]},
    ],
    6: [ # Hal 7
        {"id": "blok_6", "code": "BLOK VI", "title": "POSISI ASET & KEWAJIBAN KEUANGAN", "color": (170, 40, 0), "bbox_pt": [14.16, 30.96, 822.94, 418.51]},
        {"id": "blok_7", "code": "BLOK VII", "title": "CATATAN / KUNJUNGAN PENCACAHAN", "color": (0, 130, 180), "bbox_pt": [14.16, 424.39, 822.94, 568.32]},
    ]
}

# 1. Update templates/template_p1.json s.d. template_p7.json
print("=== 1. Memperbarui Template JSON ===")
for pno in range(7):
    pfile = TEMPLATE_DIR / f"template_p{pno+1}.json"
    with open(pfile, "r", encoding="utf-8") as f:
        tpl = json.load(f)
        
    cur_blocks = tpl.get("blocks", [])
    # Preserve sub_blocks link in block if present
    sub_block_map = {b.get("id"): b.get("sub_blocks", []) for b in cur_blocks}
    
    new_blocks = []
    for b_def in EXACT_BLOCKS[pno]:
        bid = b_def["id"]
        new_b = {
            "id": bid,
            "code": b_def["code"],
            "title": b_def["title"],
            "bbox_pt": b_def["bbox_pt"],
            "sub_blocks": sub_block_map.get(bid, [])
        }
        new_blocks.append(new_b)
        
    tpl["blocks"] = new_blocks
    with open(pfile, "w", encoding="utf-8") as f:
        json.dump(tpl, f, indent=2, ensure_ascii=False)
    print(f"  P{pno+1}: {len(new_blocks)} blok tersimpan presisi.")

# 2. Render overlays
print("\n=== 2. Merender Overlay Blok Presisi ===")
DPI = 150
K = DPI / 72.0

doc = fitz.open(PDF_PATH)

for pno in range(7):
    page = doc[pno]
    pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy()
    
    tpl_path = TEMPLATE_DIR / f"template_p{pno+1}.json"
    with open(tpl_path, "r", encoding="utf-8") as f:
        tpl = json.load(f)
        
    cells = tpl.get("cells", [])
    fields = tpl.get("fields", [])
    valid_digit_ids = set(c["id"] for c in cells if c["kind"] == "digit" and min(c["fill_rgb"]) >= 245)
    
    overlay = img.copy()
    block_defs = EXACT_BLOCKS[pno]
    
    for binfo in block_defs:
        bbox_pt = binfo["bbox_pt"]
        bx0, by0, bx1, by1 = [int(round(v * K)) for v in bbox_pt]
        b_color = binfo["color"]
        
        # Spatial filtering: center of cell inside bbox
        t_cells = [c for c in cells if bbox_pt[0] - 2 <= ((c["bbox_pt"][0]+c["bbox_pt"][2])/2) <= bbox_pt[2] + 2 and
                                      bbox_pt[1] - 2 <= ((c["bbox_pt"][1]+c["bbox_pt"][3])/2) <= bbox_pt[3] + 2]
        t_digits = [c for c in t_cells if c["id"] in valid_digit_ids]
        t_cell_ids = set(c["id"] for c in t_cells)
        t_fields = [f for f in fields if any(b in valid_digit_ids for b in f["boxes"]) and
                    any(b in t_cell_ids for b in f["boxes"])]
                    
        # 1. Shading tipis
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), b_color, -1)
        
        # 2. Border tebal Blok (3px) tepat pada garis luar
        cv2.rectangle(img, (bx0, by0), (bx1, by1), b_color, 3)
        
        # 3. Badge
        badge_code = binfo["code"]
        badge_sub = binfo["title"]
        badge_text = f" {badge_code}: {badge_sub}  [{len(t_cells)} Sel | {len(t_fields)} Field | {len(t_digits)} Kotak] "
        
        t_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        tw, th = t_size[0] + 10, t_size[1] + 8
        
        # Badge di kiri atas blok
        badge_y0 = max(2, by0 - th - 2)
        badge_y1 = badge_y0 + th
        badge_x0 = bx0
        badge_x1 = min(img.shape[1] - 2, bx0 + tw)
        
        cv2.rectangle(img, (badge_x0, badge_y0), (badge_x1, badge_y1), b_color, -1)
        cv2.rectangle(img, (badge_x0, badge_y0), (badge_x1, badge_y1), (255, 255, 255), 1)
        cv2.putText(img, badge_text, (badge_x0 + 5, badge_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                    
    # Blend shading
    alpha = 0.08
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    out_file = OUT_DIR / f"block_hal_{pno+1}_overlay.png"
    cv2.imwrite(str(out_file), img)
    shutil.copy2(out_file, ARTIFACT_PREVIEWS / out_file.name)
    print(f"  Saved: {out_file} & artifact preview")

print("\nDone! Seluruh overlay blok telah dirender ulang.")
