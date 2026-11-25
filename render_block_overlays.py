#!/usr/bin/env python3
"""
render_block_overlays.py - Merender visualisasi overlay khusus tingkat BLOK (Block Level Hierarchy)
pada kuesioner Seruti Inti 2026 PDF (halaman 1 sampai 7) dengan presisi geometris garis vektor PDF.
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
OUT_DIR = Path("output/block_overlays")
ARTIFACT_PREVIEWS = Path(r"C:\Users\ajian\.gemini\antigravity\brain\ffb6c438-eb71-4258-a214-3baaa5a8baad\previews")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PREVIEWS.mkdir(parents=True, exist_ok=True)

DPI = 150
K = DPI / 72.0

BLOCK_METADATA = {
    0: [ # Hal 1
        {"id": "blok_kesediaan", "code": "KESEDIAAN", "title": "KESEDIAAN RESPONDEN & WAKTU MULAI", "color": (140, 30, 180), "custom_bbox_pt": [14.16, 185.00, 582.00, 251.66]},
        {"id": "blok_1", "code": "BLOK I", "title": "KETERANGAN TEMPAT", "color": (180, 50, 0), "custom_bbox_pt": [14.16, 251.66, 582.00, 542.47]},
        {"id": "blok_2", "code": "BLOK II", "title": "KETERANGAN PENDATAAN", "color": (0, 130, 200), "custom_bbox_pt": [14.16, 560.23, 582.00, 809.50]},
        {"id": "blok_halaman_p1", "code": "HALAMAN", "title": "NOMOR HALAMAN 1", "color": (100, 50, 150), "custom_bbox_pt": [280.00, 824.00, 316.00, 840.00]},
    ],
    1: [ # Hal 2
        {"id": "blok_3", "code": "BLOK III", "title": "KETERANGAN RUMAH TANGGA", "color": (160, 40, 0), "custom_bbox_pt": [14.16, 14.16, 580.80, 335.93]},
        {"id": "blok_4_1", "code": "BLOK IV.1", "title": "KONSUMSI BAHAN MAKANAN (R405)", "color": (0, 150, 60), "custom_bbox_pt": [14.16, 345.17, 580.80, 406.73]},
        {"id": "blok_4_2_a_p1", "code": "BLOK IV.2.A", "title": "PENGELUARAN BUKAN MAKANAN", "color": (140, 20, 140), "custom_bbox_pt": [14.16, 415.01, 582.60, 826.06]},
        {"id": "blok_halaman_p2", "code": "HALAMAN", "title": "NOMOR HALAMAN 2", "color": (100, 50, 150), "custom_bbox_pt": [280.00, 826.50, 316.00, 840.00]},
    ],
    2: [ # Hal 3
        {"id": "blok_4_2_a_p2", "code": "BLOK IV.2.A (Lanjutan)", "title": "PENGELUARAN BUKAN MAKANAN (SEBULAN & TIGA BULAN)", "color": (140, 20, 140), "custom_bbox_pt": [14.16, 14.16, 582.60, 800.50]},
        {"id": "blok_halaman_p3", "code": "HALAMAN", "title": "NOMOR HALAMAN 3", "color": (100, 50, 150), "custom_bbox_pt": [280.00, 804.00, 316.00, 840.00]},
    ],
    3: [ # Hal 4
        {"id": "blok_4_2_b", "code": "BLOK IV.2.B", "title": "PENGELUARAN KESEHATAN DAN PENDIDIKAN", "color": (20, 110, 180), "custom_bbox_pt": [14.16, 22.56, 582.60, 318.89]},
        {"id": "blok_4_3", "code": "BLOK IV.3", "title": "PERSEPSI KONDISI EKONOMI RT (JUDUL S.D. PERTANYAAN 470)", "color": (0, 140, 120), "custom_bbox_pt": [14.16, 328.01, 582.00, 721.18]},
        {"id": "blok_catatan_p4", "code": "CATATAN", "title": "KODE KETERANGAN BLOK III", "color": (100, 100, 100), "custom_bbox_pt": [14.16, 723.58, 582.00, 821.38]},
        {"id": "blok_halaman_p4", "code": "HALAMAN", "title": "NOMOR HALAMAN 4", "color": (100, 50, 150), "custom_bbox_pt": [280.00, 824.00, 316.00, 840.00]},
    ],
    4: [ # Hal 5
        {"id": "blok_5_p1", "code": "BLOK V", "title": "PENDAPATAN, PENERIMAAN, & PENGELUARAN SELAIN KONSUMSI (TABEL 501, 502, 503)", "color": (180, 60, 0), "custom_bbox_pt": [13.44, 14.16, 822.94, 580.44]},
        {"id": "blok_halaman_p5", "code": "HALAMAN", "title": "NOMOR HALAMAN 5", "color": (100, 50, 150), "custom_bbox_pt": [403.00, 580.80, 439.00, 592.50]},
    ],
    5: [ # Hal 6
        {"id": "blok_5_p2", "code": "BLOK V (Lanjutan)", "title": "PENDAPATAN, PENERIMAAN, & PENGELUARAN SELAIN KONSUMSI (TABEL 504, 505)", "color": (160, 80, 20), "custom_bbox_pt": [13.44, 14.16, 822.94, 470.86]},
        {"id": "blok_catatan_p6", "code": "CATATAN", "title": "KLASIFIKASI LAPANGAN USAHA & PEKERJAAN", "color": (100, 100, 100), "custom_bbox_pt": [14.16, 472.66, 823.18, 579.36]},
        {"id": "blok_halaman_p6", "code": "HALAMAN", "title": "NOMOR HALAMAN 6", "color": (100, 50, 150), "custom_bbox_pt": [403.00, 580.80, 439.00, 592.50]},
    ],
    6: [ # Hal 7
        {"id": "blok_6", "code": "BLOK VI", "title": "POSISI ASET & KEWAJIBAN KEUANGAN", "color": (170, 40, 0), "custom_bbox_pt": [14.16, 30.96, 822.94, 410.11]},
        {"id": "blok_7", "code": "BLOK VII", "title": "CATATAN / KUNJUNGAN PENCACAHAN", "color": (0, 130, 180), "custom_bbox_pt": [14.16, 424.39, 822.94, 568.32]},
        {"id": "blok_waktu_selesai", "code": "WAKTU_SELESAI", "title": "WAKTU SELESAI WAWANCARA", "color": (140, 30, 180), "custom_bbox_pt": [14.16, 568.32, 200.00, 588.00]},
        {"id": "blok_halaman_p7", "code": "HALAMAN", "title": "NOMOR HALAMAN 7", "color": (100, 50, 150), "custom_bbox_pt": [403.00, 577.00, 439.00, 592.50]},
    ]
}

def render_block_overlay(pno):
    doc = fitz.open(PDF_PATH)
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
    block_defs = BLOCK_METADATA.get(pno, [])
    
    for binfo in block_defs:
        bbox_pt = binfo["custom_bbox_pt"]
        bx0, by0, bx1, by1 = [int(round(v * K)) for v in bbox_pt]
        b_color = binfo["color"] # BGR
        
        t_cells = [c for c in cells if bbox_pt[0] - 2 <= ((c["bbox_pt"][0]+c["bbox_pt"][2])/2) <= bbox_pt[2] + 2 and
                                      bbox_pt[1] - 2 <= ((c["bbox_pt"][1]+c["bbox_pt"][3])/2) <= bbox_pt[3] + 2]
        t_digits = [c for c in t_cells if c["id"] in valid_digit_ids]
        t_cell_ids = set(c["id"] for c in t_cells)
        t_fields = [f for f in fields if any(b in valid_digit_ids for b in f["boxes"]) and
                    any(b in t_cell_ids for b in f["boxes"])]
        
        # 1. Shading blok tipis
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), b_color, -1)
        
        # 2. Border tebal Blok (3px) tepat pada garis luar
        cv2.rectangle(img, (bx0, by0), (bx1, by1), b_color, 3)
        
        # 3. Header Badge Blok
        badge_code = binfo["code"]
        badge_sub = binfo["title"]
        badge_text = f" {badge_code}: {badge_sub}  [{len(t_cells)} Sel | {len(t_fields)} Field | {len(t_digits)} Kotak] "
        
        t_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        tw, th = t_size[0] + 10, t_size[1] + 8
        
        badge_y0 = max(2, by0 - th - 2)
        badge_y1 = badge_y0 + th
        if badge_code == "HALAMAN":
            cx = (bx0 + bx1) // 2
            badge_x0 = max(2, cx - tw // 2)
            badge_x1 = min(img.shape[1] - 2, badge_x0 + tw)
        else:
            badge_x0 = bx0
            badge_x1 = min(img.shape[1] - 2, bx0 + tw)
        
        cv2.rectangle(img, (badge_x0, badge_y0), (badge_x1, badge_y1), b_color, -1)
        cv2.rectangle(img, (badge_x0, badge_y0), (badge_x1, badge_y1), (255, 255, 255), 1)
        cv2.putText(img, badge_text, (badge_x0 + 5, badge_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                    
    # Blend shading tipis (alpha 0.08)
    alpha = 0.08
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    out_file = OUT_DIR / f"block_hal_{pno+1}_overlay.png"
    cv2.imwrite(str(out_file), img)
    
    # Salin ke artefak untuk pratinjau instan di panel
    art_file = ARTIFACT_PREVIEWS / out_file.name
    shutil.copy2(out_file, art_file)
    print(f"Halaman {pno+1}: Saved -> {out_file} (dan disalin ke artifact previews)")

def main():
    print(f"--- Merender Overlay Tingkat Blok (7 Halaman) dengan Presisi Garis Vektor PDF ---")
    for p in range(7):
        render_block_overlay(p)
    print("Selesai! Seluruh 7 halaman telah dirender.")

if __name__ == "__main__":
    main()
