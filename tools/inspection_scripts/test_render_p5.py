import cv2
import numpy as np
import pymupdf as fitz
import json

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

DPI = 150
K = DPI / 72.0

# Test rendering Page 5 with revised Blok V
pg5 = doc[4]
pix5 = pg5.get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
img5 = np.frombuffer(pix5.samples, np.uint8).reshape(pix5.height, pix5.width, 3)[:, :, ::-1].copy()

tpl5 = json.load(open('templates/template_p5.json', encoding='utf-8'))
cells = tpl5['cells']
fields = tpl5['fields']
valid_digit_ids = set(c["id"] for c in cells if c["kind"] == "digit" and min(c["fill_rgb"]) >= 245)

overlay = img5.copy()

blocks_p5 = [
    {"code": "BLOK V: R501", "title": "UPAH/GAJI (JUDUL BLOK V S.D. TABEL 501)", "color": (180, 60, 0), "bbox": [20.0, 14.0, 818.0, 220.0], "y_range": (14.0, 222.0)},
    {"code": "BLOK V: R502", "title": "PENDAPATAN USAHA (PERTANYAAN 502 S.D. TABEL 502)", "color": (0, 120, 200), "bbox": [20.0, 222.0, 818.0, 428.0], "y_range": (222.0, 430.0)},
    {"code": "BLOK V: R503", "title": "PENDAPATAN KEPEMILIKAN (PERTANYAAN 503 S.D. TABEL 503)", "color": (130, 30, 130), "bbox": [20.0, 430.0, 818.0, 580.0], "y_range": (430.0, 582.0)},
]

for b in blocks_p5:
    bx0, by0, bx1, by1 = [int(round(v * K)) for v in b["bbox"]]
    col = b["color"]
    y0, y1 = b["y_range"]
    
    t_cells = [c for c in cells if y0 <= c["bbox_pt"][1] and c["bbox_pt"][3] <= y1]
    t_digits = [c for c in t_cells if c["id"] in valid_digit_ids]
    t_cell_ids = set(c["id"] for c in t_cells)
    t_fields = [f for f in fields if any(bx in valid_digit_ids for bx in f["boxes"]) and any(bx in t_cell_ids for bx in f["boxes"])]
    
    cv2.rectangle(overlay, (bx0, by0), (bx1, by1), col, -1)
    cv2.rectangle(img5, (bx0, by0), (bx1, by1), col, 3)
    
    badge = f" {b['code']}: {b['title']} [{len(t_cells)} Sel | {len(t_fields)} Field | {len(t_digits)} Kotak] "
    t_sz, _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    bw, bh = t_sz[0] + 10, t_sz[1] + 8
    badge_y0 = max(0, by0 - bh - 2)
    badge_y1 = badge_y0 + bh
    
    cv2.rectangle(img5, (bx0, badge_y0), (bx0 + bw, badge_y1), col, -1)
    cv2.rectangle(img5, (bx0, badge_y0), (bx0 + bw, badge_y1), (255, 255, 255), 1)
    cv2.putText(img5, badge, (bx0 + 4, badge_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

res5 = cv2.addWeighted(overlay, 0.12, img5, 0.88, 0)
cv2.imwrite('output/test_block_hal_5.png', res5)
print("Saved output/test_block_hal_5.png successfully!")
