import pymupdf as fitz
import json

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

# Define search areas for each block on each page
BLOCK_SEARCH_TARGETS = [
    # Page 1 (0)
    (0, "BLOK I", "KETERANGAN TEMPAT", 240, 550),
    (0, "BLOK II", "KETERANGAN PENDATAAN", 550, 820),
    # Page 2 (1)
    (1, "BLOK III", "KETERANGAN RUMAH TANGGA", 10, 340),
    (1, "BLOK IV.1", "KONSUMSI BAHAN MAKANAN (R405)", 340, 410),
    (1, "BLOK IV.2.A", "PENGELUARAN BUKAN MAKANAN", 410, 830),
    # Page 3 (2)
    (2, "BLOK IV.2.A (Lanjutan)", "PENGELUARAN BUKAN MAKANAN (Lanjutan)", 10, 790),
    # Page 4 (3)
    (3, "BLOK IV.2.B", "PENGELUARAN KESEHATAN DAN PENDIDIKAN", 20, 322),
    (3, "BLOK IV.3", "PERSEPSI KONDISI EKONOMI RT", 325, 723),
    (3, "CATATAN", "KODE KETERANGAN BLOK III", 722, 825),
    # Page 5 (4)
    (4, "BLOK V", "PENDAPATAN, PENERIMAAN, & PENGELUARAN SELAIN KONSUMSI", 10, 585),
    # Page 6 (5)
    (5, "BLOK V (Lanjutan)", "PENDAPATAN, PENERIMAAN, & PENGELUARAN (Lanjutan)", 10, 472),
    (5, "CATATAN", "KLASIFIKASI LAPANGAN USAHA & PEKERJAAN", 471, 582),
    # Page 7 (6)
    (6, "BLOK VI", "POSISI ASET & KEWAJIBAN KEUANGAN", 25, 420),
    (6, "BLOK VII", "CATATAN / KUNJUNGAN PENCACAHAN", 420, 572),
]

results = []

for pno, code, title, y_start, y_end in BLOCK_SEARCH_TARGETS:
    page = doc[pno]
    dw = page.get_drawings()
    
    # Collect all drawing rectangles and lines intersecting this y-range
    rects = []
    for d in dw:
        for item in d.get("items", []):
            if item[0] == "re":
                r = item[1]
                if y_start - 2 <= r.y0 and r.y1 <= y_end + 2:
                    rects.append(r)
            elif item[0] == "l":
                p1, p2 = item[1], item[2]
                min_y, max_y = min(p1.y, p2.y), max(p1.y, p2.y)
                if y_start - 2 <= min_y and max_y <= y_end + 2:
                    rects.append(fitz.Rect(min(p1.x, p2.x), min_y, max(p1.x, p2.x), max_y))
                    
    # The outer envelope of all these border drawings
    # Filter for outer border elements:
    # 1. Top border: horizontal lines near y_start
    top_candidates = [r.y0 for r in rects if abs(r.y0 - y_start) < 15 and r.width > 200]
    if not top_candidates:
        top_candidates = [r.y0 for r in rects if abs(r.y0 - y_start) < 20]
    min_y0 = min(top_candidates) if top_candidates else min(r.y0 for r in rects)
    
    # 2. Bottom border: horizontal lines near y_end
    bot_candidates = [r.y1 for r in rects if abs(r.y1 - y_end) < 15 and r.width > 200]
    if not bot_candidates:
        bot_candidates = [r.y1 for r in rects if abs(r.y1 - y_end) < 20]
    max_y1 = max(bot_candidates) if bot_candidates else max(r.y1 for r in rects)
    
    # 3. Left border: elements near y_start..y_end
    left_candidates = [r.x0 for r in rects if min_y0 - 2 <= r.y0 and r.y1 <= max_y1 + 2 and r.x0 < 30]
    min_x0 = min(left_candidates) if left_candidates else min(r.x0 for r in rects)
    
    # 4. Right border: elements near y_start..y_end
    right_candidates = [r.x1 for r in rects if min_y0 - 2 <= r.y0 and r.y1 <= max_y1 + 2 and r.x1 > 500]
    max_x1 = max(right_candidates) if right_candidates else max(r.x1 for r in rects)
    
    results.append({
        "pno": pno,
        "page": pno + 1,
        "code": code,
        "title": title,
        "exact_bbox_pt": [round(min_x0, 2), round(min_y0, 2), round(max_x1, 2), round(max_y1, 2)],
        "height_pt": round(max_y1 - min_y0, 2),
        "width_pt": round(max_x1 - min_x0, 2)
    })

print(json.dumps(results, indent=2))
