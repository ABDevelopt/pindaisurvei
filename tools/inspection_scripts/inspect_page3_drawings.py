import pymupdf as fitz

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")
pg3 = doc[2] # 0-indexed 2

print("=== Drawings on Page 3 ===")
dw = pg3.get_drawings()
print(f"Total drawings: {len(dw)}")
rects = [d['rect'] for d in dw if d['rect'].width > 300]
print(f"Rects with width > 300: {len(rects)}")
for r in sorted(rects, key=lambda x: x.y0):
    print(f"  y0={r.y0:.2f}, y1={r.y1:.2f}, x0={r.x0:.2f}, x1={r.x1:.2f}, w={r.width:.2f}")

# Also check cells in template_p3.json
import json
tpl3 = json.load(open('templates/template_p3.json', encoding='utf-8'))
print("\nTemplate P3 tables:", [t['bbox_pt'] for t in tpl3['tables']])
print("Template P3 min/max cell y:", min(c['bbox_pt'][1] for c in tpl3['cells']), max(c['bbox_pt'][3] for c in tpl3['cells']))
