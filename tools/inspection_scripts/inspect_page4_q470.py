import json
import pymupdf as fitz

with open('templates/template_p4.json', encoding='utf-8') as f:
    tpl4 = json.load(f)

cells = tpl4.get('cells', [])
fields = tpl4.get('fields', [])

print(f"Total cells: {len(cells)}")
# Find all cells between y=590 and y=725
mid_cells = [c for c in cells if c['bbox_pt'][1] >= 580 and c['bbox_pt'][3] <= 730]
print(f"Cells between y=580 and y=730: {len(mid_cells)}")
for c in mid_cells:
    print(f"  Cell {c['id']}: kind={c['kind']}, bbox={c['bbox_pt']}, text='{c['text'][:40]}'")

# Check text words in PDF between y=580 and y=730
doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")
pg = doc[3]
words = [w for w in pg.get_text("words") if w[1] >= 580 and w[3] <= 730]
print(f"\nWords in PDF between y=580 and y=730: {len(words)}")
# Group words by approx line y
lines = {}
for w in words:
    ly = round(w[1] / 10) * 10
    lines.setdefault(ly, []).append(w)
for ly in sorted(lines.keys()):
    txt = " ".join(w[4] for w in sorted(lines[ly], key=lambda x: x[0]))
    print(f"  y~{ly}: {txt}")
