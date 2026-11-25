import json
import pymupdf as fitz

for pno in [5, 6]:
    fn = f'templates/template_p{pno}.json'
    d = json.load(open(fn, encoding='utf-8'))
    print(f"=== HALAMAN {pno} ===")
    print("Page size pt:", d.get('size_pt'))
    tables = d.get('tables', [])
    for t in tables:
        print(f"  Table {t['id']}: bbox_pt={t['bbox_pt']}")
        print(f"    x_min={t['bbox_pt'][0]}, x_max={t['bbox_pt'][2]}")
        print(f"    x_lines_pt={t['x_lines_pt'][:4]} ... {t['x_lines_pt'][-3:]}")

    # Check cells min/max x
    cells = d.get('cells', [])
    struct_cells = [c for c in cells if c['kind'] == 'struct']
    min_x = min(c['bbox_pt'][0] for c in struct_cells)
    max_x = max(c['bbox_pt'][2] for c in struct_cells)
    print(f"  Struct cells span: x=[{min_x}, {max_x}]")

# Check vector drawings in the PDF directly
doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")
for pno in [4, 5]:
    pg = doc[pno]
    drawings = pg.get_drawings()
    rects = [d['rect'] for d in drawings if d['rect'].width > 500]
    print(f"\nPDF Page {pno+1} major rects (width > 500 pt):")
    for r in rects[:10]:
        print(f"  rect: x0={r.x0:.2f}, y0={r.y0:.2f}, x1={r.x1:.2f}, y1={r.y1:.2f}, width={r.width:.2f}")
