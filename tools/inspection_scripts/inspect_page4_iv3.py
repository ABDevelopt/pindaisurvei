import pymupdf as fitz
import json

pdf_path = r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf"
doc = fitz.open(pdf_path)
pg = doc[3] # page 4 (0-indexed 3)

print("=== Text blocks on Page 4 ===")
for b in pg.get_text("blocks"):
    txt = b[4].strip()
    if any(k in txt for k in ["IV.3", "470", "464", "465", "466", "467", "468", "469", "PERSEPSI"]):
        print(f"Text block: bbox={b[:4]}, text={txt[:80]}...")

# Also inspect template_p4.json
with open('templates/template_p4.json', encoding='utf-8') as f:
    tpl4 = json.load(f)

print("\n=== Tables in template_p4.json ===")
for t in tpl4.get('tables', []):
    print(f"Table {t['id']}: bbox_pt={t['bbox_pt']}, rows={t['n_rows']}, cols={t['n_cols']}, cells={len(t['cells'])}")

# Check cells around Blok IV.3
cells = tpl4.get('cells', [])
iv3_cells = [c for c in cells if any(k in c['text'] for k in ["IV.3", "470", "464", "465", "466", "467", "468", "469"])]
print("\n=== Sample cells mentioning IV.3 or questions 464-470 ===")
for c in iv3_cells:
    print(f"Cell {c['id']}: kind={c['kind']}, bbox_pt={c['bbox_pt']}, text='{c['text'][:60]}'")
