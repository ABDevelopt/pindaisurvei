import json
import pymupdf as fitz

# Check Page 5
doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

print("=== CHECKING P5 ===")
pg5 = doc[4]
# Check cells in template_p5
tpl5 = json.load(open('templates/template_p5.json', encoding='utf-8'))
print("P5 tables:", [t['bbox_pt'] for t in tpl5['tables']])

# Check title block V text
w5 = pg5.get_text("blocks")
for b in w5:
    if "BLOK V" in b[4] or "501" in b[4] or "502" in b[4] or "503" in b[4]:
        print(f"P5: y={b[1]:.2f}-{b[3]:.2f}, x={b[0]:.2f}-{b[2]:.2f}: {b[4][:50]}")

print("\n=== CHECKING P6 ===")
pg6 = doc[5]
tpl6 = json.load(open('templates/template_p6.json', encoding='utf-8'))
print("P6 tables:", [t['bbox_pt'] for t in tpl6['tables']])

w6 = pg6.get_text("blocks")
for b in w6:
    if "BLOK V" in b[4] or "504" in b[4] or "505" in b[4] or "Kategori" in b[4]:
        print(f"P6: y={b[1]:.2f}-{b[3]:.2f}, x={b[0]:.2f}-{b[2]:.2f}: {b[4][:50]}")
