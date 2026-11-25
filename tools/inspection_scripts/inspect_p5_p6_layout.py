import pymupdf as fitz
import json

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

for pno in [4, 5]: # Page 5 (idx 4), Page 6 (idx 5)
    pg = doc[pno]
    print(f"=== PAGE {pno + 1} (Landscape {pg.rect.width} x {pg.rect.height}) ===")
    blocks = pg.get_text("blocks")
    for b in blocks:
        txt = b[4].strip()
        print(f"  y=[{b[1]:.2f} - {b[3]:.2f}], x=[{b[0]:.2f} - {b[2]:.2f}]: {txt[:70]}")
