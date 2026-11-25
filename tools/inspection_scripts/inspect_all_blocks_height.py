import pymupdf as fitz
import json

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

for pno in range(len(doc)):
    pg = doc[pno]
    print(f"\n==================== HALAMAN {pno + 1} ({pg.rect.width} x {pg.rect.height} pt) ====================")
    drawings = pg.get_drawings()
    
    # Extract horizontal lines (width > 0.6 * page width)
    min_w = 0.5 * pg.rect.width
    h_lines = []
    for d in drawings:
        r = d['rect']
        if r.width >= min_w and r.height < 10:
            h_lines.append((r.y0, r.y1, r.x0, r.x1, r.width))
            
    # Remove duplicate/near-identical lines
    unique_h = []
    for hl in sorted(h_lines, key=lambda x: x[0]):
        if not unique_h or abs(hl[0] - unique_h[-1][0]) > 2.0:
            unique_h.append(hl)
            
    print(f"Major horizontal lines ({len(unique_h)}):")
    for hl in unique_h:
        print(f"  y={hl[0]:.2f} (w={hl[4]:.1f}, x=[{hl[2]:.1f}, {hl[3]:.1f}])")
        
    # Text headers containing BLOK
    blocks = pg.get_text("blocks")
    print("Block titles / headers:")
    for b in blocks:
        txt = b[4].strip()
        if "BLOK" in txt or "CATATAN" in txt or "470." in txt or "451" in txt:
            first_line = txt.split('\n')[0][:70]
            print(f"  y=[{b[1]:.2f} - {b[3]:.2f}], x=[{b[0]:.2f} - {b[2]:.2f}]: '{first_line}'")
