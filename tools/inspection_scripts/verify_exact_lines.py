import pymupdf as fitz

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

for pno, name in [(4, "Halaman 5"), (5, "Halaman 6")]:
    pg = doc[pno]
    print(f"=== {name} ===")
    drawings = pg.get_drawings()
    # find lines and rects
    h_lines = []
    v_lines = []
    for d in drawings:
        r = d['rect']
        if r.width > 700:
            h_lines.append(r)
        if r.height > 300:
            v_lines.append(r)
            
    print(f"Long horizontal lines (width > 700): {len(h_lines)}")
    for r in sorted(h_lines, key=lambda x: x.y0):
        print(f"  y={r.y0:.2f} to {r.y1:.2f}, x0={r.x0:.2f}, x1={r.x1:.2f}, w={r.width:.2f}")
        
    print(f"Long vertical lines (height > 300): {len(v_lines)}")
    for r in sorted(v_lines, key=lambda x: x.x0):
        print(f"  x={r.x0:.2f} to {r.x1:.2f}, y0={r.y0:.2f}, y1={r.y1:.2f}, h={r.height:.2f}")
