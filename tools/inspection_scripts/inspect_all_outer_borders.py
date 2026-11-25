import pymupdf as fitz

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

for pno, page in enumerate(doc):
    print(f"\n==========================================")
    print(f"PAGE {pno+1}")
    print(f"==========================================")
    # Find all thick rects or lines (>150 pt wide, h >= 0.8 pt or border strokes)
    borders = []
    for d in page.get_drawings():
        for item in d.get("items", []):
            if item[0] == "re":
                r = item[1]
                # Check if it spans wide across page
                if r.width > 500:
                    borders.append((round(r.y0, 2), round(r.y1, 2), round(r.x0, 2), round(r.x1, 2), f"rect h={r.height:.2f}"))
            elif item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 0.5 and abs(p1.x - p2.x) > 500:
                    borders.append((round(p1.y, 2), round(p1.y, 2), round(min(p1.x, p2.x), 2), round(max(p1.x, p2.x), 2), "line"))
    # Also find texts
    tb = page.get_text("blocks")
    headers = []
    for b in tb:
        txt = b[4].strip()
        lines = [l.strip() for l in txt.split('\n') if l.strip()]
        for l in lines:
            if any(k in l.upper() for k in ["BLOK", "IV.", "CATATAN", "KETERANGAN PENDATAAN", "KETERANGAN TEMPAT"]):
                headers.append((round(b[1], 2), round(b[3], 2), round(b[0], 2), round(b[2], 2), l[:60]))
                break
                
    print("--- Headers ---")
    for h in sorted(headers, key=lambda x: x[0]):
        print(f"  y: {h[0]:.2f} - {h[1]:.2f} | x: {h[2]:.2f} - {h[3]:.2f} | {h[4]}")
        
    print("--- Outer Borders (>500 pt wide) ---")
    # Sort and unique
    borders = sorted(list(set(borders)), key=lambda x: x[0])
    for b in borders:
        print(f"  y: {b[0]:.2f} - {b[1]:.2f} | x: {b[2]:.2f} - {b[3]:.2f} | {b[4]}")
