import pymupdf as fitz

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")
pg7 = doc[6]

print("=== All text blocks in Page 7 (y from 30 to 450) ===")
tb7 = pg7.get_text("blocks")
for b in sorted(tb7, key=lambda x: x[1]):
    if 25 <= b[1] <= 450:
        first_line = b[4].strip().replace('\n', ' ')
        print(f"  [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}] : {first_line[:80]}")

print("\n=== All horizontal lines/rects in Page 7 (y from 350 to 430) ===")
h_lines = []
for d in pg7.get_drawings():
    for item in d.get("items", []):
        if item[0] == "re":
            r = item[1]
            if 350 <= r.y0 <= 430:
                h_lines.append((round(r.y0, 2), round(r.y1, 2), round(r.x0, 2), round(r.x1, 2), f"re w={r.width:.2f} h={r.height:.2f}"))
        elif item[0] == "l":
            p1, p2 = item[1], item[2]
            if 350 <= p1.y <= 430:
                h_lines.append((round(p1.y, 2), round(p1.y, 2), round(min(p1.x, p2.x), 2), round(max(p1.x, p2.x), 2), f"l len={abs(p1.x - p2.x):.2f}"))
h_lines = sorted(list(set(h_lines)), key=lambda x: x[0])
for h in h_lines:
    print(f"  y: {h[0]:.2f} - {h[1]:.2f} | x: {h[2]:.2f} - {h[3]:.2f} ({h[4]})")
