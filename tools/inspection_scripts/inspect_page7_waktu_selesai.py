import pymupdf as fitz

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")
pg7 = doc[6]

print("=== Page 7 text from y=550 to 595 ===")
for b in sorted(pg7.get_text("blocks"), key=lambda x: x[1]):
    if 550 <= b[1] <= 595:
        print(f"  [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}] : {repr(b[4].strip())}")

print("\n=== Page 7 drawings from y=560 to 595 ===")
for d in pg7.get_drawings():
    for item in d.get("items", []):
        if item[0] == "re":
            r = item[1]
            if 560 <= r.y0 <= 595 or 560 <= r.y1 <= 595:
                print(f"  re: y={r.y0:.2f}-{r.y1:.2f}, x={r.x0:.2f}-{r.x1:.2f}, w={r.width:.2f}, h={r.height:.2f}")
        elif item[0] == "l":
            p1, p2 = item[1], item[2]
            if 560 <= p1.y <= 595:
                print(f"  l: y={p1.y:.2f}, x={min(p1.x, p2.x):.2f}-{max(p1.x, p2.x):.2f}, len={abs(p1.x - p2.x):.2f}")
