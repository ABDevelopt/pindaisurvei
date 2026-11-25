import pymupdf as fitz

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")
pg1 = doc[0]

print("=== Page 1 text from y=100 to 255 ===")
for b in sorted(pg1.get_text("blocks"), key=lambda x: x[1]):
    if 100 <= b[1] <= 255:
        print(f"  [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}] : {repr(b[4].strip())}")

print("\n=== Page 1 drawings from y=180 to 255 ===")
for d in pg1.get_drawings():
    for item in d.get("items", []):
        if item[0] == "re":
            r = item[1]
            if 180 <= r.y0 <= 255 or 180 <= r.y1 <= 255:
                print(f"  re: y={r.y0:.2f}-{r.y1:.2f}, x={r.x0:.2f}-{r.x1:.2f}, w={r.width:.2f}, h={r.height:.2f}")
        elif item[0] == "l":
            p1, p2 = item[1], item[2]
            if 180 <= p1.y <= 255:
                print(f"  l: y={p1.y:.2f}, x={min(p1.x, p2.x):.2f}-{max(p1.x, p2.x):.2f}, len={abs(p1.x - p2.x):.2f}")
