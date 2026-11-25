import pymupdf as fitz
import json

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

print("==========================================")
print("1. HALAMAN 5 (TABEL 501, 502, 503)")
print("==========================================")
pg5 = doc[4]
# Find horizontal lines or rects spanning wide across table (x from ~15 to ~820)
h_p5 = []
for d in pg5.get_drawings():
    for item in d.get("items", []):
        if item[0] == "re":
            r = item[1]
            if r.width > 700:
                h_p5.append((round(r.y0, 2), round(r.y1, 2), round(r.x0, 2), round(r.x1, 2), f"rect h={r.height:.2f}"))
        elif item[0] == "l":
            p1, p2 = item[1], item[2]
            if abs(p1.y - p2.y) < 0.5 and abs(p1.x - p2.x) > 700:
                h_p5.append((round(p1.y, 2), round(p1.y, 2), round(min(p1.x, p2.x), 2), round(max(p1.x, p2.x), 2), "line"))
h_p5 = sorted(list(set(h_p5)), key=lambda x: x[0])
for h in h_p5:
    print(f"  y: {h[0]:.2f} - {h[1]:.2f} | x: {h[2]:.2f} - {h[3]:.2f} ({h[4]})")

# Also find text titles for 501, 502, 503
tb5 = pg5.get_text("blocks")
for b in tb5:
    txt = b[4].strip()
    if any(k in txt for k in ["501", "502", "503"]):
        first = txt.split('\n')[0]
        print(f"  Text [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}] : {first[:60]}")

print("\n==========================================")
print("2. HALAMAN 6 (TABEL 504, 505, CATATAN)")
print("==========================================")
pg6 = doc[5]
h_p6 = []
for d in pg6.get_drawings():
    for item in d.get("items", []):
        if item[0] == "re":
            r = item[1]
            if r.width > 700:
                h_p6.append((round(r.y0, 2), round(r.y1, 2), round(r.x0, 2), round(r.x1, 2), f"rect h={r.height:.2f}"))
        elif item[0] == "l":
            p1, p2 = item[1], item[2]
            if abs(p1.y - p2.y) < 0.5 and abs(p1.x - p2.x) > 700:
                h_p6.append((round(p1.y, 2), round(p1.y, 2), round(min(p1.x, p2.x), 2), round(max(p1.x, p2.x), 2), "line"))
h_p6 = sorted(list(set(h_p6)), key=lambda x: x[0])
for h in h_p6:
    print(f"  y: {h[0]:.2f} - {h[1]:.2f} | x: {h[2]:.2f} - {h[3]:.2f} ({h[4]})")

tb6 = pg6.get_text("blocks")
for b in tb6:
    txt = b[4].strip()
    if any(k in txt for k in ["504", "505", "Catatan"]):
        first = txt.split('\n')[0]
        print(f"  Text [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}] : {first[:60]}")
