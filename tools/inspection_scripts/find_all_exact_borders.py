import pymupdf as fitz

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

def find_borders_around(pno, y_search_ranges):
    page = doc[pno]
    dw = page.get_drawings()
    print(f"\n==========================================")
    print(f"PAGE {pno+1}")
    print(f"==========================================")
    
    # Collect all lines and rects
    h_elements = []
    v_elements = []
    for d in dw:
        for item in d.get("items", []):
            if item[0] == "re":
                r = item[1]
                if r.width > 200:
                    h_elements.append((round(r.y0, 2), round(r.y1, 2), round(r.x0, 2), round(r.x1, 2), f"rect h={r.height:.2f}"))
                if r.height > 50:
                    v_elements.append((round(r.x0, 2), round(r.x1, 2), round(r.y0, 2), round(r.y1, 2), f"v_rect w={r.width:.2f}"))
            elif item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 0.5 and abs(p1.x - p2.x) > 200:
                    h_elements.append((round(p1.y, 2), round(p1.y, 2), round(min(p1.x, p2.x), 2), round(max(p1.x, p2.x), 2), "line"))
                if abs(p1.x - p2.x) < 0.5 and abs(p1.y - p2.y) > 50:
                    v_elements.append((round(p1.x, 2), round(p1.x, 2), round(min(p1.y, p2.y), 2), round(max(p1.y, p2.y), 2), "v_line"))
                    
    # Also check cell bounding boxes in templates
    for label, (y_min, y_max) in y_search_ranges.items():
        print(f"\n--- Checking {label} (search y: {y_min} to {y_max}) ---")
        matches = [e for e in h_elements if (y_min - 20 <= e[0] <= y_min + 20) or (y_max - 20 <= e[1] <= y_max + 20)]
        matches = sorted(list(set(matches)), key=lambda x: x[0])
        print("  Candidate Horizontal Borders:")
        for m in matches:
            print(f"    y: {m[0]:.2f} - {m[1]:.2f} | x: {m[2]:.2f} - {m[3]:.2f} ({m[4]})")
            
        # Leftmost and rightmost vertical elements
        v_in_range = [v for v in v_elements if (v[2] <= y_min + 30 and v[3] >= y_max - 30)]
        v_left = sorted([v for v in v_in_range if v[0] < 50], key=lambda x: x[0])
        v_right = sorted([v for v in v_in_range if v[1] > 550], key=lambda x: x[1], reverse=True)
        print("  Outer vertical lines in range:")
        if v_left:
            print(f"    Leftmost: x={v_left[0][0]:.2f} (y: {v_left[0][2]:.2f} to {v_left[0][3]:.2f})")
        if v_right:
            print(f"    Rightmost: x={v_right[0][1]:.2f} (y: {v_right[0][2]:.2f} to {v_right[0][3]:.2f})")

# Hal 1
find_borders_around(0, {
    "BLOK I": (251.66, 541.25),
    "BLOK II": (560.23, 808.06)
})

# Hal 2
find_borders_around(1, {
    "BLOK III": (14.16, 334.75),
    "BLOK IV.1": (345.17, 405.29),
    "BLOK IV.2.A": (415.01, 825.75)
})

# Hal 3
find_borders_around(2, {
    "BLOK IV.2.A (Lanjutan)": (14.16, 784.75)
})

# Hal 4
find_borders_around(3, {
    "BLOK IV.2.B": (22.56, 318.50),
    "BLOK IV.3": (328.01, 719.74),
    "CATATAN": (723.58, 820.00)
})

# Hal 5
find_borders_around(4, {
    "BLOK V": (14.16, 580.44)
})

# Hal 6
find_borders_around(5, {
    "BLOK V (Lanjutan)": (14.16, 470.86),
    "CATATAN": (472.66, 577.92)
})

# Hal 7
find_borders_around(6, {
    "BLOK VI": (30.96, 417.07),
    "BLOK VII": (424.39, 566.88)
})
