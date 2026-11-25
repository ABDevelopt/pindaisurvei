import pymupdf as fitz

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

for pno, page in enumerate(doc):
    print(f"=== PAGE {pno+1} ({page.rect.width:.2f} x {page.rect.height:.2f}) ===")
    tb = page.get_text("blocks")
    for b in tb:
        txt = b[4].strip()
        # Look for standalone digits or page number text
        if txt == str(pno+1) or f"Halaman {pno+1}" in txt or txt.isdigit() or "hal" in txt.lower():
            print(f"  Exact match: [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}] : {repr(txt)}")
        elif len(txt) <= 5 and any(c.isdigit() for c in txt):
            print(f"  Short digit: [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}] : {repr(txt)}")
            
    # Also search words at the bottom (y > height - 60) or top (y < 60)
    words = page.get_text("words")
    for w in words:
        if (w[1] > page.rect.height - 40 or w[3] < 30) and w[4].isdigit():
            print(f"  Edge word: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}] : {repr(w[4])}")
