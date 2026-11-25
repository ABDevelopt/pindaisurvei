import pymupdf as fitz
import re

pdf_path = r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf"
doc = fitz.open(pdf_path)

for pno in [4, 5]: # 0-indexed: page 5 is 4, page 6 is 5
    pg = doc[pno]
    print(f"=== PDF Page {pno+1} ===")
    words = pg.get_text("words")
    print(f"Total words: {len(words)}")
    # Find words that look like row numbers or rincian
    codes = [w for w in words if re.match(r"^\d{1,3}\.?$", w[4])]
    print(f"Sample number tokens ({len(codes)}): {[w[4] for w in codes[:20]]}")
    # Also find headings like 501, 502, etc.
    blocks = pg.get_text("blocks")
    for b in blocks:
        txt = b[4].strip()
        if any(k in txt for k in ["501", "502", "503", "504", "505", "BLOK V"]):
            print(f"  Heading block: {txt[:60]} at {b[:4]}")
