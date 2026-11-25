import cv2
import numpy as np
import pymupdf as fitz

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

DPI = 150
K = DPI / 72.0

for pno, name, bbox in [
    (4, "Halaman 5", [14.88, 14.16, 821.50, 580.44]),
    (5, "Halaman 6", [14.88, 14.16, 821.50, 470.86]),
]:
    pg = doc[pno]
    pix = pg.get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy()
    
    bx0, by0, bx1, by1 = [int(round(v * K)) for v in bbox]
    print(f"{name}: bbox_pt={bbox} -> px=({bx0}, {by0}, {bx1}, {by1}) on img {img.shape}")
    
    # Check if the border line of the PDF exactly coincides with (bx0, by0, bx1, by1)
    # Check pixels around bx0
    col_slice = img[by0+50:by0+150, bx0-3:bx0+4]
    print(f"  Darkest pixel around x0 ({bx0}): {np.min(col_slice)}")
    col_slice_r = img[by0+50:by0+150, bx1-3:bx1+4]
    print(f"  Darkest pixel around x1 ({bx1}): {np.min(col_slice_r)}")
