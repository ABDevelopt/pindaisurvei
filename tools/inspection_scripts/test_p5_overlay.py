import cv2
import numpy as np
import pymupdf as fitz
import json

doc = fitz.open(r"D:\01_Kedinasan_BPS\SERUTI\3. 20251105 Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).pdf")

DPI = 150
K = DPI / 72.0

# Test Page 5
pg5 = doc[4]
pix5 = pg5.get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
img5 = np.frombuffer(pix5.samples, np.uint8).reshape(pix5.height, pix5.width, 3)[:, :, ::-1].copy()

# Bounding box of full BLOK V on Page 5:
# Top: y = 14.0 pt (above BLOK V title at y=17.87)
# Bottom: y = 580.0 pt (below R503 Table at y=577.75)
# Left: x = 20.0 pt
# Right: x = 818.0 pt
bx0, by0, bx1, by1 = [int(round(v * K)) for v in [20.0, 14.0, 818.0, 580.0]]

print(f"Page 5 full Blok V bbox px: ({bx0}, {by0}) to ({bx1}, {by1}) on img size {img5.shape}")

# Sub-blocks:
# 501: y = 32.0 to 220.0 pt
# 502: y = 222.0 to 428.0 pt
# 503: y = 430.0 to 580.0 pt
for code, sub_bbox in [
    ("501", [22.0, 32.0, 816.0, 219.0]),
    ("502", [22.0, 222.0, 816.0, 427.0]),
    ("503", [22.0, 429.0, 816.0, 580.0])
]:
    sx0, sy0, sx1, sy1 = [int(round(v * K)) for v in sub_bbox]
    print(f"  Sub-rincian {code}: ({sx0}, {sy0}) to ({sx1}, {sy1})")
