from rapidocr_onnxruntime import RapidOCR
import cv2

img = cv2.imread('output/seruti_inti_booklets/booklet_01/pages/hal_7_clean.png')
ocr = RapidOCR()

# Let's inspect the headers at y from 1740 to 1840, x from 200 to 1700
crop_headers = img[1740:1840, 200:1700]
res, _ = ocr(crop_headers)
print("=== Headers ===")
if res:
    for line in res:
        print(f"  {line[1]} at x={line[0][0][0]}")

# Inspect Row 1 (Kunjungan I): y 1836 to 1912
crop_r1 = img[1836:1912, 200:1700]
res, _ = ocr(crop_r1)
print("=== Row 1 (Kunjungan I) ===")
if res:
    for line in res:
        print(f"  {line[1]} at x={line[0][0][0]}")

# Inspect Row 2 (Kunjungan II): y 1912 to 1983
crop_r2 = img[1912:1983, 200:1700]
res, _ = ocr(crop_r2)
print("=== Row 2 (Kunjungan II) ===")
if res:
    for line in res:
        print(f"  {line[1]} at x={line[0][0][0]}")

# Inspect Row 3 (Kunjungan III): y 1983 to 2053
crop_r3 = img[1983:2053, 200:1700]
res, _ = ocr(crop_r3)
print("=== Row 3 (Kunjungan III) ===")
if res:
    for line in res:
        print(f"  {line[1]} at x={line[0][0][0]}")
