import sqlite3
import cv2
import os
import numpy as np

conn = sqlite3.connect('db/si_periksa.db')
c = conn.cursor()

c.execute("""
    SELECT id, booklet_id, page, variable, pred_digit, pred_conf, crop_path
    FROM hitl_queue
    WHERE corrected = 0
""")
items = c.fetchall()

print(f"Total remaining pending: {len(items)}")

blanks_detected = []
actual_ambiguous = []

for item in items:
    item_id, bid, page, var, digit, conf, cp = item
    if not cp or not os.path.exists(cp):
        actual_ambiguous.append((item, "missing_crop", 0.0))
        continue
    
    img = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
    if img is None:
        actual_ambiguous.append((item, "read_fail", 0.0))
        continue
    
    h, w = img.shape
    pad_y, pad_x = max(1, int(h * 0.12)), max(1, int(w * 0.12))
    inner = img[pad_y:h-pad_y, pad_x:w-pad_x]
    
    dark_pixels = np.sum(inner < 170)
    total_inner = inner.shape[0] * inner.shape[1]
    ink_density = dark_pixels / total_inner
    
    if ink_density < 0.035:
        blanks_detected.append((item, ink_density))
    else:
        actual_ambiguous.append((item, "has_ink", ink_density))

print(f"Pure blank / noise boxes (ink < 3.5%): {len(blanks_detected)}")
print(f"Actual handwritten ink / true ambiguous: {len(actual_ambiguous)}")

print("\nSample blank / noise boxes:")
for (it, ink) in blanks_detected[:10]:
    print(f"  {it[0]}: conf={it[5]}%, ink={ink*100:.2f}%, var={it[3]}")

print("\nSample actual handwritten ink:")
for (it, reason, ink) in actual_ambiguous[:10]:
    print(f"  {it[0]}: conf={it[5]}%, ink={ink*100:.2f}%, var={it[3]}")

conn.close()
