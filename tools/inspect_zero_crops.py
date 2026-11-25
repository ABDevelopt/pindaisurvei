import sqlite3
import cv2
import os
import numpy as np

conn = sqlite3.connect('db/si_periksa.db')
c = conn.cursor()

c.execute("""
    SELECT id, booklet_id, page, variable, pred_digit, pred_conf, crop_path
    FROM hitl_queue
    WHERE corrected = 0 AND pred_digit = '0'
    LIMIT 30
""")
rows = c.fetchall()

print(f"Inspecting sample of pending '0' items:")
zero_types = {"pure_blank": 0, "very_low_ink": 0, "clear_zero": 0, "missing_crop": 0}

for r in rows:
    item_id, bid, page, var, digit, conf, cp = r
    if not cp or not os.path.exists(cp):
        zero_types["missing_crop"] += 1
        continue
    
    img = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
    if img is None:
        zero_types["missing_crop"] += 1
        continue
    
    # Calculate ink density (pixels < 170)
    # Strip 10% border to avoid box borders
    h, w = img.shape
    pad_y, pad_x = max(1, int(h * 0.12)), max(1, int(w * 0.12))
    inner = img[pad_y:h-pad_y, pad_x:w-pad_x]
    
    dark_pixels = np.sum(inner < 170)
    total_inner = inner.shape[0] * inner.shape[1]
    ink_density = dark_pixels / total_inner
    
    if ink_density < 0.01:
        zero_types["pure_blank"] += 1
        desc = "PURE BLANK (ink < 1%)"
    elif ink_density < 0.04:
        zero_types["very_low_ink"] += 1
        desc = f"NOISE/VERY LOW INK ({ink_density*100:.1f}%)"
    else:
        zero_types["clear_zero"] += 1
        desc = f"CLEAR ZERO DIGIT ({ink_density*100:.1f}%)"
        
    print(f"  [{item_id}] conf={conf}%, ink={ink_density*100:.1f}% -> {desc}")

print("\nSummary of 30 sample zeros:")
for k, v in zero_types.items():
    print(f"  {k}: {v}")

conn.close()
