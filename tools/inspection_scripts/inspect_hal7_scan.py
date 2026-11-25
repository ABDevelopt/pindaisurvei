import cv2
import numpy as np

# Load booklet_01 hal_7_clean.png
img = cv2.imread('output/seruti_inti_booklets/booklet_01/pages/hal_7_clean.png')
H, W = img.shape[:2]
print(f"Hal 7 scan shape: {W}x{H}")

# In Hal 7 (landscape, W=3507, H=2480), Table 0 (Blok VI) occupies the top part.
# Let's locate the horizontal divider line separating Blok VI and Blok VII.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

# Horizontal projection
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
row_sums = np.sum(h_lines, axis=1)

# Find major horizontal lines in the bottom half (y > 1500)
lines_y = []
for y in range(1500, H - 100):
    if row_sums[y] > 0.4 * W * 255:
        if not lines_y or y - lines_y[-1] > 10:
            lines_y.append(y)

print(f"Major horizontal lines in bottom half (y > 1500): {lines_y}")

# Let's save a crop of Blok VII region with all lines visible
if lines_y:
    y_start = max(1400, lines_y[0] - 80)
    blok7_crop = img[y_start:H-50, :]
    cv2.imwrite('output/debug_blok7_full.png', blok7_crop)
    print(f"Saved output/debug_blok7_full.png from y={y_start} to {H-50}")
