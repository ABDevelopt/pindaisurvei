import cv2
import numpy as np

img = cv2.imread('output/seruti_inti_booklets/booklet_01/pages/hal_7_clean.png')
H, W = img.shape[:2]

# Blok VII in scan:
# Let's crop from y=1650 to y=2350, x=50 to x=3450
blok7 = img[1650:2350, 50:3450]
gray = cv2.cvtColor(blok7, cv2.COLOR_BGR2GRAY)
bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

# Find horizontal and vertical lines in Blok VII
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

h_sums = np.sum(h_lines, axis=1)
v_sums = np.sum(v_lines, axis=0)

# Detect horizontal line y positions
hy = []
for y in range(len(h_sums)):
    if h_sums[y] > 0.2 * blok7.shape[1] * 255:
        if not hy or y - hy[-1] > 15:
            hy.append(y)

# Detect vertical line x positions
vx = []
for x in range(len(v_sums)):
    if v_sums[x] > 0.2 * blok7.shape[0] * 255:
        if not vx or x - vx[-1] > 15:
            vx.append(x)

print("Blok VII Scan relative line positions:")
print("Horizontal lines (y relative to 1650):", hy)
print("  Absolute y:", [y + 1650 for y in hy])
print("Vertical lines (x relative to 50):", vx)
print("  Absolute x:", [x + 50 for x in vx])

# Draw lines on Blok VII preview
annotated = blok7.copy()
for y in hy:
    cv2.line(annotated, (0, y), (blok7.shape[1], y), (0, 0, 255), 2)
for x in vx:
    cv2.line(annotated, (x, 0), (x, blok7.shape[0]), (255, 0, 0), 2)

cv2.imwrite('output/debug_blok7_grid_detected.png', annotated)
print("Saved output/debug_blok7_grid_detected.png")
