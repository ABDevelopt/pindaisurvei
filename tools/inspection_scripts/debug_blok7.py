import cv2
import json

# Check the physical scan of booklet_01 hal_7
img = cv2.imread('output/seruti_inti_booklets/booklet_01/hal_7_clean.png')
print("Image shape:", img.shape)

# Let's crop Blok VII from the scan. In landscape (3507 x 2480):
# Table 1 in template is at y: 425.5 to 567.0 pt.
# At ~3.9 px/pt scale, 425.5 * 3.9 ~ 1660 px to 2200 px.
# Let's crop the bottom half (y from 1600 to 2400) and check or save a preview.
blok7_crop = img[1600:2400, 50:1800]
cv2.imwrite('output/debug_blok7_scan.png', blok7_crop)
print("Saved output/debug_blok7_scan.png")
