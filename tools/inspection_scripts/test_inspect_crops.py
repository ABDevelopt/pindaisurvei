import cv2

img5 = cv2.imread("output/subblock_overlays/subblock_hal_5_overlay.png")
print("Hal 5 shape:", img5.shape)

# Let's crop around the boundary of 501 and 502 (around y = 461 + 32 banner = 493 px)
crop_501_502 = img5[450:540, 20:500]
cv2.imwrite("output/subblock_overlays/test_crop_501_502.png", crop_501_502)

# Also check 502 and 503 (around y = 428.92 * 150/72 + 32 = 893 + 32 = 925 px)
crop_502_503 = img5[880:970, 20:500]
cv2.imwrite("output/subblock_overlays/test_crop_502_503.png", crop_502_503)

print("Saved test crops!")
