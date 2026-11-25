import cv2
import numpy as np
from pathlib import Path

# Let's inspect block_hal_1 to block_hal_7
OVERLAYS_DIR = Path("output/block_overlays")

for p in range(1, 8):
    img = cv2.imread(str(OVERLAYS_DIR / f"block_hal_{p}_overlay.png"))
    print(f"Hal {p}: Shape {img.shape}")
