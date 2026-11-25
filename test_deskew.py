import cv2
import numpy as np

def estimate_deskew_angle(gray: np.ndarray, max_angle: float = 3.0) -> float:
    H, W = gray.shape[:2]
    scale = 1200.0 / max(H, W)
    small = cv2.resize(gray, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h_s, w_s = small.shape[:2]

    bw = cv2.adaptiveThreshold(~small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    min_len = int(w_s * 0.12)
    lines = cv2.HoughLinesP(bw, 1, np.pi / 720, threshold=80, minLineLength=min_len, maxLineGap=15)
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    weights = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        if ang > 45:
            ang -= 90
        elif ang < -45:
            ang += 90

        if abs(ang) <= max_angle:
            angles.append(ang)
            weights.append(length)

    if not angles:
        return 0.0

    weights = np.array(weights)
    angles = np.array(angles)
    
    med = float(np.median(angles))
    valid_mask = np.abs(angles - med) <= 0.4
    if np.sum(valid_mask) > 0:
        deskew_deg = float(np.average(angles[valid_mask], weights=weights[valid_mask]))
    else:
        deskew_deg = med

    return deskew_deg

if __name__ == '__main__':
    for b in ['booklet_01', 'booklet_05', 'booklet_41', 'booklet_45', 'booklet_56', 'booklet_57']:
        for p in range(1, 8):
            p_path = f'output/seruti_inti_booklets/{b}/pages/hal_{p}_clean.png'
            im = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            ang = estimate_deskew_angle(im)
            if abs(ang) > 0.1:
                print(f'{b} Hal {p}: deskew angle = {ang:+.3f} deg')
