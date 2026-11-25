import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional, Dict, Any, List

def fixed_detect_scanned_table_frame(
    gray: np.ndarray,
    t_w_pt: Optional[float] = None,
    t_h_pt: Optional[float] = None
) -> Tuple[int, int, int, int]:
    H, W = gray.shape[:2]

    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_used = max(160, min(225, int(otsu_val)))
    _, bw = cv2.threshold(gray, thresh_used, 255, cv2.THRESH_BINARY_INV)

    # Gunakan kernel yang lebih toleran terhadap kemiringan/gutter fold
    kh_len = max(20, int(W * 0.04))
    kv_len = max(20, min(45, int(H * 0.015)))  # 40-45 px bukan 105 px!
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (kh_len, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv_len))

    lines_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
    lines_v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv)

    row_sums = np.sum(lines_h > 0, axis=1)
    col_sums = np.sum(lines_v > 0, axis=0)

    min_x_guard = 90
    max_x_guard = W - 80
    min_y_guard = 80
    max_y_guard = H - 70

    col_sums[:min_x_guard] = 0
    col_sums[max_x_guard:] = 0
    row_sums[:min_y_guard] = 0
    row_sums[max_y_guard:] = 0

    max_col = np.max(col_sums[min_x_guard:max_x_guard]) if np.max(col_sums[min_x_guard:max_x_guard]) > 0 else 1
    max_row = np.max(row_sums[min_y_guard:max_y_guard]) if np.max(row_sums[min_y_guard:max_y_guard]) > 0 else 1

    y_thresh = max(int(0.05 * W), int(0.12 * max_row))
    x_thresh = max(int(0.03 * H), int(0.12 * max_col))

    y_peaks = np.where(row_sums >= y_thresh)[0]
    x_peaks = np.where(col_sums >= x_thresh)[0]

    exp_h = int(round(3.8889 * t_h_pt)) if (t_h_pt and t_h_pt > 0) else int(H * 0.88)
    exp_w = int(round(3.8889 * t_w_pt)) if (t_w_pt and t_w_pt > 0) else int(W * 0.88)

    # Deteksi X
    if len(x_peaks) == 0:
        first_x = max(90, int((W - exp_w) / 2))
        last_x = min(W - 80, first_x + exp_w)
    else:
        fx, lx = int(x_peaks[0]), int(x_peaks[-1])
        min_exp_w = int(t_w_pt * 3.0) if (t_w_pt and t_w_pt > 0) else int(W * 0.55)
        if (lx - fx) < min_exp_w:
            # Salah satu tepi tidak terdeteksi. Periksa apakah tepi kanan atau kiri yang terdeteksi
            if lx > (W - int(W * 0.25)): # Tepi kanan kuat!
                last_x = lx
                first_x = max(90, last_x - exp_w)
            elif fx < int(W * 0.25): # Tepi kiri kuat!
                first_x = fx
                last_x = min(W - 80, first_x + exp_w)
            else:
                first_x = max(90, int((W - exp_w) / 2))
                last_x = min(W - 80, first_x + exp_w)
        else:
            first_x, last_x = fx, lx

    # Deteksi Y
    if len(y_peaks) == 0:
        first_y = max(80, int((H - exp_h) / 2))
        last_y = min(H - 70, first_y + exp_h)
    else:
        fy, ly = int(y_peaks[0]), int(y_peaks[-1])
        min_exp_h = int(t_h_pt * 3.0) if (t_h_pt and t_h_pt > 0) else int(H * 0.55)
        if (ly - fy) < min_exp_h:
            if ly > (H - int(H * 0.20)):
                last_y = ly
                first_y = max(80, last_y - exp_h)
            elif fy < int(H * 0.20):
                first_y = fy
                last_y = min(H - 70, first_y + exp_h)
            else:
                first_y = max(80, int((H - exp_h) / 2))
                last_y = min(H - 70, first_y + exp_h)
        else:
            first_y, last_y = fy, ly

    # Aspect ratio check
    if t_w_pt and t_h_pt and t_w_pt > 0 and t_h_pt > 0:
        h_px = last_y - first_y
        sy = h_px / float(t_h_pt)
        if sy < 3.2 or sy > 4.5:
            sy = 3.8889
        expected_w = int(round(sy * t_w_pt))
        w_px = last_x - first_x
        if abs(w_px - expected_w) / float(expected_w) > 0.04:
            if last_x > (W - int(W * 0.25)):
                first_x = max(90, last_x - expected_w)
            elif first_x < int(W * 0.25):
                last_x = min(W - 80, first_x + expected_w)

    return first_x, first_y, last_x, last_y

if __name__ == '__main__':
    with open('templates/template_p3.json') as f:
        t3 = json.load(f)
    from seruti_pipeline.grid_template_aligner import get_physical_grid_envelope_pt
    tx0, ty0, tx1, ty1 = get_physical_grid_envelope_pt(t3, 3)
    tw, th = tx1 - tx0, ty1 - ty0

    for b in ['booklet_05', 'booklet_41', 'booklet_45', 'booklet_55', 'booklet_56', 'booklet_57']:
        p_path = f'output/seruti_inti_booklets/{b}/pages/hal_3_clean.png'
        if not os.path.exists(p_path):
            continue
        gray = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)
        fx, fy, lx, ly = fixed_detect_scanned_table_frame(gray, tw, th)
        w = lx - fx
        h = ly - fy
        exp_w = int(round(3.8889 * tw))
        exp_h = int(round(3.8889 * th))
        print(f'{b} Hal 3: frame=[{fx}, {fy}, {lx}, {ly}] w={w} (exp={exp_w}), h={h} (exp={exp_h})')
