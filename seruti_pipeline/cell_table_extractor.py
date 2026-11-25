"""
seruti_pipeline/cell_table_extractor.py
=======================================
Ekstraksi, standardisasi sel (comb-box), dan penjajaran kisi morfologis (Lattice Snapper)
untuk pemrosesan kuesioner BPS Susenas Seruti Inti.
"""
from typing import Dict, Any, Tuple, Optional, Union, List
import cv2
import numpy as np


class PhysicalLatticeExtractor:
    """
    Ekstraksi kisi fisik dan sel tabel langsung dari morfologi citra (End-to-End Precise Grid).
    Mengadopsi algoritma rekonstruksi rangka fisik:
    1. Inverted Adaptive Binarization (Mean-C, block 15, C=-2).
    2. Ekstraksi garis morfologis H (40, 1) & V (1, 40).
    3. Union rangka tabel (cv2.add).
    4. Deteksi kontur sel fisik cv2.findContours(RETR_CCOMP).
    5. Snapping 2D untuk comb-box individual dan sel struktur.
    """
    @staticmethod
    def extract_physical_cells(
        mono_img: np.ndarray,
        h_kernel_len: int = 40,
        v_kernel_len: int = 40,
        bridge_gap: int = 15,
        min_rect: float = 0.72,
        margin_px: int = 35,
        min_area: int = 500,
        max_area: int = 800000,
        min_h: int = 15,
        max_h: int = 350,
        min_w: int = 20,
        max_w: int = 2400
    ) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Mengekstrak bounding box seluruh sel fisik dari citra monokrom dengan jaminan anti-noise:
        1. Inverted Adaptive Binarization (Mean-C, block 15, C=-2).
        2. Ekstraksi garis morfologis H (40, 1) & V (1, 40).
        3. Morphological Line Bridging: Closing 15px terarah untuk menyambung garis putus/tinta pudar.
        4. Union rangka tabel (cv2.add).
        5. Deteksi kontur sel fisik cv2.findContours(RETR_CCOMP).
        6. Rectangularity Filtering (>= 0.72) untuk membuang teks underline & coretan tangan.
        7. Perimeter Margin Clamping (margin_px) untuk membuang tepi scanner & roller marks.
        """
        if len(mono_img.shape) == 3:
            gray = cv2.cvtColor(mono_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = mono_img

        H, W = gray.shape[:2]
        bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)

        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv)

        # 1. Morphological Line Bridging (Penyambungan celah garis akibat lipatan/tinta tipis)
        if bridge_gap > 0:
            kh_br = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_gap, 1))
            kv_br = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_gap))
            h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, kh_br)
            v_lines = cv2.morphologyEx(v_lines, cv2.MORPH_CLOSE, kv_br)

        table_lattice = cv2.add(h_lines, v_lines)
        contours, _ = cv2.findContours(table_lattice, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if min_area < area < max_area and min_h < h < max_h and min_w < w < max_w:
                # 2. Perimeter Margin Clamping (Buang noise tepi scanner / roller)
                if x < margin_px or y < margin_px or x + w > W - margin_px or y + h > H - margin_px:
                    continue
                # 3. Rectangularity / Solidity Filtering (Buang teks underline & coretan non-persegi)
                c_area = cv2.contourArea(c)
                if (c_area / float(area)) >= min_rect:
                    cells.append((x, y, w, h))

        return cells, table_lattice

    @staticmethod
    def snap_box_to_physical_cells(
        predicted_box: Tuple[int, int, int, int],
        physical_cells: List[Tuple[int, int, int, int]],
        max_center_dist: float = 45.0,
        max_dim_diff: float = 20.0,
        max_h_diff: float = 22.0
    ) -> Tuple[Tuple[int, int, int, int], bool, float]:
        """
        Menyelaraskan predicted_box [bx0, by0, bx1, by1] ke sel fisik terdekat [px0, py0, px1, py1]
        dengan perlindungan integritas sel (Zero-Drift & Anti-Mis-snapping Guardrail):
        1. Prioritas 1: Multi-box contiguous contour (kotak comb berantai yang menyatu akibat garis putus/pudar).
        2. Prioritas 2: Direct single-cell match dengan validasi rasio lebar dan IoU overlap minimum (>= 0.35).
        3. Fallback: Tetap menggunakan predicted_box asli terkalibrasi jika tidak ada kontur fisik yang valid.
        """
        bx0, by0, bx1, by1 = predicted_box
        bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        bw_box, bh_box = bx1 - bx0, by1 - by0
        pred_area = max(1.0, float(bw_box * bh_box))

        # 1. Prioritas 1: Direct single-cell match dengan verifikasi IoU dan rasio lebar
        best_match = None
        best_dist = 999.0
        best_iou = 0.0

        for px, py, pw, ph in physical_cells:
            px0, py0, px1, py1 = px, py, px + pw, py + ph
            # Guardrail lebar & tinggi: tolak kolom garis sempit (< 0.70x lebar kotak) atau terlalu lebar (> 1.35x)
            if not (0.70 * bw_box <= pw <= 1.35 * bw_box and abs(ph - bh_box) < max_h_diff):
                continue

            inter_x0 = max(bx0, px0)
            inter_y0 = max(by0, py0)
            inter_x1 = min(bx1, px1)
            inter_y1 = min(by1, py1)
            if inter_x1 > inter_x0 and inter_y1 > inter_y0:
                inter_area = float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))
                iou = inter_area / (pred_area + float(pw * ph) - inter_area)
                if iou >= 0.35:
                    pcx, pcy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
                    dist = np.hypot(bcx - pcx, bcy - pcy)
                    if dist < max_center_dist and (iou > best_iou or (abs(iou - best_iou) < 0.1 and dist < best_dist)):
                        best_iou = iou
                        best_dist = dist
                        best_match = (px0, py0, px1, py1)

        if best_match is not None:
            return best_match, True, round(best_dist, 2)

        # 2. Prioritas 2: Multi-box contiguous contour match (hanya jika sel tunggal tidak ditemukan karena sekat pudar/putus)
        for px, py, pw, ph in physical_cells:
            px0, py0, px1, py1 = px, py, px + pw, py + ph
            # Batasi agar tidak membajak blok/tabel raksasa (> 13x lebar kotak)
            if bw_box * 1.35 <= pw <= bw_box * 13.0 and abs(ph - bh_box) < max_h_diff:
                if (px0 - 15 <= bcx <= px1 + 15) and (py0 - 15 <= bcy <= py1 + 15):
                    n_est = max(2, int(round(pw / max(1.0, float(bw_box)))))
                    slot_w = pw / float(n_est)
                    rel_pos = (bcx - px0) / slot_w
                    slot = int(np.clip(int(np.floor(rel_pos)), 0, n_est - 1))
                    snapped_x0 = int(round(px0 + slot * slot_w))
                    snapped_x1 = int(round(px0 + (slot + 1) * slot_w))
                    return (snapped_x0, py0, snapped_x1, py1), True, 2.0

        # 3. Fallback aman ke predicted box asli
        return predicted_box, False, 0.0

    @staticmethod
    def render_precise_grid_overlay(
        img: np.ndarray,
        physical_cells: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 220, 0),
        thickness: int = 2,
        table_envelopes: Optional[List[Tuple[int, int, int, int]]] = None,
        envelope_color: Tuple[int, int, int] = (0, 220, 0),
        envelope_thickness: int = 3
    ) -> np.ndarray:
        """
        Menggambar mesh kisi hijau bersinar di atas citra fisik pindaian
        dengan jaminan integritas segi empat terluar (Outer Envelope Clamping).
        """
        vis = img.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # 1. Gambar amplop segi empat terluar tabel (Outer Envelopes) jika tersedia
        if table_envelopes:
            for env in table_envelopes:
                x0, y0, x1, y1 = env
                cv2.rectangle(vis, (x0, y0), (x1, y1), envelope_color, envelope_thickness)

        # 2. Gambar seluruh sel fisik interior
        for x, y, w, h in physical_cells:
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        return vis


class MorphologicalLatticeSnapper:
    """
    Penjajar koordinat kisi dan batas sel menggunakan proyeksi profil morfologis.
    """
    @staticmethod
    def get_projections(mono_img: np.ndarray) -> Dict[str, Any]:
        """
        Menghasilkan proyeksi horizontal (h_proj) dan vertikal (v_proj)
        setelah isolasi garis morfologis.
        """
        H, W = mono_img.shape[:2]
        # Inversi tinta: tinta = putih (255)
        inv = 255 - mono_img

        # Isolasi garis horizontal
        kh_len = max(15, int(W * 0.04))
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (kh_len, 1))
        lines_h = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh)

        # Isolasi garis vertikal
        kv_len = max(12, int(H * 0.01))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv_len))
        lines_v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv)

        h_proj = lines_h.sum(axis=1).astype(np.float32)
        v_proj = lines_v.sum(axis=0).astype(np.float32)

        return {
            "h_proj": h_proj,
            "v_proj": v_proj,
            "shape": (H, W),
        }

    @staticmethod
    def snap_1d(profile: np.ndarray, coord: int, radius: int = 15) -> int:
        """
        Menyelaraskan koordinat 1D ke puncak intensitas terdekat dalam radius pencarian.
        """
        L = len(profile)
        c = max(0, min(L - 1, int(coord)))
        r1 = max(0, c - radius)
        r2 = min(L, c + radius + 1)
        window = profile[r1:r2]
        if len(window) == 0:
            return c

        max_val = np.max(window)
        # Jika nilai puncak melampaui ambang batas lokal
        if max_val > 0.05 * np.max(profile):
            best_idx = r1 + int(np.argmax(window))
            return best_idx
        return c


def standardize_cell_image(
    crop: np.ndarray,
    target_size: int = 28,
    border_cut_px: int = 1,
    pad_digit_px: int = 2,
    return_meta: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, bool, float]]:
    """
    Menstandarkan citra sel kuesioner ke ukuran seragam (target_size x target_size, default 28x28) dengan preservasi keutuhan digit:
    1. Evaluasi densitas tinta dan blank-check.
    2. Segmentasi goresan tinta pensil di tengah sel dengan eliminasi garis pembatas via Connected Components (tanpa pemotongan agresif).
    3. Preservasi margin pengaman (pad_digit_px) agar lengkungan/ekor pensil tidak terpotong.
    4. Normalisasi kanvas putih 255 (background putih, tinta gelap).
    """
    if crop is None or crop.size == 0:
        blank = np.full((target_size, target_size), 255, dtype=np.uint8)
        if return_meta:
            return blank, True, 0.0
        return blank

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    cH, cW = gray.shape[:2]
    # Pangkas perimeter ekstrem 1 px saja jika ada border hitam tebal
    b_cut = max(0, min(border_cut_px, 1))
    if b_cut > 0 and cH > 4 and cW > 4:
        inner = gray[b_cut:cH - b_cut, b_cut:cW - b_cut]
    else:
        inner = gray

    iH, iW = inner.shape[:2]
    # Ekstraksi tinta pensil
    ink_mask = inner < 190
    ink_density = float(np.mean(ink_mask))
    std_val = float(np.std(inner))
    is_blank = (ink_density < 0.012) or (std_val < 5.0)

    if is_blank:
        out_cell = np.full((target_size, target_size), 255, dtype=np.uint8)
    else:
        # Eliminasi garis tepi kotak fisik melalui connected components perimeter
        ink_u8 = ink_mask.astype(np.uint8) * 255
        n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(ink_u8, connectivity=8)
        clean_ink = ink_u8.copy()

        for ci in range(1, n_cc):
            cx, cy, cw, ch, ca = stats[ci]
            # Komponen garis tipis menempel di tepi luar sel
            is_edge_h = (cy <= 1 or cy + ch >= iH - 1) and cw > 0.70 * iW and ch <= max(3, int(0.15 * iH))
            is_edge_v = (cx <= 1 or cx + cw >= iW - 1) and ch > 0.70 * iH and cw <= max(3, int(0.15 * iW))
            if is_edge_h or is_edge_v or ca < 6:
                clean_ink[lab == ci] = 0

        pts = cv2.findNonZero(clean_ink)
        if pts is not None:
            x, y, w, h = cv2.boundingRect(pts)
            # Berikan padding margin di sekeliling digit agar lengkungan pensil utuh
            y0_d = max(0, y - pad_digit_px)
            y1_d = min(inner.shape[0], y + h + pad_digit_px)
            x0_d = max(0, x - pad_digit_px)
            x1_d = min(inner.shape[1], x + w + pad_digit_px)
            digit_crop = inner[y0_d:y1_d, x0_d:x1_d]
            h_d, w_d = digit_crop.shape[:2]

            # Resize dengan mempertahankan aspect ratio
            target_inner = target_size - 6
            if h_d > w_d:
                new_h = target_inner
                new_w = max(1, int(round(w_d * (target_inner / float(h_d)))))
            else:
                new_w = target_inner
                new_h = max(1, int(round(h_d * (target_inner / float(w_d)))))

            resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            out_cell = np.full((target_size, target_size), 255, dtype=np.uint8)
            start_y = (target_size - new_h) // 2
            start_x = (target_size - new_w) // 2
            out_cell[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit
        else:
            out_cell = cv2.resize(inner, (target_size, target_size), interpolation=cv2.INTER_AREA)

    if return_meta:
        return out_cell, is_blank, round(ink_density, 4)
    return out_cell


def get_default_onnx_session() -> Optional[Any]:
    """Mengembalikan sesi ONNX default jika tersedia di sistem."""
    return None
