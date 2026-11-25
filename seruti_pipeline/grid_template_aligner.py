"""
seruti_pipeline/grid_template_aligner.py
========================================
Modul Penyelarasan (Alignment/Registration) dan Pemotongan (Crop Area)
Grid Template Kuesioner BPS Susenas Seruti Inti (VSERUTI26.INTI).

Fungsi Utama:
1. Memuat berkas template grid PDF vektor (template_p1.json s.d. template_p7.json).
2. Mendeteksi batas bingkai tabel fisik pada citra pindaian kanonikal (300 DPI)
   menggunakan proyeksi morfologis horizontal dan vertikal.
3. Menghitung transformasi affine (skala sx, sy dan translasi tx, ty) untuk
   memetakan koordinat titik PDF (pt) ke piksel citra pindaian secara presisi sub-piksel.
4. Menyediakan antarmuka pemotongan area (crop area) untuk:
   - Field isian lengkap (misal 11 kotak digit comb-box nominal uang).
   - Sel/kotak digit individual (misal kotak 51x66 px untuk klasifikasi digit).
   - Sel struktur tabel dan checkbox OMR.
5. Menghasilkan visualisasi pratinjau grid overlay terdaftar (aligned preview).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import cv2
import numpy as np

# Lokasi default direktori template
DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def _interp_extrap_1d(val: float, xp: List[float], fp: List[float], default_scale: float = 3.90) -> float:
    """Interpolasi linear 1D dengan ekstrapolasi linear berbasis skala global (stabil & tidak clamp)."""
    if len(xp) < 2:
        return fp[0] if fp else val
    global_slope = (fp[-1] - fp[0]) / max(1e-6, (xp[-1] - xp[0])) if (xp[-1] - xp[0]) > 20 else default_scale
    if val <= xp[0]:
        return fp[0] + (val - xp[0]) * global_slope
    elif val >= xp[-1]:
        return fp[-1] + (val - xp[-1]) * global_slope
    else:
        return float(np.interp(val, xp, fp))


@dataclass
class GridAlignment:
    """Hasil penyelarasan koordinat template terhadap citra pindaian."""
    page_num: int
    img_width: int
    img_height: int
    template_size_pt: Tuple[float, float]
    scale_x: float
    scale_y: float
    trans_x: float
    trans_y: float
    scan_bounds: Tuple[int, int, int, int]      # [x0, y0, x1, y1] dalam piksel
    template_bounds: Tuple[float, float, float, float]  # [x0, y0, x1, y1] dalam pt
    template_data: Dict[str, Any]
    warp_x_anchors: Optional[Tuple[List[float], List[float]]] = None
    warp_y_anchors: Optional[Tuple[List[float], List[float]]] = None
    snapped_cells_px: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    physical_cells: Optional[List[Tuple[int, int, int, int]]] = None
    deskew_angle: float = 0.0
    raw_img: Optional[np.ndarray] = None

    def enable_physical_cell_snapping(self, img: np.ndarray, max_center_dist: float = 45.0) -> int:
        """
        Mendeteksi kontur sel fisik 2D dan mengunci posisi kotak comb-box secara langsung (Physical Cell Snapping).
        Mengintegrasikan algoritma morfologi rangka fisik untuk mengeliminasi residual drift menjadi 0 px.
        Dilengkapi 4-Pass Hybrid Infilling & Physical Lattice Snapping:
        - Pass 1: Direct single-cell & contiguous multi-box contour snapping untuk seluruh sel (digit & struktural).
        - Pass 2: Neighbor-Guided Fallback Infilling untuk kotak digit sebaris (misal 11 digit uang atau 4 digit waktu).
        - Pass 3: Morphological Edge Snapping untuk sel yang belum ter-snap kontur langsung (header bentang lebar, kolom deskripsi, arsir NA).
        - Pass 4: Outer Table Envelope Clamping & Structural NA Infilling (menjamin seluruh sel menempel pada garis batas fisik).
        Mengembalikan jumlah kotak digit yang berhasil terkunci presisi pada sel fisik.
        """
        from .cell_table_extractor import PhysicalLatticeExtractor

        self.raw_img = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        H, W = gray.shape[:2]

        # Ekstraksi morfologis garis horizontal dan vertikal
        bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv)

        phys_cells, _ = PhysicalLatticeExtractor.extract_physical_cells(img)
        self.physical_cells = phys_cells
        snapped = {}
        is_snapped = {}
        snapped_count = 0
        cells_by_id = {c["id"]: c for c in self.template_data.get("cells", [])}

        # Pass 1: 1-to-1 Bipartite Greedy Snapping (Zero-Collision Invariant)
        # Menjamin tidak ada 2 kotak digit yang menempel pada 1 sel fisik yang sama
        candidates = []
        for c in self.template_data.get("cells", []):
            cid = c["id"]
            pred_box = self.bbox_pt_to_px(c["bbox_pt"])
            cx_pred = (pred_box[0] + pred_box[2]) / 2.0
            cy_pred = (pred_box[1] + pred_box[3]) / 2.0
            bw_box = pred_box[2] - pred_box[0]
            bh_box = pred_box[3] - pred_box[1]

            is_checkbox = ("kesediaan" in cid.lower() or "checkbox" in cid.lower() or (0.75 <= bw_box / max(1, bh_box) <= 1.35 and bw_box <= 55))
            w_min_factor = 0.50 if is_checkbox else 0.70
            w_max_factor = 1.45 if is_checkbox else 1.35
            h_diff_max = 26 if is_checkbox else 22

            for idx, (px, py, pw, ph) in enumerate(phys_cells):
                cx_p = px + pw / 2.0
                cy_p = py + ph / 2.0
                dist = float(np.hypot(cx_pred - cx_p, cy_pred - cy_p))
                if dist <= max_center_dist and (w_min_factor * bw_box <= pw <= w_max_factor * bw_box) and abs(ph - bh_box) <= h_diff_max:
                    if is_checkbox:
                        # Pusatkan kotak berukuran kanonikal (bw_box x bh_box) presisi pada titik pusat kotak fisik
                        half_w = bw_box // 2
                        half_h = bh_box // 2
                        p_box = (
                            int(round(cx_p - half_w)),
                            int(round(cy_p - half_h)),
                            int(round(cx_p - half_w)) + bw_box,
                            int(round(cy_p - half_h)) + bh_box
                        )
                    else:
                        p_box = (px, py, px + pw, py + ph)
                    candidates.append((dist, cid, idx, p_box, pred_box, c.get("kind")))

        # Prioritaskan kandidat dengan jarak pusat terdekat (Greedy Optimal)
        candidates.sort(key=lambda x: x[0])
        used_phys_indices = set()
        for dist, cid, idx, p_box, pred_box, kind in candidates:
            if cid not in snapped and idx not in used_phys_indices:
                snapped[cid] = p_box
                is_snapped[cid] = True
                used_phys_indices.add(idx)
                if kind == "digit":
                    snapped_count += 1

        # Sel yang belum ter-snap kontur diinisialisasi dengan pred_box
        for c in self.template_data.get("cells", []):
            cid = c["id"]
            if cid not in snapped:
                snapped[cid] = self.bbox_pt_to_px(c["bbox_pt"])
                is_snapped[cid] = False

        # Pass 2: Neighbor-Guided Fallback Infilling & Monotonic Sequence Enforcement
        # Menjamin urutan digit dari kiri ke kanan (atau atas ke bawah untuk vertikal) teratur tanpa tumbukan
        for f in self.template_data.get("fields", []):
            boxes = [b for b in f.get("boxes", []) if b in is_snapped]
            if len(boxes) < 2:
                continue

            # Deteksi orientasi field: vertikal (misal checkbox kesediaan) vs horizontal
            first_pred = self.bbox_pt_to_px(cells_by_id[boxes[0]]["bbox_pt"]) if boxes[0] in cells_by_id else snapped[boxes[0]]
            last_pred = self.bbox_pt_to_px(cells_by_id[boxes[-1]]["bbox_pt"]) if boxes[-1] in cells_by_id else snapped[boxes[-1]]
            is_vertical = (abs(last_pred[1] - first_pred[1]) > abs(last_pred[0] - first_pred[0]) + 15)

            if is_vertical:
                # Verifikasi urutan monoton vertikal: kotak i+1 harus di bawah kotak i
                for i in range(len(boxes) - 1):
                    b0, b1 = boxes[i], boxes[i+1]
                    if is_snapped[b0] and is_snapped[b1]:
                        if snapped[b1][1] <= snapped[b0][1] + 5:
                            is_snapped[b1] = False

                snapped_indices = [i for i, b in enumerate(boxes) if is_snapped[b]]
                if not snapped_indices or len(snapped_indices) == len(boxes):
                    continue

                heights = [snapped[boxes[i]][3] - snapped[boxes[i]][1] for i in snapped_indices]
                med_h = int(round(float(np.median(heights))))

                # Infill forward (atas ke bawah)
                for i in range(len(boxes)):
                    bid = boxes[i]
                    if not is_snapped[bid] and i > 0 and is_snapped[boxes[i-1]]:
                        ref = snapped[boxes[i-1]]
                        snapped[bid] = (ref[0], ref[3], ref[2], ref[3] + med_h)
                        is_snapped[bid] = True
                        snapped_count += 1

                # Infill backward (bawah ke atas)
                for i in range(len(boxes) - 1, -1, -1):
                    bid = boxes[i]
                    if not is_snapped[bid] and i < len(boxes) - 1 and is_snapped[boxes[i+1]]:
                        ref = snapped[boxes[i+1]]
                        snapped[bid] = (ref[0], ref[1] - med_h, ref[2], ref[1])
                        is_snapped[bid] = True
                        snapped_count += 1
            else:
                # Verifikasi urutan monoton horizontal: kotak i+1 harus di sebelah kanan kotak i
                for i in range(len(boxes) - 1):
                    b0, b1 = boxes[i], boxes[i+1]
                    if is_snapped[b0] and is_snapped[b1]:
                        if snapped[b1][0] <= snapped[b0][0] + 5:
                            # Terjadi kolaps / tumbukan: lepaskan snap kotak kedua agar diinfill teratur
                            is_snapped[b1] = False

                snapped_indices = [i for i, b in enumerate(boxes) if is_snapped[b]]
                if not snapped_indices or len(snapped_indices) == len(boxes):
                    continue

                widths = [snapped[boxes[i]][2] - snapped[boxes[i]][0] for i in snapped_indices]
                med_w = int(round(float(np.median(widths))))

                # Infill forward (kiri ke kanan)
                for i in range(len(boxes)):
                    bid = boxes[i]
                    if not is_snapped[bid] and i > 0 and is_snapped[boxes[i-1]]:
                        ref = snapped[boxes[i-1]]
                        snapped[bid] = (ref[2], ref[1], ref[2] + med_w, ref[3])
                        is_snapped[bid] = True
                        snapped_count += 1

                # Infill backward (kanan ke kiri)
                for i in range(len(boxes) - 1, -1, -1):
                    bid = boxes[i]
                    if not is_snapped[bid] and i < len(boxes) - 1 and is_snapped[boxes[i+1]]:
                        ref = snapped[boxes[i+1]]
                        snapped[bid] = (ref[0] - med_w, ref[1], ref[0], ref[3])
                        is_snapped[bid] = True
                        snapped_count += 1

        # Pass 3: Morphological Edge Snapping untuk sel yang belum ter-snap kontur langsung
        # (header bentang lebar, kolom deskripsi, arsir NA, atau sel bergaris tipis)
        search_r = 18
        for c in self.template_data.get("cells", []):
            cid = c["id"]
            if not is_snapped.get(cid, False):
                x0, y0, x1, y1 = snapped[cid]
                orig_w = max(1, x1 - x0)
                orig_h = max(1, y1 - y0)

                # Snap horizontal edges y0 dan y1
                w_y0 = range(max(0, y0 - search_r), min(H, y0 + search_r + 1))
                x_sub0 = max(0, min(W - 1, x0 + 4))
                x_sub1 = max(x_sub0 + 1, min(W, x1 - 4))
                sc0 = [float(h_lines[yy, x_sub0:x_sub1].sum()) for yy in w_y0]
                sy0 = w_y0[int(np.argmax(sc0))] if (sc0 and max(sc0) > 0) else y0

                w_y1 = range(max(0, y1 - search_r), min(H, y1 + search_r + 1))
                sc1 = [float(h_lines[yy, x_sub0:x_sub1].sum()) for yy in w_y1]
                sy1 = w_y1[int(np.argmax(sc1))] if (sc1 and max(sc1) > 0) else y1

                # Snap vertical edges x0 dan x1
                y_sub0 = max(0, min(H - 1, y0 + 4))
                y_sub1 = max(y_sub0 + 1, min(H, y1 - 4))
                w_x0 = range(max(0, x0 - search_r), min(W, x0 + search_r + 1))
                scx0 = [float(v_lines[y_sub0:y_sub1, xx].sum()) for xx in w_x0]
                sx0 = w_x0[int(np.argmax(scx0))] if (scx0 and max(scx0) > 0) else x0

                w_x1 = range(max(0, x1 - search_r), min(W, x1 + search_r + 1))
                scx1 = [float(v_lines[y_sub0:y_sub1, xx].sum()) for xx in w_x1]
                sx1 = w_x1[int(np.argmax(scx1))] if (scx1 and max(scx1) > 0) else x1

                new_w = sx1 - sx0
                new_h = sy1 - sy0
                is_cb = ("kesediaan" in cid.lower() or "checkbox" in cid.lower())
                if is_cb and new_w > 0 and new_h > 0 and (sx0 < sx1) and (sy0 < sy1):
                    # Untuk checkbox / OMR, pertahankan dimensi kanonikal orig_w x orig_h terpusat pada titik tengah fisik
                    cx = (sx0 + sx1) / 2.0
                    cy = (sy0 + sy1) / 2.0
                    half_w = orig_w // 2
                    half_h = orig_h // 2
                    snapped[cid] = (int(round(cx - half_w)), int(round(cy - half_h)), int(round(cx - half_w)) + orig_w, int(round(cy - half_h)) + orig_h)
                    is_snapped[cid] = True
                    if c.get("kind") == "digit":
                        snapped_count += 1
                elif new_w > 0 and new_h > 0 and 0.65 * orig_w <= new_w <= 1.35 * orig_w and 0.65 * orig_h <= new_h <= 1.35 * orig_h:
                    snapped[cid] = (sx0, sy0, sx1, sy1)
                    is_snapped[cid] = True
                    if c.get("kind") == "digit":
                        snapped_count += 1

        # Pass 4: Outer Table Envelope Clamping & Structural NA Infilling
        # Menjamin bahwa sel arsir NA dan perimeter luar tabel selalu menempel tepat di garis batas fisik
        table_envelopes = self.get_snapped_table_envelopes(img)
        for env in table_envelopes:
            tx0, ty0, tx1, ty1 = env["bbox_px"]
            t_idx = env["table_index"] - 1
            tables_list = self.template_data.get("tables", [])
            t_meta = tables_list[t_idx] if t_idx < len(tables_list) else None
            if t_meta:
                tx0_pt, ty0_pt, tx1_pt, ty1_pt = t_meta["bbox_pt"]
                for c in self.template_data.get("cells", []):
                    cid = c["id"]
                    cx_pt = (c["bbox_pt"][0] + c["bbox_pt"][2]) / 2.0
                    cy_pt = (c["bbox_pt"][1] + c["bbox_pt"][3]) / 2.0
                    if (c.get("table") == t_idx or 
                        (tx0_pt - 5 <= cx_pt <= tx1_pt + 5 and ty0_pt - 5 <= cy_pt <= ty1_pt + 5)):
                        bx0, by0, bx1, by1 = snapped[cid]
                        # Clamp cell to table envelope
                        cx0 = max(tx0, min(tx1, bx0))
                        cy0 = max(ty0, min(ty1, by0))
                        cx1 = max(tx0, min(tx1, bx1))
                        cy1 = max(ty0, min(ty1, by1))
                        if cx1 > cx0 and cy1 > cy0:
                            snapped[cid] = (cx0, cy0, cx1, cy1)

        # Pass 5: Dedicated Adaptive Checkbox Snapper untuk Persetujuan Responden (Hal 1)
        # Menjamin sel b00, b01, b02 terkunci presisi sub-piksel pada kotak fisik tanpa terpotong
        if any("kesediaan" in c.get("id", "").lower() for c in self.template_data.get("cells", [])):
            self._snap_kesediaan_checkboxes(img, snapped, is_snapped)

        self.snapped_cells_px = snapped
        self.is_cell_snapped = is_snapped
        return snapped_count

    def _snap_kesediaan_checkboxes(
        self,
        img: np.ndarray,
        snapped: Dict[str, Tuple[int, int, int, int]],
        is_snapped: Dict[str, bool]
    ) -> None:
        """
        Deteksi adaptif kotak persetujuan responden (b00, b01, b02) pada Halaman 1.
        Memanfaatkan orientasi kolom vertikal dan jarak antar kotak (vertical pitch)
        untuk mengunci seluruh kotak dengan toleransi 0 px terpotong (Zero-Cut Invariant).
        """
        H, W = img.shape[:2]
        rx0, ry0 = self.pt_to_px(72.0, 188.0)
        rx1, ry1 = self.pt_to_px(102.0, 238.0)

        rx0 = max(0, rx0 - 25)
        ry0 = max(0, ry0 - 25)
        rx1 = min(W, rx1 + 25)
        ry1 = min(H, ry1 + 25)

        roi = img[ry0:ry1, rx0:rx1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        th = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        cnts, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if 24 <= w <= 38 and 24 <= h <= 38:
                ar = w / float(h)
                if 0.75 <= ar <= 1.30:
                    candidates.append((rx0 + x, ry0 + y, w, h))

        unique = []
        for cand in sorted(candidates, key=lambda c: c[1]):
            if not any(abs(cand[0] - u[0]) < 6 and abs(cand[1] - u[1]) < 6 for u in unique):
                unique.append(cand)

        x_groups = {}
        for cand in unique:
            matched = False
            for gx in x_groups:
                if abs(cand[0] - gx) <= 8:
                    x_groups[gx].append(cand)
                    matched = True
                    break
            if not matched:
                x_groups[cand[0]] = [cand]

        exp_x, _ = self.pt_to_px(82.0, 200.0)
        best_boxes = []
        best_dist = 9999

        for gx, g_boxes in x_groups.items():
            avg_x = np.mean([b[0] for b in g_boxes])
            dist = abs(avg_x - exp_x)
            score = dist - len(g_boxes) * 30
            if score < best_dist:
                best_dist = score
                best_boxes = sorted(g_boxes, key=lambda b: b[1])

        exp_pitch = int(round(11.5 * self.scale_y))
        boxes_res = [None, None, None]
        _, y_exp_0 = self.pt_to_px(82.0, 196.75)
        _, y_exp_1 = self.pt_to_px(82.0, 208.25)
        _, y_exp_2 = self.pt_to_px(82.0, 219.75)

        for b in best_boxes:
            bx, by, bw, bh = b
            d0 = abs(by - y_exp_0)
            d1 = abs(by - y_exp_1)
            d2 = abs(by - y_exp_2)
            min_d = min(d0, d1, d2)
            if min_d <= 25:
                if min_d == d0 and boxes_res[0] is None:
                    boxes_res[0] = b
                elif min_d == d1 and boxes_res[1] is None:
                    boxes_res[1] = b
                elif min_d == d2 and boxes_res[2] is None:
                    boxes_res[2] = b

        known = [b for b in boxes_res if b is not None]
        target_ids = ["p1_c_kesediaan_1", "p1_c_kesediaan_2", "p1_c_kesediaan_3"]
        if known:
            ref_w = int(round(float(np.median([b[2] for b in known]))))
            ref_h = int(round(float(np.median([b[3] for b in known]))))
            ref_x = int(round(float(np.median([b[0] for b in known]))))

            if boxes_res[1] and boxes_res[2]:
                pitch = boxes_res[2][1] - boxes_res[1][1]
            elif boxes_res[0] and boxes_res[1]:
                pitch = boxes_res[1][1] - boxes_res[0][1]
            elif boxes_res[0] and boxes_res[2]:
                pitch = (boxes_res[2][1] - boxes_res[0][1]) // 2
            else:
                pitch = exp_pitch

            if boxes_res[1] is None and boxes_res[0] is not None:
                boxes_res[1] = (ref_x, boxes_res[0][1] + pitch, ref_w, ref_h)
            elif boxes_res[1] is None and boxes_res[2] is not None:
                boxes_res[1] = (ref_x, boxes_res[2][1] - pitch, ref_w, ref_h)

            if boxes_res[0] is None and boxes_res[1] is not None:
                boxes_res[0] = (ref_x, boxes_res[1][1] - pitch, ref_w, ref_h)

            if boxes_res[2] is None and boxes_res[1] is not None:
                boxes_res[2] = (ref_x, boxes_res[1][1] + pitch, ref_w, ref_h)

            for i, tid in enumerate(target_ids):
                by = boxes_res[i][1]
                # Simpan dengan margin 1px agar garis tepi fisik kotak utuh 100%
                snapped[tid] = (
                    max(0, ref_x - 1),
                    max(0, by - 1),
                    min(W, ref_x + ref_w + 1),
                    min(H, by + ref_h + 1)
                )
                is_snapped[tid] = True
        else:
            for i, (tid, pt_y) in enumerate(zip(target_ids, [196.75, 208.25, 219.75])):
                snapped[tid] = self.bbox_pt_to_px([82.0, pt_y, 90.0, pt_y + 7.75], pad=1)
                is_snapped[tid] = True

    def get_snapping_metrics_by_role(self) -> Dict[str, Dict[str, int]]:
        """
        Mengembalikan ringkasan snapping terperinci berdasarkan peran sel (role):
        - questionnaire_digit (isian kuesioner responden)
        - triwulan (metadata periode administratif)
        - page_number (footer nomor halaman)
        """
        metrics = {
            "questionnaire_digit": {"total": 0, "snapped": 0},
            "triwulan": {"total": 0, "snapped": 0},
            "page_number": {"total": 0, "snapped": 0},
        }
        for c in self.template_data.get("cells", []):
            if c.get("kind") != "digit" or min(c.get("fill_rgb", [255, 255, 255])) < 245:
                continue
            role = c.get("role", "questionnaire_digit")
            if role not in metrics:
                metrics[role] = {"total": 0, "snapped": 0}
            metrics[role]["total"] += 1
            if getattr(self, "is_cell_snapped", {}).get(c["id"], False):
                metrics[role]["snapped"] += 1
        return metrics

    def snap_table_envelope_to_lines(
        self,
        img: np.ndarray,
        approx_bbox: Tuple[int, int, int, int],
        search_radius: int = 45
    ) -> Tuple[int, int, int, int]:
        """
        Menyelaraskan batas amplop tabel (x0, y0, x1, y1) secara langsung ke garis fisik
        horizontal dan vertikal pada citra pindaian menggunakan proyeksi profil morfologis.
        Menjamin potongan tabel jatuh tepat di atas garis batas fisik (Zero-Cut Invariant).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        H, W = gray.shape[:2]

        bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv)

        x0, y0, x1, y1 = approx_bbox
        inner_m = 30
        x_start = max(0, min(W - 1, x0 + inner_m))
        x_end = max(0, min(W, x1 - inner_m))
        y_start = max(0, min(H - 1, y0 + inner_m))
        y_end = max(0, min(H, y1 - inner_m))
        if x_end <= x_start:
            x_start, x_end = max(0, x0), min(W, x1)
        if y_end <= y_start:
            y_start, y_end = max(0, y0), min(H, y1)

        def _pick_best_line(indices, scores, target):
            if not scores or max(scores) <= 0:
                return target
            max_s = max(scores)
            thresh = 0.75 * max_s
            cands = [(abs(idx - target), -score, idx) for idx, score in zip(indices, scores) if score >= thresh]
            if not cands:
                return target
            cands.sort()
            return cands[0][2]

        # 1. Snap y0 (Garis horizontal atas dengan prioritas jarak terdekat ke target)
        w_y0 = list(range(max(0, y0 - search_radius), min(H, y0 + search_radius + 1)))
        y0_scores = [float(h_lines[y, x_start:x_end].sum()) for y in w_y0]
        snapped_y0 = _pick_best_line(w_y0, y0_scores, y0)

        # 2. Snap y1 (Garis horizontal bawah dengan prioritas jarak terdekat ke target)
        w_y1 = list(range(max(0, y1 - search_radius), min(H, y1 + search_radius + 1)))
        y1_scores = [float(h_lines[y, x_start:x_end].sum()) for y in w_y1]
        snapped_y1 = _pick_best_line(w_y1, y1_scores, y1)

        # 3. Snap x0 (Garis vertikal kiri dengan prioritas jarak terdekat ke target)
        w_x0 = list(range(max(0, x0 - search_radius), min(W, x0 + search_radius + 1)))
        x0_scores = [float(v_lines[y_start:y_end, x].sum()) for x in w_x0]
        snapped_x0 = _pick_best_line(w_x0, x0_scores, x0)

        # 4. Snap x1 (Garis vertikal kanan dengan prioritas jarak terdekat ke target)
        w_x1 = list(range(max(0, x1 - search_radius), min(W, x1 + search_radius + 1)))
        x1_scores = [float(v_lines[y_start:y_end, x].sum()) for x in w_x1]
        snapped_x1 = _pick_best_line(w_x1, x1_scores, x1)

        if snapped_x1 <= snapped_x0:
            snapped_x0, snapped_x1 = x0, x1
        if snapped_y1 <= snapped_y0:
            snapped_y0, snapped_y1 = y0, y1

        return (int(snapped_x0), int(snapped_y0), int(snapped_x1), int(snapped_y1))

    def get_snapped_table_envelopes(self, img: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Menghasilkan amplop segi empat terluar yang terkunci presisi pada kontur fisik untuk setiap tabel.
        Menjamin bahwa perimeter luar setiap tabel SELALU 100% berbentuk segi empat beraturan yang tertutup
        dan terpotong persis pada garis fisik batas terluar tabel.
        """
        target_img = img if img is not None else getattr(self, "raw_img", None)
        table_envelopes = []
        for t_idx, t in enumerate(self.template_data.get("tables", [])):
            tx0, ty0, tx1, ty1 = self.bbox_pt_to_px(t["bbox_pt"])

            if target_img is not None:
                tx0, ty0, tx1, ty1 = self.snap_table_envelope_to_lines(target_img, (tx0, ty0, tx1, ty1))
            else:
                cells_in_t = [
                    c for c in (self.physical_cells or [])
                    if tx0 - 20 <= c[0] and c[0] + c[2] <= tx1 + 20
                    and ty0 - 20 <= c[1] and c[1] + c[3] <= ty1 + 20
                ]
                if cells_in_t:
                    fit_x0 = min(c[0] for c in cells_in_t)
                    fit_y0 = min(c[1] for c in cells_in_t)
                    fit_x1 = max(c[0] + c[2] for c in cells_in_t)
                    fit_y1 = max(c[1] + c[3] for c in cells_in_t)
                    tx0 = fit_x0 if abs(fit_x0 - tx0) <= 25 else tx0
                    ty0 = fit_y0 if abs(fit_y0 - ty0) <= 25 else ty0
                    tx1 = fit_x1 if abs(fit_x1 - tx1) <= 25 else tx1
                    ty1 = fit_y1 if abs(fit_y1 - ty1) <= 25 else ty1

            table_envelopes.append({
                "table_index": t_idx + 1,
                "bbox_pt": t.get("bbox_pt"),
                "bbox_px": (tx0, ty0, tx1, ty1),
                "width": tx1 - tx0,
                "height": ty1 - ty0
            })
        return table_envelopes

    def get_cell_bbox_px(self, cell_id: str, pad: Union[int, Tuple[int, int]] = 0) -> Optional[Tuple[int, int, int, int]]:
        """Mengembalikan koordinat piksel [x0, y0, x1, y1] dari sel, memprioritaskan hasil snapping fisik."""
        if isinstance(pad, (tuple, list)):
            pad_x, pad_y = pad
        else:
            pad_x = pad_y = pad

        if self.snapped_cells_px and cell_id in self.snapped_cells_px:
            x0, y0, x1, y1 = self.snapped_cells_px[cell_id]
            if pad_x != 0 or pad_y != 0:
                x0 = max(0, x0 - pad_x)
                y0 = max(0, y0 - pad_y)
                x1 = min(self.img_width, x1 + pad_x)
                y1 = min(self.img_height, y1 + pad_y)
            return x0, y0, x1, y1

        cells_by_id = {c["id"]: c for c in self.template_data.get("cells", [])}
        c = cells_by_id.get(cell_id)
        if c:
            return self.bbox_pt_to_px(c["bbox_pt"], pad=pad)
        return None

    def pt_to_px(self, x_pt: float, y_pt: float) -> Tuple[int, int]:
        """Konversi koordinat (x, y) dalam pt PDF ke koordinat piksel citra."""
        if self.warp_x_anchors and len(self.warp_x_anchors[0]) >= 2:
            x_px = int(round(_interp_extrap_1d(x_pt, self.warp_x_anchors[0], self.warp_x_anchors[1], default_scale=self.scale_x)))
        else:
            x_px = int(round(x_pt * self.scale_x + self.trans_x))

        if self.warp_y_anchors and len(self.warp_y_anchors[0]) >= 2:
            y_px = int(round(_interp_extrap_1d(y_pt, self.warp_y_anchors[0], self.warp_y_anchors[1], default_scale=self.scale_y)))
        else:
            y_px = int(round(y_pt * self.scale_y + self.trans_y))

        x_px = max(0, min(self.img_width - 1, x_px))
        y_px = max(0, min(self.img_height - 1, y_px))
        return x_px, y_px

    def bbox_pt_to_px(self, bbox_pt: Union[List[float], Tuple[float, float, float, float]], pad: Union[int, Tuple[int, int]] = 0) -> Tuple[int, int, int, int]:
        """Konversi [x0, y0, x1, y1] dalam pt ke koordinat piksel [x0, y0, x1, y1]."""
        if isinstance(pad, (tuple, list)):
            pad_x, pad_y = pad
        else:
            pad_x = pad_y = pad

        x0_pt, y0_pt, x1_pt, y1_pt = bbox_pt
        x0, y0 = self.pt_to_px(x0_pt, y0_pt)
        x1, y1 = self.pt_to_px(x1_pt, y1_pt)

        # Terapkan padding batas opsional
        if pad_x != 0 or pad_y != 0:
            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y)
            x1 = min(self.img_width, x1 + pad_x)
            y1 = min(self.img_height, y1 + pad_y)

        return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)

    def crop_bbox(self, img: np.ndarray, bbox_pt: List[float], pad: Union[int, Tuple[int, int]] = 0) -> np.ndarray:
        """Memotong area citra berdasarkan bounding box dalam pt."""
        x0, y0, x1, y1 = self.bbox_pt_to_px(bbox_pt, pad=pad)
        return img[y0:y1, x0:x1].copy()

    def crop_field(self, img: np.ndarray, field_id_or_label: str, pad: int = 0) -> Optional[Dict[str, Any]]:
        """
        Memotong seluruh area field isian (misal 'p7_F000' atau label baris '601').
        Mengembalikan dictionary berisi citra crop dan metadata field.
        Memprioritaskan batas amplop fisik dari sel-sel anak yang ter-snap (Physical Cell Snapping).
        """
        for fld in self.template_data.get("fields", []):
            if fld["id"] == field_id_or_label or fld.get("row_label") == field_id_or_label:
                boxes = fld.get("boxes", [])
                if boxes and self.snapped_cells_px and all(b in self.snapped_cells_px for b in boxes):
                    snapped_boxes = [self.snapped_cells_px[b] for b in boxes]
                    bx0 = max(0, min(b[0] for b in snapped_boxes) - pad)
                    by0 = max(0, min(b[1] for b in snapped_boxes) - pad)
                    bx1 = min(self.img_width, max(b[2] for b in snapped_boxes) + pad)
                    by1 = min(self.img_height, max(b[3] for b in snapped_boxes) + pad)
                    crop = img[by0:by1, bx0:bx1].copy()
                    return {
                        "field_id": fld["id"],
                        "row_label": fld.get("row_label"),
                        "parent_text": fld.get("parent_text"),
                        "n_boxes": fld.get("n_boxes", 1),
                        "bbox_px": [bx0, by0, bx1, by1],
                        "bbox_pt": fld["bbox_pt"],
                        "crop": crop,
                    }
                crop = self.crop_bbox(img, fld["bbox_pt"], pad=pad)
                return {
                    "field_id": fld["id"],
                    "row_label": fld.get("row_label"),
                    "parent_text": fld.get("parent_text"),
                    "n_boxes": fld.get("n_boxes", 1),
                    "bbox_px": list(self.bbox_pt_to_px(fld["bbox_pt"], pad=pad)),
                    "bbox_pt": fld["bbox_pt"],
                    "crop": crop,
                }
        return None

    def crop_boxes(self, img: np.ndarray, field_id_or_label: str, pad: int = 0) -> Optional[List[Dict[str, Any]]]:
        """
        Memotong kotak-kotak digit individual di dalam field tertentu.
        Mengembalikan list citra potongan per kotak digit dengan batas terkunci presisi fisik (Physical Cell Snapping).
        """
        cells_by_id = {c["id"]: c for c in self.template_data.get("cells", [])}
        for fld in self.template_data.get("fields", []):
            if fld["id"] == field_id_or_label or fld.get("row_label") == field_id_or_label:
                results = []
                for box_idx, cid in enumerate(fld.get("boxes", [])):
                    c = cells_by_id.get(cid)
                    if c is None:
                        continue
                    b_px = self.get_cell_bbox_px(cid, pad=pad)
                    if b_px is not None:
                        x0, y0, x1, y1 = b_px
                        crop = img[y0:y1, x0:x1].copy()
                    else:
                        crop = self.crop_bbox(img, c["bbox_pt"], pad=pad)
                        b_px = self.bbox_pt_to_px(c["bbox_pt"], pad=pad)
                    results.append({
                        "box_index": box_idx,
                        "cell_id": cid,
                        "bbox_px": list(b_px),
                        "bbox_pt": c["bbox_pt"],
                        "crop": crop,
                    })
                return results
        return None

    def crop_all_fields(self, img: np.ndarray, include_boxes: bool = False, pad: int = 0) -> List[Dict[str, Any]]:
        """Memotong seluruh field isian pada halaman dengan presisi fisik."""
        all_crops = []
        cells_by_id = {c["id"]: c for c in self.template_data.get("cells", [])} if include_boxes else {}

        for fld in self.template_data.get("fields", []):
            boxes = fld.get("boxes", [])
            if boxes and self.snapped_cells_px and all(b in self.snapped_cells_px for b in boxes):
                snapped_boxes = [self.snapped_cells_px[b] for b in boxes]
                bx0 = max(0, min(b[0] for b in snapped_boxes) - pad)
                by0 = max(0, min(b[1] for b in snapped_boxes) - pad)
                bx1 = min(self.img_width, max(b[2] for b in snapped_boxes) + pad)
                by1 = min(self.img_height, max(b[3] for b in snapped_boxes) + pad)
                f_crop = img[by0:by1, bx0:bx1].copy()
                f_bbox_px = [bx0, by0, bx1, by1]
            else:
                f_crop = self.crop_bbox(img, fld["bbox_pt"], pad=pad)
                f_bbox_px = list(self.bbox_pt_to_px(fld["bbox_pt"], pad=pad))

            item = {
                "field_id": fld["id"],
                "row_label": fld.get("row_label"),
                "parent_text": fld.get("parent_text"),
                "n_boxes": fld.get("n_boxes", 1),
                "bbox_px": f_bbox_px,
                "bbox_pt": fld["bbox_pt"],
                "crop": f_crop,
            }
            if include_boxes:
                boxes = []
                for box_idx, cid in enumerate(fld.get("boxes", [])):
                    c = cells_by_id.get(cid)
                    if c is not None:
                        b_px = self.get_cell_bbox_px(cid, pad=pad)
                        if b_px is not None:
                            x0, y0, x1, y1 = b_px
                            b_crop = img[y0:y1, x0:x1].copy()
                        else:
                            b_crop = self.crop_bbox(img, c["bbox_pt"], pad=pad)
                            b_px = self.bbox_pt_to_px(c["bbox_pt"], pad=pad)
                        boxes.append({
                            "box_index": box_idx,
                            "cell_id": cid,
                            "bbox_px": list(b_px),
                            "crop": b_crop,
                        })
                item["boxes"] = boxes
            all_crops.append(item)
        return all_crops

    def get_blocks(self) -> List[Dict[str, Any]]:
        """Mendapatkan daftar seluruh blok makro yang terdaftar pada halaman ini."""
        return self.template_data.get("blocks", [])

    def crop_block(self, img: np.ndarray, block_id_or_code: str, pad: int = 0) -> Optional[Dict[str, Any]]:
        """
        Memotong area citra satu Blok Makro tertentu (misal 'blok_1' atau 'BLOK I').
        Mengembalikan dictionary berisi citra crop dan metadata blok.
        """
        for blk in self.template_data.get("blocks", []):
            if blk["id"] == block_id_or_code or blk.get("code") == block_id_or_code:
                crop = self.crop_bbox(img, blk["bbox_pt"], pad=pad)
                return {
                    "block_id": blk["id"],
                    "code": blk.get("code"),
                    "name": blk.get("name"),
                    "bbox_px": list(self.bbox_pt_to_px(blk["bbox_pt"], pad=pad)),
                    "bbox_pt": blk["bbox_pt"],
                    "crop": crop,
                }
        return None

    def crop_all_blocks(self, img: np.ndarray, pad: int = 0) -> List[Dict[str, Any]]:
        """Memotong seluruh blok makro kuesioner pada halaman ini."""
        results = []
        for blk in self.template_data.get("blocks", []):
            crop = self.crop_bbox(img, blk["bbox_pt"], pad=pad)
            results.append({
                "block_id": blk["id"],
                "code": blk.get("code"),
                "name": blk.get("name"),
                "bbox_px": list(self.bbox_pt_to_px(blk["bbox_pt"], pad=pad)),
                "bbox_pt": blk["bbox_pt"],
                "crop": crop,
            })
        return results

    def crop_all_table_envelopes(self, img: np.ndarray, pad: int = 0) -> List[Dict[str, Any]]:
        """
        Memotong seluruh amplop kisi fisik tabel terluar pada halaman ini (Precise Grid).
        Menjamin setiap crop tabel berbentuk segi empat utuh dan terpotong presisi sesuai garis batas fisik (Zero-Cut Invariant).
        """
        envelopes = self.get_snapped_table_envelopes(img=img)
        results = []
        for env in envelopes:
            x0, y0, x1, y1 = env["bbox_px"]
            if pad != 0:
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(self.img_width, x1 + pad)
                y1 = min(self.img_height, y1 + pad)
            crop = img[y0:min(self.img_height, y1 + 1), x0:min(self.img_width, x1 + 1)].copy()

            # Buat juga citra visualisasi lattice kisi di dalam amplop ini (hanya sel struktural murni)
            lattice_crop = crop.copy()
            if len(lattice_crop.shape) == 2:
                lattice_crop = cv2.cvtColor(lattice_crop, cv2.COLOR_GRAY2BGR)

            table_grids = self.get_precise_table_grid_segments(img, struct_only=True)
            tg = next((g for g in table_grids if g["table_index"] == env["table_index"] - 1), None)
            struct_cells = tg["cells"] if tg else []
            for c_info in struct_cells:
                cx0, cy0, cx1, cy1 = c_info["bbox_px"]
                rel_x0 = max(0, cx0 - x0)
                rel_y0 = max(0, cy0 - y0)
                rel_x1 = min(x1 - x0, cx1 - x0)
                rel_y1 = min(y1 - y0, cy1 - y0)
                cv2.rectangle(lattice_crop, (rel_x0, rel_y0), (rel_x1, rel_y1), (0, 220, 0), 1)

            results.append({
                "table_index": env["table_index"],
                "bbox_px": [x0, y0, x1, y1],
                "bbox_pt": env["bbox_pt"],
                "width": x1 - x0,
                "height": y1 - y0,
                "crop": crop,
                "lattice_preview": lattice_crop,
                "cell_count": len(struct_cells)
            })
        return results

    def crop_all_table_rows(
        self,
        img: np.ndarray,
        pad: int = 0,
        snap_to_lines: bool = True,
        data_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Memotong seluruh baris (rows / roster lines) pada setiap sub-blok di halaman kuesioner.
        Setiap baris mengacu pada sub-blok induknya dan dibatasi secara horizontal oleh batas sub-blok (sb_x0 s.d. sb_x1).
        
        Karakteristik Utama:
        - Bounded by Sub-Block: Pada halaman multi-kolom (seperti Hal 7 Blok VI & VII), baris tidak melebar melintasi
          kolom sub-blok lain, melainkan terisolasi presisi per sub-blok (misal: R601-R607 vs R608-R612 terpisah).
        - Physical Line-Snapped (Zero-Cut Invariant): batas horizontal dan vertikal baris terkunci ke garis fisik.
        - Memetakan field, sel, dan label spesifik sub-blok yang berada di baris tersebut.
        - Menyertakan koordinat relatif field di dalam citra baris untuk VLM / annotator.
        """
        table_envelopes = self.get_snapped_table_envelopes(img=img)
        results = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        H, W = gray.shape[:2]

        h_lines = None
        v_lines = None
        if snap_to_lines:
            bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
            kh = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
            kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv)

        search_radius = 20
        all_fields = self.template_data.get("fields", [])
        all_cells = self.template_data.get("cells", [])
        tables = self.template_data.get("tables", [])

        # Iterasi seluruh sub-blok untuk menghasilkan baris yang mengacu ke sub-blok
        for sb in self.template_data.get("sub_blocks", []):
            sb_id = sb["id"]
            sb_code = sb.get("code") or sb_id
            sb_name = sb.get("name", "")
            sb_x0_pt, sb_y0_pt, sb_x1_pt, sb_y1_pt = sb["bbox_pt"]

            # Filter field yang menjadi milik sub-blok ini
            sb_field_ids = set(sb.get("fields", []))
            sb_fields = [f for f in all_fields if f["id"] in sb_field_ids]

            # Cari tabel yang tumpang tindih secara vertikal dengan sub-blok ini
            best_t = None
            best_ov = 0.0
            for t_idx, t in enumerate(tables):
                ty0, ty1 = t["bbox_pt"][1], t["bbox_pt"][3]
                inter = max(0.0, min(sb_y1_pt, ty1) - max(sb_y0_pt, ty0))
                if inter > 15 and inter > best_ov:
                    best_ov = inter
                    best_t = (t_idx, t)

            if not sb_fields and not best_t:
                continue

            # Hitung batas horizontal piksel sub-blok
            approx_sb_x0, approx_sb_y0, approx_sb_x1, approx_sb_y1 = self.bbox_pt_to_px(sb["bbox_pt"])

            if snap_to_lines and v_lines is not None:
                w_x0 = range(max(0, approx_sb_x0 - 25), min(W, approx_sb_x0 + 26))
                sc0 = [float(v_lines[max(0, approx_sb_y0 + 10):min(H, approx_sb_y1 - 10), x].sum()) for x in w_x0]
                sb_x0 = w_x0[int(np.argmax(sc0))] if (sc0 and max(sc0) > 0) else approx_sb_x0

                w_x1 = range(max(0, approx_sb_x1 - 25), min(W, approx_sb_x1 + 26))
                sc1 = [float(v_lines[max(0, approx_sb_y0 + 10):min(H, approx_sb_y1 - 10), x].sum()) for x in w_x1]
                sb_x1 = w_x1[int(np.argmax(sc1))] if (sc1 and max(sc1) > 0) else approx_sb_x1
            else:
                sb_x0, sb_x1 = approx_sb_x0, approx_sb_x1

            if sb_x1 <= sb_x0:
                sb_x0, sb_x1 = approx_sb_x0, approx_sb_x1

            # Jika terhubung ke tabel kisi fisik
            if best_t:
                t_idx, tbl = best_t
                env = next((e for e in table_envelopes if e["table_index"] == t_idx + 1), None)
                if env:
                    tx0, _, tx1, _ = env["bbox_px"]
                    sb_x0 = max(tx0, min(tx1, sb_x0))
                    sb_x1 = max(tx0, min(tx1, sb_x1))

                y_lines = [y for y in tbl.get("y_lines_pt", []) if sb_y0_pt - 4 <= y <= sb_y1_pt + 4]
                if len(y_lines) < 2:
                    continue

                for r_idx in range(len(y_lines) - 1):
                    y_top_pt = y_lines[r_idx]
                    y_bot_pt = y_lines[r_idx + 1]

                    # Cari sel di dalam baris sub-blok ini
                    row_cells = [
                        c for c in all_cells
                        if (sb_x0_pt - 5 <= (c["bbox_pt"][0] + c["bbox_pt"][2]) / 2.0 <= sb_x1_pt + 5 and
                            y_top_pt - 3 <= (c["bbox_pt"][1] + c["bbox_pt"][3]) / 2.0 <= y_bot_pt + 3)
                    ]

                    # Cari fields di dalam baris sub-blok ini
                    row_fields = [
                        f for f in sb_fields
                        if y_top_pt - 3 <= (f["bbox_pt"][1] + f["bbox_pt"][3]) / 2.0 <= y_bot_pt + 3
                    ]

                    texts = [c.get("text", "").strip() for c in row_cells if c.get("text")]
                    is_total = any("JUMLAH" in t.upper() or "TOTAL" in t.upper() for t in texts)
                    if is_total:
                        row_type = "total"
                    elif row_fields or any(c.get("kind") in ("comb", "box", "check") for c in row_cells):
                        row_type = "data"
                    else:
                        row_type = "header"

                    if data_only and row_type == "header":
                        continue

                    # Tentukan row_label terbaik
                    row_label = ""
                    for f in row_fields:
                        if f.get("row_label"):
                            cand = str(f["row_label"]).strip()
                            if cand and not all(c in "….-_ " for c in cand):
                                row_label = cand
                                break
                    if not row_label and texts:
                        for t in texts:
                            t_clean = t.strip()
                            if t_clean and not all(c in "….-_ " for c in t_clean):
                                if len(t_clean) <= 25 and (t_clean.isdigit() or t_clean.startswith("R") or t_clean.startswith("0")):
                                    row_label = t_clean
                                    break
                        if not row_label:
                            for t in texts:
                                t_clean = t.strip()
                                if t_clean and not all(c in "….-_ " for c in t_clean):
                                    row_label = t_clean[:25]
                                    break
                    if not row_label:
                        row_label = f"row_{r_idx:02d}"

                    # Koordinat piksel approximate dari affine
                    approx_x0, approx_y0, approx_x1, approx_y1 = self.bbox_pt_to_px(
                        [sb_x0_pt, y_top_pt, sb_x1_pt, y_bot_pt]
                    )
                    x0, x1 = sb_x0, sb_x1

                    if snap_to_lines and h_lines is not None:
                        w_y0 = range(max(0, approx_y0 - search_radius), min(H, approx_y0 + search_radius + 1))
                        sc_y0 = [float(h_lines[y, max(0, x0 + 15):min(W, x1 - 15)].sum()) for y in w_y0]
                        snapped_y0 = w_y0[int(np.argmax(sc_y0))] if (sc_y0 and max(sc_y0) > 0) else approx_y0

                        w_y1 = range(max(0, approx_y1 - search_radius), min(H, approx_y1 + search_radius + 1))
                        sc_y1 = [float(h_lines[y, max(0, x0 + 15):min(W, x1 - 15)].sum()) for y in w_y1]
                        snapped_y1 = w_y1[int(np.argmax(sc_y1))] if (sc_y1 and max(sc_y1) > 0) else approx_y1
                    else:
                        snapped_y0, snapped_y1 = approx_y0, approx_y1

                    if snapped_y1 <= snapped_y0:
                        snapped_y0, snapped_y1 = approx_y0, approx_y1

                    if pad != 0:
                        y0_c = max(0, snapped_y0 - pad)
                        y1_c = min(H, snapped_y1 + pad)
                        x0_c = max(0, x0 - pad)
                        x1_c = min(W, x1 + pad)
                    else:
                        y0_c, y1_c, x0_c, x1_c = snapped_y0, snapped_y1, x0, x1

                    if y1_c <= y0_c or x1_c <= x0_c:
                        continue

                    crop = img[y0_c:y1_c, x0_c:x1_c].copy()

                    # Koordinat relatif field di dalam potongan baris
                    field_details = []
                    for f in row_fields:
                        fb_px = self.bbox_pt_to_px(f["bbox_pt"])
                        rel_bx = [
                            max(0, fb_px[0] - x0_c),
                            max(0, fb_px[1] - y0_c),
                            min(x1_c - x0_c, fb_px[2] - x0_c),
                            min(y1_c - y0_c, fb_px[3] - y0_c)
                        ]
                        field_details.append({
                            "field_id": f["id"],
                            "row_label": f.get("row_label", ""),
                            "n_boxes": f.get("n_boxes", 1),
                            "rel_bbox_px": rel_bx,
                            "abs_bbox_px": fb_px
                        })

                    results.append({
                        "row_id": f"p{self.page_num}_{sb_id}_r{r_idx:02d}",
                        "sub_block_id": sb_id,
                        "sub_block_code": sb_code,
                        "sub_block_name": sb_name,
                        "table_index": t_idx,
                        "row_index": r_idx,
                        "row_type": row_type,
                        "row_label": row_label,
                        "description": " ".join(texts)[:60],
                        "bbox_px": [x0_c, y0_c, x1_c, y1_c],
                        "bbox_pt": [sb_x0_pt, y_top_pt, sb_x1_pt, y_bot_pt],
                        "dimensions": [crop.shape[1], crop.shape[0]],
                        "fields_count": len(row_fields),
                        "fields": [f["id"] for f in row_fields],
                        "field_details": field_details,
                        "cells_count": len(row_cells),
                        "crop": crop
                    })

            # Jika sub-blok tidak memiliki tabel fisik tapi memiliki field isian (seperti R470, Kesediaan, Waktu)
            elif sb_fields:
                # Urutkan field berdasarkan posisi vertikal (y)
                sb_fields_sorted = sorted(sb_fields, key=lambda f: (f["bbox_pt"][1], f["bbox_pt"][0]))
                for r_idx, f in enumerate(sb_fields_sorted):
                    fb_pt = f["bbox_pt"]
                    fb_px = self.bbox_pt_to_px(fb_pt)
                    lbl = f.get("row_label") or f"r{r_idx:02d}"

                    # Rentang vertikal baris field
                    approx_y0 = max(0, fb_px[1] - 8)
                    approx_y1 = min(H, fb_px[3] + 8)

                    if snap_to_lines and h_lines is not None:
                        w_y0 = range(max(0, approx_y0 - 10), min(H, approx_y0 + 11))
                        sc_y0 = [float(h_lines[y, max(0, sb_x0 + 10):min(W, sb_x1 - 10)].sum()) for y in w_y0]
                        snapped_y0 = w_y0[int(np.argmax(sc_y0))] if (sc_y0 and max(sc_y0) > 0) else approx_y0

                        w_y1 = range(max(0, approx_y1 - 10), min(H, approx_y1 + 11))
                        sc_y1 = [float(h_lines[y, max(0, sb_x0 + 10):min(W, sb_x1 - 10)].sum()) for y in w_y1]
                        snapped_y1 = w_y1[int(np.argmax(sc_y1))] if (sc_y1 and max(sc_y1) > 0) else approx_y1
                    else:
                        snapped_y0, snapped_y1 = approx_y0, approx_y1

                    if snapped_y1 <= snapped_y0:
                        snapped_y0, snapped_y1 = approx_y0, approx_y1

                    y0_c = max(0, snapped_y0 - pad) if pad != 0 else snapped_y0
                    y1_c = min(H, snapped_y1 + pad) if pad != 0 else snapped_y1
                    x0_c = max(0, sb_x0 - pad) if pad != 0 else sb_x0
                    x1_c = min(W, sb_x1 + pad) if pad != 0 else sb_x1

                    if y1_c <= y0_c or x1_c <= x0_c:
                        continue

                    crop = img[y0_c:y1_c, x0_c:x1_c].copy()

                    rel_bx = [
                        max(0, fb_px[0] - x0_c),
                        max(0, fb_px[1] - y0_c),
                        min(x1_c - x0_c, fb_px[2] - x0_c),
                        min(y1_c - y0_c, fb_px[3] - y0_c)
                    ]

                    results.append({
                        "row_id": f"p{self.page_num}_{sb_id}_r{r_idx:02d}",
                        "sub_block_id": sb_id,
                        "sub_block_code": sb_code,
                        "sub_block_name": sb_name,
                        "table_index": None,
                        "row_index": r_idx,
                        "row_type": "data",
                        "row_label": lbl,
                        "description": f.get("parent_text", "")[:60],
                        "bbox_px": [x0_c, y0_c, x1_c, y1_c],
                        "bbox_pt": [sb_x0_pt, fb_pt[1], sb_x1_pt, fb_pt[3]],
                        "dimensions": [crop.shape[1], crop.shape[0]],
                        "fields_count": 1,
                        "fields": [f["id"]],
                        "field_details": [{
                            "field_id": f["id"],
                            "row_label": lbl,
                            "n_boxes": f.get("n_boxes", 1),
                            "rel_bbox_px": rel_bx,
                            "abs_bbox_px": fb_px
                        }],
                        "cells_count": 0,
                        "crop": crop
                    })

        return results

    crop_all_sub_block_rows = crop_all_table_rows

    def crop_all_grid_cells(self, img: np.ndarray, pad: int = 0, struct_only: bool = True) -> List[Dict[str, Any]]:
        """
        Memotong seluruh sel kisi fisik individual terkecil (atomic grid cells) pada seluruh tabel.
        Secara default (struct_only=True), hanya mencakup sel struktural tabel (header, label, nomor rincian, kolom nilai/satuan)
        dan mengecualikan sel comb-box individual yang telah ditangani secara independen oleh layer mikro/digit.
        """
        results = []
        for c in self.template_data.get("cells", []):
            if struct_only and c.get("kind") == "digit":
                continue
            cid = c["id"]
            b_px = self.get_cell_bbox_px(cid, pad=pad)
            if b_px is None:
                b_px = self.bbox_pt_to_px(c["bbox_pt"], pad=pad)
            x0, y0, x1, y1 = b_px
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(self.img_width, x1)
            y1 = min(self.img_height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img[y0:y1, x0:x1].copy()
            t_idx = (c.get("table", 0) + 1) if c.get("table") is not None else 1
            results.append({
                "cell_id": cid,
                "table_index": t_idx,
                "row": c.get("row"),
                "col": c.get("col"),
                "rowspan": c.get("rowspan", 1),
                "colspan": c.get("colspan", 1),
                "kind": c.get("kind", "struct"),
                "fill": c.get("fill", "white"),
                "text": c.get("text", ""),
                "bbox_px": [x0, y0, x1, y1],
                "bbox_pt": c.get("bbox_pt"),
                "crop": crop
            })
        return results

    def crop_all_digit_boxes(self, img: np.ndarray, pad: int = 0) -> List[Dict[str, Any]]:
        """
        Memotong seluruh kotak digit individual (comb box) yang sah pada halaman.
        Hanya mencakup sel dengan kind == 'digit' dan mengabaikan sel abu-abu/palsu (fill_rgb < 245).
        Menyertakan metadata lengkap terstandarisasi: field_id, field_label, digit_label, box_index, dsb.
        """
        results = []
        for c in self.template_data.get("cells", []):
            if c.get("kind") != "digit":
                continue
            if min(c.get("fill_rgb", [255, 255, 255])) < 245:
                continue
            cid = c["id"]
            b_px = self.get_cell_bbox_px(cid, pad=pad)
            if b_px is None:
                b_px = self.bbox_pt_to_px(c["bbox_pt"], pad=pad)
            x0, y0, x1, y1 = b_px
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(self.img_width, x1)
            y1 = min(self.img_height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img[y0:y1, x0:x1].copy()
            results.append({
                "cell_id": cid,
                "field_id": c.get("field_id"),
                "field_label": c.get("field_label"),
                "digit_label": c.get("label", f"{cid}_digit"),
                "box_index": c.get("box_index", 0),
                "total_boxes": c.get("total_boxes", 1),
                "bbox_px": [x0, y0, x1, y1],
                "bbox_pt": c.get("bbox_pt"),
                "crop": crop
            })
        return results

    def get_sub_blocks(self) -> List[Dict[str, Any]]:
        """Mendapatkan daftar seluruh sub-blok yang terdaftar pada halaman ini."""
        return self.template_data.get("sub_blocks", [])

    def crop_sub_block(self, img: np.ndarray, sub_block_id_or_code: str, pad: int = 0) -> Optional[Dict[str, Any]]:
        """
        Memotong area citra satu Sub-Blok tertentu (misal 'sub_501' atau 'R501').
        Mengembalikan dictionary berisi citra crop dan metadata sub-blok.
        """
        for sb in self.template_data.get("sub_blocks", []):
            if sb["id"] == sub_block_id_or_code or sb.get("code") == sub_block_id_or_code:
                crop = self.crop_bbox(img, sb["bbox_pt"], pad=pad)
                return {
                    "sub_block_id": sb["id"],
                    "code": sb.get("code"),
                    "name": sb.get("name"),
                    "type": sb.get("type"),
                    "bbox_px": list(self.bbox_pt_to_px(sb["bbox_pt"], pad=pad)),
                    "bbox_pt": sb["bbox_pt"],
                    "cells_count": sb.get("cells_count", 0),
                    "fields_count": sb.get("fields_count", 0),
                    "digits_count": sb.get("digits_count", 0),
                    "fields": sb.get("fields", []),
                    "crop": crop,
                }
        return None

    def crop_all_sub_blocks(self, img: np.ndarray, pad: int = 0) -> List[Dict[str, Any]]:
        """Memotong seluruh sub-blok yang ada pada halaman ini."""
        results = []
        for sb in self.template_data.get("sub_blocks", []):
            crop = self.crop_bbox(img, sb["bbox_pt"], pad=pad)
            results.append({
                "sub_block_id": sb["id"],
                "code": sb.get("code"),
                "name": sb.get("name"),
                "type": sb.get("type"),
                "bbox_px": list(self.bbox_pt_to_px(sb["bbox_pt"], pad=pad)),
                "bbox_pt": sb["bbox_pt"],
                "cells_count": sb.get("cells_count", 0),
                "fields_count": sb.get("fields_count", 0),
                "digits_count": sb.get("digits_count", 0),
                "fields": sb.get("fields", []),
                "crop": crop,
            })
        return results

    def get_precise_table_grid_segments(
        self,
        img: Optional[np.ndarray] = None,
        struct_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Mengekstrak geometri kisi fisik presisi (envelope terluar dan koordinat sel 2D) untuk seluruh tabel.
        Secara default (struct_only=True), hanya mencakup sel struktural tabel (header, kolom, baris)
        dan mengecualikan sel digit comb-box agar kisi tabel bersih dan tidak tumpang-tindih dengan layer field.
        Mengadopsi tracing kontur fisik 2D dan perataan morfologis h_lines/v_lines untuk mengeliminasi drift.
        """
        target_img = img if img is not None else getattr(self, "raw_img", None)
        all_cells = self.template_data.get("cells", [])
        results = []

        table_envelopes = self.get_snapped_table_envelopes(target_img) if target_img is not None else []

        for t_idx, t in enumerate(self.template_data.get("tables", [])):
            tx0_pt, ty0_pt, tx1_pt, ty1_pt = t["bbox_pt"]
            env = next((e for e in table_envelopes if e["table_index"] == t_idx + 1), None)
            if env:
                tx0, ty0, tx1, ty1 = env["bbox_px"]
            elif target_img is not None:
                tx0, ty0, tx1, ty1 = self.snap_table_envelope_to_lines(target_img, self.bbox_pt_to_px(t["bbox_pt"]))
            else:
                tx0, ty0, tx1, ty1 = self.bbox_pt_to_px(t["bbox_pt"])

            t_cells = [
                c for c in all_cells
                if (not struct_only or c.get("kind") != "digit") and (
                    c.get("table") == t_idx or (
                        tx0_pt - 5 <= (c["bbox_pt"][0] + c["bbox_pt"][2]) / 2.0 <= tx1_pt + 5 and
                        ty0_pt - 5 <= (c["bbox_pt"][1] + c["bbox_pt"][3]) / 2.0 <= ty1_pt + 5
                    )
                )
            ]

            cell_boxes = []
            for c in t_cells:
                cid = c["id"]
                if self.snapped_cells_px and cid in self.snapped_cells_px:
                    cx0, cy0, cx1, cy1 = self.snapped_cells_px[cid]
                else:
                    cx0, cy0, cx1, cy1 = self.bbox_pt_to_px(c["bbox_pt"])
                cx0 = max(tx0, min(tx1, cx0))
                cy0 = max(ty0, min(ty1, cy0))
                cx1 = max(tx0, min(tx1, cx1))
                cy1 = max(ty0, min(ty1, cy1))
                if cx1 > cx0 and cy1 > cy0:
                    cell_boxes.append({
                        "id": cid,
                        "kind": c.get("kind"),
                        "bbox_px": (cx0, cy0, cx1, cy1)
                    })

            results.append({
                "table_index": t_idx,
                "bbox_pt": t["bbox_pt"],
                "envelope_px": (tx0, ty0, tx1, ty1),
                "cells": cell_boxes
            })
        return results

    def draw_alignment_overlay(
        self,
        img: np.ndarray,
        draw_tables: bool = True,
        draw_fields: bool = True,
        draw_labels: bool = True,
        draw_sub_blocks: bool = False,
        draw_text_fields: bool = True,
        struct_only_tables: bool = True,
        alpha: float = 0.55
    ) -> np.ndarray:
        """
        Menggambar overlay grid tabel dan field di atas citra pindaian untuk verifikasi visual.
        - Garis hijau: kisi tabel presisi berbasis tracing sel fisik 2D & envelope snap (murni sel struktural)
        - Kotak merah: field isian (crop areas) yang membungkus kotak digit secara presisi
        - Kotak cyan: bidang teks bebas / tulisan tangan (Nama KRT, Alamat, Roster ART, Catatan)
        - Label biru: kode rincian baris
        - Border ungu: batas sub-blok
        """
        overlay = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 0. Gambar batas Sub-Blok (jika diaktifkan)
        if draw_sub_blocks:
            for sb in self.template_data.get("sub_blocks", []):
                sx0, sy0, sx1, sy1 = self.bbox_pt_to_px(sb["bbox_pt"])
                cv2.rectangle(overlay, (sx0, sy0), (sx1, sy1), (180, 0, 180), 3)
                sb_badge = f"{sb.get('code')}: {sb.get('name')[:30]}"
                cv2.putText(
                    overlay, sb_badge, (sx0 + 5, sy0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 0, 180), 2, cv2.LINE_AA
                )

        # 1. Gambar kisi garis tabel (Tracing Sel 2D & Kisi Fisik)
        if draw_tables:
            if self.snapped_cells_px is not None:
                # Menggunakan tracing kisi sel fisik 2D presisi (murni sel struktural)
                table_grids = self.get_precise_table_grid_segments(img, struct_only=struct_only_tables)
                for tg in table_grids:
                    tx0, ty0, tx1, ty1 = tg["envelope_px"]
                    # Gambar amplop segi empat terluar (Outer Envelope)
                    cv2.rectangle(overlay, (tx0, ty0), (tx1, ty1), (0, 230, 0), 2)
                    # Gambar setiap batas sel fisik 2D interior (menjaga merged header/colspan tetap utuh)
                    for c_info in tg["cells"]:
                        cx0, cy0, cx1, cy1 = c_info["bbox_px"]
                        cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), (0, 230, 0), 1)
            else:
                # Fallback jika physical snapping dinonaktifkan
                for t in self.template_data.get("tables", []):
                    t_x0, t_y0, t_x1, t_y1 = self.bbox_pt_to_px(t["bbox_pt"])
                    for xl in t.get("x_lines_pt", []):
                        x_px, _ = self.pt_to_px(xl, 0)
                        cv2.line(overlay, (x_px, t_y0), (x_px, t_y1), (0, 255, 0), 1)
                    for yl in t.get("y_lines_pt", []):
                        _, y_px = self.pt_to_px(0, yl)
                        cv2.line(overlay, (t_x0, y_px), (t_x1, y_px), (0, 255, 0), 1)

        # 2. Gambar kotak field isian
        if draw_fields:
            for fld in self.template_data.get("fields", []):
                # Jika field memiliki boxes dan telah di-snap, bungkus seluruh kotak digitnya secara presisi
                fld_boxes = [
                    self.snapped_cells_px[b]
                    for b in fld.get("boxes", [])
                    if self.snapped_cells_px and b in self.snapped_cells_px
                ]
                if fld_boxes:
                    x0 = min(b[0] for b in fld_boxes)
                    y0 = min(b[1] for b in fld_boxes)
                    x1 = max(b[2] for b in fld_boxes)
                    y1 = max(b[3] for b in fld_boxes)
                else:
                    x0, y0, x1, y1 = self.bbox_pt_to_px(fld["bbox_pt"])

                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
                lbl = fld.get("row_label")
                if draw_labels and lbl:
                    cv2.putText(
                        overlay, lbl, (x0, max(y0 - 4, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA
                    )

        # 3. Gambar kotak text fields (Bidang Teks Bebas / Tulisan Tangan)
        if draw_text_fields:
            for tf in self.template_data.get("text_fields", []):
                cid = tf.get("cell_id")
                b_px = self.get_cell_bbox_px(cid) if cid else None
                if b_px is not None:
                    tx0, ty0, tx1, ty1 = b_px
                else:
                    tx0, ty0, tx1, ty1 = self.bbox_pt_to_px(tf.get("bbox_pt", [0, 0, 0, 0]), pad=2)

                # Warna Cyan / Teal (212, 182, 6) BGR
                cv2.rectangle(overlay, (tx0, ty0), (tx1, ty1), (212, 182, 6), 2)
                lbl = tf.get("question_code") or tf.get("label") or tf.get("id")
                if draw_labels and lbl:
                    cv2.putText(
                        overlay, f"[TEKS] {lbl}", (tx0, max(ty0 - 4, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (212, 182, 6), 2, cv2.LINE_AA
                    )

        # Blend transparan
        base_bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(overlay, alpha, base_bgr, 1.0 - alpha, 0)


def detect_scanned_table_frame(
    gray: np.ndarray,
    t_w_pt: Optional[float] = None,
    t_h_pt: Optional[float] = None
) -> Tuple[int, int, int, int]:
    """
    Mendeteksi batas luar kisi tabel pada citra pindaian monokrom
    menggunakan binarisasi Otsu adaptif dan proyeksi akumulasi profil garis morfologis.
    Dilengkapi anti-dummy centering & asymmetric edge anchoring untuk mengatasi distorsi lipatan tengah (gutter).
    Mengembalikan (first_x, first_y, last_x, last_y) dalam piksel.
    """
    H, W = gray.shape[:2]

    # 1. Binarisasi Otsu Adaptif dengan batas pengaman [160, 225]
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_used = max(160, min(225, int(otsu_val)))
    _, bw = cv2.threshold(gray, thresh_used, 255, cv2.THRESH_BINARY_INV)

    # 2. Kernel morfologi garis horizontal dan vertikal
    # Menggunakan kernel vertikal adaptif (40-45 px) agar tahan terhadap kemiringan dan kelengkungan gutter
    kh_len = max(20, int(W * 0.04))
    kv_len = max(20, min(45, int(H * 0.015)))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (kh_len, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv_len))

    lines_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
    lines_v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv)

    row_sums = np.sum(lines_h > 0, axis=1)
    col_sums = np.sum(lines_v > 0, axis=0)

    # Margin guard: buang noise tepi scanner, punch hole, dan roller (< 90 px tepi)
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

    # Deteksi X dengan Anti-Dummy Centering & Asymmetric Edge Anchoring
    if len(x_peaks) == 0:
        first_x = max(90, int((W - exp_w) / 2))
        last_x = min(W - 80, first_x + exp_w)
    else:
        fx, lx = int(x_peaks[0]), int(x_peaks[-1])
        min_exp_w = int(t_w_pt * 3.0) if (t_w_pt and t_w_pt > 0) else int(W * 0.55)
        if (lx - fx) < min_exp_w:
            # Salah satu tepi terhalang noise/gutter: jangkar pada tepi yang terdeteksi pasti
            if lx > (W - int(W * 0.25)):  # Tepi kanan kuat!
                last_x = lx
                first_x = max(90, last_x - exp_w)
            elif fx < int(W * 0.25):  # Tepi kiri kuat!
                first_x = fx
                last_x = min(W - 80, first_x + exp_w)
            else:
                first_x = max(90, int((W - exp_w) / 2))
                last_x = min(W - 80, first_x + exp_w)
        else:
            first_x, last_x = fx, lx

    # Deteksi Y dengan Anti-Dummy Centering & Asymmetric Edge Anchoring
    if len(y_peaks) == 0:
        first_y = max(80, int((H - exp_h) / 2))
        last_y = min(H - 70, first_y + exp_h)
    else:
        fy, ly = int(y_peaks[0]), int(y_peaks[-1])
        min_exp_h = int(t_h_pt * 3.0) if (t_h_pt and t_h_pt > 0) else int(H * 0.55)
        if (ly - fy) < min_exp_h:
            if ly > (H - int(H * 0.20)):  # Tepi bawah kuat!
                last_y = ly
                first_y = max(80, last_y - exp_h)
            elif fy < int(H * 0.20):  # Tepi atas kuat!
                first_y = fy
                last_y = min(H - 70, first_y + exp_h)
            else:
                first_y = max(80, int((H - exp_h) / 2))
                last_y = min(H - 70, first_y + exp_h)
        else:
            first_y, last_y = fy, ly

    # 3. Guardrail Kesetaraan Skala X dan Y jika t_w_pt & t_h_pt disediakan
    if t_w_pt and t_h_pt and t_w_pt > 0 and t_h_pt > 0:
        h_px = last_y - first_y
        sy = h_px / float(t_h_pt)
        if sy < 3.2 or sy > 4.5:
            sy = 3.8889  # Fallback 300 DPI kanonikal
        expected_w = int(round(sy * t_w_pt))
        w_px = last_x - first_x

        # Jika deviasi lebar melampaui 4% terhadap expected_w
        if abs(w_px - expected_w) / float(expected_w) > 0.04:
            if last_x > (W - int(W * 0.25)):
                first_x = max(90, last_x - expected_w)
            elif first_x < int(W * 0.25):
                last_x = min(W - 80, first_x + expected_w)
            elif 110 <= first_x <= 230:
                last_x = first_x + expected_w
            elif 110 <= (last_x - expected_w) <= 230:
                first_x = last_x - expected_w
            else:
                first_x = int((W - expected_w) / 2)
                last_x = first_x + expected_w

    return first_x, first_y, last_x, last_y


def get_physical_grid_envelope_pt(tpl_data: Dict[str, Any], page_num: int) -> Tuple[float, float, float, float]:
    """
    Mendapatkan bounding box (x0, y0, x1, y1) dalam pt dari struktur fisik
    kuesioner yang benar-benar memiliki garis pembatas horizontal dan vertikal penuh.
    Menghindari distorsi akibat elemen terbuka/parsial tanpa garis lebar
    (misal: Kesediaan pada Hal 1, Waktu Selesai pada Hal 7, dan Blok Nomor Halaman).
    """
    content_blocks = [b for b in tpl_data.get("blocks", []) if "halaman" not in b.get("id", "")]
    bounded_blocks = []
    for b in content_blocks:
        bid = b.get("id", "")
        if page_num == 1 and ("kesediaan" in bid or "header" in bid):
            continue
        if page_num == 7 and "waktu_selesai" in bid:
            continue
        bounded_blocks.append(b)

    if not bounded_blocks:
        bounded_blocks = content_blocks or tpl_data.get("tables", [])

    t_x0 = min(b["bbox_pt"][0] for b in bounded_blocks)
    t_y0 = min(b["bbox_pt"][1] for b in bounded_blocks)
    t_x1 = max(b["bbox_pt"][2] for b in bounded_blocks)
    t_y1 = max(b["bbox_pt"][3] for b in bounded_blocks)
    return t_x0, t_y0, t_x1, t_y1


def align_template_to_image(
    img: np.ndarray,
    page_num: int,
    template_dir: Optional[Union[str, Path]] = None,
    use_lattice_anchors: bool = True,
    use_physical_snapping: bool = True
) -> GridAlignment:
    """
    Menyelaraskan template grid JSON terhadap citra pindaian halaman tertentu.
    Mendukung fusi hibrida ganda:
    1. Template Grid Vektor + Morphological Lattice Projections (Neighbor-Guided 1D Anchors).
    2. Physical Cell Snapping 2D (End-to-End Precise Grid) untuk mengunci comb-box ke batas fisik 0 px.
    
    Args:
        img: Citra pindaian (BGR atau Grayscale).
        page_num: Nomor halaman kuesioner (1 s.d. 7).
        template_dir: Direktori berisi template_p{X}.json (default: templates/).
        use_lattice_anchors: Apakah menggunakan anchoring kisi morfologis 1D (default: True).
        use_physical_snapping: Apakah mengunci batas comb-box langsung ke kontur fisik 2D (default: True).
        
    Returns:
        Objek GridAlignment terkonfigurasi lengkap.
    """
    if template_dir is None:
        template_dir = DEFAULT_TEMPLATE_DIR
    template_dir = Path(template_dir)

    template_file = template_dir / f"template_p{page_num}.json"
    if not template_file.exists():
        raise FileNotFoundError(f"Template grid tidak ditemukan: {template_file}")

    with open(template_file, "r", encoding="utf-8") as f:
        tpl_data = json.load(f)

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # 1. Ambil batas luar konten fisik bergaris dari template PDF
    t_x0, t_y0, t_x1, t_y1 = get_physical_grid_envelope_pt(tpl_data, page_num)
    t_w_pt = max(1.0, t_x1 - t_x0)
    t_h_pt = max(1.0, t_y1 - t_y0)

    # 2. Deteksi bingkai tabel pada citra pindaian dengan panduan aspek rasio template
    s_x0, s_y0, s_x1, s_y1 = detect_scanned_table_frame(gray, t_w_pt=t_w_pt, t_h_pt=t_h_pt)

    # Kalibrasi awal untuk Halaman 1 jika s_y0 menangkap garis tipis kop logo/judul
    if page_num == 1 and s_y0 < 800:
        sx_init = (s_x1 - s_x0) / max(1.0, (t_x1 - t_x0))
        exp_y0 = int(round(s_y1 - sx_init * (t_y1 - t_y0)))
        w0 = max(0, exp_y0 - 80)
        w1 = min(H, exp_y0 + 80)
        _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.15), 1))
        lines_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
        row_sums = np.sum(lines_h > 0, axis=1)
        pks = np.where(row_sums[w0:w1] > int(0.12 * W))[0]
        if len(pks) > 0:
            s_y0 = w0 + int(pks[0])

    # 3. Hitung transformasi affine presisi dengan aspect-ratio guardrail
    sx = (s_x1 - s_x0) / max(1.0, (t_x1 - t_x0))
    raw_sy = (s_y1 - s_y0) / max(1.0, (t_y1 - t_y0))

    # Guardrail rasio aspek (maksimal toleransi 4% perbedaan antara sx dan sy)
    if abs(raw_sy - sx) / sx > 0.04:
        expected_y0 = int(round(s_y1 - sx * (t_y1 - t_y0)))
        _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.10), 1))
        lines_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
        row_sums = np.sum(lines_h > 0, axis=1)

        win_y0 = max(0, expected_y0 - 80)
        win_y1 = min(H, expected_y0 + 80)
        local_peaks = np.where(row_sums[win_y0:win_y1] > int(0.12 * W))[0]
        if len(local_peaks) > 0:
            s_y0 = win_y0 + int(local_peaks[0])
            sy = (s_y1 - s_y0) / max(1.0, (t_y1 - t_y0))
        else:
            sy = sx
            s_y0 = expected_y0
    else:
        sy = raw_sy

    tx = s_x0 - sx * t_x0
    ty = s_y0 - sy * t_y0

    warp_x_anchors = None
    warp_y_anchors = None

    # 4. Integrasi Fusi Morphological Lattice Projections sebagai Anchors
    if use_lattice_anchors:
        _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        # Multi-scale dynamic morphological kernels
        kh_len = max(16, int(W * 0.035))
        kv_len = max(16, int(H * 0.018))
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (kh_len, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv_len))
        lines_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh)
        lines_v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv)
        row_sums = np.sum(lines_h > 0, axis=1)
        col_sums = np.sum(lines_v > 0, axis=0)

        content_blocks = [b for b in tpl_data.get("blocks", []) if "halaman" not in b.get("id", "")]

        # Template X lines (garis vertikal tabel dan kolom)
        tpl_x = set()
        for t in tpl_data.get("tables", []):
            for x in t.get("x_lines_pt", []):
                tpl_x.add(round(x, 2))
        for b in content_blocks:
            tpl_x.add(round(b["bbox_pt"][0], 2))
            tpl_x.add(round(b["bbox_pt"][2], 2))
        tpl_x = sorted([x for x in tpl_x if t_x0 < x < t_x1])

        # Predictive Neighbor-Guided Tracking untuk Sumbu X
        matched_tx = [t_x0]
        matched_ax = [s_x0]
        current_sx = sx

        for x_pt in tpl_x:
            d_pt = x_pt - matched_tx[-1]
            xp_global = int(round(x_pt * sx + tx))
            xp_neighbor = int(round(matched_ax[-1] + d_pt * current_sx))
            xp = int(round(0.5 * xp_global + 0.5 * xp_neighbor))
            rad = max(6, min(18, int(round(d_pt * current_sx * 0.40))))

            w0 = max(s_x0, xp - rad)
            w1 = min(s_x1, xp + rad + 1)
            if w1 > w0:
                sub = col_sums[w0:w1]
                if len(sub) > 0 and np.max(sub) > int(0.03 * H):
                    peak_idx = w0 + int(np.argmax(sub))
                    if peak_idx > matched_ax[-1] + 2:
                        step_sx = (peak_idx - matched_ax[-1]) / max(1e-3, d_pt)
                        current_sx = 0.7 * current_sx + 0.3 * step_sx
                        matched_tx.append(x_pt)
                        matched_ax.append(peak_idx)
        matched_tx.append(t_x1)
        matched_ax.append(s_x1)

        # Optical Rigidity Guardrail untuk sumbu X:
        # Sensor optik pemindai pada sumbu ortogonal terhadap ADF feed bersifat kaku (rigid).
        # Tolak anchor yang menyimpang > 14 px dari transformasi afin linier (menghindari false-lock).
        is_portrait = H > W
        clean_tx, clean_ax = [matched_tx[0]], [matched_ax[0]]
        for i in range(1, len(matched_tx) - 1):
            x_pt = matched_tx[i]
            a_px = matched_ax[i]
            linear_px = x_pt * sx + tx
            max_allowed = 14.0 if is_portrait else 18.0
            if abs(a_px - linear_px) <= max_allowed:
                clean_tx.append(x_pt)
                clean_ax.append(a_px)
        clean_tx.append(matched_tx[-1])
        clean_ax.append(matched_ax[-1])

        # Template Y lines (garis horizontal tabel dan pemisah baris)
        # 1. Khusus halaman ber-tabel pada orientasi portrait:
        # Gunakan Table-Specific Morphological Line Extraction (Adaptive Threshold + Wide Kernel)
        # untuk mengisolasi murni garis kisi fisik tabel secara presisi tanpa terganggu comb-box/teks.
        is_portrait = H > W
        tables = tpl_data.get("tables", [])
        table_anchors_found = False

        if is_portrait and tables:
            th_table = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
            all_tx0, all_tx1 = [], []
            table_y_pts = set()
            for t in tables:
                x_pts = t.get("x_lines_pt", [])
                if x_pts:
                    all_tx0.append(min(x_pts))
                    all_tx1.append(max(x_pts))
                for y in t.get("y_lines_pt", []):
                    table_y_pts.add(round(y, 2))
            if all_tx0 and all_tx1:
                tbl_x0_px = max(0, int(min(all_tx0) * sx + tx))
                tbl_x1_px = min(W, int(max(all_tx1) * sx + tx))
                tw = max(100, tbl_x1_px - tbl_x0_px)
                k_table_len = max(140, int(tw * 0.08))
                k_table = cv2.getStructuringElement(cv2.MORPH_RECT, (k_table_len, 1))
                h_table = cv2.morphologyEx(th_table, cv2.MORPH_OPEN, k_table)
                table_row_sums = (h_table[:, tbl_x0_px:tbl_x1_px] > 0).sum(axis=1)
                min_w = int(tw * 0.25)
                phys_table_lines = []
                in_pk = False
                st = 0
                for y, s in enumerate(table_row_sums):
                    if s > min_w:
                        if not in_pk:
                            in_pk = True
                            st = y
                    else:
                        if in_pk:
                            in_pk = False
                            phys_table_lines.append(st + int(np.argmax(table_row_sums[st:y])))
                sorted_ty = sorted(table_y_pts)
                tbl_matched_ty = []
                tbl_matched_ay = []
                curr_idx = 0
                cur_sy = sy
                cur_offset = ty
                for y_pt in sorted_ty:
                    exp_y = y_pt * cur_sy + cur_offset
                    best_idx = None
                    best_dist = 9999
                    for pi in range(curr_idx, min(len(phys_table_lines), curr_idx + 3)):
                        d = abs(phys_table_lines[pi] - exp_y)
                        if d < best_dist:
                            best_dist = d
                            best_idx = pi
                    if best_idx is not None and best_dist < 40:
                        tbl_matched_ty.append(y_pt)
                        tbl_matched_ay.append(phys_table_lines[best_idx])
                        curr_idx = best_idx + 1
                        cur_offset = 0.7 * cur_offset + 0.3 * (phys_table_lines[best_idx] - y_pt * cur_sy)

                if len(tbl_matched_ty) >= max(3, int(0.6 * len(sorted_ty))):
                    clean_ty = [t_y0] + tbl_matched_ty + [t_y1]
                    clean_ay = [s_y0] + tbl_matched_ay + [s_y1]
                    table_anchors_found = True

        if not table_anchors_found:
            tpl_y = set()
            for t in tpl_data.get("tables", []):
                for y in t.get("y_lines_pt", []):
                    tpl_y.add(round(y, 2))
            for b in content_blocks:
                tpl_y.add(round(b["bbox_pt"][1], 2))
                tpl_y.add(round(b["bbox_pt"][3], 2))
            tpl_y = sorted([y for y in tpl_y if t_y0 < y < t_y1])

            # Predictive Neighbor-Guided Tracking untuk Sumbu Y
            matched_ty = [t_y0]
            matched_ay = [s_y0]
            current_sy = sy

            for y_pt in tpl_y:
                d_pt = y_pt - matched_ty[-1]
                yp_global = int(round(y_pt * sy + ty))
                yp_neighbor = int(round(matched_ay[-1] + d_pt * current_sy))
                yp = int(round(0.5 * yp_global + 0.5 * yp_neighbor))
                rad = max(6, min(20, int(round(d_pt * current_sy * 0.40))))

                w0 = max(s_y0, yp - rad)
                w1 = min(s_y1, yp + rad + 1)
                if w1 > w0:
                    sub = row_sums[w0:w1]
                    if len(sub) > 0 and np.max(sub) > int(0.04 * W):
                        peak_idx = w0 + int(np.argmax(sub))
                        if peak_idx > matched_ay[-1] + 2:
                            step_sy = (peak_idx - matched_ay[-1]) / max(1e-3, d_pt)
                            current_sy = 0.7 * current_sy + 0.3 * step_sy
                            matched_ty.append(y_pt)
                            matched_ay.append(peak_idx)
            matched_ty.append(t_y1)
            matched_ay.append(s_y1)

            # Optical Rigidity Guardrail untuk sumbu Y pada landscape:
            clean_ty, clean_ay = [matched_ty[0]], [matched_ay[0]]
            for i in range(1, len(matched_ty) - 1):
                y_pt = matched_ty[i]
                a_py = matched_ay[i]
                linear_py = y_pt * sy + ty
                max_allowed = 45.0 if is_portrait else 16.0  # Portrait ADF slip diizinkan hingga 45 px (Hal 3)
                if abs(a_py - linear_py) <= max_allowed:
                    clean_ty.append(y_pt)
                    clean_ay.append(a_py)
            clean_ty.append(matched_ty[-1])
            clean_ay.append(matched_ay[-1])

        if len(clean_tx) >= 2:
            warp_x_anchors = (clean_tx, clean_ax)
        if len(clean_ty) >= 2:
            warp_y_anchors = (clean_ty, clean_ay)

    size_pt_tuple = (float(tpl_data["size_pt"][0]), float(tpl_data["size_pt"][1]))

    # Estimasi residual skew untuk diagnostik
    deskew_angle = 0.0
    if page_num in (5, 6, 7):
        from .orientation_handler import estimate_skew_angle
        deskew_angle = estimate_skew_angle(gray, max_angle_deg=3.0, is_landscape=True)

    alignment = GridAlignment(
        page_num=page_num,
        img_width=W,
        img_height=H,
        template_size_pt=size_pt_tuple,
        scale_x=sx,
        scale_y=sy,
        trans_x=tx,
        trans_y=ty,
        scan_bounds=(s_x0, s_y0, s_x1, s_y1),
        template_bounds=(t_x0, t_y0, t_x1, t_y1),
        template_data=tpl_data,
        warp_x_anchors=warp_x_anchors,
        warp_y_anchors=warp_y_anchors,
        deskew_angle=deskew_angle,
    )

    if use_physical_snapping:
        alignment.enable_physical_cell_snapping(gray)

    return alignment


def align_and_deskew_image(
    img: np.ndarray,
    page_num: int,
    template_dir: Optional[Union[str, Path]] = None,
    use_lattice_anchors: bool = True,
    use_physical_snapping: bool = True
) -> Tuple[np.ndarray, GridAlignment]:
    """
    Menyelaraskan citra pindaian dengan pra-koreksi kemiringan (deskew) otomatis jika terdeteksi skew.
    Mengembalikan tuple (deskewed_image, alignment).
    """
    from .orientation_handler import estimate_skew_angle, rotate_image
    H, W = img.shape[:2]
    is_land = (W > H)
    ang = estimate_skew_angle(img, max_angle_deg=3.5, is_landscape=is_land)
    if abs(ang) >= 0.025:
        img_deskewed = rotate_image(img, ang, interp=cv2.INTER_CUBIC)
    else:
        img_deskewed = img

    al = align_template_to_image(
        img_deskewed,
        page_num=page_num,
        template_dir=template_dir,
        use_lattice_anchors=use_lattice_anchors,
        use_physical_snapping=use_physical_snapping
    )
    al.deskew_angle = ang
    return img_deskewed, al
