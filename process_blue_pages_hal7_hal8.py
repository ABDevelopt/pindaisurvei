#!/usr/bin/env python3
"""
process_blue_pages_hal7_hal8.py
===============================
Skrip Pemrosesan Halaman Kertas Biru (Halaman 7 dan Halaman 8)
Kuesioner BPS Susenas Seruti Inti (VSERUTI26.INTI)

Fokus:
1. Akurasi & Presisi Maksimal Pembacaan Kuesioner (Presisi > Latensi).
2. Isolasi Kanal Biru (I = B) & Adaptive Illumination Flat-Fielding (Background Division)
   untuk mengubah kertas biru berserat/berbayang menjadi latar belakang 255 putih salju murni.
3. Penajaman Kontras S-Curve LUT & CamScanner Magic Filter (White Point Snapping >= 232 -> 255).
4. Penanganan Halaman 7 (Blok VI & VII):
   - Auto-Deskew sub-derajat presisi (< 5.0 deg).
   - Auto-Rotate 90° Searah Jarum Jam (Portrait 3507x2480 -> Landscape 2480x3507).
   - Hybrid Split Detection pada Banner 'BLOK VII. CATATAN' (y ~ 1700 px) untuk
     memisahkan Blok VI (Tabel Finansial Landscape) dari Blok VII (Catatan Lapangan).
   - Proteksi teks deskripsi Kolom 2 (x < 1200 px) & narasi Blok VII (y >= split_y).
   - Snapping kisi mikro sub-piksel (GroundTruth / Lattice Snapper) & ekstraksi comb-box digit.
   - Eliminasi artefak border tabel (Connected Components + Center-of-Mass Spatial Moments).
   - Batched ONNX CNN Digit Inference & perakitan nilai nominal rupiah.
5. Penanganan Halaman 8 (Sampul Belakang):
   - Deteksi Sampul Kosong (Blank Cover Detection) melalui std deviasi (std < 2.5) dan rasio piksel gelap.
   - Jika kosong: Hasilkan kanvas 255 putih murni bebas derau scanner roller.
   - Jika berisi catatan/tanda tangan: Pemutihan latar biru & isolasi goresan tinta pensil/pulpen.
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np

# Impor konfigurasi & modul pipeline Seruti Inti
from seruti_inti.config import (
    HAL7_MARGIN_PX,
    DEFAULT_MARGIN_PX,
    HAL7_SPLIT_Y_DEFAULT,
    HAL7_SPLIT_Y_RATIO,
    PRIMARY_ONNX_PATH,
    FALLBACK_ONNX_PATHS,
    A4_CANONICAL_HEIGHT,
    A4_CANONICAL_WIDTH,
)
from seruti_pipeline.channel_extractor import (
    is_blue_paper,
    flatten_illumination_flat_fielding,
    apply_scurve_lut,
    apply_camscanner_magic_filter,
    clean_scanner_margins,
)
from seruti_pipeline.fold_seam_detector import split_spread_to_pages
from seruti_pipeline.orientation_handler import auto_deskew, detect_hal7_split_y
from seruti_pipeline.ground_truth_snapper import GroundTruthGuidedSnapper
from seruti_pipeline.cell_table_extractor import (
    standardize_cell_image,
    MorphologicalLatticeSnapper,
    get_default_onnx_session,
)


def _init_onnx_session(session: Optional[Any] = None) -> Optional[Any]:
    """Inisialisasi sesi ONNX Runtime untuk klasifikasi digit CNN."""
    if session is not None:
        return session
    try:
        import onnxruntime as ort
        candidates = [PRIMARY_ONNX_PATH] + FALLBACK_ONNX_PATHS
        for p in candidates:
            if p.exists():
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                return ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
    except Exception:
        pass
    return get_default_onnx_session()


def clean_blue_page_image(
    page_img: np.ndarray,
    page_id: str,
    margin_px: int = 65,
    unsharp_strength: float = 0.50,
    white_cutoff: int = 222,
    gamma: float = 2.0,
    contrast_mode: str = "high",
    binarize: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pipeline pemutihan kertas biru khusus & penajaman kontras tinggi (Hal 7 & Hal 8):
    1. Deteksi dominansi kanal biru (B - R > 12).
    2. Ekstraksi kanal Blue (I = B) untuk mengubah latar biru menjadi terang.
    3. Flat-Fielding (Morphological Closing Division) mengeliminasi bayangan scanner.
    4. S-Curve Dynamic Contrast LUT untuk mempertegas goresan pensil.
    5. CamScanner Magic Filter Mutakhir Berkontras Tinggi:
       - Dynamic Black Point Clamping (menarik tinta tergelap ke level jet-black 0-30).
       - Power-law gamma expansion (membuat tulisan tangan pensil 2B tebal, pekat, dan jelas).
       - High-frequency unsharp masking untuk ketajaman batas huruf/angka.
       - White Point Snapping (menghapus residu abu-abu latar menjadi 255 putih salju murni).
    6. Pembersihan margin perimeter scanner (65 px).
    """
    diag: Dict[str, Any] = {"page_id": page_id, "input_shape": list(page_img.shape), "contrast_mode": contrast_mode}

    if contrast_mode == "normal":
        gamma = 1.6
        bp_offset = 12.0
        unsharp_strength = 0.35
        white_cutoff = 228
    elif contrast_mode == "ultra":
        gamma = 2.4
        bp_offset = 35.0
        unsharp_strength = 0.65
        white_cutoff = 215
    else:  # "high" (default)
        gamma = 2.0
        bp_offset = 22.0
        unsharp_strength = 0.50
        white_cutoff = 222

    if len(page_img.shape) == 3:
        b_mean = float(np.mean(page_img[:, :, 0]))
        r_mean = float(np.mean(page_img[:, :, 2]))
        blue_diff = b_mean - r_mean
        is_blue, _ = is_blue_paper(page_img)
        diag.update({"mean_b": round(b_mean, 2), "mean_r": round(r_mean, 2), "blue_diff": round(blue_diff, 2), "is_blue_paper": is_blue})
        mono_raw = page_img[:, :, 0] if is_blue else cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    else:
        mono_raw = page_img.copy()
        diag.update({"is_blue_paper": False, "blue_diff": 0.0})

    # 1. Flat-fielding adaptif
    flat = flatten_illumination_flat_fielding(mono_raw, ksize=None)
    # 2. Magic White Point Snapping & Dynamic Contrast (Single-pass natural enhancement)
    magic = apply_camscanner_magic_filter(
        flat,
        unsharp_strength=unsharp_strength,
        white_cutoff=white_cutoff,
        gamma=gamma,
        bp_offset=bp_offset
    )

    if binarize:
        _, bin_img = cv2.threshold(magic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        clean = clean_scanner_margins(bin_img, margin_px=margin_px)
    else:
        clean = clean_scanner_margins(magic, margin_px=margin_px)

    diag["output_mean"] = round(float(np.mean(clean)), 2)
    diag["output_std"] = round(float(np.std(clean)), 2)
    return clean, diag


def process_seruti_inti_hal7(
    page7_img: np.ndarray,
    booklet_id: str = "booklet_01",
    onnx_session: Optional[Any] = None,
    contrast_mode: str = "high",
    binarize: bool = False
) -> Dict[str, Any]:
    """Pemrosesan lengkap Halaman 7 (Tabel Finansial Blok VI & Catatan Blok VII)."""
    t_start = time.perf_counter()

    # 1. Auto-Deskew Sub-Derajat LANGSUNG PADA CITRA RAW MENTAH (Sebelum Filter)
    deskewed_raw, skew_angle = auto_deskew(page7_img, max_angle_deg=5.0)

    # 2. Pembersihan Kertas Biru & Penajaman Kontras pada citra RAW yang sudah tegak
    clean_mono, clean_diag = clean_blue_page_image(
        deskewed_raw, page_id="hal_7", margin_px=HAL7_MARGIN_PX, contrast_mode=contrast_mode, binarize=binarize
    )

    # 3. Auto-Rotate 90 Derajat Searah Jarum Jam jika Portrait (H > W) (Lossless Permutation)
    was_rotated = False
    if clean_mono.shape[0] > clean_mono.shape[1]:
        rotated_mono = cv2.rotate(clean_mono, cv2.ROTATE_90_CLOCKWISE)
        was_rotated = True
    else:
        rotated_mono = clean_mono.copy()

    h_rot, w_rot = rotated_mono.shape[:2]

    # 4. Deteksi Pemisahan Hibrida Blok VI vs Blok VII (Banner 'BLOK VII. CATATAN')
    try:
        split_y = detect_hal7_split_y(rotated_mono)
    except Exception:
        split_y = int(round(h_rot * HAL7_SPLIT_Y_RATIO))

    blok6_img = rotated_mono[:split_y, :].copy()
    blok7_img = rotated_mono[split_y:, :].copy()

    # 5. Snapping Kisi Mikro Finansial Presisi
    cells: List[Dict[str, Any]] = []
    snapping_engine = "ground_truth_guided"
    try:
        cells, _ = GroundTruthGuidedSnapper.extract_guided_cells(
            rotated_mono, "VSERUTI26_INTI_HAL_7", booklet_id
        )
        for c in cells:
            cy = c.get("bbox", [0, 0, 0, 0])[1]
            c["block_id"] = "blok_6" if cy < split_y else "blok_7"
    except Exception:
        snapping_engine = "morphological_lattice"
        # Fallback snapping kisi morfologis
        meta = MorphologicalLatticeSnapper.get_projections(rotated_mono)
        hp, vp = meta["h_proj"], meta["v_proj"]
        row_names = [f"R6{i:02d}" for i in range(1, 15)]
        start_y = int(round(h_rot * 0.16))
        row_step = int(round((split_y - start_y) / max(1, len(row_names))))
        x_start, x_end = int(round(w_rot * 0.70)), int(round(w_rot * 0.96))
        comb_w = (x_end - x_start) // 10

        for r_idx, r_name in enumerate(row_names):
            ry1 = start_y + r_idx * row_step
            ry2 = ry1 + row_step - 4
            sy1 = MorphologicalLatticeSnapper.snap_1d(hp, ry1, radius=20)
            sy2 = MorphologicalLatticeSnapper.snap_1d(hp, ry2, radius=20)
            if sy2 <= sy1: sy1, sy2 = ry1, ry2

            for d_idx in range(10):
                cx1 = x_start + d_idx * comb_w
                cx2 = cx1 + comb_w - 2
                scx1 = MorphologicalLatticeSnapper.snap_1d(vp, cx1, radius=12)
                scx2 = MorphologicalLatticeSnapper.snap_1d(vp, cx2, radius=12)
                if scx2 <= scx1: scx1, scx2 = cx1, cx2

                crop = rotated_mono[sy1:sy2, scx1:scx2]
                std28, is_b, ink_d = standardize_cell_image(crop, target_size=28, return_meta=True)
                cells.append({
                    "cell_id": f"{booklet_id}_hal7_{r_name}_d{d_idx+1}",
                    "field_name": r_name,
                    "sub_index": d_idx + 1,
                    "cell_type": "comb_digit",
                    "bbox": [scx1, sy1, scx2 - scx1, sy2 - sy1],
                    "std28": std28,
                    "is_blank": is_b,
                    "block_id": "blok_6",
                })

    # 6. Batched ONNX Inference untuk Digit Pensil
    onnx_sess = _init_onnx_session(onnx_session)
    comb_cells = [c for c in cells if c.get("cell_type") == "comb_digit"]
    filled_cells = [c for c in comb_cells if not c.get("is_blank", True)]
    confidences: List[float] = []

    if filled_cells and onnx_sess is not None:
        inv_crops = [(255.0 - c["std28"].astype(np.float32)) / 255.0 for c in filled_cells]
        batch_input = np.expand_dims(np.array(inv_crops, dtype=np.float32), axis=1)
        input_name = onnx_sess.get_inputs()[0].name
        outs = onnx_sess.run(None, {input_name: batch_input})[0]

        for i, logits in enumerate(outs):
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))
            filled_cells[i]["pred"] = pred
            filled_cells[i]["conf"] = round(conf, 4)
            confidences.append(conf)

    # 7. Perakitan Nominal Rupiah
    field_groups = defaultdict(list)
    for c in cells:
        field_groups[c.get("field_name", "FIELD")].append(c)

    nominals: Dict[str, Any] = {}
    for f_name, f_cells in field_groups.items():
        digit_cells = [c for c in f_cells if c.get("cell_type") == "comb_digit"]
        if not digit_cells: continue
        digit_cells.sort(key=lambda x: x.get("sub_index", 0))
        d_chars, d_confs, has_ink = [], [], False
        for dc in digit_cells:
            if not dc.get("is_blank", True) and dc.get("pred") is not None:
                d_chars.append(str(dc["pred"]))
                d_confs.append(dc.get("conf", 1.0))
                has_ink = True
            else:
                d_chars.append(" ")
        d_str = "".join(d_chars).strip()
        val = int("".join(c for c in d_str if c.isdigit())) if (has_ink and d_str) else None
        nominals[f_name] = {
            "field_name": f_name,
            "digit_string": d_str if has_ink else None,
            "value": val,
            "formatted": f"Rp {val:,.0f}" if val is not None else None,
            "avg_confidence": round(float(np.mean(d_confs)), 4) if d_confs else 1.0,
        }

    return {
        "page_id": "hal_7",
        "booklet_id": booklet_id,
        "is_blue_paper": clean_diag.get("is_blue_paper", True),
        "blue_diff": clean_diag.get("blue_diff", 0.0),
        "skew_angle_deg": round(skew_angle, 3),
        "was_rotated_cw": was_rotated,
        "clean_shape_landscape": list(rotated_mono.shape),
        "split_y": int(split_y),
        "snapping_engine": snapping_engine,
        "metrics": {
            "total_cells": len(cells),
            "comb_cells": len(comb_cells),
            "filled_comb_cells": len(filled_cells),
            "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 1.0,
            "elapsed_sec": round(time.perf_counter() - t_start, 4),
        },
        "nominals": nominals,
        "clean_image": rotated_mono,
        "blocks": {"blok_6": blok6_img, "blok_7": blok7_img},
    }


def process_seruti_inti_hal8(
    page8_img: np.ndarray,
    booklet_id: str = "booklet_01",
    contrast_mode: str = "high"
) -> Dict[str, Any]:
    """Pemrosesan lengkap Halaman 8 (Sampul Belakang Kertas Biru)."""
    t_start = time.perf_counter()
    is_blue, blue_diff = False, 0.0
    if len(page8_img.shape) == 3:
        is_blue, blue_diff = is_blue_paper(page8_img)
        mono_raw = page8_img[:, :, 0] if is_blue else cv2.cvtColor(page8_img, cv2.COLOR_BGR2GRAY)
    else:
        mono_raw = page8_img.copy()

    std_val = float(np.std(mono_raw))
    dark_ratio = float(np.mean(mono_raw < 150))
    is_blank = (std_val < 2.5) or (dark_ratio < 0.005)

    if is_blank:
        # Rekonstruksi kanvas putih murni 255 bebas noise pindaian
        clean_img = np.full((A4_CANONICAL_HEIGHT, A4_CANONICAL_WIDTH), 255, dtype=np.uint8)
        skew_angle, mode = 0.0, "pure_white_synthetic_canvas"
    else:
        # Deskew RAW terlebih dahulu sebelum filter
        deskewed_raw, skew_angle = auto_deskew(page8_img, max_angle_deg=5.0)
        clean_img, _ = clean_blue_page_image(
            deskewed_raw, page_id="hal_8", margin_px=DEFAULT_MARGIN_PX, contrast_mode=contrast_mode
        )
        mode = "handwritten_notes_whitened"

    return {
        "page_id": "hal_8",
        "booklet_id": booklet_id,
        "is_blue_paper": is_blue,
        "blue_diff": round(blue_diff, 2),
        "is_blank_cover": is_blank,
        "processing_mode": mode,
        "skew_angle_deg": round(skew_angle, 3),
        "clean_shape": list(clean_img.shape),
        "metrics": {
            "output_mean": round(float(np.mean(clean_img)), 2),
            "output_std": round(float(np.std(clean_img)), 2),
            "elapsed_sec": round(time.perf_counter() - t_start, 4),
        },
        "clean_image": clean_img,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pemrosesan Halaman Kertas Biru (Hal 7 & Hal 8) Susenas Seruti Inti"
    )
    parser.add_argument("--booklet-dir", type=Path, help="Folder buklet hasil perakitan (berisi folder pages/)")
    parser.add_argument("--hal7", type=Path, help="Path berkas pindaian Halaman 7")
    parser.add_argument("--hal8", type=Path, help="Path berkas pindaian Halaman 8")
    parser.add_argument("--spread2", type=Path, help="Path berkas spread 2 mentah A3 (Hal 2 + Hal 7)")
    parser.add_argument("--spread1", type=Path, help="Path berkas spread 1 mentah A3 (Hal 8 + Hal 1)")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("output/blue_pages_processed"), help="Folder output")
    parser.add_argument("--booklet-id", type=str, default="booklet_01", help="ID buklet")
    parser.add_argument("--contrast", choices=["normal", "high", "ultra"], default="high", help="Tingkat ketegasan kontras (normal, high, ultra)")
    parser.add_argument("--binarize", action="store_true", help="Hasilkan citra biner 1-bit hitam-putih murni (Otsu)")
    parser.add_argument("--save-blocks", action="store_true", help="Simpan potongan terpisah Blok VI dan Blok VII")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"booklet_id": args.booklet_id, "timestamp": time.time(), "pages": {}}

    print("=" * 65)
    print(" PEMROSESAN HALAMAN KERTAS BIRU SERUTI INTI (HAL 7 & HAL 8)")
    print("=" * 65)

    # 1. Resolusi Halaman 7
    img_hal7 = None
    if args.hal7 and args.hal7.exists():
        img_hal7 = cv2.imread(str(args.hal7))
        print(f"[*] Memuat Halaman 7 dari: {args.hal7}")
    elif args.spread2 and args.spread2.exists():
        spread2_bgr = cv2.imread(str(args.spread2))
        l_a4, r_a4, _ = split_spread_to_pages(spread2_bgr)
        # Halaman 7 adalah sisi kertas biru / Blok VI pada Spread 2
        is_blue_l, _ = is_blue_paper(l_a4)
        is_blue_r, _ = is_blue_paper(r_a4)
        img_hal7 = r_a4 if is_blue_r else l_a4
        print(f"[*] Memuat Halaman 7 dari potongan Spread 2: {args.spread2.name}")
    elif args.booklet_dir and args.booklet_dir.exists():
        pages_dir = args.booklet_dir / "pages"
        candidates = [
            pages_dir / "hal_7_raw.jpg",
            pages_dir / "hal_7_vlm.jpg",
            pages_dir / "hal_7_clean.png",
        ]
        for c in candidates:
            if c.exists():
                img_hal7 = cv2.imread(str(c))
                print(f"[*] Memuat Halaman 7 dari buklet: {c.name}")
                break

    # 2. Resolusi Halaman 8
    img_hal8 = None
    if args.hal8 and args.hal8.exists():
        img_hal8 = cv2.imread(str(args.hal8))
        print(f"[*] Memuat Halaman 8 dari: {args.hal8}")
    elif args.spread1 and args.spread1.exists():
        spread1_bgr = cv2.imread(str(args.spread1))
        l_a4, r_a4, _ = split_spread_to_pages(spread1_bgr)
        # Halaman 8 adalah sampul belakang (std lebih rendah dibanding Hal 1 sampul depan)
        std_l = float(np.std(l_a4))
        std_r = float(np.std(r_a4))
        img_hal8 = l_a4 if std_l < std_r else r_a4
        print(f"[*] Memuat Halaman 8 dari potongan Spread 1: {args.spread1.name}")
    elif args.booklet_dir and args.booklet_dir.exists():
        pages_dir = args.booklet_dir / "pages"
        candidates = [
            pages_dir / "hal_8_raw.jpg",
            pages_dir / "hal_8_vlm.jpg",
            pages_dir / "hal_8_clean.png",
        ]
        for c in candidates:
            if c.exists():
                img_hal8 = cv2.imread(str(c))
                print(f"[*] Memuat Halaman 8 dari buklet: {c.name}")
                break

    # Eksekusi Pemrosesan Halaman 7
    if img_hal7 is not None:
        mode_label = f"{args.contrast.upper()}" + (" + BINARY" if args.binarize else "")
        print(f"\n[+] Memproses Halaman 7 (Blok VI & VII) [Kontras: {mode_label}]...")
        res7 = process_seruti_inti_hal7(
            img_hal7, booklet_id=args.booklet_id, contrast_mode=args.contrast, binarize=args.binarize
        )
        clean_hal7_path = args.out_dir / f"{args.booklet_id}_hal_7_clean_magic.png"
        cv2.imwrite(str(clean_hal7_path), res7["clean_image"])

        if args.save_blocks and "blocks" in res7:
            cv2.imwrite(str(args.out_dir / f"{args.booklet_id}_hal_7_blok6.png"), res7["blocks"]["blok_6"])
            cv2.imwrite(str(args.out_dir / f"{args.booklet_id}_hal_7_blok7.png"), res7["blocks"]["blok_7"])

        # Sanitasi data hasil agar tidak memuat raw image array di JSON
        res7_json = {k: v for k, v in res7.items() if k not in ["clean_image", "blocks"]}
        results["pages"]["hal_7"] = res7_json

        print(f"    - Bentuk Bersih     : {res7['clean_shape_landscape']}")
        print(f"    - Sudut Kemiringan  : {res7['skew_angle_deg']}°")
        print(f"    - Rotasi Landscape  : {res7['was_rotated_cw']}")
        print(f"    - Split Banner Y    : y = {res7['split_y']} px")
        print(f"    - Snapping Engine   : {res7['snapping_engine']}")
        print(f"    - Total Comb-Boxes  : {res7['metrics']['comb_cells']} sel")
        print(f"    - Durasi Eksekusi   : {res7['metrics']['elapsed_sec']} detik")
        print(f"    - Berkas Tersimpan  : {clean_hal7_path.name}")

    # Eksekusi Pemrosesan Halaman 8
    if img_hal8 is not None:
        print(f"\n[+] Memproses Halaman 8 (Sampul Belakang) [Kontras: {args.contrast.upper()}]...")
        res8 = process_seruti_inti_hal8(img_hal8, booklet_id=args.booklet_id, contrast_mode=args.contrast)
        clean_hal8_path = args.out_dir / f"{args.booklet_id}_hal_8_clean_magic.png"
        cv2.imwrite(str(clean_hal8_path), res8["clean_image"])

        res8_json = {k: v for k, v in res8.items() if k not in ["clean_image"]}
        results["pages"]["hal_8"] = res8_json

        print(f"    - Mode Pemrosesan   : {res8['processing_mode']}")
        print(f"    - Sampul Kosong     : {res8['is_blank_cover']}")
        print(f"    - Mean Intensitas   : {res8['metrics']['output_mean']} (Target 255.0)")
        print(f"    - Std Deviasi       : {res8['metrics']['output_std']}")
        print(f"    - Durasi Eksekusi   : {res8['metrics']['elapsed_sec']} detik")
        print(f"    - Berkas Tersimpan  : {clean_hal8_path.name}")

    # Simpan ringkasan metadata JSON
    json_path = args.out_dir / f"{args.booklet_id}_blue_pages_summary.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[OK] Seluruh hasil dan ringkasan disimpan ke: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
