#!/usr/bin/env python3
"""
render_text_scan_overlays.py
============================
Merender visualisasi scan overlay khusus isian teks bebas (Freeform Handwriting Text Fields)
kuesioner BPS Seruti Inti 2026 (VSERUTI26.INTI) langsung di atas citra pindaian fisik asli (300 DPI).

Bidang yang divisualisasikan:
1. Halaman 1:
   - Nama Kepala Rumah Tangga (R110)
   - Alamat Tempat Tinggal Responden (R111)
   - Nama Petugas Pencacah (R201)
   - Nama Petugas Pengawas / Pemeriksa (R202)
2. Halaman 2:
   - Roster Nama Anggota Rumah Tangga No. 01 s.d. 10 (Blok III Kolom 2)
3. Halaman 7:
   - Blok VII Catatan Khusus Petugas Lapangan (R700)

Fitur Utama:
- Shading transparan dengan palet warna semantik BPS yang elegan (tidak menutupi tulisan tangan).
- Bounding box berpresisi sub-piksel dengan affine alignment.
- Badge identifikasi kode rincian dan label pertanyaan.
- Integrasi anotasi hasil inferensi OCR Google Gemini Vision (opsional, dengan masking Zero-PII).
- Header banner profesional dengan legenda warna.
- Ekspor resolusi penuh (300 DPI) dan pratinjau web responsif untuk UI laporan.
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from seruti_pipeline.grid_template_aligner import align_template_to_image, GridAlignment, DEFAULT_TEMPLATE_DIR
from seruti_pipeline.stage_gate_validator import DUPLICATE_BOOKLET_IDS, REJECTED_BOOKLET_IDS

ARTIFACT_PREVIEWS = Path(r"C:\Users\ajian\.gemini\antigravity\brain\ffb6c438-eb71-4258-a214-3baaa5a8baad\previews")
ARTIFACT_PREVIEWS.mkdir(parents=True, exist_ok=True)

# Palet warna semantik BGR untuk jenis bidang teks bebas
FIELD_PALETTE = {
    "nama_krt": {
        "color": (160, 40, 140),       # Royal Purple / Violet
        "label": "Nama Kepala Rumah Tangga (R110)",
        "legend": "Nama KRT"
    },
    "alamat": {
        "color": (0, 130, 230),        # Warm Amber / Orange
        "label": "Alamat Tempat Tinggal (R111)",
        "legend": "Alamat Responden"
    },
    "pencacah": {
        "color": (150, 120, 0),        # Teal / Turquoise
        "label": "Petugas Pencacah (R201)",
        "legend": "Pencacah Lapangan"
    },
    "pengawas": {
        "color": (170, 90, 20),        # Sky Blue / Ocean
        "label": "Petugas Pengawas (R202)",
        "legend": "Pengawas Lapangan"
    },
    "art": {
        "color": (140, 50, 160),       # Purple / Magenta
        "label": "Nama Anggota Rumah Tangga",
        "legend": "Roster ART"
    },
    "catatan": {
        "color": (30, 150, 50),        # Emerald Green
        "label": "Catatan Lapangan (Blok VII)",
        "legend": "Catatan Lapangan"
    },
    "default": {
        "color": (100, 80, 160),
        "label": "Bidang Teks Bebas",
        "legend": "Teks Bebas"
    }
}


def mask_text_preview(text: str) -> str:
    """Masking PII konsol terminal & visualisasi publik sesuai kaidah Zero-PII Cloud Egress."""
    text = str(text).strip()
    if text in ["KOSONG", "FAILED", "", "None", "nan"]:
        return text
    if len(text) <= 3:
        return text[0] + "**"
    return text[:2] + "*" * (min(len(text) - 3, 5)) + text[-1:]


def get_field_style(field_id: str) -> Dict[str, Any]:
    """Mengembalikan konfigurasi warna dan label semantik berdasarkan ID bidang."""
    fid = field_id.lower()
    if "nama_krt" in fid:
        return FIELD_PALETTE["nama_krt"]
    elif "alamat" in fid:
        return FIELD_PALETTE["alamat"]
    elif "pencacah" in fid:
        return FIELD_PALETTE["pencacah"]
    elif "pengawas" in fid:
        return FIELD_PALETTE["pengawas"]
    elif "art" in fid:
        return FIELD_PALETTE["art"]
    elif "catatan" in fid:
        return FIELD_PALETTE["catatan"]
    return FIELD_PALETTE["default"]


def create_header_banner(
    width: int,
    title_text: str,
    meta_text: str,
    legend_items: List[Tuple[str, Tuple[int, int, int]]],
    banner_height: int = 60
) -> np.ndarray:
    """Membuat banner informasi profesional di bagian atas overlay scan."""
    banner = np.full((banner_height, width, 3), 246, dtype=np.uint8)
    # Garis pemisah bawah
    cv2.line(banner, (0, banner_height - 1), (width, banner_height - 1), (180, 180, 180), 2)

    # Judul Kiri
    cv2.putText(banner, title_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(banner, meta_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 90, 90), 1, cv2.LINE_AA)

    # Legenda Kanan
    if legend_items:
        item_w = 175
        lx = width - (len(legend_items) * item_w + 20)
        for l_txt, l_col in legend_items:
            box_y0, box_y1 = 18, 42
            # Kotak warna legenda dengan border tegas
            cv2.rectangle(banner, (lx, box_y0), (lx + 20, box_y1), l_col, -1)
            cv2.rectangle(banner, (lx, box_y0), (lx + 20, box_y1), (40, 40, 40), 1)
            cv2.putText(banner, l_txt, (lx + 28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1, cv2.LINE_AA)
            lx += item_w

    return banner


def load_ocr_results(booklet_id: str) -> Dict[str, Dict[str, Any]]:
    """Memuat hasil inferensi OCR teks jika tersedia (dari hasil_teks.csv atau hasil_uji_teks.csv)."""
    ocr_map = {}
    csv_candidates = [
        BASE_DIR / "output" / "test_e2e_10_booklets" / "hasil_uji_teks.csv",
        BASE_DIR / "hasil_labeling_teks" / "hasil_teks.csv"
    ]
    for csv_p in csv_candidates:
        if csv_p.exists():
            try:
                df = pd.read_csv(csv_p)
                sub_df = df[df["id_booklet"] == booklet_id]
                for _, row in sub_df.iterrows():
                    fid = str(row.get("field_id", ""))
                    ocr_map[fid] = {
                        "text": str(row.get("teks_hasil", "")),
                        "confidence": float(row.get("confidence", 0.0)),
                        "is_blank": bool(row.get("is_blank", False)),
                        "model": str(row.get("model_used", ""))
                    }
                if ocr_map:
                    break
            except Exception:
                pass
    return ocr_map


def render_scan_text_overlay(
    base_img: np.ndarray,
    alignment: GridAlignment,
    booklet_id: str,
    page_num: int,
    image_type: str = "clean",
    show_ocr: bool = True,
    mask_pii: bool = True
) -> np.ndarray:
    """
    Merender visualisasi overlay isian teks bebas kuesioner Seruti Inti
    langsung di atas citra scan pindaian fisik asli.
    """
    tpl = alignment.template_data
    text_fields = tpl.get("text_fields", [])
    blocks = tpl.get("blocks", [])

    vis = base_img.copy() if len(base_img.shape) == 3 else cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    H, W = vis.shape[:2]

    # 1. Gambar outline makro blok formulir tipis sebagai referensi spasial konteks
    relevant_blocks = {
        1: ["blok_1", "blok_2"],
        2: ["blok_3"],
        7: ["blok_7"]
    }.get(page_num, [])

    for b in blocks:
        if b.get("id") in relevant_blocks:
            bx0, by0, bx1, by1 = alignment.bbox_pt_to_px(b["bbox_pt"])
            cv2.rectangle(vis, (bx0, by0), (bx1, by1), (130, 130, 130), 2)
            # Label blok samar
            blbl = b.get("name", "")
            if blbl:
                cv2.putText(vis, blbl, (bx0 + 8, by0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 100, 100), 2, cv2.LINE_AA)

    # Muat OCR map jika diminta
    ocr_map = load_ocr_results(booklet_id) if show_ocr else {}

    # 2. Gambar Shading Transparan Bidang Teks (Alpha 0.22)
    for tf in text_fields:
        fid = tf["id"]
        cid = tf.get("cell_id")
        b_px = alignment.get_cell_bbox_px(cid) if cid else None
        if b_px is not None:
            x0, y0, x1, y1 = b_px
        else:
            x0, y0, x1, y1 = alignment.bbox_pt_to_px(tf["bbox_pt"], pad=2)
        style = get_field_style(fid)
        color = style["color"]

        # Shading kotak teks
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)

    vis = cv2.addWeighted(overlay, 0.22, vis, 0.78, 0)

    # 3. Gambar Border Presisi, Label Field, dan Anotasi Nilai OCR
    for tf in text_fields:
        fid = tf["id"]
        cid = tf.get("cell_id")
        b_px = alignment.get_cell_bbox_px(cid) if cid else None
        if b_px is not None:
            x0, y0, x1, y1 = b_px
        else:
            x0, y0, x1, y1 = alignment.bbox_pt_to_px(tf["bbox_pt"], pad=2)
        style = get_field_style(fid)
        color = style["color"]
        field_lbl = tf.get("label", fid)
        q_code = tf.get("question_code", "")

        # Border luar tebal 3px
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 3)

        # Badge Label Utama di sudut kiri atas
        badge_text = f"[{q_code}] {field_lbl}" if q_code else field_lbl
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        t_size, _ = cv2.getTextSize(badge_text, font, font_scale, thickness)

        badge_x0 = x0
        badge_y0 = max(y0 - t_size[1] - 8, 4)
        badge_x1 = badge_x0 + t_size[0] + 16
        badge_y1 = badge_y0 + t_size[1] + 8

        # Latar badge putih dengan border warna field
        cv2.rectangle(vis, (badge_x0, badge_y0), (badge_x1, badge_y1), (255, 255, 255), -1)
        cv2.rectangle(vis, (badge_x0, badge_y0), (badge_x1, badge_y1), color, 2)
        cv2.putText(vis, badge_text, (badge_x0 + 8, badge_y1 - 5), font, font_scale, (20, 20, 20), 1, cv2.LINE_AA)

        # Badge Status OCR di sudut kanan atas bidang teks
        if fid in ocr_map:
            res = ocr_map[fid]
            is_b = res["is_blank"]
            raw_t = res["text"]
            conf = res["confidence"]

            if is_b or raw_t == "KOSONG":
                ocr_badge_text = "[KOSONG]"
                ocr_bg_col = (235, 235, 235)
                ocr_txt_col = (100, 100, 100)
                ocr_border_col = (160, 160, 160)
            else:
                disp_t = mask_text_preview(raw_t) if mask_pii else raw_t
                ocr_badge_text = f"OCR: {disp_t} (conf: {conf:.2f})"
                ocr_bg_col = (240, 255, 240)
                ocr_txt_col = (10, 120, 30)
                ocr_border_col = (20, 160, 40)

            ocr_size, _ = cv2.getTextSize(ocr_badge_text, font, font_scale, thickness)
            ocr_x1 = x1
            ocr_x0 = max(ocr_x1 - ocr_size[0] - 16, badge_x1 + 8)
            ocr_y0 = badge_y0
            ocr_y1 = badge_y1

            cv2.rectangle(vis, (ocr_x0, ocr_y0), (ocr_x1, ocr_y1), ocr_bg_col, -1)
            cv2.rectangle(vis, (ocr_x0, ocr_y0), (ocr_x1, ocr_y1), ocr_border_col, 2)
            cv2.putText(vis, ocr_badge_text, (ocr_x0 + 8, ocr_y1 - 5), font, font_scale, ocr_txt_col, 1, cv2.LINE_AA)

    # 4. Header Banner
    title_text = f"SERUTI INTI 2026 — [{booklet_id.upper()}] HALAMAN {page_num} ({image_type.upper()}) — SCAN OVERLAY ISIAN TEKS BEBAS"
    meta_text = (
        f"Resolusi Scan: {W}x{H} px (300 DPI) | Total Bidang: {len(text_fields)} Bidang Teks | "
        f"Affine: sx={alignment.scale_x:.4f}, sy={alignment.scale_y:.4f}, tx={alignment.trans_x:.1f}, ty={alignment.trans_y:.1f} | "
        f"Model OCR: Google Gemini-3.8-Flash"
    )

    legend_items = [
        (FIELD_PALETTE["nama_krt"]["legend"], FIELD_PALETTE["nama_krt"]["color"]),
        (FIELD_PALETTE["alamat"]["legend"], FIELD_PALETTE["alamat"]["color"]),
        (FIELD_PALETTE["pencacah"]["legend"], FIELD_PALETTE["pencacah"]["color"]),
        (FIELD_PALETTE["catatan"]["legend"], FIELD_PALETTE["catatan"]["color"]),
    ]

    banner = create_header_banner(W, title_text, meta_text, legend_items, banner_height=60)
    return np.vstack([banner, vis])


def process_booklet_text_overlays(
    booklet_id: str,
    image_type: str = "clean",
    pages: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    show_ocr: bool = True,
    save_previews: bool = True
) -> Dict[str, Any]:
    """Merender scan overlay isian teks bebas untuk satu buklet."""
    booklets_base = BASE_DIR / "output" / "seruti_inti_booklets"
    b_dir = booklets_base / booklet_id
    pages_dir = b_dir / "pages"

    if not pages_dir.exists():
        print(f"[ERROR] Halaman buklet tidak ditemukan di: {pages_dir}")
        return {}

    # Target halaman yang memuat bidang teks bebas (default: 1, 2, 7)
    if pages is None:
        pages = [1, 2, 7]

    if output_dir is None:
        out_dir = BASE_DIR / "output" / "scan_overlays" / booklet_id / image_type
    else:
        out_dir = output_dir / booklet_id / image_type
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "booklet_id": booklet_id,
        "image_type": image_type,
        "pages_rendered": []
    }

    print(f"\n[*] Merender Scan Overlay Teks Bebas: {booklet_id} ({image_type.upper()})")

    for p in pages:
        # Cari file citra
        if image_type == "raw":
            img_file = pages_dir / f"hal_{p}_raw.jpg"
        elif image_type == "vlm":
            img_file = pages_dir / f"hal_{p}_vlm.jpg"
        else:
            img_file = pages_dir / f"hal_{p}_clean.png"

        clean_file = pages_dir / f"hal_{p}_clean.png"

        if not img_file.exists():
            print(f"  [!] Berkas tidak ditemukan: {img_file}")
            continue

        scan_img = cv2.imread(str(img_file))
        if scan_img is None:
            print(f"  [!] Gagal memuat citra: {img_file}")
            continue

        clean_img = cv2.imread(str(clean_file)) if clean_file.exists() else scan_img
        alignment = align_template_to_image(clean_img, page_num=p)
        alignment.enable_physical_cell_snapping(clean_img)

        text_fields = alignment.template_data.get("text_fields", [])
        if not text_fields:
            print(f"  [-] Halaman {p}: Tidak ada bidang teks bebas di template.")
            continue

        # Render Text Scan Overlay
        text_overlay = render_scan_text_overlay(
            scan_img, alignment, booklet_id, p,
            image_type=image_type, show_ocr=show_ocr, mask_pii=True
        )

        # Simpan resolusi penuh (300 DPI)
        out_file = out_dir / f"hal_{p}_scan_text_overlay.jpg"
        cv2.imwrite(str(out_file), text_overlay, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"  [✓] Halaman {p}: {len(text_fields)} bidang teks -> {out_file.name}")

        # Simpan pratinjau web (~1280 px) untuk UI Artefak
        if save_previews:
            H, W = text_overlay.shape[:2]
            prev_w = 1280
            prev_h = int(round(H * (prev_w / float(W))))
            prev_img = cv2.resize(text_overlay, (prev_w, prev_h), interpolation=cv2.INTER_AREA)

            p_file = ARTIFACT_PREVIEWS / f"scan_{booklet_id}_hal_{p}_text_overlay.jpg"
            cv2.imwrite(str(p_file), prev_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            # Juga salin versi kanonikal halaman jika booklet_01
            if booklet_id == "booklet_01":
                p_file_canon = ARTIFACT_PREVIEWS / f"scan_hal_{p}_text_overlay.jpg"
                cv2.imwrite(str(p_file_canon), prev_img, [cv2.IMWRITE_JPEG_QUALITY, 88])

        summary["pages_rendered"].append({
            "page_num": p,
            "text_fields_count": len(text_fields),
            "file": str(out_file.relative_to(BASE_DIR))
        })

    return summary


from concurrent.futures import ProcessPoolExecutor

def _worker_process_booklet(args_tuple):
    b_id, image_type, pages, show_ocr, save_previews = args_tuple
    return process_booklet_text_overlays(
        booklet_id=b_id,
        image_type=image_type,
        pages=pages,
        show_ocr=show_ocr,
        save_previews=save_previews
    )


def main():
    parser = argparse.ArgumentParser(description="Render Scan Overlay Isian Teks Bebas Seruti Inti")
    parser.add_argument("--booklet", type=str, default="booklet_01", help="ID buklet (default: booklet_01, atau 'all')")
    parser.add_argument("--workers", type=int, default=4, help="Jumlah worker multiprocessing paralel (default: 4)")
    parser.add_argument("--limit", type=int, default=None, help="Batas maksimal buklet jika memproses batch")
    parser.add_argument("--pages", type=int, nargs="+", default=[1, 2, 7], help="Halaman target (default: 1 2 7)")
    parser.add_argument("--image-type", choices=["clean", "raw", "vlm"], default="clean", help="Jenis citra scan")
    parser.add_argument("--no-ocr", action="store_true", help="Jangan tampilkan badge hasil teks OCR")
    args = parser.parse_args()

    print("=" * 80)
    print("  🎨 RENDER SCAN OVERLAY ISIAN TEKS BEBAS (FREEFORM HANDWRITING OVERLAY)")
    print(f"  Target        : {args.booklet}")
    print(f"  Halaman       : {args.pages}")
    print(f"  Mode Citra    : {args.image_type.upper()}")
    print(f"  Worker Pool   : {args.workers} worker(s)")
    print(f"  Anotasi OCR   : {'NONAKTIF' if args.no_ocr else 'AKTIF (Zero-PII Masked)'}")
    print("=" * 80)

    booklets_base = BASE_DIR / "output" / "seruti_inti_booklets"
    if args.booklet.lower() == "all":
        b_dirs = sorted([
            d.name for d in booklets_base.iterdir()
            if d.is_dir() and d.name.startswith("booklet_")
            and d.name not in DUPLICATE_BOOKLET_IDS
            and d.name not in REJECTED_BOOKLET_IDS
        ])
        if args.limit:
            b_dirs = b_dirs[:args.limit]
    else:
        b_dirs = [args.booklet]

    print(f"[*] Menemukan {len(b_dirs)} buklet sah untuk diproses...")
    total_rendered = 0

    if args.workers > 1 and len(b_dirs) > 1:
        task_payloads = [
            (b_id, args.image_type, args.pages, not args.no_ocr, True)
            for b_id in b_dirs
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for idx_b, res in enumerate(executor.map(_worker_process_booklet, task_payloads), 1):
                b_name = res.get("booklet_id", f"booklet_{idx_b}")
                count_p = len(res.get("pages_rendered", []))
                total_rendered += count_p
                print(f"  [{idx_b:02d}/{len(b_dirs):02d}] {b_name} Selesai: {count_p} halaman overlay tersimpan (Kumulatif: {total_rendered})")
    else:
        for idx_b, b_id in enumerate(b_dirs, 1):
            res = process_booklet_text_overlays(
                booklet_id=b_id,
                image_type=args.image_type,
                pages=args.pages,
                show_ocr=not args.no_ocr,
                save_previews=True
            )
            count_p = len(res.get("pages_rendered", []))
            total_rendered += count_p
            print(f"  [{idx_b:02d}/{len(b_dirs):02d}] {b_id} Selesai: {count_p} halaman overlay tersimpan")

    print("\n" + "=" * 80)
    print(f"  ✅ SELESAI: Berhasil merender {total_rendered} halaman scan overlay teks bebas.")
    print(f"  Direktori Output: {BASE_DIR / 'output' / 'scan_overlays'}")
    print(f"  Pratinjau UI    : {ARTIFACT_PREVIEWS}")
    print("=" * 80)


if __name__ == "__main__":
    main()
