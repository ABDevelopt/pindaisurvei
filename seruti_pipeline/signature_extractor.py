"""
seruti_pipeline/signature_extractor.py
Modul ekstraksi dan penyimpanan crop tanda tangan petugas (Pendata/PCL & Pemeriksa/PML).
Berdasarkan koordinat terkalibrasi presisi pada Blok II Kuesioner Seruti Inti 2026 (VSERUTI26.INTI).
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
BOOKLET_DIR_INTI = BASE_DIR / "output" / "seruti_inti_booklets"
CROPS_DIR = BASE_DIR / "output" / "crops"
ARCHIVE_DIR = BASE_DIR / "archive" / "duplicates" / "seruti_inti_booklets"

# Koordinat baku pada dimensi 2480 x 3507 (A4 300 DPI terkalibrasi)
# Menghindari garis batas tabel di sisi kiri (x=1817-1820), kanan (x=2396), atas (y=2478), tengah (y=2697), bawah (y=2916)
SIGNATURE_BOXES = {
    "pencacah": (1836, 2486, 2384, 2690),
    "pcl": (1836, 2486, 2384, 2690),
    "pendata": (1836, 2486, 2384, 2690),
    "pengawas": (1836, 2704, 2384, 2908),
    "pml": (1836, 2704, 2384, 2908),
    "pemeriksa": (1836, 2704, 2384, 2908),
}

CANONICAL_NAMES = {
    "pencacah": "ttd_pencacah.png",
    "pcl": "ttd_pencacah.png",
    "pendata": "ttd_pencacah.png",
    "pengawas": "ttd_pengawas.png",
    "pml": "ttd_pengawas.png",
    "pemeriksa": "ttd_pengawas.png",
}


def find_hal1_image(booklet_id: str) -> Optional[Path]:
    """Cari berkas pindaian halaman 1 untuk booklet_id tertentu."""
    candidates = [
        BOOKLET_DIR_INTI / booklet_id / "pages" / "hal_1_clean.png",
        BOOKLET_DIR_INTI / booklet_id / "pages" / "hal_1_raw.jpg",
        BOOKLET_DIR_INTI / booklet_id / "hal_1_clean.png",
        BOOKLET_DIR_INTI / booklet_id / "hal_1.png",
        ARCHIVE_DIR / booklet_id / "pages" / "hal_1_clean.png",
        ARCHIVE_DIR / booklet_id / "pages" / "hal_1_raw.jpg",
        ARCHIVE_DIR / booklet_id / "hal_1_clean.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def crop_signature_from_image(img_path: Path, officer_type: str) -> Optional[Image.Image]:
    """Melakukan cropping tanda tangan dari citra pindaian halaman 1."""
    key = officer_type.lower().strip()
    if key not in SIGNATURE_BOXES:
        return None

    try:
        with Image.open(img_path) as im:
            w, h = im.size
            sx = w / 2480.0
            sy = h / 3507.0

            x1_base, y1_base, x2_base, y2_base = SIGNATURE_BOXES[key]
            x1 = int(round(x1_base * sx))
            y1 = int(round(y1_base * sy))
            x2 = int(round(x2_base * sx))
            y2 = int(round(y2_base * sy))

            # Batasi agar tidak melebihi batas gambar
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            crop = im.crop((x1, y1, x2, y2))
            return crop.copy()
    except Exception as e:
        print(f"[SignatureExtractor] Gagal crop {img_path}: {e}")
        return None


def get_or_create_signature_crop(booklet_id: str, officer_type: str) -> Optional[Path]:
    """
    Mengambil berkas crop tanda tangan. Jika belum ada di disk,
    langsung memotong dari pindaian halaman 1 dan menyimpannya.
    """
    key = officer_type.lower().strip()
    filename = CANONICAL_NAMES.get(key)
    if not filename:
        return None

    target_dir = CROPS_DIR / booklet_id / "signatures"
    target_path = target_dir / filename

    if target_path.exists():
        return target_path

    # Cari hal_1 dan crop
    img_path = find_hal1_image(booklet_id)
    if not img_path:
        return None

    crop = crop_signature_from_image(img_path, key)
    if crop is None:
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    crop.save(target_path, format="PNG")

    # Buat juga salinan alias (ttd_pcl / ttd_pml)
    if "pencacah" in filename:
        alias_path = target_dir / "ttd_pcl.png"
        if not alias_path.exists():
            crop.save(alias_path, format="PNG")
    elif "pengawas" in filename:
        alias_path = target_dir / "ttd_pml.png"
        if not alias_path.exists():
            crop.save(alias_path, format="PNG")

    return target_path


def extract_all_booklet_signatures(booklet_id: str) -> Dict[str, str]:
    """Ekstraksi dan simpan kedua tanda tangan (pencacah & pengawas) untuk 1 buklet."""
    res = {}
    p_path = get_or_create_signature_crop(booklet_id, "pencacah")
    if p_path and p_path.exists():
        res["pencacah"] = f"/output/crops/{booklet_id}/signatures/ttd_pencacah.png"
        res["pcl"] = f"/output/crops/{booklet_id}/signatures/ttd_pcl.png"

    w_path = get_or_create_signature_crop(booklet_id, "pengawas")
    if w_path and w_path.exists():
        res["pengawas"] = f"/output/crops/{booklet_id}/signatures/ttd_pengawas.png"
        res["pml"] = f"/output/crops/{booklet_id}/signatures/ttd_pml.png"

    return res


def batch_extract_all() -> Dict[str, int]:
    """Jalankan batch ekstraksi tanda tangan untuk semua buklet yang tersedia."""
    all_bids = set()
    if BOOKLET_DIR_INTI.exists():
        all_bids.update([f.name for f in BOOKLET_DIR_INTI.iterdir() if f.is_dir()])
    if ARCHIVE_DIR.exists():
        all_bids.update([f.name for f in ARCHIVE_DIR.iterdir() if f.is_dir()])

    total = len(all_bids)
    success = 0
    for bid in sorted(all_bids):
        res = extract_all_booklet_signatures(bid)
        if res.get("pencacah") or res.get("pengawas"):
            success += 1

    print(f"[SignatureExtractor] Batch selesai: {success}/{total} buklet berhasil diproses.")
    return {"total": total, "success": success}


if __name__ == "__main__":
    batch_extract_all()
