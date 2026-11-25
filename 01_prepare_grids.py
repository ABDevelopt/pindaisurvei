import argparse
import glob
import json
import os
import zipfile
import cv2
import numpy as np

STATE_DIR = "state"
GRIDS_DIR = "grids_to_read"
MANIFEST_PATH = os.path.join(STATE_DIR, "manifest.json")

GRID_COLS = 10
GRID_ROWS = 5
CELLS_PER_GRID = GRID_COLS * GRID_ROWS  # 50 sel per grid
CELL_DISPLAY_SIZE = 64  # 64x64 px per sel pada montase

def is_cell_empty(img_bytes_or_mat, threshold_variance=120.0, ink_min_val=170):
    """Deteksi cepat apakah sel kosong menggunakan varians dan nilai piksel tergelap."""
    if isinstance(img_bytes_or_mat, (bytes, bytearray)):
        arr = np.frombuffer(img_bytes_or_mat, np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    else:
        gray = img_bytes_or_mat if len(img_bytes_or_mat.shape) == 2 else cv2.cvtColor(img_bytes_or_mat, cv2.COLOR_BGR2GRAY)
        
    if gray is None or gray.size == 0:
        return True
    
    # Potong margin 15% untuk menghindari border kuesioner fisik
    h, w = gray.shape
    dy, dx = int(h * 0.15), int(w * 0.15)
    crop = gray[dy:h-dy, dx:w-dx]
    if crop.size == 0:
        crop = gray
        
    var = float(np.var(crop))
    min_val = int(np.min(crop))
    
    # Kosong jika varians rendah atau tidak ada goresan tinta pensil gelap
    if var < threshold_variance or min_val > ink_min_val:
        return True
    return False

def buat_grid_image(cell_images_with_indices):
    """Menyusun sel-sel mentah ke dalam satu citra grid dengan nomor indeks (00, 01, ...)."""
    grid_w = GRID_COLS * CELL_DISPLAY_SIZE
    grid_h = GRID_ROWS * CELL_DISPLAY_SIZE
    canvas = np.full((grid_h, grid_w, 3), 245, dtype=np.uint8)
    
    for idx, (cell_img, idx_str) in enumerate(cell_images_with_indices):
        r = idx // GRID_COLS
        c = idx % GRID_COLS
        y0 = r * CELL_DISPLAY_SIZE
        x0 = c * CELL_DISPLAY_SIZE
        
        pad = 2
        dw = CELL_DISPLAY_SIZE - (2 * pad)
        dh = CELL_DISPLAY_SIZE - (2 * pad)
        
        if len(cell_img.shape) == 2:
            cell_bgr = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2BGR)
        else:
            cell_bgr = cell_img
            
        resized = cv2.resize(cell_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        canvas[y0+pad:y0+pad+dh, x0+pad:x0+pad+dw] = resized
        
        # Garis pembatas grid
        cv2.rectangle(canvas, (x0, y0), (x0 + CELL_DISPLAY_SIZE, y0 + CELL_DISPLAY_SIZE), (200, 200, 200), 1)
        
        # Overlay badge nomor indeks (00, 01, ...)
        badge_w, badge_h = 22, 13
        cv2.rectangle(canvas, (x0+1, y0+1), (x0+1+badge_w, y0+1+badge_h), (30, 30, 30), -1)
        cv2.putText(canvas, idx_str, (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        
    return canvas

def siapkan_grid(source_path="output/crops", booklet=None, page=None, use_zip_if_present=True):
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(GRIDS_DIR, exist_ok=True)

    kosong_otomatis = []
    berisi_paths = []

    # 1. Cek apakah menggunakan ZIP atau Directory langsung
    is_zip = False
    if use_zip_if_present and os.path.exists("raw.zip") and source_path == "raw.zip":
        is_zip = True
        print("[*] Menggunakan arsip 'raw.zip' sebagai sumber data...")
        with zipfile.ZipFile("raw.zip", "r") as zf:
            valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
            all_files = [f for f in zf.namelist() if not f.endswith("/") and f.lower().endswith(valid_exts)]
            print(f"    -> Ditemukan {len(all_files):,} citra di raw.zip")
            for zpath in all_files:
                img_bytes = zf.read(zpath)
                if is_cell_empty(img_bytes):
                    kosong_otomatis.append(zpath)
                else:
                    berisi_paths.append(zpath)
    else:
        # Pindai langsung direktori raw crops (misal: output/crops/booklet_73/hal_1/mikro/raw)
        print(f"[*] Memindai citra raw langsung dari direktori: {source_path}...")
        
        if booklet and page:
            pattern = os.path.join(source_path, booklet, page, "mikro", "raw", "**", "*.png")
        elif booklet:
            pattern = os.path.join(source_path, booklet, "hal_*", "mikro", "raw", "**", "*.png")
        else:
            pattern = os.path.join(source_path, "booklet_*", "hal_*", "mikro", "raw", "**", "*.png")
            
        all_files = sorted(glob.glob(pattern, recursive=True))
        if not all_files:
            # Fallback jika pola spesifik langsung mengarah ke folder
            pattern2 = os.path.join(source_path, "**", "*.png")
            all_files = sorted(glob.glob(pattern2, recursive=True))
            
        print(f"    -> Ditemukan {len(all_files):,} citra raw micro crops")
        
        for p in all_files:
            # Baca citra
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            norm_p = p.replace("\\", "/")
            if is_cell_empty(im):
                kosong_otomatis.append(norm_p)
            else:
                berisi_paths.append(norm_p)

    print(f"[+] Hasil Deteksi Awal:")
    print(f"    -> Sel Kosong Otomatis (OpenCV): {len(kosong_otomatis):,} citra")
    print(f"    -> Sel Berisi (Perlu Pelabelan) : {len(berisi_paths):,} citra")

    # 2. Susun sel berisi ke dalam grid montase
    grids_dict = {}
    total_grids = (len(berisi_paths) + CELLS_PER_GRID - 1) // CELLS_PER_GRID
    print(f"[*] Menyusun {len(berisi_paths):,} sel berisi ke dalam {total_grids} citra grid montase...")

    if is_zip:
        with zipfile.ZipFile("raw.zip", "r") as zf:
            for g_idx in range(total_grids):
                grid_name = f"grid_{g_idx:04d}.png"
                batch_paths = berisi_paths[g_idx * CELLS_PER_GRID : (g_idx + 1) * CELLS_PER_GRID]
                cells_with_idx = []
                for i, p in enumerate(batch_paths):
                    arr = np.frombuffer(zf.read(p), np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    cells_with_idx.append((img, f"{i:02d}"))
                grid_img = buat_grid_image(cells_with_idx)
                cv2.imwrite(os.path.join(GRIDS_DIR, grid_name), grid_img)
                grids_dict[grid_name] = batch_paths
    else:
        for g_idx in range(total_grids):
            grid_name = f"grid_{g_idx:04d}.png"
            batch_paths = berisi_paths[g_idx * CELLS_PER_GRID : (g_idx + 1) * CELLS_PER_GRID]
            cells_with_idx = []
            for i, p in enumerate(batch_paths):
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    img = np.full((50, 50, 3), 200, dtype=np.uint8)
                cells_with_idx.append((img, f"{i:02d}"))
            grid_img = buat_grid_image(cells_with_idx)
            cv2.imwrite(os.path.join(GRIDS_DIR, grid_name), grid_img)
            grids_dict[grid_name] = batch_paths

    # Simpan manifest.json
    manifest = {
        "is_zip": is_zip,
        "source_path": source_path,
        "total_citra": len(all_files),
        "kosong_otomatis": kosong_otomatis,
        "grids": grids_dict
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Berkas state tersimpan di: {MANIFEST_PATH}")
    print(f"[+] Citra grid montase tersimpan di folder: {GRIDS_DIR}/ (Total {len(grids_dict)} berkas)")
    print("\nLangkah Berikutnya: Agent AI siap membaca grid di 'grids_to_read/' dan menghasilkan 'state/seed_labels.json'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siapkan grid montase sel kuesioner Seruti.")
    parser.add_argument("--source", type=str, default="output/crops", help="Direktori sumber atau berkas raw.zip")
    parser.add_argument("--booklet", type=str, default=None, help="Filter buklet spesifik (misal: booklet_73)")
    parser.add_argument("--page", type=str, default=None, help="Filter halaman spesifik (misal: hal_1)")
    args = parser.parse_args()
    
    siapkan_grid(source_path=args.source, booklet=args.booklet, page=args.page)
