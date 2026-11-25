#!/usr/bin/env python3
"""
lattice_overlay.py - Deteksi kisi (grid) formulir dengan Morphological Lattice
Projections, lalu timpa (overlay) ke gambar dan ekspor koordinat sel ke JSON.

Metode:
  1. Binarisasi tinta (adaptive threshold, tinta = putih)
  2. Opening morfologis dengan kernel garis panjang:
        horizontal (W x 1)  -> hanya garis horizontal yang bertahan
        vertikal   (1 x H)  -> hanya garis vertikal yang bertahan
     Teks dan tulisan tangan terhapus karena tidak cukup panjang.
  3. Proyeksi horizontal  -> posisi garis baris (puncak profil per-baris)
  4. Per pita baris (antara dua garis baris) lakukan proyeksi vertikal lokal
     -> posisi garis kolom khusus pita itu (kisi tidak harus lurus global)
  5. Sel = persegi antara dua garis baris dan dua garis kolom berurutan,
     disaring dengan uji "terkurung" (garis atas & bawah sel harus ada) dan
     filter ukuran.

Masukan sebaiknya gambar yang sudah dibersihkan (bersihkan_scan.py) dan
sudah tegak. Mendukung gambar grayscale maupun berwarna.

Pemakaian:
  python lattice_overlay.py halaman_kiri_bersih.png
  python lattice_overlay.py halaman.png -o hasil/ --min-cell 12 --no-fill
  python lattice_overlay.py inti/ -o hasil_lattice_inti/ --workers 4

Keluaran (di folder -o):
  <nama>_lattice.png    gambar dengan overlay (merah=garis baris, biru=garis kolom)
  <nama>_lattice.json   daftar baris dan sel {x0,y0,x1,y1,row,col}

Dependensi: pip install opencv-python numpy
"""
import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def group(idx, gap):
    """Kelompokkan indeks berurutan (jarak <= gap) menjadi satu puncak."""
    out, g = [], []
    for i in idx:
        if g and i - g[-1] > gap:
            out.append(g)
            g = []
        g.append(int(i))
    if g:
        out.append(g)
    return out


def odd(n):
    n = int(n)
    return n if n % 2 else n + 1


def line_mask(gray):
    """Respons garis gelap-tipis dengan black-hat pada gambar yang sudah
    dinormalisasi latarnya. Black-hat kebal terhadap bayangan dan area arsiran,
    sehingga garis di dalam sel abu-abu tetap terdeteksi."""
    H = gray.shape[0]
    bg = cv2.GaussianBlur(cv2.medianBlur(cv2.dilate(gray, np.ones((41, 41), np.uint8)), 31), (0, 0), 15)
    n = cv2.divide(gray, bg, scale=255)
    k = odd(max(9, H * 0.0055))
    bh = cv2.morphologyEx(n, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    t, _ = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (bh > max(12, 0.7 * t)).astype(np.uint8) * 255


def peaks(profile, thr, gap):
    return [int(np.mean(g)) for g in group(np.where(profile >= thr)[0], gap)]


def merge_close(xs, tol):
    out = []
    for x in sorted(xs):
        if out and x - out[-1][-1] <= tol:
            out[-1].append(x)
        else:
            out.append([x])
    return [int(round(np.mean(g))) for g in out]


def lattice(gray, h_frac=0.05, v_px=None, row_min=0.05, col_span=0.62,
            glob_support=0.2, local_support=0.35, min_cell=14,
            max_w_frac=0.92, max_h_frac=0.25):
    H, W = gray.shape
    v_px = v_px or max(12, int(H * 0.0099))
    bw = line_mask(gray)

    # --- opening morfologis: isolasi garis -------------------------------- #
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(W * h_frac)), 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_px))
    d = np.ones((3, 3), np.uint8)
    hor = cv2.dilate(cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh), d)
    ver = cv2.dilate(cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv), d)

    # --- proyeksi horizontal: garis baris --------------------------------- #
    ph = (hor > 0).sum(1)
    rows = peaks(ph, W * row_min, 4)
    heights = np.diff(rows) if len(rows) > 1 else np.array([v_px])
    med_h = float(np.median(heights[heights >= v_px - 4])) if (heights >= v_px - 4).any() else v_px

    # --- kelompokkan pita baris menjadi tabel (celah besar = tabel baru) -- #
    bands = [(y0, y1) for y0, y1 in zip(rows[:-1], rows[1:]) if y1 - y0 >= v_px - 4]
    tables, cur = [], []
    for b in bands:
        if cur and b[0] - cur[-1][1] > 1.2 * med_h:
            tables.append(cur)
            cur = []
        cur.append(b)
    if cur:
        tables.append(cur)

    # --- bangun masker kisi: garis terdeteksi + garis konsensus ------------- #
    grid = cv2.dilate(cv2.bitwise_or(hor, ver), d)
    band_cols = {}
    for tb in tables:
        ty0, ty1 = tb[0][0], tb[-1][1]
        pv_t = (ver[ty0:ty1] > 0).sum(0)
        glob = peaks(pv_t, max(glob_support * (ty1 - ty0), 2 * v_px), 3)
        tx = np.where(hor[ty0:ty1].sum(0) > 0)[0]
        xl, xr = (int(tx.min()), int(tx.max())) if len(tx) else (0, W - 1)
        for y0, y1 in tb:
            band = ver[y0 + 3:y1 - 2]
            pv = (band > 0).sum(0)
            loc = peaks(pv, max(col_span * (y1 - y0 - 5), 8), 3)
            extra = [x for x in glob
                     if pv[max(x - 3, 0):x + 4].max() >= local_support * (y1 - y0 - 5)]
            cols = merge_close(loc + extra, 5)
            band_cols[(y0, y1)] = cols
            for x in cols:                       # suntik garis kolom yang putus
                cv2.line(grid, (x, y0), (x, y1), 255, 3)
            cv2.line(grid, (xl, y0), (xr, y0), 255, 3)   # garis baris penuh
            cv2.line(grid, (xl, y1), (xr, y1), 255, 3)
    # jembatani celah kecil antar segmen garis
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # --- sel = komponen latar yang terkurung kisi -------------------------- #
    inv = cv2.bitwise_not(grid)
    num, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = st[i]
        if w < min_cell or h < min_cell - 2 or w > max_w_frac * W or h > max_h_frac * H:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        if area < 0.35 * w * h:          # bentuk tak persegi -> bukan sel
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))

    # indeks baris/kolom dari proyeksi: kelompokkan y0 lalu urutkan x0
    ys = merge_close([b[1] for b in boxes], max(6, int(0.4 * med_h)))
    cells = []
    for (x0, y0, x1, y1) in sorted(boxes, key=lambda b: (b[1], b[0])):
        r = int(np.argmin([abs(y0 - y) for y in ys]))
        cells.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "row": r})
    cells.sort(key=lambda c: (c["row"], c["x0"]))
    last, k = None, 0
    for c in cells:
        k = 0 if c["row"] != last else k + 1
        c["col"], last = k, c["row"]
    return hor, ver, rows, cells


def draw(gray, cells, fill=True):
    o = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if fill:
        ov = o.copy()
        for c in cells:
            cv2.rectangle(ov, (c["x0"], c["y0"]), (c["x1"], c["y1"]), (0, 190, 0), -1)
        o = cv2.addWeighted(ov, 0.16, o, 0.84, 0)
    for c in cells:
        cv2.line(o, (c["x0"], c["y0"]), (c["x1"], c["y0"]), (0, 0, 230), 1)
        cv2.line(o, (c["x0"], c["y1"]), (c["x1"], c["y1"]), (0, 0, 230), 1)
        cv2.line(o, (c["x0"], c["y0"]), (c["x0"], c["y1"]), (230, 90, 0), 1)
        cv2.line(o, (c["x1"], c["y0"]), (c["x1"], c["y1"]), (230, 90, 0), 1)
    return o


def coverage(cells, rows, shape):
    """Persentase luas tabel (antara garis baris pertama-terakhir) yang tercakup sel."""
    if len(rows) < 2 or not cells:
        return 0.0
    m = np.zeros(shape, np.uint8)
    for c in cells:
        m[c['y0']:c['y1'], c['x0']:c['x1']] = 1
    xs0 = min(c['x0'] for c in cells); xs1 = max(c['x1'] for c in cells)
    area = 0; got = 0
    for y0, y1 in zip(rows[:-1], rows[1:]):
        if y1 - y0 > shape[0] * 0.25:
            continue
        area += (y1 - y0) * (xs1 - xs0)
        got += int(m[y0:y1, xs0:xs1].sum())
    return got / max(area, 1)


def process_image(img_path, out_dir, min_cell=14, h_frac=0.05, glob_support=0.2, no_fill=False, verbose=True):
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            if verbose:
                print(f"[ERROR] Tidak bisa membaca {img_path}", file=sys.stderr)
            return {"image": img_path.name, "success": False, "error": "Cannot read image"}

        hor, ver, rows, cells = lattice(img, h_frac=h_frac, min_cell=min_cell, glob_support=glob_support)

        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{img_path.stem}_lattice.png"
        js = out_dir / f"{img_path.stem}_lattice.json"

        cv2.imwrite(str(png), draw(img, cells, fill=not no_fill))
        js.write_text(json.dumps({"image": img_path.name, "size": [img.shape[1], img.shape[0]],
                                  "rows": rows, "cells": cells}, indent=1), encoding="utf-8")
        cov = coverage(cells, rows, img.shape)
        if verbose:
            print(f"[{img_path.name}] {len(rows)} garis baris, {len(cells)} sel, cakupan area tabel {cov:.0%}")
        return {
            "image": img_path.name,
            "success": True,
            "rows": len(rows),
            "cells": len(cells),
            "coverage": cov,
            "png": str(png),
            "json": str(js)
        }
    except Exception as e:
        if verbose:
            print(f"[ERROR {img_path.name}] {e}", file=sys.stderr)
        return {"image": img_path.name, "success": False, "error": str(e)}


def _worker_task(task_args):
    img_path, target_out, min_cell, h_frac, glob_support, no_fill, verbose = task_args
    return process_image(img_path, target_out, min_cell=min_cell, h_frac=h_frac,
                         glob_support=glob_support, no_fill=no_fill, verbose=verbose)


def main():
    p = argparse.ArgumentParser(description="Overlay kisi dengan Morphological Lattice Projections.")
    p.add_argument("input", type=Path, help="Berkas gambar atau direktori gambar")
    p.add_argument("-o", "--output", type=Path, default=Path("hasil_lattice"), help="Folder output")
    p.add_argument("--min-cell", type=int, default=14, help="Lebar sel minimum (px)")
    p.add_argument("--h-frac", type=float, default=0.05, help="Panjang kernel horizontal (pecahan lebar)")
    p.add_argument("--glob-support", type=float, default=0.2,
                   help="Dukungan minimum (pecahan tinggi tabel) agar garis kolom dipakai di seluruh tabel")
    p.add_argument("--no-fill", action="store_true", help="Tanpa warna isi sel")
    p.add_argument("--workers", type=int, default=4, help="Jumlah worker paralel untuk pemrosesan folder (default: 4)")
    p.add_argument("--limit", type=int, default=None, help="Batasi jumlah gambar yang diproses")
    p.add_argument("--quiet", action="store_true", help="Sembunyikan log per gambar, hanya tampilkan agregat")
    a = p.parse_args()

    t_start = time.time()
    if a.input.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = [f for f in sorted(a.input.rglob("*")) if f.is_file() and f.suffix.lower() in image_exts]
        if not images:
            raise SystemExit(f"Tidak ada gambar ditemukan di direktori {a.input}")
        if a.limit:
            images = images[:a.limit]

        total = len(images)
        print(f"=== Memulai Lattice Overlay pada {total} berkas scan (workers={a.workers}) ===")
        print(f"Input: {a.input.resolve()}")
        print(f"Output: {a.output.resolve()}\n")

        tasks = []
        for img_path in images:
            rel_dir = img_path.parent.relative_to(a.input) if img_path.parent != a.input else Path()
            target_out = a.output / rel_dir
            tasks.append((img_path, target_out, a.min_cell, a.h_frac, a.glob_support, a.no_fill, not a.quiet))

        results = []
        done_count = 0
        if a.workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as executor:
                futures = [executor.submit(_worker_task, t) for t in tasks]
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    results.append(res)
                    done_count += 1
                    if done_count % 10 == 0 or done_count == total:
                        print(f"Progress: [{done_count}/{total}] berkas selesai ({(done_count/total)*100:.1f}%)")
        else:
            for t in tasks:
                res = _worker_task(t)
                results.append(res)
                done_count += 1
                if done_count % 10 == 0 or done_count == total:
                    print(f"Progress: [{done_count}/{total}] berkas selesai ({(done_count/total)*100:.1f}%)")

        successes = [r for r in results if r.get("success")]
        failures = [r for r in results if not r.get("success")]
        elapsed = time.time() - t_start

        avg_rows = np.mean([r["rows"] for r in successes]) if successes else 0
        avg_cells = np.mean([r["cells"] for r in successes]) if successes else 0
        avg_cov = np.mean([r["coverage"] for r in successes]) if successes else 0

        print("\n" + "="*50)
        print(f"RINGKASAN EKSEKUSI LATTICE OVERLAY:")
        print(f"Total Dokumen Diproses : {total}")
        print(f"Berhasil               : {len(successes)}")
        print(f"Gagal                  : {len(failures)}")
        print(f"Rata-rata Garis Baris  : {avg_rows:.1f}")
        print(f"Rata-rata Sel per Hal  : {avg_cells:.1f}")
        print(f"Rata-rata Cakupan Area : {avg_cov:.1%}")
        print(f"Total Durasi           : {elapsed:.2f} detik ({elapsed/total:.2f} s/gambar)")
        print(f"Folder Output          : {a.output.resolve()}")
        print("="*50)
    else:
        if not a.input.exists():
            raise SystemExit(f"Berkas tidak ditemukan: {a.input}")
        res = process_image(a.input, a.output, min_cell=a.min_cell, h_frac=a.h_frac,
                            glob_support=a.glob_support, no_fill=a.no_fill, verbose=True)
        if not res.get("success"):
            sys.exit(1)
        print(f"Selesai dalam {time.time() - t_start:.2f} detik -> Output: {a.output}")


if __name__ == "__main__":
    main()
