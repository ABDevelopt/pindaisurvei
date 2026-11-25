#!/usr/bin/env python3
"""
build_template_from_pdf.py - Membuat template grid dari kuesioner PDF (vektor).

Karena PDF dibuat dari Word, semua garis tabel adalah objek vektor. Skrip ini:
  1. Menghapus teks dari halaman (redaksi teks saja) lalu merender grafiknya
     (garis, kotak, arsiran) pada 288 dpi (4 px per pt)
  2. Mengambil "face" (area terkurung garis) lewat connected components pada
     latar yang tidak tertutup garis; hanya face yang berbentuk persegi diterima
  3. Membedakan kotak digit (face kecil yang bersarang di dalam sel, atau
     kelompok kotak yang berdiri sendiri) dari sel struktur tabel
  4. Mengelompokkan sel menjadi tabel dan memberi indeks baris/kolom/span
  5. Mengelompokkan kotak digit menjadi field isian (+ label baris, misalnya
     "401", "406a", "01") dan memecah field di tanda ':' atau '/'
  6. Mengklasifikasi arsiran: header / title / band / na (sel tidak diisi)
  7. Menyimpan JSON per halaman (titik dan koordinat ternormalisasi) dan
     gambar pratinjau overlay

Pemakaian:
  python build_template_from_pdf.py kuesioner.pdf -o template/
  python build_template_from_pdf.py kuesioner.pdf -o template/ --pages 2 7

Dependensi: pip install pymupdf opencv-python numpy
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # versi lama
    import fitz

DPI = 288
S = DPI / 72.0            # piksel per pt
CODE_RE = re.compile(r"^\d{1,3}\.?$")
SUB_RE = re.compile(r"^([a-z])\.$")


# --------------------------------------------------------------------------- #
# Render grafik tanpa teks
# --------------------------------------------------------------------------- #
def render_graphics(pdf, pno):
    doc = fitz.open(pdf)
    pg = doc[pno]
    words = [(w[0], w[1], w[2], w[3], w[4]) for w in pg.get_text("words")]
    for w in words:
        pg.add_redact_annot(fitz.Rect(w[:4]))
    pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                        graphics=fitz.PDF_REDACT_LINE_ART_NONE)
    pix = pg.get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    return rgb, words, (pg.rect.width, pg.rect.height)


# --------------------------------------------------------------------------- #
# Ekstraksi face
# --------------------------------------------------------------------------- #
def ink_mask(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    strong = g < 120
    weak = ((g >= 120) & (g < 215)).astype(np.uint8)
    opened = cv2.morphologyEx(weak, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    thin = (weak > 0) & (opened == 0)         # garis abu-abu tipis, bukan arsiran
    ink = (strong | thin).astype(np.uint8)
    return cv2.dilate(ink, np.ones((3, 3), np.uint8))


def extract_faces(rgb):
    H, W = rgb.shape[:2]
    inv = (1 - ink_mask(rgb)).astype(np.uint8)
    num, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    faces = []
    for i in range(1, num):
        x, y, w, h, area = [int(v) for v in st[i]]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue                              # latar halaman
        if w / S < 2.5 or h / S < 2.5 or area > 0.5 * W * H:
            continue
        crop = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, hier = cv2.findContours(crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        outer = [c for c, hh in zip(cnts, hier[0]) if hh[3] == -1]
        holes = [c for c, hh in zip(cnts, hier[0]) if hh[3] != -1 and cv2.contourArea(c) >= 12]
        carea = max(cv2.contourArea(c) for c in outer)
        if carea <= 0 or carea / (w * h) < 0.9:
            continue                              # bukan persegi
        sel = lab[y:y + h, x:x + w] == i
        pix = rgb[y:y + h, x:x + w][sel]
        med = np.median(pix, axis=0)
        faces.append({
            "x0": (x - 2) / S, "y0": (y - 2) / S, "x1": (x + w + 2) / S, "y1": (y + h + 2) / S,
            "leaf": len(holes) == 0, "rgb": [int(v) for v in med],
        })
    return faces


def classify_fill(rgb):
    r, g, b = rgb
    if min(rgb) >= 250:
        return "white"
    if b - r >= 15:
        return "title"
    if b - r >= 6:
        return "header"
    lum = (r + g + b) / 3
    return "na" if lum < 200 else "band"


def contains(a, b, m=0.6):
    return a["x0"] - m <= b["x0"] and a["y0"] - m <= b["y0"] and a["x1"] + m >= b["x1"] and a["y1"] + m >= b["y1"]


def touches(a, b, tol=2.5):
    return not (a["x1"] + tol < b["x0"] or b["x1"] + tol < a["x0"] or a["y1"] + tol < b["y0"] or b["y1"] + tol < a["y0"])


def label_digits(faces):
    """Tandai kotak digit: face kecil berdaun yang bersarang dalam sel berlubang,
    atau yang hanya bersentuhan dengan face kecil lain (kotak berdiri sendiri)."""
    for f in faces:
        w, h = f["x1"] - f["x0"], f["y1"] - f["y0"]
        f["small_leaf"] = f["leaf"] and 5.5 <= w <= 24 and 7 <= h <= 26
        f["parent"] = None
    holed = [i for i, f in enumerate(faces) if not f["leaf"]]
    for i, f in enumerate(faces):
        if not f["small_leaf"]:
            continue
        best = None
        for j in holed:
            g = faces[j]
            if contains(g, f) and (best is None or (g["x1"] - g["x0"]) * (g["y1"] - g["y0"]) < best[1]):
                best = (j, (g["x1"] - g["x0"]) * (g["y1"] - g["y0"]))
        f["parent"] = best[0] if best else None
    for i, f in enumerate(faces):
        if not f["small_leaf"]:
            f["kind"] = "struct"
            continue
        w, h = f["x1"] - f["x0"], f["y1"] - f["y0"]
        if f["parent"] is not None or (min(f["rgb"]) >= 250 and 0.4 <= w / h <= 1.3):
            f["kind"] = "digit"                   # bersarang, atau kotak putih seukuran digit
            continue
        nb = [g for j, g in enumerate(faces) if j != i and touches(f, g)]
        f["kind"] = "digit" if all(g["small_leaf"] for g in nb) else "struct"


# --------------------------------------------------------------------------- #
# Teks, tabel, field
# --------------------------------------------------------------------------- #
def assign_text(faces, words):
    for f in faces:
        f["words"] = []
    order = sorted(range(len(faces)), key=lambda i: (faces[i]["x1"] - faces[i]["x0"]) * (faces[i]["y1"] - faces[i]["y0"]))
    for w in words:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        for i in order:
            f = faces[i]
            if f["x0"] <= cx <= f["x1"] and f["y0"] <= cy <= f["y1"]:
                f["words"].append(w)
                break
    for f in faces:
        ws = sorted(f["words"], key=lambda w: (round(w[1] / 3), w[0]))
        f["text"] = " ".join(w[4] for w in ws).strip()


def snap(vals, tol):
    vals = sorted(vals)
    groups, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def nearest(arr, v):
    return int(np.argmin([abs(a - v) for a in arr]))


def build_tables(faces):
    idx = [i for i, f in enumerate(faces) if f["kind"] == "struct"]
    parent = {i: i for i in idx}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a_i, a in enumerate(idx):
        for b in idx[a_i + 1:]:
            if touches(faces[a], faces[b], 2.0):
                parent[find(a)] = find(b)
    groups = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    tables = []
    for members in groups.values():
        if len(members) < 2:
            for i in members:                      # sel tunggal tetap dicatat
                faces[i]["table"] = None
            continue
        xs = snap([faces[i][k] for i in members for k in ("x0", "x1")], 2.2)
        ys = snap([faces[i][k] for i in members for k in ("y0", "y1")], 2.2)
        tid = len(tables)
        for i in members:
            f = faces[i]
            f["x0"], f["x1"] = xs[nearest(xs, f["x0"])], xs[nearest(xs, f["x1"])]
            f["y0"], f["y1"] = ys[nearest(ys, f["y0"])], ys[nearest(ys, f["y1"])]
            f["table"] = tid
            f["row"], f["row_end"] = nearest(ys, f["y0"]), nearest(ys, f["y1"])
            f["col"], f["col_end"] = nearest(xs, f["x0"]), nearest(xs, f["x1"])
        tables.append({"id": tid, "members": members, "xs": xs, "ys": ys,
                       "bbox": [xs[0], ys[0], xs[-1], ys[-1]]})
    return tables


def build_fields(faces, tables):
    digits = [i for i, f in enumerate(faces) if f["kind"] == "digit"]
    structs = [f for f in faces if f["kind"] == "struct"]
    rows = {}
    for i in digits:
        f = faces[i]
        cy = (f["y0"] + f["y1"]) / 2
        key = None
        for k in rows:
            if k[0] == f["parent"] and abs(k[1] - cy) < 0.5 * (f["y1"] - f["y0"]):
                key = k
                break
        rows.setdefault(key or (f["parent"], cy), []).append(i)
    fields = []
    for (par, cy), members in sorted(rows.items(), key=lambda kv: (kv[0][1], faces[kv[1][0]]["x0"])):
        members.sort(key=lambda i: faces[i]["x0"])
        # pecah di ':' atau '/' pada celah antar kotak (jam, tanggal)
        segs, cur = [], [members[0]]
        for a, b in zip(members[:-1], members[1:]):
            fa, fb = faces[a], faces[b]
            gap_words = []
            for f in faces:
                for w in f["words"]:
                    wx = (w[0] + w[2]) / 2
                    wy = (w[1] + w[3]) / 2
                    if fa["x1"] - 1 <= wx <= fb["x0"] + 1 and fa["y0"] <= wy <= fa["y1"] + 2:
                        gap_words.append(w[4])
            cyb = (fa["y0"] + fa["y1"]) / 2
            on_col_line = any(
                t["bbox"][1] - 1 <= cyb <= t["bbox"][3] + 1
                and any(fa["x1"] - 1.5 <= xv <= fb["x0"] + 1.5 for xv in t["xs"])
                for t in tables)
            if (any(t in (":", "/") for t in gap_words) or fb["x0"] - fa["x1"] > 14
                    or on_col_line):
                segs.append(cur)
                cur = []
            cur.append(b)
        segs.append(cur)
        for seg in segs:
            fields.append({"parent": par, "boxes": seg, "cy": cy})
    # label baris
    for fd in fields:
        bx = [faces[i] for i in fd["boxes"]]
        x0 = min(b["x0"] for b in bx)
        y0, y1 = min(b["y0"] for b in bx), max(b["y1"] for b in bx)
        ref = faces[fd["parent"]] if fd["parent"] is not None else None
        ry0, ry1 = y0, y1                           # rentang baris = tinggi field sendiri
        code, sub = None, None
        cands = []
        for s in structs:
            if s["x1"] > x0 + 1 or not s["text"]:
                continue
            ov = min(s["y1"], ry1) - max(s["y0"], ry0)
            if ov < 0.5 * min(ry1 - ry0, s["y1"] - s["y0"]):
                continue
            cands.append(s)
        cands.sort(key=lambda c: -c["x1"])          # dari sel terdekat ke yang terjauh
        for s in cands:
            tok = s["text"].split()[0]
            if CODE_RE.match(tok):
                code = tok.rstrip(".")
                break                                # kode ditemukan: berhenti
            m = SUB_RE.match(tok)
            if m and sub is None:
                sub = m.group(1)
        if sub and not code:                        # sub-baris (b, c, d) mewarisi kode di atasnya
            best = None
            for s in structs:
                toks = s["text"].split()
                if toks and CODE_RE.match(toks[0]) and s["x1"] <= x0 + 1 and s["y1"] <= ry0 + 2:
                    d = (ry0 - s["y1"], x0 - s["x1"])
                    if best is None or d < best[0]:
                        best = (d, toks[0].rstrip("."))
            code = best[1] if best else None
        fd["row_label"] = (code or "") + (sub or "")
        small_par = ref is not None and (ref["y1"] - ref["y0"]) <= 60
        fd["parent_text"] = ref["text"][:60] if small_par else ""
        fd["printed"] = "".join(faces[i]["text"] for i in fd["boxes"])
    return fields


# --------------------------------------------------------------------------- #
# Keluaran
# --------------------------------------------------------------------------- #
def r2(v):
    return round(float(v), 2)


def make_json(pno, size, faces, tables, fields, pdf_name):
    W, H = size
    ids = {}
    cells = []
    for i, f in enumerate(faces):
        cid = f"p{pno + 1}_c{len(cells):03d}"
        ids[i] = cid
        c = {
            "id": cid, "kind": f["kind"],
            "bbox_pt": [r2(f["x0"]), r2(f["y0"]), r2(f["x1"]), r2(f["y1"])],
            "bbox_n": [round(f["x0"] / W, 5), round(f["y0"] / H, 5), round(f["x1"] / W, 5), round(f["y1"] / H, 5)],
            "fill": classify_fill(f["rgb"]), "fill_rgb": f["rgb"], "text": f["text"],
        }
        if f["kind"] == "struct" and f.get("table") is not None:
            c.update({"table": f["table"], "row": f["row"], "col": f["col"],
                      "rowspan": f["row_end"] - f["row"], "colspan": f["col_end"] - f["col"]})
        if f["kind"] == "digit" and f["text"]:
            c["printed"] = f["text"]
        cells.append(c)
    flds = []
    for k, fd in enumerate(fields):
        bx = [faces[i] for i in fd["boxes"]]
        flds.append({
            "id": f"p{pno + 1}_F{k:03d}", "row_label": fd["row_label"], "n_boxes": len(bx),
            "boxes": [ids[i] for i in fd["boxes"]],
            "parent": ids[fd["parent"]] if fd["parent"] is not None else None,
            "parent_text": fd["parent_text"], "printed": fd["printed"] or None,
            "bbox_pt": [r2(min(b["x0"] for b in bx)), r2(min(b["y0"] for b in bx)),
                        r2(max(b["x1"] for b in bx)), r2(max(b["y1"] for b in bx))],
        })
    tabs = []
    for t in tables:
        mem = [ids[i] for i in t["members"]]
        tabs.append({"id": t["id"], "bbox_pt": [r2(v) for v in t["bbox"]],
                     "n_rows": len(t["ys"]) - 1, "n_cols": len(t["xs"]) - 1,
                     "x_lines_pt": [r2(v) for v in t["xs"]], "y_lines_pt": [r2(v) for v in t["ys"]],
                     "cells": mem})
    return {"source": pdf_name, "page": pno + 1, "size_pt": [r2(W), r2(H)],
            "coords": "bbox_pt dalam titik PDF (asal kiri-atas, y ke bawah); bbox_n = dibagi ukuran halaman",
            "counts": {"struct": sum(1 for c in cells if c["kind"] == "struct"),
                       "digit": sum(1 for c in cells if c["kind"] == "digit"),
                       "fields": len(flds), "tables": len(tabs)},
            "tables": tabs, "cells": cells, "fields": flds}


def preview(pdf, pno, tpl, out_png, dpi=100):
    doc = fitz.open(pdf)
    pix = doc[pno].get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy()
    k = dpi / 72.0
    ov = img.copy()
    for c in tpl["cells"]:
        x0, y0, x1, y1 = [int(v * k) for v in c["bbox_pt"]]
        if c["kind"] == "struct":
            if c["fill"] == "na":
                cv2.rectangle(ov, (x0, y0), (x1, y1), (60, 60, 230), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 150, 0), 1)
        else:
            cv2.rectangle(ov, (x0, y0), (x1, y1), (230, 140, 20), -1)
    img = cv2.addWeighted(ov, 0.30, img, 0.70, 0)
    for f in tpl["fields"]:
        x0, y0, x1, y1 = [int(v * k) for v in f["bbox_pt"]]
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (0, 120, 255), 1)
        if f["row_label"]:
            cv2.putText(img, f["row_label"], (x0, max(y0 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), img)


def main():
    ap = argparse.ArgumentParser(description="Template grid dari kuesioner PDF vektor.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("template"))
    ap.add_argument("--pages", type=int, nargs="*", help="nomor halaman (mulai 1); default semua")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    n = len(fitz.open(a.pdf))
    pages = [p - 1 for p in a.pages] if a.pages else range(n)
    for pno in pages:
        rgb, words, size = render_graphics(a.pdf, pno)
        faces = extract_faces(rgb)
        label_digits(faces)
        assign_text(faces, words)
        tables = build_tables(faces)
        fields = build_fields(faces, tables)
        tpl = make_json(pno, size, faces, tables, fields, a.pdf.name)
        js = a.output / f"template_p{pno + 1}.json"
        js.write_text(json.dumps(tpl, ensure_ascii=False, indent=1))
        preview(a.pdf, pno, tpl, a.output / f"template_p{pno + 1}_preview.png")
        c = tpl["counts"]
        print(f"hal {pno + 1}: {c['tables']} tabel, {c['struct']} sel struktur, "
              f"{c['digit']} kotak digit, {c['fields']} field -> {js.name}")


if __name__ == "__main__":
    main()
