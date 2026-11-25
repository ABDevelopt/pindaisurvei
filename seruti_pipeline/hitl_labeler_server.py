#!/usr/bin/env python3
"""
seruti_pipeline/hitl_labeler_server.py
======================================
Aplikasi Human-in-the-Loop (HITL) Web Server untuk Koreksi & Verifikasi Label Manual.
Prioritas otomatis: Sel Berkonfidensi Rendah (< 0.85), Sel Ambiguitas, dan Diskrepansi Aritmetika.

Fitur Utama:
1. Keyboard-First Rapid Review: Tekan 0-9 untuk koreksi instan + otomatis lompat ke sel berikutnya.
2. Sumber Crop Utuh (Context Viewer): Menampilkan baris field utuh (dengan highlight kotak merah),
   kotak mentah (raw uncleaned scan), dan tabel sub-blok (meso).
3. Filter Cerdas: Antrean prioritas konfidensi rendah (<0.85), diskrepansi tri-model, atau per buklet.
4. Tampilan Citra Adaptif: Citra 28x28 diperbesar 6x tajam (168x168 px), toggle latar kertas/inversi.
5. Sinkronisasi Real-Time: Menyimpan audit log ke dataset/manual_corrections_log.jsonl,
   memperbarui ground truth JSON dan array .npz secara presisi.
"""

import argparse
import atexit
import base64
import copy
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from seruti_pipeline.validator import SerutiArithmeticValidator
except ImportError:
    try:
        from validator import SerutiArithmeticValidator
    except ImportError:
        SerutiArithmeticValidator = None

try:
    from seruti_pipeline.advanced_filter import remove_comb_box_border, letterbox_cell_28x28, is_blank_cell
except ImportError:
    try:
        from advanced_filter import remove_comb_box_border, letterbox_cell_28x28, is_blank_cell
    except ImportError:
        remove_comb_box_border = None
        letterbox_cell_28x28 = None
        is_blank_cell = None

DATASET_DIR = BASE_DIR / "dataset"
GT_DIR = DATASET_DIR / "ground_truth"
SPLITS_DIR = DATASET_DIR / "splits"
ARRAYS_DIR = DATASET_DIR / "data_arrays"
CROPS_DIR = BASE_DIR / "output" / "crops"
LOG_FILE = DATASET_DIR / "manual_corrections_log.jsonl"


class HITLDataManager:
    """Manajer data untuk antrean verifikasi manual dan penyimpanan audit."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items = []
        self.lookup_npz = {}
        self.npz_data = {}
        self.manifest_cache = {}
        self.dirty = False
        self.last_edit_time = 0.0
        self.last_save_time = time.time()
        self.auto_save_enabled = True
        self.refresh_dataset()
        self._start_autosave_worker()

    def _start_autosave_worker(self):
        """Worker background daemon untuk melakukan auto-save ke disk jika terdapat perubahan tertunda."""
        def worker():
            while True:
                time.sleep(1.0)
                if self.auto_save_enabled and self.dirty:
                    if time.time() - self.last_edit_time >= 1.5:
                        try:
                            res = self.sync_to_disk()
                            print(f"[*] Background Auto-Save sukses: {res.get('elapsed_seconds')}s pada {res.get('last_saved')}")
                        except Exception as e:
                            print(f"[-] Background Auto-Save error: {e}")
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def refresh_dataset(self):
        with self.lock:
            t0 = time.time()
            all_manifests = []
            for split_name in ["train", "val", "test"]:
                m_path = SPLITS_DIR / f"{split_name}_manifest.csv"
                if m_path.exists():
                    df = pd.read_csv(m_path)
                    df["split"] = split_name
                    df["sample_idx_in_split"] = range(len(df))
                    all_manifests.append(df)

            if not all_manifests:
                self.items = []
                return

            df_all = pd.concat(all_manifests, ignore_index=True)

            # Muat NPZ dalam format mutable dictionary
            for sp in ["train", "val", "test"]:
                npz_path = ARRAYS_DIR / f"{sp}.npz"
                if npz_path.exists():
                    loaded = np.load(npz_path, allow_pickle=True)
                    self.npz_data[sp] = {k: np.copy(loaded[k]) for k in loaded.files}

            # Bangun list item dengan skema fleksibel kompatibel multi-versi
            items = []
            for idx, row in df_all.iterrows():
                b_id = str(row.get("booklet_id") or "")
                if not b_id and "cell_id" in row and str(row["cell_id"]).startswith("booklet_"):
                    b_id = str(row["cell_id"]).split("_p")[0]

                p_num = int(row.get("page", 1))
                f_id = str(row.get("field_id", ""))
                v_name = str(row.get("var_name") or row.get("field_name") or f_id)
                b_idx = int(row.get("box_index") if pd.notna(row.get("box_index")) else row.get("box_idx", 0))
                lbl = int(row.get("label") if pd.notna(row.get("label")) else row.get("digit", 0))
                conf = float(row.get("conf") if pd.notna(row.get("conf")) else row.get("confidence", 1.0))
                reason = str(row.get("reason") or row.get("arbitration_reason") or row.get("provenance") or "")
                sample_idx = int(row.get("sample_idx_in_split", idx))

                prio_score = (1.0 - conf) * 100.0
                if "ARITHMETIC" in reason or "OVERRIDE" in reason or "TIE_BREAKER" in reason:
                    prio_score += 25.0
                if "BORDER" in reason or "SLIVER" in reason:
                    prio_score += 20.0
                if lbl == 1:
                    prio_score += 5.0

                items.append({
                    "id": int(idx),
                    "split": str(row["split"]),
                    "booklet_id": b_id,
                    "page": p_num,
                    "field_id": f_id,
                    "var_name": v_name,
                    "box_index": b_idx,
                    "label": lbl,
                    "conf": round(conf, 4),
                    "reason": reason,
                    "sample_idx": sample_idx,
                    "priority_score": round(prio_score, 2),
                    "reviewed": False,
                })

            # Baca audit log manual jika ada
            if LOG_FILE.exists():
                reviewed_keys = set()
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                log_entry = json.loads(line)
                                k = (log_entry["booklet_id"], int(log_entry["page"]), log_entry["field_id"], int(log_entry["box_index"]))
                                reviewed_keys.add(k)
                            except Exception:
                                pass
                for it in items:
                    if (it["booklet_id"], it["page"], it["field_id"], it["box_index"]) in reviewed_keys:
                        it["reviewed"] = True

            # Urutkan berdasarkan prioritas
            items.sort(key=lambda x: (x["reviewed"], -x["priority_score"]))
            self.items = items
            print(f"[*] HITL Data Manager dimuat: {len(self.items):,} sel dalam {round(time.time() - t0, 2)}s.")

    def get_stats(self):
        with self.lock:
            total = len(self.items)
            reviewed = sum(1 for x in self.items if x["reviewed"])
            low_conf = sum(1 for x in self.items if x["conf"] < 0.85)
            low_conf_pending = sum(1 for x in self.items if x["conf"] < 0.85 and not x["reviewed"])
            non_digits = sum(1 for x in self.items if x["label"] == -1)
            border_sliver_cnt = sum(1 for x in self.items if "BORDER" in x["reason"] or "SLIVER" in x["reason"])
            disagreements_cnt = sum(1 for x in self.items if "OVERRIDE" in x["reason"] or "ARITHMETIC" in x["reason"] or "TIE_BREAKER" in x["reason"])
            digit_1_cnt = sum(1 for x in self.items if x["label"] == 1)
            booklets = sorted(list(set(x["booklet_id"] for x in self.items if x["booklet_id"])))
            return {
                "total_cells": total,
                "reviewed_cells": reviewed,
                "pending_cells": total - reviewed,
                "low_confidence_total": low_conf,
                "low_confidence_pending": low_conf_pending,
                "border_sliver_total": border_sliver_cnt,
                "disagreements_total": disagreements_cnt,
                "digit_1_total": digit_1_cnt,
                "non_digit_cells": non_digits,
                "review_progress_pct": round(reviewed / max(1, total) * 100, 2),
                "booklets": booklets,
                "dirty": self.dirty,
                "auto_save_enabled": self.auto_save_enabled,
                "last_saved": datetime.fromtimestamp(self.last_save_time).strftime("%H:%M:%S") if self.last_save_time else None,
            }

    def _get_page_image(self, booklet_id: str, page: int) -> Optional[np.ndarray]:
        """Cache citra halaman pindaian resolusi tinggi untuk pemotongan on-the-fly."""
        if not hasattr(self, "_page_cache"):
            self._page_cache = {}
            self._page_cache_keys = []

        key = (booklet_id, page)
        if key in self._page_cache:
            return self._page_cache[key]

        page_path = BASE_DIR / "output" / "seruti_inti_booklets" / booklet_id / "pages" / f"hal_{page}_clean.png"
        if not page_path.exists():
            return None

        im = cv2.imread(str(page_path))
        if im is None:
            return None

        # Simpan di LRU cache kapasitas 8 halaman
        if len(self._page_cache_keys) >= 8:
            oldest = self._page_cache_keys.pop(0)
            self._page_cache.pop(oldest, None)

        self._page_cache[key] = im
        self._page_cache_keys.append(key)
        return im

    def extract_intact_digit_for_item(self, booklet_id: str, page: int, field_id: str, box_index: int) -> Optional[np.ndarray]:
        """
        Ekstraksi digit 100% utuh bebas bocor tetangga (Anti-Neighbor Bleed & 0-Px Drift):
        1. Prioritas 1: Baca clean_native (66x49) langsung dari disk jika tersedia.
        2. Prioritas 2: Baca raw_file crop dan bersihkan garis kisi in-situ via remove_comb_box_border.
        3. Prioritas 3 (Fallback): Potong langsung dari pindaian halaman sesuai [bx0, by0, bx1, by1] (0-px lateral padding)
           lalu terapkan pembersihan morfologi.
        """
        page = int(page)
        box_index = int(box_index)
        cache_key = (str(booklet_id), page)
        m_path = CROPS_DIR / booklet_id / f"hal_{page}" / "crop_manifest.json"
        manifest = None
        if m_path.exists():
            try:
                mtime = m_path.stat().st_mtime
                cached_entry = self.manifest_cache.get(cache_key)
                if cached_entry is None or cached_entry.get("_mtime") != mtime:
                    with open(m_path, "r", encoding="utf-8") as mf:
                        manifest = json.load(mf)
                        manifest["_mtime"] = mtime
                        self.manifest_cache[cache_key] = manifest
                else:
                    manifest = cached_entry
            except Exception:
                manifest = self.manifest_cache.get(cache_key)

        if not manifest:
            return None

        target_box = None
        target_item = None
        for m in manifest.get("tiers", {}).get("mikro", []):
            if m.get("field_id") == field_id and int(m.get("box_index", -1)) == box_index:
                target_box = m.get("bbox_px")
                target_item = m
                break

        # Prioritas 1: Baca clean_native_file langsung dari disk jika tersedia
        if target_item and target_item.get("clean_native_file"):
            cnf_path = BASE_DIR / target_item["clean_native_file"]
            if cnf_path.exists():
                im = cv2.imread(str(cnf_path))
                if im is not None:
                    return im

        # Prioritas 2: Baca raw_file crop dan bersihkan garis kisi in-situ (resolusi asli ~66x49)
        if target_item and target_item.get("raw_file"):
            rf_path = BASE_DIR / target_item["raw_file"]
            if rf_path.exists():
                im = cv2.imread(str(rf_path))
                if im is not None:
                    if remove_comb_box_border:
                        return remove_comb_box_border(im)
                    return im

        # Prioritas 3: Baca clean_file langsung dari disk jika tersedia
        if target_item and target_item.get("clean_file"):
            cf_path = BASE_DIR / target_item["clean_file"]
            if cf_path.exists():
                im = cv2.imread(str(cf_path))
                if im is not None:
                    return im

        # Prioritas 4 (Fallback): Potong langsung dari citra halaman utuh sesuai koordinat fisik m_box
        if not target_box or len(target_box) != 4:
            return None

        bx0, by0, bx1, by1 = target_box
        page_img = self._get_page_image(booklet_id, page)
        if page_img is None:
            return None

        H_img, W_img = page_img.shape[:2]
        crop = page_img[max(0, by0):min(H_img, by1), max(0, bx0):min(W_img, bx1)].copy()
        if remove_comb_box_border:
            return remove_comb_box_border(crop)
        return crop

    def get_queue(self, filter_mode="low_conf", offset=0, limit=20, booklet_id=None, query=None):
        with self.lock:
            filtered = self.items

            if booklet_id and booklet_id != "all":
                filtered = [x for x in filtered if x["booklet_id"] == booklet_id]

            if query and query.strip():
                q_lower = query.strip().lower()
                filtered = [
                    x for x in filtered 
                    if q_lower in str(x.get("var_name", "")).lower() 
                    or q_lower in str(x.get("field_id", "")).lower()
                    or q_lower in f"box_{x.get('box_index', '')}".lower()
                    or q_lower in f"p{x.get('page', '')}".lower()
                ]

            if filter_mode == "low_conf":
                filtered = [x for x in filtered if x["conf"] < 0.85]
            elif filter_mode == "disagreements":
                filtered = [x for x in filtered if "OVERRIDE" in x["reason"] or "ARITHMETIC" in x["reason"] or "TIE_BREAKER" in x["reason"]]
            elif filter_mode == "border_sliver":
                filtered = [x for x in filtered if "BORDER" in x["reason"] or "SLIVER" in x["reason"]]
            elif filter_mode == "digit_1":
                filtered = [x for x in filtered if x["label"] == 1]
            elif filter_mode == "non_digit":
                filtered = [x for x in filtered if x["label"] == -1]
            elif filter_mode == "pending":
                filtered = [x for x in filtered if not x["reviewed"]]
            elif filter_mode == "reviewed":
                filtered = [x for x in filtered if x["reviewed"]]

            total_filtered = len(filtered)
            page_items = filtered[offset : offset + limit]

            # Tambahkan image data base64 untuk render instan di browser
            result = []
            for item in page_items:
                item_copy = dict(item)
                sp = item["split"]
                s_idx = item["sample_idx"]
                im_base64 = ""

                # Prioritas 1: Ekstraksi citra resolusi asli (Native Clean ~66x49) bebas distorsi & bebas garis kotak
                b_id = str(item["booklet_id"])
                p_num = int(item["page"])
                fid = str(item["field_id"])
                b_idx = int(item["box_index"])
                extracted = self.extract_intact_digit_for_item(b_id, p_num, fid, b_idx)
                print(f"[DEBUG get_queue] {b_id} p{p_num} {fid} b{b_idx}: extracted={extracted.shape if extracted is not None else None}", flush=True)
                if extracted is not None:
                    _, buf = cv2.imencode(".png", extracted)
                    im_base64 = base64.b64encode(buf).decode("utf-8")

                # Prioritas 2 (Fallback): Ambil dari memori npz_data jika ekstraksi native belum tersedia
                if not im_base64 and sp in self.npz_data and s_idx < len(self.npz_data[sp]["X"]):
                    raw_28 = self.npz_data[sp]["X"][s_idx]
                    if cv2.countNonZero(raw_28) > 0:
                        _, buf = cv2.imencode(".png", raw_28)
                        im_base64 = base64.b64encode(buf).decode("utf-8")

                item_copy["image_base64"] = im_base64
                result.append(item_copy)

            return {
                "total": total_filtered,
                "offset": offset,
                "limit": limit,
                "items": result,
            }

    def get_context_image(self, booklet_id: str, page: int, field_id: str, box_index: int, tier: str = "field") -> bytes:
        """
        Menghasilkan citra sumber crop utuh (field strip, raw box, atau meso table)
        dengan anotasi highlight kotak aktif.
        """
        print(f"[DEBUG context_image] booklet={booklet_id}, page={page}, field={field_id}, box={box_index}, tier={tier}", flush=True)
        with self.lock:
            cache_key = (booklet_id, page)
            m_path = CROPS_DIR / booklet_id / f"hal_{page}" / "crop_manifest.json"

            # Auto-reload manifest jika berkas di disk berubah (mtime check) atau belum termuat
            if m_path.exists():
                try:
                    mtime = m_path.stat().st_mtime
                    cached_entry = self.manifest_cache.get(cache_key)
                    if cached_entry is None or cached_entry.get("_mtime") != mtime:
                        with open(m_path, "r", encoding="utf-8") as mf:
                            manifest_data = json.load(mf)
                            manifest_data["_mtime"] = mtime
                            self.manifest_cache[cache_key] = manifest_data
                except Exception:
                    pass
            else:
                self.manifest_cache[cache_key] = None

            manifest = self.manifest_cache.get(cache_key)
            if not manifest:
                print(f"[DEBUG context_image] Manifest not found for {cache_key}", flush=True)
                # Buat citra placeholder kosong jika manifest tidak ada
                blank = np.ones((80, 400, 3), dtype=np.uint8) * 240
                cv2.putText(blank, "Manifest Crop Tidak Ditemukan", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
                _, buf = cv2.imencode(".png", blank)
                return buf.tobytes()

            tiers = manifest.get("tiers", {})
            mikro_list = tiers.get("mikro", [])
            rows_list = tiers.get("rows", [])
            fields_list = tiers.get("fields", [])
            meso_list = tiers.get("meso", [])

            # Cari item mikro yang sesuai
            target_mikro = None
            for m in mikro_list:
                if m.get("field_id") == field_id and int(m.get("box_index", -1)) == box_index:
                    target_mikro = m
                    break

            # 1. Tier Raw Box (Kotak Mentah Asli dengan Padded Context & Penanda Batas Comb)
            if tier == "raw":
                # Prioritaskan pemotongan langsung dari pindaian asli beresolusi tinggi dengan padding lapang
                page_img = self._get_page_image(booklet_id, page)
                if page_img is not None and target_mikro and "bbox_px" in target_mikro:
                    bx0, by0, bx1, by1 = target_mikro["bbox_px"]
                    pad_y = 12
                    pad_x = 8
                    y0 = max(0, by0 - pad_y)
                    y1 = min(page_img.shape[0], by1 + pad_y)
                    x0 = max(0, bx0 - pad_x)
                    x1 = min(page_img.shape[1], bx1 + pad_x)
                    crop = page_img[y0:y1, x0:x1].copy()

                    # Tandai batas fisik comb box asli dengan outline tipis penjelas
                    fx0 = bx0 - x0
                    fy0 = by0 - y0
                    fx1 = bx1 - x0
                    fy1 = by1 - y0
                    cv2.rectangle(crop, (fx0, fy0), (fx1, fy1), (0, 215, 255), 1)

                    im_up = cv2.resize(crop, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
                    _, buf = cv2.imencode(".png", im_up)
                    return buf.tobytes()

                # Fallback ke raw_file jika pindaian booklet belum termuat
                if target_mikro and target_mikro.get("raw_file"):
                    rf = BASE_DIR / target_mikro["raw_file"]
                    if rf.exists():
                        im = cv2.imread(str(rf))
                        if im is not None:
                            im_up = cv2.resize(im, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
                            _, buf = cv2.imencode(".png", im_up)
                            return buf.tobytes()

            # 2. Tier Field (Baris Field Penuh dengan Highlight Kotak Aktif)
            target_field = None
            for f in fields_list:
                if f.get("field_id") == field_id:
                    target_field = f
                    break

            if tier == "field":
                # Prioritas 1: Baca file crop field dari disk
                if target_field and target_field.get("file"):
                    ff = BASE_DIR / target_field["file"]
                    if ff.exists():
                        im = cv2.imread(str(ff))
                        if im is not None:
                            # Gambar kotak highlight jika koordinat mikro ditemukan
                            if target_mikro and "bbox_px" in target_mikro and "bbox_px" in target_field:
                                f_box = target_field["bbox_px"]
                                m_box = target_mikro["bbox_px"]
                                rx1 = max(0, m_box[0] - f_box[0])
                                ry1 = max(0, m_box[1] - f_box[1])
                                rx2 = min(im.shape[1], m_box[2] - f_box[0])
                                ry2 = min(im.shape[0], m_box[3] - f_box[1])

                                # Gambar kotak merah tebal & border ganda cerah
                                cv2.rectangle(im, (rx1, ry1), (rx2, ry2), (0, 0, 240), 2)
                                cv2.rectangle(im, (max(0, rx1-1), max(0, ry1-1)), (min(im.shape[1], rx2+1), min(im.shape[0], ry2+1)), (0, 255, 255), 1)

                            # Skala perbesar 2x agar teks tajam
                            im_up = cv2.resize(im, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                            _, buf = cv2.imencode(".png", im_up)
                            return buf.tobytes()

                # Prioritas 2 (Fallback): Potong langsung dari citra halaman utuh
                page_img = self._get_page_image(booklet_id, page)
                if page_img is not None and target_field and "bbox_px" in target_field:
                    fbx0, fby0, fbx1, fby1 = target_field["bbox_px"]
                    pad = 4
                    y0 = max(0, fby0 - pad)
                    y1 = min(page_img.shape[0], fby1 + pad)
                    x0 = max(0, fbx0 - pad)
                    x1 = min(page_img.shape[1], fbx1 + pad)
                    im = page_img[y0:y1, x0:x1].copy()

                    if target_mikro and "bbox_px" in target_mikro:
                        mb = target_mikro["bbox_px"]
                        rx1 = max(0, mb[0] - x0)
                        ry1 = max(0, mb[1] - y0)
                        rx2 = min(im.shape[1], mb[2] - x0)
                        ry2 = min(im.shape[0], mb[3] - y0)
                        cv2.rectangle(im, (rx1, ry1), (rx2, ry2), (0, 0, 240), 2)
                        cv2.rectangle(im, (max(0, rx1-1), max(0, ry1-1)), (min(im.shape[1], rx2+1), min(im.shape[0], ry2+1)), (0, 255, 255), 1)

                    im_up = cv2.resize(im, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                    _, buf = cv2.imencode(".png", im_up)
                    return buf.tobytes()

            # 3. Tier Row (Baris Roster / Data Horizontal Utuh dengan Highlight Kotak Aktif)
            if tier == "row":
                target_row = None
                # Cari baris yang memuat field_id
                for r in rows_list:
                    if field_id in r.get("fields", []):
                        target_row = r
                        break

                # Fallback: cari baris berdasarkan interseksi vertikal bounding box
                if not target_row and target_mikro and "bbox_px" in target_mikro:
                    mb = target_mikro["bbox_px"]
                    for r in rows_list:
                        rb = r.get("bbox_px", [0, 0, 0, 0])
                        if rb[1] <= mb[1] and rb[3] >= mb[3]:
                            target_row = r
                            break

                print(f"[DEBUG tier==row] rows_count={len(rows_list)}, target_row={target_row is not None}", flush=True)

                # Prioritas 1: Baca file crop row dari disk
                if target_row and target_row.get("file"):
                    rf_path = BASE_DIR / target_row["file"]
                    print(f"[DEBUG target_row] file={target_row.get('file')}, exists={rf_path.exists()}", flush=True)
                    if rf_path.exists():
                        im = cv2.imread(str(rf_path))
                        if im is not None:
                            # Gambar kotak highlight jika koordinat mikro ditemukan
                            if target_mikro and "bbox_px" in target_mikro and "bbox_px" in target_row:
                                r_box = target_row["bbox_px"]
                                m_box = target_mikro["bbox_px"]
                                rx1 = max(0, m_box[0] - r_box[0])
                                ry1 = max(0, m_box[1] - r_box[1])
                                rx2 = min(im.shape[1], m_box[2] - r_box[0])
                                ry2 = min(im.shape[0], m_box[3] - r_box[1])

                                cv2.rectangle(im, (rx1, ry1), (rx2, ry2), (0, 0, 240), 2)
                                cv2.rectangle(im, (max(0, rx1-1), max(0, ry1-1)), (min(im.shape[1], rx2+1), min(im.shape[0], ry2+1)), (0, 255, 255), 1)

                            if im.shape[0] < 120:
                                im = cv2.resize(im, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

                            _, buf = cv2.imencode(".png", im)
                            return buf.tobytes()

                # Prioritas 2 (Fallback): Potong baris langsung dari citra halaman utuh
                page_img = self._get_page_image(booklet_id, page)
                if page_img is not None and target_row and "bbox_px" in target_row:
                    rbx0, rby0, rbx1, rby1 = target_row["bbox_px"]
                    im = page_img[rby0:rby1, rbx0:rbx1].copy()
                    if target_mikro and "bbox_px" in target_mikro:
                        mb = target_mikro["bbox_px"]
                        rx1 = max(0, mb[0] - rbx0)
                        ry1 = max(0, mb[1] - rby0)
                        rx2 = min(im.shape[1], mb[2] - rbx0)
                        ry2 = min(im.shape[0], mb[3] - rby0)
                        cv2.rectangle(im, (rx1, ry1), (rx2, ry2), (0, 0, 240), 2)
                        cv2.rectangle(im, (max(0, rx1-1), max(0, ry1-1)), (min(im.shape[1], rx2+1), min(im.shape[0], ry2+1)), (0, 255, 255), 1)

                    if im.shape[0] < 120:
                        im = cv2.resize(im, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

                    _, buf = cv2.imencode(".png", im)
                    return buf.tobytes()

            # 3. Tier Meso (Tabel Sub-Blok Pertanyaan)
            if tier == "meso":
                if target_field and "bbox_px" in target_field:
                    fb = target_field["bbox_px"]
                    for me in meso_list:
                        mb = me.get("bbox_px", [0, 0, 0, 0])
                        if mb[0] <= fb[0] and mb[1] <= fb[1] and mb[2] >= fb[2] and mb[3] >= fb[3]:
                            mf_file = BASE_DIR / me["file"]
                            if mf_file.exists():
                                im = cv2.imread(str(mf_file))
                                if im is not None:
                                    rx1 = max(0, fb[0] - mb[0])
                                    ry1 = max(0, fb[1] - mb[1])
                                    rx2 = min(im.shape[1], fb[2] - mb[0])
                                    ry2 = min(im.shape[0], fb[3] - mb[1])
                                    cv2.rectangle(im, (rx1, ry1), (rx2, ry2), (0, 0, 240), 3)
                                    _, buf = cv2.imencode(".png", im)
                                    return buf.tobytes()
                    # Fallback ke meso pertama jika ada
                    if meso_list and meso_list[0].get("file"):
                        mf_file = BASE_DIR / meso_list[0]["file"]
                        if mf_file.exists():
                            im = cv2.imread(str(mf_file))
                            if im is not None:
                                _, buf = cv2.imencode(".png", im)
                                return buf.tobytes()

            # Placeholder fallback
            blank = np.ones((80, 400, 3), dtype=np.uint8) * 240
            cv2.putText(blank, f"Konteks {tier} Belum Tersedia", (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)
            _, buf = cv2.imencode(".png", blank)
            return buf.tobytes()

    def save_correction(self, booklet_id, page, field_id, box_index, new_label, reviewer="Human Verifier"):
        with self.lock:
            found = False
            prev_label = None
            split_name = None
            sample_idx = None
            var_name = field_id

            for it in self.items:
                if (it["booklet_id"] == booklet_id and it["page"] == page and
                        it["field_id"] == field_id and it["box_index"] == box_index):
                    prev_label = it["label"]
                    it["label"] = int(new_label)
                    it["reviewed"] = True
                    it["reason"] = f"HUMAN_VERIFIED ({reviewer})" if int(new_label) >= 0 else f"HUMAN_REJECT_NON_DIGIT ({reviewer})"
                    it["conf"] = 1.0
                    split_name = it["split"]
                    sample_idx = it["sample_idx"]
                    var_name = it.get("var_name", field_id)
                    found = True
                    break

            if not found:
                return {"success": False, "error": "Item tidak ditemukan"}

            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "booklet_id": booklet_id,
                "page": page,
                "field_id": field_id,
                "var_name": var_name,
                "box_index": box_index,
                "previous_label": prev_label,
                "corrected_label": int(new_label),
                "reviewer": reviewer,
            }
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry) + "\n")

            val_summary = None
            gt_file = GT_DIR / f"{booklet_id}.json"
            if gt_file.exists():
                try:
                    with open(gt_file, "r", encoding="utf-8") as gf:
                        gt_doc = json.load(gf)

                    # Perbarui raw_strings pada karakter box_index
                    raw_target_key = None
                    if "raw_strings" in gt_doc:
                        if var_name in gt_doc["raw_strings"]:
                            raw_target_key = var_name
                        elif field_id in gt_doc["raw_strings"]:
                            raw_target_key = field_id

                    if raw_target_key and raw_target_key in gt_doc["raw_strings"]:
                        old_raw = list(str(gt_doc["raw_strings"][raw_target_key]))
                        if 0 <= box_index < len(old_raw):
                            new_char = str(new_label) if int(new_label) >= 0 else "_"
                            old_raw[box_index] = new_char
                            updated_raw = "".join(old_raw)
                            gt_doc["raw_strings"][raw_target_key] = updated_raw

                            clean_digits = updated_raw.replace("_", "").strip()
                            if "fields" in gt_doc:
                                gt_doc["fields"][raw_target_key] = int(clean_digits) if clean_digits.isdigit() else None

                    gt_doc["summary"]["manual_corrections"] = gt_doc["summary"].get("manual_corrections", 0) + 1

                    # Revalidasi konsistensi aritmetika secara deterministik
                    try:
                        if SerutiArithmeticValidator is not None:
                            validator = SerutiArithmeticValidator()
                            report = validator.validate_booklet(gt_doc, auto_reconcile=False)
                            gt_doc["validation_report"] = report.summary.to_dict()
                            gt_doc["summary"]["rules_passed"] = report.summary.rules_passed
                            gt_doc["summary"]["pass_rate_pct"] = report.summary.pass_rate_pct
                            gt_doc["summary"]["total_discrepancy_rupiah"] = report.summary.total_discrepancy_rupiah
                            val_summary = report.summary.to_dict()
                    except Exception as ve:
                        print(f"[-] Revalidasi aritmetika lewati: {ve}")

                    with open(gt_file, "w", encoding="utf-8") as gf:
                        json.dump(gt_doc, gf, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[-] Gagal update ground truth JSON: {e}")

            if split_name and split_name in self.npz_data and sample_idx is not None:
                try:
                    self.npz_data[split_name]["y"][sample_idx] = int(new_label)
                except Exception as e:
                    print(f"[-] Gagal update memory npz: {e}")

            self.dirty = True
            self.last_edit_time = time.time()

            return {
                "success": True,
                "booklet_id": booklet_id,
                "field_id": field_id,
                "var_name": var_name,
                "box_index": box_index,
                "prev_label": prev_label,
                "new_label": int(new_label),
                "dirty": self.dirty,
                "validation_report": val_summary,
            }

    def sync_to_disk(self):
        with self.lock:
            t0 = time.time()
            for sp, data in self.npz_data.items():
                npz_file = ARRAYS_DIR / f"{sp}.npz"
                save_kwargs = {"X": data["X"], "y": data["y"]}
                for extra in ["booklet_ids", "field_ids", "var_names", "confs", "verified"]:
                    if extra in data:
                        save_kwargs[extra] = data[extra]
                np.savez_compressed(npz_file, **save_kwargs)

            # Perbarui file manifest CSV jika ada perubahan
            for sp in ["train", "val", "test"]:
                m_path = SPLITS_DIR / f"{sp}_manifest.csv"
                if m_path.exists():
                    try:
                        df_m = pd.read_csv(m_path)
                        col_digit = "digit" if "digit" in df_m.columns else ("label" if "label" in df_m.columns else None)
                        col_prov = "provenance" if "provenance" in df_m.columns else ("arbitration_reason" if "arbitration_reason" in df_m.columns else None)
                        
                        updated = False
                        for it in self.items:
                            if it["split"] == sp and it["reviewed"]:
                                s_idx = it["sample_idx"]
                                if s_idx < len(df_m):
                                    if col_digit:
                                        df_m.at[s_idx, col_digit] = it["label"]
                                    if col_prov:
                                        df_m.at[s_idx, col_prov] = it["reason"]
                                    updated = True
                        if updated:
                            df_m.to_csv(m_path, index=False)
                    except Exception as me:
                        print(f"[-] Gagal sinkron manifest {sp}: {me}")

            self.dirty = False
            self.last_save_time = time.time()
            return {
                "success": True,
                "elapsed_seconds": round(time.time() - t0, 2),
                "last_saved": datetime.fromtimestamp(self.last_save_time).strftime("%H:%M:%S")
            }


# Inisialisasi Singleton Data Manager
DATA_MANAGER = HITLDataManager()

# Register atexit handler untuk menjamin perubahan otomatis disimpan saat shutdown
def _on_server_exit():
    if DATA_MANAGER.dirty:
        print("[*] Server shutdown terdeteksi: Melakukan flush auto-save perubahan tertunda ke disk...")
        try:
            res = DATA_MANAGER.sync_to_disk()
            print(f"[+] Flush auto-save tuntas dalam {res.get('elapsed_seconds')}s.")
        except Exception as e:
            print(f"[-] Gagal flush auto-save: {e}")

atexit.register(_on_server_exit)


# =============================================================================
# TEMPLATE HTML / UI (DENGAN VIEWER SUMBER CROP UTUH & LIGHTBOX MODAL)
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BPS Seruti Inti - HITL Labeling Verifier</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .pixelated { image-rendering: pixelated; image-rendering: -moz-crisp-edges; image-rendering: crisp-edges; }
    .glow-active { box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.4); }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col antialiased select-none">

  <!-- Header Navigasi -->
  <header class="bg-slate-900 border-b border-slate-800 px-6 py-3 flex items-center justify-between sticky top-0 z-50 shadow-md">
    <div class="flex items-center gap-3">
      <div class="w-9 h-9 rounded-lg bg-emerald-600 flex items-center justify-center font-bold text-white shadow-lg shadow-emerald-900/40">
        BP
      </div>
      <div>
        <h1 class="text-base font-bold text-white tracking-tight flex items-center gap-2">
          BPS Seruti Inti <span class="bg-emerald-500/10 text-emerald-400 text-xs px-2 py-0.5 rounded border border-emerald-500/20 font-medium">HITL Verifier</span>
        </h1>
        <p class="text-xs text-slate-400">Verifikasi & Koreksi Manual Kuesioner (Prioritas Confidence Rendah)</p>
      </div>
    </div>

    <!-- Quick Stats -->
    <div class="flex items-center gap-4 text-xs">
      <div class="bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700/60">
        <span class="text-slate-400">Total:</span>
        <span id="stat-total" class="font-bold text-slate-200 ml-1">0</span>
      </div>
      <div class="bg-amber-500/10 px-3 py-1.5 rounded-lg border border-amber-500/20 text-amber-400">
        <span>Conf Rendah:</span>
        <span id="stat-low-conf" class="font-bold ml-1">0</span>
      </div>
      <div class="bg-purple-500/10 px-3 py-1.5 rounded-lg border border-purple-500/20 text-purple-400" title="Perbedaan prediksi multi-model AI">
        <span>Disagreements:</span>
        <span id="stat-disagreements" class="font-bold ml-1">0</span>
      </div>
      <div class="bg-cyan-500/10 px-3 py-1.5 rounded-lg border border-cyan-500/20 text-cyan-400" title="Residu garis kisi comb-box yang disterilkan">
        <span>Anti-False 1:</span>
        <span id="stat-border-sliver" class="font-bold ml-1">0</span>
      </div>
      <div class="bg-emerald-500/10 px-3 py-1.5 rounded-lg border border-emerald-500/20 text-emerald-400">
        <span>Reviewed:</span>
        <span id="stat-reviewed" class="font-bold ml-1">0</span>
        <span id="stat-pct" class="text-emerald-500/80 ml-0.5">(0%)</span>
      </div>

      <!-- Auto-Save & Manual Save Controls -->
      <div class="flex items-center gap-2 bg-slate-900/90 p-1 rounded-xl border border-slate-700/80 shadow-inner">
        <!-- Auto-Save Toggle -->
        <label class="flex items-center gap-1.5 cursor-pointer px-2 py-1 rounded-lg hover:bg-slate-800 transition select-none" title="Aktifkan/Nonaktifkan Auto-Save otomatis ke disk">
          <input type="checkbox" id="toggle-autosave" onchange="toggleAutoSave(this.checked)" checked class="w-3.5 h-3.5 text-emerald-500 rounded bg-slate-800 border-slate-600 focus:ring-0 cursor-pointer accent-emerald-500">
          <span class="text-[11px] font-semibold text-slate-300">Auto-Save</span>
        </label>

        <!-- Save Status Badge -->
        <div id="save-status" class="text-[11px] font-mono px-2 py-1 rounded-md text-slate-400 flex items-center gap-1.5 bg-slate-950/60 border border-slate-800">
          <span class="w-2 h-2 rounded-full bg-emerald-400" id="save-indicator"></span>
          <span id="save-text" class="text-emerald-400">Tersimpan</span>
        </div>

        <!-- Tombol Simpan Manual -->
        <button id="btn-save-manual" onclick="manualSave()" class="bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1.5 rounded-lg font-bold text-xs transition flex items-center gap-1.5 shadow-md shadow-emerald-950 active:scale-95" title="Simpan perubahan ke disk sekarang [Ctrl + S]">
          <svg id="save-icon" class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"/></svg>
          <span>Simpan [Ctrl+S]</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Content Layout -->
  <main class="flex-1 flex overflow-hidden">

    <!-- Panel Kontrol & Filter Kiri -->
    <aside class="w-72 bg-slate-900/60 border-r border-slate-800/80 p-4 flex flex-col gap-4 overflow-y-auto">
      <!-- Pilihan Buklet -->
      <div>
        <label class="text-xs font-semibold text-slate-400 tracking-wider uppercase block mb-1.5">Pilih Buklet Target</label>
        <select id="booklet-filter" onchange="changeBooklet()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500">
          <option value="all" selected>Semua Buklet (76 Buklet)</option>
        </select>
      </div>

      <!-- Filter Antrean -->
      <div>
        <label class="text-xs font-semibold text-slate-400 tracking-wider uppercase block mb-1.5">Filter Antrean AI</label>
        <select id="filter-mode" onchange="changeFilter()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500">
          <option value="low_conf" selected>Prioritas: Conf Rendah (&lt; 0.85)</option>
          <option value="disagreements">Perbedaan Prediksi / TrOCR Tie-Break</option>
          <option value="border_sliver">Residu Garis Kisi (Anti-False 1)</option>
          <option value="digit_1">Kandidat Angka 1 (Verifikasi Tepi)</option>
          <option value="non_digit">Sel Ditolak / Bukan Angka (-1)</option>
          <option value="pending">Sel Belum Direview</option>
          <option value="reviewed">Sel Sudah Direview</option>
          <option value="all">Semua Sel Karakter</option>
        </select>
      </div>

      <!-- Pencarian Cepat Variabel / Field ID / Box Index -->
      <div>
        <label class="text-xs font-semibold text-slate-400 tracking-wider uppercase block mb-1.5">Cari Variabel / Field ID</label>
        <div class="relative">
          <input type="text" id="search-box" oninput="handleSearch()" placeholder="Misal: b6_r602a atau p7_F002..." class="w-full bg-slate-800 border border-slate-700 rounded-lg pl-8 pr-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 placeholder-slate-500">
          <svg class="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
        </div>
      </div>

      <!-- Panduan Tombol Pintas -->
      <div class="bg-slate-800/50 rounded-xl p-3 border border-slate-700/50">
        <h3 class="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
          <svg class="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/></svg>
          Keyboard Shortcuts
        </h3>
        <div class="space-y-1.5 text-xs text-slate-400">
          <div class="flex justify-between items-center"><span class="bg-emerald-950/80 border border-emerald-600/50 text-emerald-300 px-1.5 py-0.5 rounded mono font-bold">Ctrl + S</span><span>Simpan ke Disk</span></div>
          <div class="flex justify-between items-center"><span class="bg-slate-700 px-1.5 py-0.5 rounded text-white mono font-bold">0 - 9</span><span>Koreksi & Lompat Otomatis</span></div>
          <div class="flex justify-between items-center"><span class="bg-rose-950/80 border border-rose-600/50 text-rose-300 px-1.5 py-0.5 rounded mono font-bold">X / Del</span><span>Bukan Angka / Tolak (-1)</span></div>
          <div class="flex justify-between items-center"><span class="bg-slate-700 px-1.5 py-0.5 rounded text-white mono font-bold">Space / Enter</span><span>Terima Prediksi Saat Ini</span></div>
          <div class="flex justify-between items-center"><span class="bg-slate-700 px-1.5 py-0.5 rounded text-white mono font-bold">&larr; / &rarr;</span><span>Navigasi Tanpa Ubah</span></div>
          <div class="flex justify-between items-center"><span class="bg-slate-700 px-1.5 py-0.5 rounded text-white mono font-bold">I</span><span>Invert Warna (Hitam/Putih)</span></div>
          <div class="flex justify-between items-center"><span class="bg-slate-700 px-1.5 py-0.5 rounded text-white mono font-bold">Z</span><span>Zoom Sumber Crop Penuh</span></div>
        </div>
      </div>

      <!-- Daftar Antrean Item Kecil -->
      <div class="flex-1 flex flex-col min-h-0">
        <div class="flex justify-between items-center mb-1.5">
          <span class="text-xs font-semibold text-slate-400 tracking-wider uppercase">Daftar Antrean</span>
          <span id="queue-count" class="text-xs text-slate-500">0 item</span>
        </div>
        <div id="queue-list" class="flex-1 overflow-y-auto space-y-1 pr-1">
          <!-- Populated by JS -->
        </div>
      </div>
    </aside>

    <!-- Area Verifikasi Tengah (Focus Card) -->
    <section class="flex-1 p-6 flex flex-col items-center justify-start overflow-y-auto">
      <div id="review-card" class="w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl flex flex-col items-center gap-5 relative transition-all">
        
        <!-- Status Badges Top -->
        <div class="w-full flex justify-between items-center">
          <div class="flex items-center gap-2">
            <span id="badge-booklet" class="bg-slate-800 text-slate-300 text-xs px-2.5 py-1 rounded-md border border-slate-700 font-medium mono">booklet_00</span>
            <span id="badge-page" class="bg-slate-800 text-slate-300 text-xs px-2.5 py-1 rounded-md border border-slate-700 font-medium">Hal 1</span>
            <span id="badge-split" class="bg-blue-500/10 text-blue-400 border border-blue-500/20 text-xs px-2.5 py-1 rounded-md font-medium uppercase">test</span>
          </div>
          <div id="badge-status" class="bg-amber-500/10 text-amber-400 border border-amber-500/20 text-xs px-2.5 py-1 rounded-md font-medium flex items-center gap-1">
            <span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>
            Belum Diverifikasi
          </div>
        </div>

        <!-- Kanvas Citra Crop Native (Resolusi Asli ~66x49 Preservasi Rasio Aspek) -->
        <div class="flex items-center gap-6 w-full justify-center">
          <div class="flex flex-col items-center gap-1.5">
            <div id="crop-container" class="w-36 h-36 bg-white border-4 border-slate-700 rounded-xl p-1.5 flex items-center justify-center shadow-inner overflow-hidden transition-colors">
              <img id="cell-image" src="" alt="Digit Crop" class="w-28 h-28 object-contain filter contrast-110 drop-shadow-sm transition-transform">
            </div>
            <button onclick="toggleInvert()" class="text-xs text-slate-400 hover:text-slate-200 transition flex items-center gap-1">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/></svg>
              Invert [I]
            </button>
          </div>

          <!-- Informasi Semantik Ringkas di Samping -->
          <div class="flex-1 bg-slate-950/60 rounded-xl p-4 border border-slate-800/80 space-y-1.5">
            <div class="text-xs text-slate-500 uppercase tracking-wider font-semibold">Nama Variabel Kuesioner:</div>
            <div class="text-base font-bold text-emerald-400 tracking-tight" id="info-varname">b4_r452_kesehatan_oop</div>
            <div class="text-xs text-slate-400 grid grid-cols-2 gap-1 pt-1 border-t border-slate-800/60">
              <div>Field ID: <span class="mono text-slate-200 font-semibold" id="info-fieldid">p4_F022</span></div>
              <div>Kotak ke-: <span class="mono text-slate-200 font-semibold" id="info-boxindex">0</span></div>
              <div>Konfidensi: <span class="mono font-bold text-amber-400" id="info-conf">0.78</span></div>
              <div>Prediksi AI: <span class="mono font-bold text-emerald-300 text-sm" id="info-current-label">0</span></div>
            </div>
            <div class="text-xs text-slate-500 italic truncate pt-1" id="info-reason">V2_V3_CONSENSUS_OVERRIDE_BASELINE</div>
          </div>
        </div>

        <!-- =============================================================== -->
        <!-- FITUR BARU: PANEL SUMBER CROP UTUH (KONTEKS LAPANGAN HIERARKIS) -->
        <!-- =============================================================== -->
        <div class="w-full bg-slate-950/80 border border-slate-800/90 rounded-xl p-4 flex flex-col gap-2.5 shadow-inner">
          <div class="flex justify-between items-center">
            <div class="flex items-center gap-2">
              <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
              <span class="text-xs font-bold text-slate-200 uppercase tracking-wider">Sumber Crop Utuh (Konteks Lapangan)</span>
            </div>
            <!-- Tab Pilihan Tingkat Konteks -->
            <div class="flex gap-1 text-xs">
              <button id="tab-btn-field" onclick="setContextTier('field')" class="px-2.5 py-1 rounded-md bg-emerald-600 text-white font-medium transition active:scale-95 shadow">
                Baris Field (Highlight)
              </button>
              <button id="tab-btn-row" onclick="setContextTier('row')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition active:scale-95">
                Baris Roster [R]
              </button>
              <button id="tab-btn-raw" onclick="setContextTier('raw')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition active:scale-95">
                Kotak Mentah Asli
              </button>
              <button id="tab-btn-meso" onclick="setContextTier('meso')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition active:scale-95">
                Tabel Sub-Blok
              </button>
            </div>
          </div>

          <!-- Viewport Gambar Konteks -->
          <div class="relative w-full min-h-[90px] max-h-[180px] bg-slate-900 rounded-lg border border-slate-800/80 flex items-center justify-center p-2 overflow-auto group">
            <img id="context-image" src="" alt="Konteks Field Lapangan" class="max-h-[160px] object-contain rounded shadow cursor-zoom-in transition group-hover:brightness-105" onclick="openModalZoom()">
            <div id="context-loading" class="absolute inset-0 bg-slate-900/80 flex items-center justify-center text-xs text-slate-400 hidden">
              <svg class="animate-spin w-4 h-4 text-emerald-400 mr-2" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
              Memuat konteks lapangan...
            </div>
          </div>
          
          <div class="flex justify-between items-center text-[11px] text-slate-400">
            <span id="context-hint" class="flex items-center gap-1.5">
              <span class="w-3 h-3 bg-red-600/80 border border-yellow-400 inline-block rounded-sm"></span>
              Kotak yang sedang diverifikasi ditandai kotak merah terang di baris field.
            </span>
            <button onclick="openModalZoom()" class="text-emerald-400 hover:text-emerald-300 font-medium flex items-center gap-1 active:scale-95">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"/></svg>
              Perbesar Penuh [Z]
            </button>
          </div>
        </div>

        <!-- Bar Status Validasi Aritmetika Deterministik -->
        <div id="arithmetic-bar" class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-2.5 flex items-center justify-between text-xs shadow-inner">
          <div class="flex items-center gap-2.5">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-400" id="arithmetic-dot"></span>
            <span class="text-slate-400 font-medium">Audit Aritmetika Buklet:</span>
            <span id="arithmetic-status" class="font-bold text-emerald-400 mono">Memuat status konsistensi...</span>
          </div>
          <button onclick="revalidateActiveBooklet()" class="text-xs bg-slate-800 hover:bg-slate-700 text-emerald-400 px-3 py-1 rounded-lg font-semibold flex items-center gap-1.5 border border-slate-700 transition active:scale-95" title="Audit ulang konsistensi 19 relasi aritmetika buklet ini">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
            <span>Audit Ulang [A]</span>
          </button>
        </div>

        <!-- Tombol Digit Cepat (Numpad 0-9 & Bukan Angka) -->
        <div class="w-full flex flex-col items-center gap-2">
          <div class="text-xs font-semibold text-slate-400 tracking-wider uppercase">Pilih Digit Yang Benar (Atau Tekan Angka 0-9 di Keyboard)</div>
          <div class="grid grid-cols-10 gap-1.5 w-full">
            <button onclick="submitCorrection(0)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">0</button>
            <button onclick="submitCorrection(1)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">1</button>
            <button onclick="submitCorrection(2)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">2</button>
            <button onclick="submitCorrection(3)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">3</button>
            <button onclick="submitCorrection(4)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">4</button>
            <button onclick="submitCorrection(5)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">5</button>
            <button onclick="submitCorrection(6)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">6</button>
            <button onclick="submitCorrection(7)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">7</button>
            <button onclick="submitCorrection(8)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">8</button>
            <button onclick="submitCorrection(9)" class="btn-digit bg-slate-800 hover:bg-emerald-600 hover:text-white text-slate-200 py-2.5 rounded-lg font-bold text-lg mono transition border border-slate-700 active:scale-95">9</button>
          </div>

          <!-- Tombol Khusus Bukan Angka / Crop Meleset / Garis Kotak / Noda -->
          <div class="w-full pt-1">
            <button id="btn-non-digit" onclick="submitCorrection(-1)" class="w-full py-2.5 px-4 rounded-xl font-bold text-xs uppercase tracking-wider transition flex items-center justify-center gap-2 border active:scale-95 bg-rose-950/40 hover:bg-rose-600 hover:text-white text-rose-300 border-rose-800/60 shadow-lg shadow-rose-950/30">
              <svg class="w-4 h-4 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
              <span>✖ Bukan Angka / Crop Meleset / Garis Kotak / Noda (Tekan X / Del)</span>
            </button>
          </div>
        </div>

        <!-- Tombol Aksi Navigasi Bawah -->
        <div class="w-full flex justify-between items-center pt-2 border-t border-slate-800">
          <button onclick="navigate(-1)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg text-xs font-semibold transition flex items-center gap-1 active:scale-95">
            &larr; Sebelumnya
          </button>
          <button onclick="confirmCurrent()" class="bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2 rounded-lg text-xs font-bold transition shadow-lg shadow-emerald-950 flex items-center gap-2 active:scale-95">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
            Terima Label Ini [Space/Enter]
          </button>
          <button onclick="navigate(1)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-lg text-xs font-semibold transition flex items-center gap-1 active:scale-95">
            Selanjutnya &rarr;
          </button>
        </div>

      </div>
    </section>
  </main>

  <!-- Modal Lightbox Zoom Penuh -->
  <div id="modal-zoom" class="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-sm hidden flex flex-col items-center justify-center p-6" onclick="closeModalZoom()">
    <div class="relative max-w-5xl max-h-[90vh] bg-slate-900 border border-slate-700 rounded-2xl p-4 shadow-2xl flex flex-col items-center overflow-auto" onclick="event.stopPropagation()">
      <div class="w-full flex justify-between items-center mb-2 pb-2 border-b border-slate-800">
        <span class="text-sm font-bold text-emerald-400" id="modal-title">Konteks Baris Field (Perbesaran Penuh)</span>
        <button onclick="closeModalZoom()" class="text-slate-400 hover:text-white p-1 rounded-lg bg-slate-800 text-xs font-bold">✕ Tutup [Esc]</button>
      </div>
      <img id="modal-image" src="" alt="Zoom Konteks" class="max-w-full max-h-[75vh] object-contain rounded border border-slate-800">
      <div class="text-xs text-slate-400 mt-2 text-center" id="modal-caption">Kotak yang diverifikasi ditandai dengan garis merah tebal.</div>
    </div>
  </div>

  <!-- Floating Toast Notification -->
  <div id="toast-msg" class="fixed bottom-5 right-5 z-50 opacity-0 translate-y-2 pointer-events-none transition-all duration-300 bg-slate-900 border border-slate-700 text-slate-200 px-4 py-2.5 rounded-xl text-xs font-semibold shadow-2xl flex items-center gap-2"></div>

  <script>
    let queue = [];
    let currentIndex = 0;
    let isInverted = false;
    let currentContextTier = 'field';
    let autoSaveEnabled = true;
    let autoSaveTimer = null;
    let isSaving = false;
    let hasUnsavedChanges = false;
    let lastSavedTime = null;

    async function loadStats() {
      try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('stat-total').innerText = (data.total_cells || 0).toLocaleString();
        document.getElementById('stat-low-conf').innerText = (data.low_confidence_pending || 0).toLocaleString();
        if (document.getElementById('stat-disagreements')) {
          document.getElementById('stat-disagreements').innerText = (data.disagreements_total || 0).toLocaleString();
        }
        if (document.getElementById('stat-border-sliver')) {
          document.getElementById('stat-border-sliver').innerText = (data.border_sliver_total || 0).toLocaleString();
        }
        document.getElementById('stat-reviewed').innerText = (data.reviewed_cells || 0).toLocaleString();
        document.getElementById('stat-pct').innerText = `(${data.review_progress_pct || 0}%)`;

        // Isi pilihan buklet jika belum terisi
        const bSelect = document.getElementById('booklet-filter');
        if (bSelect && data.booklets && data.booklets.length > 0 && bSelect.options.length <= 1) {
          data.booklets.forEach(b => {
            const opt = document.createElement('option');
            opt.value = b;
            opt.innerText = b;
            bSelect.appendChild(opt);
          });
        }

        if (data.last_saved && !hasUnsavedChanges) {
          lastSavedTime = data.last_saved;
          setSaveStatus('saved', `Tersimpan ${lastSavedTime}`);
        }
      } catch (err) {
        console.error(err);
      }
    }

    let searchDebounceTimer = null;
    function handleSearch() {
      if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
      searchDebounceTimer = setTimeout(() => {
        loadQueue(0);
      }, 250);
    }

    async function loadQueue(offset = 0) {
      const mode = document.getElementById('filter-mode').value;
      const bSelect = document.getElementById('booklet-filter');
      const booklet = bSelect ? bSelect.value : 'all';
      const searchBox = document.getElementById('search-box');
      const query = searchBox ? searchBox.value.trim() : '';

      try {
        let url = `/api/queue?filter=${mode}&booklet=${booklet}&offset=${offset}&limit=50`;
        if (query) {
          url += `&q=${encodeURIComponent(query)}`;
        }
        const res = await fetch(url);
        const data = await res.json();
        queue = data.items;
        document.getElementById('queue-count').innerText = `${(data.total || 0).toLocaleString()} item`;
        renderQueueList();
        if (queue.length > 0) {
          currentIndex = 0;
          renderCurrentItem();
        }
      } catch (err) {
        console.error(err);
      }
    }

    function changeBooklet() {
      loadQueue(0);
    }

    function renderQueueList() {
      const container = document.getElementById('queue-list');
      container.innerHTML = '';
      queue.forEach((it, idx) => {
        const div = document.createElement('div');
        div.className = `p-2 rounded-lg border text-xs cursor-pointer transition flex items-center justify-between ${idx === currentIndex ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-200' : 'bg-slate-800/40 border-slate-700/40 text-slate-300 hover:bg-slate-800'}`;
        div.onclick = () => { currentIndex = idx; renderCurrentItem(); };
        
        let labelDisplay = `<span class="font-bold mono">[${it.label}]</span>`;
        if (it.label === -1) {
          labelDisplay = `<span class="font-bold mono text-rose-400 bg-rose-950/60 px-1 rounded border border-rose-800/60">[✖ NON]</span>`;
        }
        div.innerHTML = `
          <div class="truncate pr-1 flex items-center gap-1">
            ${labelDisplay}
            <span class="truncate text-slate-400">${it.var_name}</span>
          </div>
          <span class="mono text-[10px] ${it.conf < 0.85 ? 'text-amber-400 font-bold' : 'text-slate-500'}">${(it.conf*100).toFixed(0)}%</span>
        `;
        container.appendChild(div);
      });
    }

    function renderCurrentItem() {
      if (currentIndex < 0 || currentIndex >= queue.length) return;
      const it = queue[currentIndex];

      document.getElementById('badge-booklet').innerText = it.booklet_id;
      document.getElementById('badge-page').innerText = `Hal ${it.page}`;
      document.getElementById('badge-split').innerText = it.split;

      const badgeStatus = document.getElementById('badge-status');
      if (it.reviewed) {
        badgeStatus.className = 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs px-2.5 py-1 rounded-md font-medium flex items-center gap-1';
        badgeStatus.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>Sudah Diverifikasi';
      } else {
        badgeStatus.className = 'bg-amber-500/10 text-amber-400 border border-amber-500/20 text-xs px-2.5 py-1 rounded-md font-medium flex items-center gap-1';
        badgeStatus.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>Belum Diverifikasi';
      }

      // Gambar 28x28
      const img = document.getElementById('cell-image');
      if (it.image_base64) {
        img.src = `data:image/png;base64,${it.image_base64}`;
      } else {
        img.src = '';
      }

      document.getElementById('info-varname').innerText = it.var_name;
      document.getElementById('info-fieldid').innerText = it.field_id;
      document.getElementById('info-boxindex').innerText = it.box_index;
      document.getElementById('info-conf').innerText = it.conf.toFixed(4);
      
      const labelBadge = document.getElementById('info-current-label');
      const btnNonDigit = document.getElementById('btn-non-digit');

      if (it.label === -1) {
        labelBadge.innerText = '✖ BUKAN ANGKA (-1)';
        labelBadge.className = 'mono font-bold text-rose-400 bg-rose-950/60 px-2 py-0.5 rounded border border-rose-700 text-xs';
        if (btnNonDigit) {
          btnNonDigit.className = 'w-full py-2.5 px-4 rounded-xl font-bold text-xs uppercase tracking-wider transition flex items-center justify-center gap-2 border active:scale-95 bg-rose-600 text-white border-rose-400 shadow-lg shadow-rose-950';
        }
      } else {
        labelBadge.innerText = it.label;
        labelBadge.className = 'mono font-bold text-emerald-300 text-sm';
        if (btnNonDigit) {
          btnNonDigit.className = 'w-full py-2.5 px-4 rounded-xl font-bold text-xs uppercase tracking-wider transition flex items-center justify-center gap-2 border active:scale-95 bg-rose-950/40 hover:bg-rose-600 hover:text-white text-rose-300 border-rose-800/60 shadow-lg shadow-rose-950/30';
        }
      }

      // Format AI Reason badge dengan warna semantik
      const reasonEl = document.getElementById('info-reason');
      const reasonText = it.reason || 'Prediksi Model OCR';
      reasonEl.innerText = reasonText;
      if (reasonText.includes('HUMAN_VERIFIED')) {
        reasonEl.className = 'text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 inline-block';
      } else if (reasonText.includes('BORDER') || reasonText.includes('SLIVER')) {
        reasonEl.className = 'text-xs font-semibold px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 inline-block';
      } else if (reasonText.includes('ARITHMETIC') || reasonText.includes('OVERRIDE')) {
        reasonEl.className = 'text-xs font-semibold px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/40 inline-block';
      } else if (reasonText.includes('TIE_BREAKER')) {
        reasonEl.className = 'text-xs font-semibold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 inline-block';
      } else {
        reasonEl.className = 'text-xs text-slate-400 italic truncate pt-1';
      }

      // Highlight active button for 0-9
      document.querySelectorAll('.btn-digit').forEach((btn, d) => {
        if (d === it.label) {
          btn.classList.add('bg-emerald-600', 'text-white', 'border-emerald-400');
          btn.classList.remove('bg-slate-800');
        } else {
          btn.classList.remove('bg-emerald-600', 'text-white', 'border-emerald-400');
          btn.classList.add('bg-slate-800');
        }
      });

      // Audit konsistensi aritmetika buklet aktif
      revalidateActiveBooklet(it.booklet_id);

      // Muat gambar konteks sumber crop utuh
      updateContextImage();
      renderQueueList();
    }

    async function revalidateActiveBooklet(bId) {
      const targetBId = bId || (queue[currentIndex] ? queue[currentIndex].booklet_id : null);
      if (!targetBId) return;
      try {
        const res = await fetch(`/api/revalidate?booklet=${targetBId}`);
        const data = await res.json();
        if (data.success && data.validation_report) {
          updateArithmeticBadge(data.validation_report);
        }
      } catch (e) {
        console.error(e);
      }
    }

    function updateArithmeticBadge(rep) {
      const dot = document.getElementById('arithmetic-dot');
      const text = document.getElementById('arithmetic-status');
      if (!dot || !text) return;
      const passed = rep.rules_passed;
      const total = rep.total_rules;
      const disc = rep.total_discrepancy_rupiah;
      if (disc === 0 || passed === total) {
        dot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400';
        text.className = 'font-bold text-emerald-400 mono';
        text.innerText = `${passed}/${total} Aturan Lulus (Rp 0 Selisih - 100% Konsisten)`;
      } else {
        dot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse';
        text.className = 'font-bold text-amber-400 mono';
        text.innerText = `${passed}/${total} Aturan Lulus (Selisih: Rp ${disc.toLocaleString()})`;
      }
    }

    function updateContextImage() {
      if (currentIndex < 0 || currentIndex >= queue.length) return;
      const it = queue[currentIndex];
      const img = document.getElementById('context-image');
      const loader = document.getElementById('context-loading');
      loader.classList.remove('hidden');

      const srcUrl = `/api/context_image?booklet=${it.booklet_id}&page=${it.page}&field_id=${it.field_id}&box_index=${it.box_index}&tier=${currentContextTier}&t=${Date.now()}`;
      
      img.onload = () => { loader.classList.add('hidden'); };
      img.onerror = () => {
        loader.classList.remove('hidden');
        loader.innerText = 'Gambar sumber crop tidak ditemukan di disk.';
      };
      img.src = srcUrl;
    }

    function setContextTier(tier) {
      currentContextTier = tier;
      ['field', 'row', 'raw', 'meso'].forEach(t => {
        const btn = document.getElementById(`tab-btn-${t}`);
        if (!btn) return;
        if (t === tier) {
          btn.className = 'px-2.5 py-1 rounded-md bg-emerald-600 text-white font-medium transition active:scale-95 shadow';
        } else {
          btn.className = 'px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition active:scale-95';
        }
      });

      const hint = document.getElementById('context-hint');
      if (tier === 'field') {
        hint.innerHTML = '<span class="w-3 h-3 bg-red-600/80 border border-yellow-400 inline-block rounded-sm"></span> Kotak yang diverifikasi ditandai kotak merah terang di baris field.';
      } else if (tier === 'row') {
        hint.innerHTML = '<span class="w-3 h-3 bg-red-600/80 border border-yellow-400 inline-block rounded-sm"></span> Kotak yang diverifikasi ditandai kotak merah di baris roster/data tabel horizontal.';
      } else if (tier === 'raw') {
        hint.innerHTML = 'Citra mentah asli hasil scan kotak (sebelum pembersihan garis tepi).';
      } else if (tier === 'meso') {
        hint.innerHTML = 'Tabel sub-blok pertanyaan lengkap (baris field ditandai kotak merah tebal).';
      }

      updateContextImage();
    }

    function openModalZoom() {
      const modal = document.getElementById('modal-zoom');
      const modalImg = document.getElementById('modal-image');
      const currentSrc = document.getElementById('context-image').src;
      modalImg.src = currentSrc;
      document.getElementById('modal-title').innerText = `Sumber Crop Utuh: ${document.getElementById('info-varname').innerText} (Kotak ke-${document.getElementById('info-boxindex').innerText})`;
      modal.classList.remove('hidden');
    }

    function closeModalZoom() {
      document.getElementById('modal-zoom').classList.add('hidden');
    }

    async function submitCorrection(digit) {
      if (currentIndex < 0 || currentIndex >= queue.length) return;
      const it = queue[currentIndex];

      try {
        const res = await fetch('/api/correct', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            booklet_id: it.booklet_id,
            page: it.page,
            field_id: it.field_id,
            box_index: it.box_index,
            new_label: digit,
            reviewer: 'Human Verifier'
          })
        });
        const data = await res.json();
        if (data.success) {
          it.label = digit;
          it.reviewed = true;
          it.conf = 1.0;
          it.reason = digit >= 0 ? 'HUMAN_VERIFIED (Human Verifier)' : 'HUMAN_REJECT_NON_DIGIT (Human Verifier)';
          if (data.validation_report) {
            updateArithmeticBadge(data.validation_report);
            showToast(`✓ Label diperbarui: ${digit >= 0 ? digit : 'NON'} (${data.validation_report.rules_passed}/${data.validation_report.total_rules} aturan lulus)`, 'success');
          } else {
            showToast(`✓ Label diperbarui: ${digit >= 0 ? digit : 'NON'}`, 'success');
          }
          loadStats();
          scheduleAutoSave();
          navigate(1);
        }
      } catch (err) {
        console.error(err);
      }
    }

    function confirmCurrent() {
      if (currentIndex < 0 || currentIndex >= queue.length) return;
      submitCorrection(queue[currentIndex].label);
    }

    function navigate(delta) {
      const next = currentIndex + delta;
      if (next >= 0 && next < queue.length) {
        currentIndex = next;
        renderCurrentItem();
      } else if (next >= queue.length) {
        loadQueue(queue.length);
      }
    }

    function changeFilter() {
      loadQueue(0);
    }

    function toggleInvert() {
      isInverted = !isInverted;
      const cont = document.getElementById('crop-container');
      const img = document.getElementById('cell-image');
      if (isInverted) {
        cont.classList.remove('bg-white');
        cont.classList.add('bg-slate-900');
        img.classList.add('invert');
      } else {
        cont.classList.remove('bg-slate-900');
        cont.classList.add('bg-white');
        img.classList.remove('invert');
      }
    }

    function setSaveStatus(state, msg) {
      const indEl = document.getElementById('save-indicator');
      const textEl = document.getElementById('save-text');
      if (!indEl || !textEl) return;
      
      if (state === 'saved') {
        indEl.className = 'w-2 h-2 rounded-full bg-emerald-400';
        textEl.innerText = msg || (lastSavedTime ? `Tersimpan ${lastSavedTime}` : 'Tersimpan');
        textEl.className = 'text-emerald-400 font-mono text-[11px] font-semibold';
        hasUnsavedChanges = false;
      } else if (state === 'pending') {
        indEl.className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse';
        textEl.innerText = msg || 'Ada perubahan...';
        textEl.className = 'text-amber-400 font-mono text-[11px] font-semibold';
        hasUnsavedChanges = true;
      } else if (state === 'saving') {
        indEl.className = 'w-2 h-2 rounded-full bg-blue-400 animate-ping';
        textEl.innerText = 'Menyimpan...';
        textEl.className = 'text-blue-400 font-mono text-[11px] font-semibold';
      } else if (state === 'error') {
        indEl.className = 'w-2 h-2 rounded-full bg-rose-500';
        textEl.innerText = msg || 'Gagal Simpan';
        textEl.className = 'text-rose-400 font-mono text-[11px] font-semibold';
      }
    }

    function toggleAutoSave(enabled) {
      autoSaveEnabled = enabled;
      showToast(enabled ? 'Auto-Save diaktifkan (otomatis simpan saat idle)' : 'Auto-Save dinonaktifkan (tekan Ctrl+S untuk simpan)', enabled ? 'success' : 'warning');
      if (enabled && hasUnsavedChanges) {
        scheduleAutoSave(300);
      } else if (!enabled && hasUnsavedChanges) {
        setSaveStatus('pending', 'Belum disimpan');
      }
    }

    function scheduleAutoSave(delayMs = 1500) {
      if (!autoSaveEnabled) {
        setSaveStatus('pending', 'Belum disimpan');
        return;
      }
      setSaveStatus('pending', 'Auto-save...');
      if (autoSaveTimer) clearTimeout(autoSaveTimer);
      autoSaveTimer = setTimeout(() => {
        executeSave(true);
      }, delayMs);
    }

    async function executeSave(isAuto = false) {
      if (isSaving) return;
      isSaving = true;
      setSaveStatus('saving');
      
      const saveBtn = document.getElementById('btn-save-manual');
      if (saveBtn) {
        saveBtn.classList.add('opacity-75', 'cursor-wait');
      }

      try {
        const res = await fetch('/api/sync_npz', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          const now = new Date();
          lastSavedTime = data.last_saved || now.toLocaleTimeString();
          setSaveStatus('saved', `Tersimpan ${lastSavedTime}`);
          showToast(`✓ Perubahan berhasil disimpan ke disk (${data.elapsed_seconds || 0}s)`, 'success');
        } else {
          setSaveStatus('error', 'Gagal Simpan');
          showToast('Gagal menyimpan ke disk: ' + (data.error || 'Unknown error'), 'error');
        }
      } catch (err) {
        setSaveStatus('error', 'Koneksi Putus');
        showToast('Gagal menghubungi server: ' + err, 'error');
      } finally {
        isSaving = false;
        if (saveBtn) {
          saveBtn.classList.remove('opacity-75', 'cursor-wait');
        }
      }
    }

    function manualSave() {
      if (autoSaveTimer) clearTimeout(autoSaveTimer);
      executeSave(false);
    }

    function showToast(message, type = 'info') {
      const toast = document.getElementById('toast-msg');
      if (!toast) return;
      toast.innerText = message;
      if (type === 'success') {
        toast.className = 'fixed bottom-5 right-5 z-50 bg-emerald-950/95 border border-emerald-500/80 text-emerald-200 px-4 py-2.5 rounded-xl text-xs font-semibold shadow-2xl flex items-center gap-2 transition-all duration-300 opacity-100 transform translate-y-0';
      } else if (type === 'warning') {
        toast.className = 'fixed bottom-5 right-5 z-50 bg-amber-950/95 border border-amber-500/80 text-amber-200 px-4 py-2.5 rounded-xl text-xs font-semibold shadow-2xl flex items-center gap-2 transition-all duration-300 opacity-100 transform translate-y-0';
      } else if (type === 'error') {
        toast.className = 'fixed bottom-5 right-5 z-50 bg-rose-950/95 border border-rose-500/80 text-rose-200 px-4 py-2.5 rounded-xl text-xs font-semibold shadow-2xl flex items-center gap-2 transition-all duration-300 opacity-100 transform translate-y-0';
      } else {
        toast.className = 'fixed bottom-5 right-5 z-50 bg-slate-900/95 border border-slate-700 text-slate-200 px-4 py-2.5 rounded-xl text-xs font-semibold shadow-2xl flex items-center gap-2 transition-all duration-300 opacity-100 transform translate-y-0';
      }
      setTimeout(() => {
        toast.classList.add('opacity-0', 'translate-y-2');
      }, 2500);
    }

    // Keyboard bindings
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        manualSave();
        return;
      }

      if (e.key === 'Escape') {
        closeModalZoom();
      } else if (e.key.toLowerCase() === 'z') {
        e.preventDefault();
        const modal = document.getElementById('modal-zoom');
        if (modal.classList.contains('hidden')) {
          openModalZoom();
        } else {
          closeModalZoom();
        }
      } else if (e.key.toLowerCase() === 'a') {
        e.preventDefault();
        revalidateActiveBooklet();
      } else if (e.key >= '0' && e.key <= '9') {
        e.preventDefault();
        submitCorrection(parseInt(e.key));
      } else if (e.key.toLowerCase() === 'x' || e.key === 'Delete' || e.key === 'Backspace' || e.key === '-') {
        e.preventDefault();
        submitCorrection(-1);
      } else if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        confirmCurrent();
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        navigate(1);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        navigate(-1);
      } else if (e.key.toLowerCase() === 'i') {
        e.preventDefault();
        toggleInvert();
      } else if (e.key.toLowerCase() === 'r') {
        e.preventDefault();
        setContextTier(currentContextTier === 'row' ? 'field' : 'row');
      }
    });

    window.addEventListener('beforeunload', (e) => {
      if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = 'Perhatian: Terdapat perubahan label yang belum disimpan ke disk!';
      }
    });

    // Inisialisasi awal
    loadStats();
    loadQueue(0);
  </script>
</body>
</html>
"""


class HITLRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Handler HTTP REST API dan Frontend UI."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/stats":
            stats = DATA_MANAGER.get_stats()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode("utf-8"))
            return

        elif path == "/api/queue":
            qs = parse_qs(parsed.query)
            filter_mode = qs.get("filter", ["low_conf"])[0]
            offset = int(qs.get("offset", [0])[0])
            limit = int(qs.get("limit", [20])[0])
            booklet_id = qs.get("booklet", [None])[0]
            query_str = qs.get("q", [None])[0]

            res = DATA_MANAGER.get_queue(filter_mode=filter_mode, offset=offset, limit=limit, booklet_id=booklet_id, query=query_str)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        elif path == "/api/context_image":
            qs = parse_qs(parsed.query)
            booklet_id = qs.get("booklet", [""])[0]
            page = int(qs.get("page", [1])[0])
            field_id = qs.get("field_id", [""])[0]
            box_index = int(qs.get("box_index", [0])[0])
            tier = qs.get("tier", ["field"])[0]

            img_bytes = DATA_MANAGER.get_context_image(booklet_id, page, field_id, box_index, tier=tier)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(img_bytes)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(img_bytes)
            return

        elif path == "/api/revalidate":
            qs = parse_qs(parsed.query)
            booklet_id = qs.get("booklet", ["booklet_01"])[0]
            gt_file = GT_DIR / f"{booklet_id}.json"
            if gt_file.exists():
                try:
                    with open(gt_file, "r", encoding="utf-8") as f:
                        gt_doc = json.load(f)
                    if SerutiArithmeticValidator is not None:
                        validator = SerutiArithmeticValidator()
                        report = validator.validate_booklet(gt_doc, auto_reconcile=False)
                        res = {
                            "success": True,
                            "booklet_id": booklet_id,
                            "validation_report": report.summary.to_dict()
                        }
                    else:
                        res = {"success": False, "error": "Validator module not available"}
                except Exception as e:
                    res = {"success": False, "error": str(e)}
            else:
                res = {"success": False, "error": f"{booklet_id} tidak ditemukan"}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)

        if path == "/api/correct":
            try:
                payload = json.loads(post_body.decode("utf-8"))
                booklet_id = payload["booklet_id"]
                page = int(payload["page"])
                field_id = payload["field_id"]
                box_index = int(payload["box_index"])
                new_label = int(payload["new_label"])
                reviewer = payload.get("reviewer", "Human Verifier")

                res = DATA_MANAGER.save_correction(booklet_id, page, field_id, box_index, new_label, reviewer)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
                return

        elif path == "/api/sync_npz":
            res = DATA_MANAGER.sync_to_disk()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_server(port=8585):
    with ThreadedTCPServer(("127.0.0.1", port), HITLRequestHandler) as httpd:
        print("=" * 80)
        print("  BPS SERUTI INTI - HITL LABELING & VERIFIKASI SERVER AKTIF (MULTI-THREADED)")
        print(f"  URL Aplikasi : http://localhost:{port}")
        print("  Fitur Baru   : Dynamic Stroke Preservation, Zero-Drift Snapping, dan Padded Raw Context")
        print("  Pemberhentian: Tekan Ctrl+C di terminal")
        print("=" * 80)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server dimatikan.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPS Seruti HITL Labeling Server")
    parser.add_argument("--port", type=int, default=8585, help="Port HTTP server")
    args = parser.parse_args()

    run_server(port=args.port)
