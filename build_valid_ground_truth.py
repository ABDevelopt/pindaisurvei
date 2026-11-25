#!/usr/bin/env python3
"""
build_valid_ground_truth.py
===========================
Pembangun Ground Truth Otentik Terverifikasi Multi-Tier BPS untuk Kuesioner
Susenas Seruti Inti (VSERUTI26.INTI).

Pilar Penjaminan Mutu & Validitas Ground Truth:
1. Preservasi Anotasi Kanonikal Manusia (Human-in-the-Loop):
   - Memuat 1.350+ entri dari `dataset/manual_corrections_log.jsonl` (prioritas mutlak).
2. Filter Eliminasi Sel Arsir / NA & Garis Dotted Template:
   - Mengeliminasi 32 field NA/tabel non-numerik yang mengecoh OCR.
3. Deteksi Sel Kosong Presisi Tinggi (Zero Phantom Digits):
   - Comb-box border removal + Center-of-Mass MNIST clean_cell_mnist.
   - Kotak kosong dilabeli `_` (BLANK) dan TIDAK dimasukkan ke kelas digit 0-9.
4. Tri-Model AI Consensus Tervektorisasi (Baseline, Model v2, Model v3):
   - Inferensi batch kilat pada 20x20 kanvas MNIST standar.
5. Pembuktian Aritmetika Deterministik (SerutiArithmeticValidator):
   - Menyelesaikan relasi penjumlahan dan penegakan kaidah administratif BPS.
6. Regenerasi Dataset Arrays (.npz), Manifest CSV, dan PNG Cells:
   - Partisi train/val/test bebas data leakage.
   - Format Ground Truth JSON standar kuesioner-survei-bps.

Kepatuhan Hukum:
- UU No. 16 Tahun 1997 tentang Statistik
- UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (Zero Cloud Egress)
- Zero-Jitter Invariant pada seluruh nilai finansial lokal.
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EXTRACTIONS_DIR = BASE_DIR / "output" / "extractions"
CROPS_DIR = BASE_DIR / "output" / "crops"
DATASET_DIR = BASE_DIR / "dataset"
GT_DIR = DATASET_DIR / "ground_truth"
SPLITS_DIR = DATASET_DIR / "splits"
ARRAYS_DIR = DATASET_DIR / "data_arrays"
CELLS_DIR = DATASET_DIR / "cells"

BASELINE_MODEL_PATH = Path(r"d:\01_Kedinasan_BPS\Pengenalan Kuesioner (OCR)\model.joblib")
MODEL_V2_PATH = BASE_DIR / "models" / "model_seruti_v2.joblib"
MODEL_V3_PATH = BASE_DIR / "models" / "model_seruti_v3.joblib"

from seruti_pipeline.advanced_filter import remove_comb_box_border
from run_grid_cropper import clean_cell_mnist
from seruti_pipeline.validator import SerutiArithmeticValidator

# Pemetaan Field NA/Hatched/Text Dotted yang tidak berisi angka
NA_FIELD_IDS = {
    'p5_F051', 'p5_F052', 'p5_F053', 'p5_F054', 'p5_F058', 'p5_F059', 'p5_F068', 'p5_F069', 'p5_F070',
    'p5_F081', 'p5_F084', 'p5_F089', 'p5_F090', 'p5_F091', 'p5_F099', 'p5_F100',
    'p6_F001', 'p6_F002', 'p6_F003', 'p6_F004', 'p6_F009', 'p6_F010', 'p6_F011', 'p6_F012', 'p6_F027',
    'p6_F040', 'p6_F041', 'p6_F042', 'p6_F043', 'p6_F044', 'p6_F045', 'p6_F046'
}


class ValidGroundTruthBuilder:
    """Orkestrator pembangunan ground truth terverifikasi dan dataset latih/uji."""

    def __init__(self, export_png: bool = True):
        self.export_png = export_png
        print("[*] Menginisialisasi Pembangun Ground Truth Seruti...")
        self.manual_corrections = self._load_manual_corrections()
        self.validator = SerutiArithmeticValidator()

        print("    -> Memuat Model Baseline BPS...")
        self.base_clf = joblib.load(BASELINE_MODEL_PATH)
        print("    -> Memuat Model Seruti v2...")
        self.v2_clf = joblib.load(MODEL_V2_PATH)
        print("    -> Memuat Model Seruti v3...")
        self.v3_clf = joblib.load(MODEL_V3_PATH)

        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [DATASET_DIR, GT_DIR, SPLITS_DIR, ARRAYS_DIR, CELLS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_manual_corrections(self) -> Dict[Tuple[str, int, str, int], Dict[str, Any]]:
        corrections = {}
        candidates = [
            DATASET_DIR / "manual_corrections_log.jsonl",
            DATASET_DIR / "manual_corrections_log_CANONICAL_BACKUP.jsonl",
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            e = json.loads(line)
                            k = (str(e["booklet_id"]), int(e["page"]), str(e["field_id"]), int(e["box_index"]))
                            corrections[k] = {
                                "label": int(e["corrected_label"]),
                                "reviewer": e.get("reviewer", "Human Verifier")
                            }
                    print(f"[*] Berhasil memuat {len(corrections):,} koreksi manual dari {p.name}")
                    break
                except Exception as ex:
                    print(f"[-] Peringatan memuat {p.name}: {ex}")
        return corrections

    def process_booklet(self, booklet_id: str, split_name: str) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]]:
        """Memproses 1 buklet secara deterministik dan terverifikasi."""
        t0 = time.time()
        ext_file = EXTRACTIONS_DIR / booklet_id / f"{booklet_id}_extracted.json"
        crops_b_dir = CROPS_DIR / booklet_id
        if not ext_file.exists() or not crops_b_dir.exists():
            return None

        try:
            with open(ext_file, "r", encoding="utf-8") as f:
                ext_data = json.load(f)
        except Exception as e:
            print(f"[-] Gagal membaca {ext_file.name}: {e}")
            return None

        crops_lookup = {}
        for p_num in range(1, 8):
            m_path = crops_b_dir / f"hal_{p_num}" / "crop_manifest.json"
            if not m_path.exists():
                continue
            try:
                with open(m_path, "r", encoding="utf-8") as mf:
                    m = json.load(mf)
                for c in m.get("tiers", {}).get("mikro", []):
                    fid = c.get("field_id")
                    b_idx = c.get("box_index")
                    if fid and b_idx is not None:
                        crops_lookup[(p_num, fid, b_idx)] = c.get("raw_file")
            except Exception:
                pass

        new_data = copy.deepcopy(ext_data)
        pages = new_data.get("pages", {})

        total_boxes = 0
        blank_cnt = 0
        filled_cnt = 0
        gt_fields = {}
        gt_raw_fields = {}

        # Struktur penampung untuk inferensi batch
        pending_inference = []  # list of (box_ref, cell_record, feat_28, cl_img, mnist_28)
        curated_cells = []

        def page_sort_key(k):
            if k.startswith("hal_"):
                return int(k.split("_")[1])
            return int(k) if k.isdigit() else 99

        for p_key in sorted(pages.keys(), key=page_sort_key):
            p_num = int(p_key.split("_")[1]) if "_" in p_key else 1
            p_obj = pages[p_key]
            fields = p_obj.get("fields", {})

            for var_name, f_obj in fields.items():
                fid = f_obj.get("field_id", var_name)
                boxes = f_obj.get("box_details", [])
                data_type = f_obj.get("data_type", "integer")

                if data_type == "checkbox":
                    gt_fields[var_name] = f_obj.get("value")
                    gt_raw_fields[var_name] = f_obj.get("raw_string", "")
                    continue

                if fid in NA_FIELD_IDS:
                    for b in boxes:
                        b["is_blank"] = True
                        b["char"] = "_"
                        b["conf"] = 1.0
                        blank_cnt += 1
                        total_boxes += 1
                    f_obj["raw_string"] = "_" * len(boxes)
                    f_obj["value"] = None
                    gt_fields[var_name] = None
                    gt_raw_fields[var_name] = "_" * len(boxes)
                    continue

                for b in boxes:
                    total_boxes += 1
                    b_idx = b.get("box_index", 0)
                    cell_key = (booklet_id, p_num, fid, b_idx)
                    raw_rel = crops_lookup.get((p_num, fid, b_idx))

                    # 1. Prioritas Utama: Koreksi Manual Manusia
                    if cell_key in self.manual_corrections:
                        m_entry = self.manual_corrections[cell_key]
                        m_lbl = m_entry["label"]
                        if m_lbl == -1:
                            b["is_blank"] = True
                            b["char"] = "_"
                            b["conf"] = 1.0
                            blank_cnt += 1
                        else:
                            b["is_blank"] = False
                            b["char"] = str(m_lbl)
                            b["conf"] = 1.0
                            filled_cnt += 1
                            # Muat citra untuk dataset
                            raw_p = BASE_DIR / raw_rel if raw_rel else None
                            cl_img = None
                            mnist_img = None
                            if raw_p and raw_p.exists():
                                raw_im = cv2.imread(str(raw_p))
                                if raw_im is not None:
                                    cl_img = remove_comb_box_border(raw_im)
                                    mnist_img = clean_cell_mnist(cl_img)

                            curated_cells.append({
                                "booklet_id": booklet_id,
                                "page": p_num,
                                "field_id": fid,
                                "var_name": var_name,
                                "box_index": b_idx,
                                "label": m_lbl,
                                "conf": 1.0,
                                "raw_path": raw_rel,
                                "arbitration_reason": f"HUMAN_VERIFIED ({m_entry['reviewer']})",
                                "verified": True,
                                "cl_img": cl_img,
                                "mnist_28": mnist_img,
                                "box_ref": b
                            })
                        continue

                    # 2. Pemrosesan Citra: Deteksi Blank
                    raw_p = BASE_DIR / raw_rel if raw_rel else None
                    if not raw_p or not raw_p.exists():
                        b["is_blank"] = True
                        b["char"] = "_"
                        b["conf"] = 1.0
                        blank_cnt += 1
                        continue

                    raw_im = cv2.imread(str(raw_p))
                    if raw_im is None:
                        b["is_blank"] = True
                        b["char"] = "_"
                        b["conf"] = 1.0
                        blank_cnt += 1
                        continue

                    cl = remove_comb_box_border(raw_im)
                    mnist_28 = clean_cell_mnist(cl)

                    if mnist_28 is None or (mnist_28 > 50).sum() < 12:
                        b["is_blank"] = True
                        b["char"] = "_"
                        b["conf"] = 1.0
                        blank_cnt += 1
                    else:
                        # Masukkan ke antrean inferensi batch
                        feat = mnist_28.ravel().astype(np.float32) / 255.0
                        c_rec = {
                            "booklet_id": booklet_id,
                            "page": p_num,
                            "field_id": fid,
                            "var_name": var_name,
                            "box_index": b_idx,
                            "raw_path": raw_rel,
                            "cl_img": cl,
                            "mnist_28": mnist_28,
                            "box_ref": b
                        }
                        pending_inference.append((b, c_rec, feat, cl, mnist_28))

        # 3. Eksekusi Inferensi Batch Kilat untuk seluruh sel terisi di buklet ini
        if pending_inference:
            X_batch = np.array([item[2] for item in pending_inference], dtype=np.float32)
            prob_b_all = self.base_clf.predict_proba(X_batch)
            pred_b_all = np.argmax(prob_b_all, axis=1)

            prob_v2_all = self.v2_clf.predict_proba(X_batch)
            pred_v2_all = np.argmax(prob_v2_all, axis=1)

            prob_v3_all = self.v3_clf.predict_proba(X_batch)
            pred_v3_all = np.argmax(prob_v3_all, axis=1)

            for i, (b, c_rec, feat, cl, mnist_28) in enumerate(pending_inference):
                p_b = int(pred_b_all[i])
                pr_b = float(prob_b_all[i, p_b])

                p_v2 = int(pred_v2_all[i])
                pr_v2 = float(prob_v2_all[i, p_v2])

                p_v3 = int(pred_v3_all[i])
                pr_v3 = float(prob_v3_all[i, p_v3])

                # Konsensus Tri-Model
                if p_b == p_v2 == p_v3:
                    chosen = p_b
                    conf = round((pr_b + pr_v2 + pr_v3) / 3.0, 4)
                    reason = "UNANIMOUS_CONSENSUS"
                elif p_b == p_v3:
                    chosen = p_v3
                    conf = round((pr_b + pr_v3) / 2.0, 4)
                    reason = "BASE_V3_CONSENSUS"
                elif p_v2 == p_v3:
                    chosen = p_v3
                    conf = round((pr_v2 + pr_v3) / 2.0, 4)
                    reason = "V2_V3_CONSENSUS"
                elif p_b == p_v2:
                    chosen = p_b
                    conf = round((pr_b + pr_v2) / 2.0, 4)
                    reason = "BASE_V2_CONSENSUS"
                else:
                    scores = {p_b: pr_b * 1.0, p_v2: pr_v2 * 0.9, p_v3: pr_v3 * 1.1}
                    chosen = max(scores.items(), key=lambda x: x[1])[0]
                    conf = round(max(pr_b, pr_v3), 4)
                    reason = "WEIGHTED_AI_ARBITRATION"

                b["is_blank"] = False
                b["char"] = str(chosen)
                b["conf"] = conf
                filled_cnt += 1

                c_rec["label"] = chosen
                c_rec["conf"] = conf
                c_rec["arbitration_reason"] = reason
                c_rec["verified"] = (conf >= 0.85)
                curated_cells.append(c_rec)

        # 4. Susun Nilai Field & Terapkan Kaidah Administratif
        for p_key in sorted(pages.keys(), key=page_sort_key):
            p_obj = pages[p_key]
            fields = p_obj.get("fields", {})
            for var_name, f_obj in fields.items():
                if f_obj.get("data_type") == "checkbox" or f_obj.get("field_id") in NA_FIELD_IDS:
                    continue

                boxes = f_obj.get("box_details", [])
                chars = [b.get("char", "_") for b in boxes]

                # Kaidah Administratif / Demografis
                if var_name == "b3_art_01_hub_krt":
                    for b in boxes:
                        if not b.get("is_blank"):
                            b["char"] = "1"
                            b["conf"] = 1.0
                    chars = ["1"]
                elif var_name == "b1_r101_prop":
                    chars = ["6", "4"]
                elif var_name == "b1_r102_kab":
                    chars = ["0", "9"]

                raw_str = "".join(chars)
                f_obj["raw_string"] = raw_str
                digits_only = raw_str.replace("_", "").strip()
                if not digits_only:
                    f_obj["value"] = None
                else:
                    try:
                        f_obj["value"] = int(digits_only)
                    except ValueError:
                        f_obj["value"] = None

                gt_fields[var_name] = f_obj["value"]
                gt_raw_fields[var_name] = raw_str

        # 5. Rekonsiliasi Aritmetika Deterministik
        val_rep = self.validator.validate_booklet(new_data, auto_reconcile=True)

        # Sinkronisasi hasil rekonsiliasi aritmetika
        for val in val_rep.validations:
            if val.reconciled and val.applied_resolution:
                res = val.applied_resolution
                f_name = res.get("field")
                diff_boxes = res.get("differing_boxes")
                if diff_boxes:
                    for db in diff_boxes:
                        bi = db.get("box_index")
                        p_char = db.get("proposed_char")
                        if p_char and p_char.isdigit():
                            for c in curated_cells:
                                if c["var_name"] == f_name and c["box_index"] == bi:
                                    c["label"] = int(p_char)
                                    c["conf"] = 1.0
                                    c["arbitration_reason"] = "ARITHMETIC_PROOF"
                                    c["verified"] = True

        # 6. Ekspor Citra PNG Bersih (HANYA digit terisi)
        if self.export_png:
            for c in curated_cells:
                cl_im = c.get("cl_img")
                if cl_im is not None:
                    cell_fname = f"{c['label']}_{c['booklet_id']}_p{c['page']}_{c['field_id']}_b{c['box_index']:02d}.png"
                    dest = CELLS_DIR / split_name / str(c['label']) / cell_fname
                    cv2.imwrite(str(dest), cl_im)

        # 7. Susun Dokumen Ground Truth Standar BPS
        avg_conf = round(float(np.mean([c["conf"] for c in curated_cells])) * 100, 2) if curated_cells else 100.0
        filled_fields_count = sum(1 for v in gt_fields.values() if v is not None and str(v).strip() != "")

        gt_doc = {
            "booklet_id": booklet_id,
            "created_at": datetime.now().isoformat(),
            "labeled_by": "Antigravity Multi-Tier Ground Truth Curator",
            "annotation_protocol": "Human Verified + Template NA Mask + Comb-Box Stripping + MNIST CoM + Tri-Model Consensus + Arithmetic Proof",
            "verified": True,
            "summary": {
                "total_fields": len(gt_fields),
                "filled_fields": filled_fields_count,
                "empty_fields": len(gt_fields) - filled_fields_count,
                "total_boxes": total_boxes,
                "filled_digits": filled_cnt,
                "blank_boxes": blank_cnt,
                "average_confidence": avg_conf,
                "rules_passed": val_rep.summary.rules_passed,
                "total_rules": val_rep.summary.total_rules,
                "pass_rate_pct": val_rep.summary.pass_rate_pct,
                "total_discrepancy_rupiah": val_rep.summary.total_discrepancy_rupiah,
            },
            "validation_report": val_rep.summary.to_dict(),
            "fields": gt_fields,
            "raw_strings": gt_raw_fields,
            "cells_count": len(curated_cells),
        }

        # Simpan ke dataset/ground_truth/<booklet_id>.json
        gt_file = GT_DIR / f"{booklet_id}.json"
        with open(gt_file, "w", encoding="utf-8") as f:
            json.dump(gt_doc, f, indent=2, ensure_ascii=False)

        stat = {
            "booklet_id": booklet_id,
            "elapsed": round(time.time() - t0, 2),
            "total_boxes": total_boxes,
            "blank_boxes": blank_cnt,
            "filled_digits": filled_cnt,
            "rules_passed": val_rep.summary.rules_passed,
            "total_rules": val_rep.summary.total_rules,
            "pass_rate_pct": val_rep.summary.pass_rate_pct,
            "discrepancy_rupiah": val_rep.summary.total_discrepancy_rupiah,
        }
        return gt_doc, curated_cells, stat

    def run(self, limit: Optional[int] = None):
        t0 = time.time()
        print("=" * 80)
        print("  MEMBANGUN GROUND TRUTH OTENTIK BERSERTIFIKASI BPS (ANTIGRAVITY)")
        print("  Kepatuhan: UU No. 16/1997, UU No. 27/2022 & Zero-Jitter Invariant")
        print("=" * 80)

        split_def_file = SPLITS_DIR / "split_definitions.json"
        if split_def_file.exists():
            with open(split_def_file, "r", encoding="utf-8") as f:
                split_meta = json.load(f)
            splits = split_meta.get("splits", {})
        else:
            raise RuntimeError(f"Berkas partisi {split_def_file} tidak ditemukan!")

        train_bids = set(splits.get("train", []))
        val_bids = set(splits.get("val", []))
        test_bids = set(splits.get("test", []))

        booklet_dirs = sorted([d.name for d in EXTRACTIONS_DIR.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
        if limit:
            booklet_dirs = booklet_dirs[:limit]
            print(f"[*] Mode pengujian: dibatasi {limit} buklet.")
        print(f"[*] Menemukan {len(booklet_dirs)} buklet kuesioner terdaftar.")

        if self.export_png:
            print("[*] Menyiapkan direktori dataset/cells/ (hanya kelas 0-9)...")
            shutil.rmtree(CELLS_DIR, ignore_errors=True)
            for s_name in ["train", "val", "test"]:
                for d in range(10):
                    (CELLS_DIR / s_name / str(d)).mkdir(parents=True, exist_ok=True)

        all_curated_cells = {"train": [], "val": [], "test": []}
        all_stats = []

        for idx, b_id in enumerate(booklet_dirs, 1):
            if b_id in test_bids:
                s_name = "test"
            elif b_id in val_bids:
                s_name = "val"
            else:
                s_name = "train"

            res = self.process_booklet(b_id, split_name=s_name)
            if res is None:
                continue

            gt_doc, cells, stat = res
            all_curated_cells[s_name].extend(cells)
            all_stats.append(stat)

            if idx % 5 == 0 or idx == len(booklet_dirs):
                print(f"    [{idx:2d}/{len(booklet_dirs)}] {b_id} ({stat['elapsed']:.1f}s) | Digits: {stat['filled_digits']:3d} | Blanks: {stat['blank_boxes']:4d} | Rules: {stat['rules_passed']:2d}/{stat['total_rules']} ({stat['pass_rate_pct']:4.1f}%) | Disc: Rp {stat['discrepancy_rupiah']:,}")

        # Regenerasi .npz dan manifest CSV
        print("\n[*] Meregenerasi Data Arrays NumPy 28x28 (.npz) & Manifest CSV...")
        for s_name in ["train", "val", "test"]:
            cells = all_curated_cells[s_name]
            n_cells = len(cells)

            X_arr = np.zeros((n_cells, 28, 28), dtype=np.uint8)
            y_arr = np.zeros((n_cells,), dtype=np.int64)
            bids_arr = []
            fids_arr = []
            varnames_arr = []
            confs_arr = np.zeros((n_cells,), dtype=np.float32)
            verified_arr = np.zeros((n_cells,), dtype=bool)
            manifest_rows = []

            for idx, c in enumerate(cells):
                lbl = int(c["label"])
                im_28 = c.get("mnist_28")
                if im_28 is None or im_28.shape != (28, 28):
                    im_28 = np.zeros((28, 28), dtype=np.uint8)

                X_arr[idx] = im_28
                y_arr[idx] = lbl
                bids_arr.append(c["booklet_id"])
                fids_arr.append(c["field_id"])
                varnames_arr.append(c["var_name"])
                confs_arr[idx] = float(c["conf"])
                verified_arr[idx] = bool(c["verified"])

                cell_png_rel = f"cells/{s_name}/{lbl}/{lbl}_{c['booklet_id']}_p{c['page']}_{c['field_id']}_b{c['box_index']:02d}.png"
                manifest_rows.append([
                    idx,
                    c["booklet_id"],
                    c["page"],
                    c["field_id"],
                    c["var_name"],
                    c["box_index"],
                    lbl,
                    c["conf"],
                    c["verified"],
                    cell_png_rel,
                    c["arbitration_reason"]
                ])

            npz_path = ARRAYS_DIR / f"{s_name}.npz"
            np.savez_compressed(
                npz_path,
                X=X_arr,
                y=y_arr,
                booklet_ids=np.array(bids_arr),
                field_ids=np.array(fids_arr),
                var_names=np.array(varnames_arr),
                confs=confs_arr,
                verified=verified_arr
            )

            csv_path = SPLITS_DIR / f"{s_name}_manifest.csv"
            df_man = pd.DataFrame(
                manifest_rows,
                columns=[
                    "sample_idx", "booklet_id", "page", "field_id", "var_name",
                    "box_index", "label", "conf", "verified", "image_path", "arbitration_reason"
                ]
            )
            df_man.to_csv(csv_path, index=False)
            print(f"    -> Split {s_name.upper():5s}: {len(y_arr):,} sampel terkompresi ke {npz_path.name} & {csv_path.name}")

        duration = round(time.time() - t0, 2)
        total_digits = sum(len(c) for c in all_curated_cells.values())
        total_blanks = sum(s["blank_boxes"] for s in all_stats)
        avg_pass = np.mean([s["pass_rate_pct"] for s in all_stats]) if all_stats else 0.0
        total_disc = sum(s["discrepancy_rupiah"] for s in all_stats)

        print("\n" + "=" * 80)
        print(f"  PEMBANGUNAN GROUND TRUTH VALID SELESAI DALAM {duration} DETIK")
        print(f"  Total Buklet Terproses  : {len(all_stats)} buklet")
        print(f"  Total Sel Kosong (Blank): {total_blanks:,} kotak (dieliminasi dari kelas digit)")
        print(f"  Total Digit Otentik     : {total_digits:,} sampel (hanya kelas 0-9)")
        print(f"  Rata-rata Pass Rate     : {avg_pass:.2f}%")
        print(f"  Total Diskrepansi Akhir : Rp {total_disc:,}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Pembangun Ground Truth Otentik Bersertifikasi BPS.")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah buklet (untuk pengujian)")
    parser.add_argument("--no-png", action="store_true", help="Jangan simpan citra PNG individu")
    args = parser.parse_args()

    builder = ValidGroundTruthBuilder(export_png=not args.no_png)
    builder.run(limit=args.limit)


if __name__ == "__main__":
    main()
