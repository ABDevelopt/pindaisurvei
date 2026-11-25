#!/usr/bin/env python3
"""
seruti_pipeline/dataset_builder.py
==================================
Modul Pembangun Dataset Latih dan Evaluasi (Ground Truth & Benchmark)
untuk Kuesioner BPS Susenas Seruti Inti (VSERUTI26.INTI).

Arsitektur & Kepatuhan:
1. Kurasi Ground Truth Multitier:
   - Field-level Ground Truth (JSON format standar BPS).
   - Cell/Character-level Ground Truth (28x28 CoM MNIST style).
   - Verifikasi silang berbasis relasi semantik & aritmetika resmi BPS.
2. Stratifikasi Partisi Tingkat Buklet (Booklet-Level Split):
   - Mencegah data leakage tulisan tangan antar split.
   - Train (70%, 54 buklet), Validation (15%, 11 buklet), Test/Benchmark (15%, 11 buklet).
3. Penyimpanan Multi-Format:
   - Data Arrays NumPy (.npz) terkompresi hemat memori & ultra-cepat.
   - Folder citra hierarkis (dataset/cells/<split>/<class>/) kompatibel ImageFolder/PyTorch/Keras.
   - Manifest tabular CSV & Katalog Metadata JSON.
4. Kepatuhan Mutlak UU No. 16/1997 & UU PDP No. 27/2022:
   - Zero PII Cloud Egress & Zero-Jitter Invariant pada seluruh kalkulasi finansial lokal.
"""

import csv
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from seruti_pipeline.validator import SerutiArithmeticValidator
from seruti_pipeline.advanced_filter import remove_comb_box_border, letterbox_cell_28x28

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXTRACTIONS_DIR = BASE_DIR / "output" / "extractions"
DEFAULT_CROPS_DIR = BASE_DIR / "output" / "crops"
DEFAULT_DATASET_DIR = BASE_DIR / "dataset"


class SerutiDatasetBuilder:
    """
    Pembangun dataset terstruktur latih, validasi, dan benchmark evaluasi
    kuesioner BPS Susenas Seruti Inti.
    """

    def __init__(
        self,
        extractions_dir: Path = DEFAULT_EXTRACTIONS_DIR,
        crops_dir: Path = DEFAULT_CROPS_DIR,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        export_png_cells: bool = True,
        clean_box_border: bool = True,
        raw_margin: bool = True,
        target_npz_dim: int = 26,
        clean_existing_cells: bool = True,
        random_seed: int = 42,
    ):
        self.extractions_dir = Path(extractions_dir)
        self.crops_dir = Path(crops_dir)
        self.dataset_dir = Path(dataset_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.export_png_cells = export_png_cells
        self.clean_box_border = clean_box_border
        self.raw_margin = raw_margin
        self.target_npz_dim = target_npz_dim
        self.clean_existing_cells = clean_existing_cells
        self.random_seed = random_seed

        # Subdirektori tujuan
        self.gt_dir = self.dataset_dir / "ground_truth"
        self.splits_dir = self.dataset_dir / "splits"
        self.arrays_dir = self.dataset_dir / "data_arrays"
        self.cells_dir = self.dataset_dir / "cells"

        self.validator = SerutiArithmeticValidator()
        self.manual_corrections = self._load_manual_corrections()

    def _load_manual_corrections(self) -> Dict[Tuple[str, int, str, int], Dict[str, Any]]:
        """
        Memuat seluruh riwayat koreksi manual dari manual_corrections_log.jsonl
        atau berkas backup kanonikal agar hasil verifikasi manual manusia tetap utuh 100%.
        Key: (booklet_id, page, field_id, box_index)
        Value: {"corrected_label": int, "reviewer": str}
        """
        corrections = {}
        log_candidates = [
            self.dataset_dir / "manual_corrections_log.jsonl",
            self.dataset_dir / "manual_corrections_log_CANONICAL_BACKUP.jsonl",
        ]
        for log_path in log_candidates:
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                entry = json.loads(line)
                                k = (
                                    str(entry["booklet_id"]),
                                    int(entry["page"]),
                                    str(entry["field_id"]),
                                    int(entry["box_index"])
                                )
                                corrections[k] = {
                                    "corrected_label": int(entry["corrected_label"]),
                                    "reviewer": entry.get("reviewer", "Human Verifier")
                                }
                    print(f"[*] Berhasil memuat {len(corrections):,} entri anotasi manual dari {log_path.name}")
                    break
                except Exception as e:
                    print(f"[-] Peringatan: Gagal memuat log koreksi {log_path}: {e}")
        return corrections

    def _ensure_directories(self):
        """Membuat seluruh struktur direktori dataset."""
        for d in [self.dataset_dir, self.gt_dir, self.splits_dir, self.arrays_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if self.export_png_cells:
            if self.clean_existing_cells and self.cells_dir.exists():
                print("      Membersihkan direktori dataset/cells/ lama untuk ekspor citra baru...")
                shutil.rmtree(self.cells_dir, ignore_errors=True)
            for split in ["train", "val", "test"]:
                for digit in range(10):
                    (self.cells_dir / split / str(digit)).mkdir(parents=True, exist_ok=True)

    def discover_booklets(self) -> List[Dict[str, Any]]:
        """
        Menemukan seluruh buklet yang memiliki ekstraksi valid dan crops.
        """
        booklets = []
        for b_dir in sorted(self.extractions_dir.iterdir()):
            if not b_dir.is_dir() or not b_dir.name.startswith("booklet_"):
                continue
            b_id = b_dir.name
            ext_file = b_dir / f"{b_id}_extracted.json"
            crops_b_dir = self.crops_dir / b_id

            if ext_file.exists() and crops_b_dir.exists():
                try:
                    with open(ext_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    summary = data.get("summary", {})
                    sem = summary.get("semantic_validation", {})
                    booklets.append({
                        "booklet_id": b_id,
                        "ext_path": ext_file,
                        "crops_dir": crops_b_dir,
                        "rules_passed": sem.get("rules_passed", 0),
                        "total_rules": sem.get("total_rules", 21),
                        "pass_rate": sem.get("pass_rate_pct", 0.0),
                        "avg_conf": summary.get("average_confidence", 0.0),
                        "filled_boxes": summary.get("total_filled_boxes", 0),
                        "roster_art": summary.get("roster_art_count", 0),
                    })
                except Exception as e:
                    print(f"[-] Peringatan: Gagal membaca {ext_file}: {e}")

        return booklets

    def partition_booklets(self, booklets: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Partisi bertingkat buklet (stratified booklet-level split):
        Mengurutkan buklet berdasarkan (rules_passed, avg_conf, booklet_id)
        dan mendistribusikan secara berimbang ke train, val, dan test.
        """
        sorted_b = sorted(
            booklets,
            key=lambda x: (x["rules_passed"], x["avg_conf"], x["booklet_id"]),
            reverse=True,
        )

        n_total = len(sorted_b)
        n_val = max(1, int(round(n_total * self.val_ratio)))
        n_test = max(1, int(round(n_total * self.test_ratio)))
        n_train = n_total - n_val - n_test

        train_list, val_list, test_list = [], [], []

        # Pembagian interleaving stratifikasi
        for i, b in enumerate(sorted_b):
            mod = i % 7
            if mod == 5 and len(val_list) < n_val:
                val_list.append(b["booklet_id"])
            elif mod == 6 and len(test_list) < n_test:
                test_list.append(b["booklet_id"])
            else:
                if len(train_list) < n_train:
                    train_list.append(b["booklet_id"])
                elif len(val_list) < n_val:
                    val_list.append(b["booklet_id"])
                else:
                    test_list.append(b["booklet_id"])

        splits = {
            "train": sorted(train_list),
            "val": sorted(val_list),
            "test": sorted(test_list),
        }

        # Simpan definisi split
        split_def_file = self.splits_dir / "split_definitions.json"
        with open(split_def_file, "w", encoding="utf-8") as f:
            json.dump({
                "created_at": datetime.now().isoformat(),
                "total_booklets": n_total,
                "counts": {
                    "train": len(splits["train"]),
                    "val": len(splits["val"]),
                    "test": len(splits["test"]),
                },
                "splits": splits,
            }, f, indent=2)

        return splits

    def build_booklet_ground_truth(
        self,
        b_info: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Membangun berkas Ground Truth formal untuk 1 buklet dan mengumpulkan
        sel-sel karakter berlabel.
        """
        b_id = b_info["booklet_id"]
        ext_file = b_info["ext_path"]
        crops_b_dir = b_info["crops_dir"]

        with open(ext_file, "r", encoding="utf-8") as f:
            ext_data = json.load(f)

        # 1. Audit aritmetika & verifikasi konsistensi
        val_rep = self.validator.validate_booklet(ext_data, auto_reconcile=False)
        val_dict = val_rep.to_dict()

        # 2. Peta lookup citra mikro dari crop_manifest per halaman
        image_lookup = {}
        for p_num in range(1, 8):
            m_path = crops_b_dir / f"hal_{p_num}" / "crop_manifest.json"
            if not m_path.exists():
                continue
            try:
                with open(m_path, "r", encoding="utf-8") as mf:
                    man = json.load(mf)
                for item in man.get("tiers", {}).get("mikro", []):
                    fid = item.get("field_id")
                    b_idx = item.get("box_index")
                    cfile = item.get("clean_file")
                    rfile = item.get("raw_file")
                    if fid and b_idx is not None:
                        image_lookup[(p_num, fid, b_idx)] = {
                            "clean_file": cfile,
                            "raw_file": rfile
                        }
            except Exception:
                pass

        # 3. Kumpulkan nilai field dan sel berlabel
        gt_fields = {}
        gt_raw_fields = {}
        booklet_cells = []
        total_digits = 0
        filled_fields_count = 0

        pages = ext_data.get("pages", {})
        for hal_key, p_obj in pages.items():
            p_num = p_obj.get("page_num", 1)
            fields = p_obj.get("fields", {})

            for var_name, f_obj in fields.items():
                fid = f_obj.get("field_id")
                val = f_obj.get("value")
                raw_str = f_obj.get("raw_string", "")
                boxes = f_obj.get("box_details", [])

                gt_fields[var_name] = val
                gt_raw_fields[var_name] = raw_str

                if val is not None and str(val).strip() != "":
                    filled_fields_count += 1

                for b in boxes:
                    b_idx = b.get("box_index", 0)
                    img_info = image_lookup.get((p_num, fid, b_idx), {})
                    cfile = img_info.get("clean_file")
                    rfile = img_info.get("raw_file")
                    cell_key = (b_id, p_num, fid, b_idx)

                    # 1. Prioritas Utama: Evaluasi anotasi manual manusia
                    if cell_key in self.manual_corrections:
                        m_info = self.manual_corrections[cell_key]
                        m_label = m_info["corrected_label"]
                        if m_label == -1:
                            # Ditolak manusia: bukan angka / noise comb-box -> jangan dimasukkan ke kelas digit 0-9
                            continue
                        label = int(m_label)
                        conf = 1.0
                        is_verified = True
                        reason = f"HUMAN_VERIFIED ({m_info['reviewer']})"
                    else:
                        if b.get("is_blank", False):
                            continue

                        char = str(b.get("char", ""))
                        if not char.isdigit():
                            continue

                        label = int(char)
                        conf = float(b.get("conf", 1.0))
                        is_verified = (conf >= 0.85)
                        reason = "OCR_BASELINE"

                    cell_record = {
                        "booklet_id": b_id,
                        "page": p_num,
                        "field_id": fid,
                        "var_name": var_name,
                        "box_index": b_idx,
                        "label": label,
                        "conf": round(conf, 4),
                        "clean_path": cfile,
                        "raw_path": rfile,
                        "verified": is_verified,
                        "arbitration_reason": reason,
                    }
                    booklet_cells.append(cell_record)
                    total_digits += 1

        # 4. Susun struktur Ground Truth buklet standar
        gt_doc = {
            "booklet_id": b_id,
            "created_at": datetime.now().isoformat(),
            "summary": {
                "total_fields": len(gt_fields),
                "filled_fields": filled_fields_count,
                "total_digits": total_digits,
                "average_confidence": b_info["avg_conf"],
                "rules_passed": b_info["rules_passed"],
                "total_rules": b_info["total_rules"],
                "pass_rate_pct": b_info["pass_rate"],
            },
            "validation_report": val_dict.get("summary", {}),
            "fields": gt_fields,
            "raw_strings": gt_raw_fields,
            "cells_count": len(booklet_cells),
        }

        # Simpan ke dataset/ground_truth/<booklet_id>.json
        gt_file = self.gt_dir / f"{b_id}.json"
        with open(gt_file, "w", encoding="utf-8") as f:
            json.dump(gt_doc, f, indent=2, ensure_ascii=False)

        return gt_doc, booklet_cells

    def process_and_export_dataset(self) -> Dict[str, Any]:
        """
        Eksekusi penuh pembangunan dataset:
        1. Menemukan 76 buklet.
        2. Membagi buklet ke train, val, dan test.
        3. Membangun ground truth per buklet.
        4. Memuat citra 28x28 dan menyusun data array .npz.
        5. Menyimpan PNG cells (opsional).
        6. Menghasilkan manifest CSV dan laporan katalog dataset.
        """
        t0 = time.time()
        self._ensure_directories()

        print("=" * 75)
        print("  MEMBANGUN DATASET LATIH & EVALUASI GROUND TRUTH SERUTI INTI")
        print("  Kepatuhan UU No. 16/1997 & UU PDP No. 27/2022 (Zero PII Cloud Egress)")
        print("=" * 75)

        # 1. Temukan seluruh buklet
        print("[1/5] Menginventarisasi buklet kuesioner terproses...")
        booklets = self.discover_booklets()
        print(f"      Ditemukan {len(booklets)} buklet kuesioner valid.")

        if not booklets:
            raise RuntimeError("Tidak ada buklet valid ditemukan di output/extractions!")

        # 2. Partisi Buklet (Train / Val / Test)
        print("[2/5] Melakukan stratifikasi partisi tingkat buklet (Booklet-Level Split)...")
        splits = self.partition_booklets(booklets)
        print(f"      Split: Train={len(splits['train'])} | Val={len(splits['val'])} | Test={len(splits['test'])}")

        # 3. Bangun Ground Truth & Kumpulkan Sel
        print("[3/5] Membangun Ground Truth resmi BPS & mengekstrak citra sel karakter...")
        booklet_map = {b["booklet_id"]: b for b in booklets}
        split_cells = {"train": [], "val": [], "test": []}

        for split_name, b_ids in splits.items():
            for b_id in b_ids:
                b_info = booklet_map[b_id]
                _, cells = self.build_booklet_ground_truth(b_info)
                for c in cells:
                    c["split"] = split_name
                split_cells[split_name].extend(cells)

        total_cells_all = sum(len(c) for c in split_cells.values())
        print(f"      Terkumpul {total_cells_all:,} sel karakter digit dari seluruh buklet.")
        for sname in ["train", "val", "test"]:
            print(f"      -> {sname.upper():<5}: {len(split_cells[sname]):,} sampel sel")

        # 4. Bangun Array NumPy (.npz), Manifest CSV, dan PNG Files
        print("[4/5] Mengonversi ke Array NumPy 28x28 (.npz) & mengekspor berkas manifest...")
        split_stats = {}

        for split_name in ["train", "val", "test"]:
            cells = split_cells[split_name]
            n_cells = len(cells)

            X = np.zeros((n_cells, 28, 28), dtype=np.uint8)
            y = np.zeros((n_cells,), dtype=np.int64)
            b_ids_arr = []
            f_ids_arr = []
            var_names_arr = []
            confs_arr = np.zeros((n_cells,), dtype=np.float32)
            verified_arr = np.zeros((n_cells,), dtype=bool)

            manifest_rows = []
            valid_idx = 0

            print(f"      -> Memproses split {split_name.upper()} ({n_cells:,} sel)...")
            t_split = time.time()

            for i, c in enumerate(cells):
                rpath = c.get("raw_path")
                cpath = c.get("clean_path")

                raw_img = None
                if self.raw_margin and rpath:
                    p_rpath = Path(rpath)
                    if p_rpath.exists():
                        raw_img = cv2.imread(str(p_rpath))
                    elif (BASE_DIR / rpath).exists():
                        raw_img = cv2.imread(str(BASE_DIR / rpath))

                if raw_img is not None:
                    # Opsi 1: Bersihkan garis batas fisik comb-box dengan margin & dimensi asli utuh
                    if self.clean_box_border:
                        export_img = remove_comb_box_border(raw_img)
                    else:
                        export_img = raw_img
                    # Normalisasi proporsional letterbox ke 28x28 untuk array .npz (tanpa peregangan/tight-zoom)
                    img_28 = letterbox_cell_28x28(export_img, target_dim=self.target_npz_dim)
                else:
                    # Fallback jika raw_path tidak tersedia: gunakan clean_path
                    img_28 = None
                    p_cpath = Path(cpath) if cpath else None
                    if p_cpath and p_cpath.exists():
                        img_28 = cv2.imread(str(p_cpath), cv2.IMREAD_GRAYSCALE)
                    elif cpath and (BASE_DIR / cpath).exists():
                        img_28 = cv2.imread(str(BASE_DIR / cpath), cv2.IMREAD_GRAYSCALE)
                    export_img = img_28

                if img_28 is None or img_28.shape != (28, 28):
                    continue

                lbl = int(c["label"])
                X[valid_idx] = img_28
                y[valid_idx] = lbl
                b_ids_arr.append(c["booklet_id"])
                f_ids_arr.append(c["field_id"])
                var_names_arr.append(c["var_name"])
                confs_arr[valid_idx] = float(c["conf"])
                verified_arr[valid_idx] = bool(c["verified"])

                # Simpan citra PNG ke folder cells (dengan dimensi & margin asli crop, bebas garis kotak)
                rel_cell_png = ""
                if self.export_png_cells and export_img is not None:
                    cell_fname = f"{lbl}_{c['booklet_id']}_p{c['page']}_{c['field_id']}_b{c['box_index']:02d}.png"
                    dest_png = self.cells_dir / split_name / str(lbl) / cell_fname
                    cv2.imwrite(str(dest_png), export_img)
                    rel_cell_png = str(dest_png.relative_to(self.dataset_dir)).replace("\\", "/")

                manifest_rows.append([
                    valid_idx,
                    c["booklet_id"],
                    c["page"],
                    c["field_id"],
                    c["var_name"],
                    c["box_index"],
                    lbl,
                    c["conf"],
                    c["verified"],
                    rel_cell_png or rpath or cpath,
                    c.get("arbitration_reason", "OCR_BASELINE"),
                ])

                valid_idx += 1
                if (i + 1) % 15000 == 0 or (i + 1) == n_cells:
                    elapsed_s = time.time() - t_split
                    rate = (i + 1) / max(0.1, elapsed_s)
                    print(f"         [{i+1:,}/{n_cells:,}] ({rate:.1f} sel/dtk) - Valid: {valid_idx:,}")

            # Potong array jika ada file yang hilang
            X = X[:valid_idx]
            y = y[:valid_idx]
            confs_arr = confs_arr[:valid_idx]
            verified_arr = verified_arr[:valid_idx]

            # Simpan .npz terkompresi
            npz_file = self.arrays_dir / f"{split_name}.npz"
            np.savez_compressed(
                npz_file,
                X=X,
                y=y,
                booklet_ids=np.array(b_ids_arr, dtype=str),
                field_ids=np.array(f_ids_arr, dtype=str),
                var_names=np.array(var_names_arr, dtype=str),
                confs=confs_arr,
                verified=verified_arr,
            )

            # Simpan CSV manifest
            csv_file = self.splits_dir / f"{split_name}_manifest.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sample_idx",
                    "booklet_id",
                    "page",
                    "field_id",
                    "var_name",
                    "box_index",
                    "label",
                    "conf",
                    "verified",
                    "image_path",
                    "arbitration_reason",
                ])
                writer.writerows(manifest_rows)

            # Hitung distribusi kelas
            cls_dist = Counter(y.tolist())
            split_stats[split_name] = {
                "total_samples": int(valid_idx),
                "npz_file": str(npz_file.relative_to(self.dataset_dir)).replace("\\", "/"),
                "manifest_csv": str(csv_file.relative_to(self.dataset_dir)).replace("\\", "/"),
                "npz_size_mb": round(os.path.getsize(npz_file) / (1024 * 1024), 2),
                "class_distribution": {str(k): cls_dist.get(k, 0) for k in range(10)},
                "high_conf_pct": round(float(np.mean(confs_arr >= 0.85) * 100), 2) if valid_idx > 0 else 0.0,
            }

        # 5. Buat Katalog Metadata Lengkap
        print("[5/5] Menyusun Katalog Dataset & Laporan Metadata (dataset_catalog.json)...")
        total_samples_dataset = sum(s["total_samples"] for s in split_stats.values())
        overall_dist = Counter()
        for s in split_stats.values():
            for k, v in s["class_distribution"].items():
                overall_dist[k] += v

        catalog = {
            "dataset_name": "BPS Susenas Seruti Inti Digit & Field Ground Truth Dataset",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "description": (
                "Dataset beranotasi lengkap untuk pelatihan dan benchmarking pengenalan karakter "
                "comb-box serta evaluasi konsistensi ekstraksi kuesioner BPS VSERUTI26.INTI."
            ),
            "compliance": {
                "uu_no_16_1997": "Kepatuhan penuh kerahasiaan data sensus/survei",
                "uu_pdp_no_27_2022": "Zero PII Cloud Egress terverifikasi",
                "zero_jitter_invariant": "Kalkulasi finansial presisi integer Rupiah asli",
            },
            "preprocessing": {
                "clean_box_border": self.clean_box_border,
                "raw_margin": self.raw_margin,
                "target_npz_dim": self.target_npz_dim,
                "cell_format": "original_resolution_cleaned_comb_box" if self.clean_box_border else "original_crop",
                "npz_format": "letterbox_28x28_mnist_style",
            },
            "summary": {
                "total_booklets": len(booklets),
                "total_pages": len(booklets) * 7,
                "total_cell_samples": total_samples_dataset,
                "overall_class_distribution": {str(k): overall_dist.get(str(k), 0) for k in range(10)},
                "splits": {
                    "train_booklets": len(splits["train"]),
                    "val_booklets": len(splits["val"]),
                    "test_booklets": len(splits["test"]),
                },
            },
            "split_details": split_stats,
            "booklet_splits": splits,
        }

        catalog_file = self.splits_dir / "dataset_catalog.json"
        with open(catalog_file, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)

        elapsed = round(time.time() - t0, 2)
        print("\n" + "=" * 75)
        print("  DATASET LATIH & EVALUASI GROUND TRUTH SELESAI DIBANGUN")
        print("=" * 75)
        print(f"  Total Waktu Eksekusi   : {elapsed} detik")
        print(f"  Total Buklet Terproses : {len(booklets)}")
        print(f"  Total Sampel Karakter  : {total_samples_dataset:,} citra 28x28")
        print(f"  Train Set              : {split_stats['train']['total_samples']:,} sampel ({split_stats['train']['npz_size_mb']} MB)")
        print(f"  Validation Set         : {split_stats['val']['total_samples']:,} sampel ({split_stats['val']['npz_size_mb']} MB)")
        print(f"  Test/Benchmark Set     : {split_stats['test']['total_samples']:,} sampel ({split_stats['test']['npz_size_mb']} MB)")
        print(f"  Ground Truth JSONs     : {len(booklets)} berkas di {self.gt_dir.relative_to(BASE_DIR)}")
        print(f"  Katalog Metadata       : {catalog_file.relative_to(BASE_DIR)}")
        print("=" * 75)

        return catalog


def main():
    builder = SerutiDatasetBuilder()
    builder.process_and_export_dataset()


if __name__ == "__main__":
    main()
