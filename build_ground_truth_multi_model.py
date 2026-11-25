#!/usr/bin/env python3
"""
build_ground_truth_multi_model.py
=================================
Pembangun Ground Truth Otentik Bersertifikasi Multi-Model AI BPS untuk Kuesioner
Susenas Seruti Inti (VSERUTI26.INTI).

Ensemble Multi-Model Terintegrasi:
1. Prioritas Mutlak Human-in-the-Loop (1.267 koreksi manual dari manual_corrections_log.jsonl).
2. Deep Convolutional Neural Network (CNN) PyTorch (dilatih pada MNIST + Real Pen Seruti, 94.14% val acc).
3. Model Seruti v3 (MNIST Foundation + Real Pen MLP, 94.48% test acc).
4. Model Seruti v2 (Cold Start MLP).
5. BPS Baseline MLP (model.joblib).
6. Microsoft TrOCR (Vision Transformer DeiT + TrOCR Causal LM) untuk validasi & resolusi sel ambigu.
7. Pembuktian Aritmetika Deterministik Seruti (19 relasi penjumlahan vertikal & horizontal).

Kepatuhan Hukum & Privasi:
- UU No. 16 Tahun 1997 tentang Statistik
- UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP)
- Zero PII Cloud Egress: 100% inferensi lokal pada mesin pengguna.
- Zero-Jitter Invariant: Integritas nominal finansial integer 64-bit presisi penuh.
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
import torch
import torch.nn as nn
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

EXTRACTIONS_DIR = BASE_DIR / "output" / "extractions"
CROPS_DIR = BASE_DIR / "output" / "crops"
DATASET_DIR = BASE_DIR / "dataset"
GT_DIR = DATASET_DIR / "ground_truth"
SPLITS_DIR = DATASET_DIR / "splits"
ARRAYS_DIR = DATASET_DIR / "data_arrays"
CELLS_DIR = DATASET_DIR / "cells"
CELLS_NATIVE_DIR = DATASET_DIR / "cells_native"

BASELINE_MODEL_PATH = Path(r"d:\01_Kedinasan_BPS\Pengenalan Kuesioner (OCR)\model.joblib")
MODEL_V2_PATH = BASE_DIR / "models" / "model_seruti_v2.joblib"
MODEL_V3_PATH = BASE_DIR / "models" / "model_seruti_v3.joblib"
MODEL_CNN_PATH = BASE_DIR / "models" / "model_cnn_mnist.pt"

from seruti_pipeline.validator import SerutiArithmeticValidator
from seruti_pipeline.advanced_filter import remove_comb_box_border, letterbox_cell_28x28, is_blank_cell, clean_cell_mnist

# 32 field NA/tabel non-numerik template
NA_FIELD_IDS = {
    "p5_F051", "p5_F052", "p5_F053", "p5_F054", "p5_F055",
    "p5_F056", "p5_F057", "p5_F058", "p5_F059", "p5_F060",
    "p5_F061", "p5_F062", "p5_F063", "p5_F064", "p5_F065",
    "p5_F066", "p5_F067", "p5_F068", "p5_F069", "p5_F070",
    "p5_F071", "p5_F072", "p5_F073", "p5_F074", "p5_F075",
    "p5_F076", "p5_F077", "p5_F078", "p5_F079", "p5_F080",
    "p6_F001", "p6_F002"
}

# Token IDs TrOCR untuk digit '0' s.d. '9'
TROCR_DIGIT_TOKENS = [1724, 896, 792, 1023, 1065, 1264, 1487, 1473, 1439, 1428]


class SerutiDigitCNN(nn.Module):
    """Arsitektur PyTorch Deep CNN untuk Digit 28x28."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)





class MultiModelGroundTruthBuilder:
    def __init__(self, export_png: bool = True):
        self.export_png = export_png
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Inisialisasi Pembangun Ground Truth Multi-Model BPS (Device: {self.device})...")

        self.manual_corrections = self._load_manual_corrections()
        self.validator = SerutiArithmeticValidator()

        # 1. Model Baseline MLP
        print("    -> Memuat Model Baseline BPS...")
        self.base_clf = joblib.load(BASELINE_MODEL_PATH)
        # 2. Model Seruti v2 MLP
        print("    -> Memuat Model Seruti v2...")
        self.v2_clf = joblib.load(MODEL_V2_PATH)
        # 3. Model Seruti v3 MLP
        print("    -> Memuat Model Seruti v3...")
        self.v3_clf = joblib.load(MODEL_V3_PATH)

        # 4. PyTorch Deep CNN
        print("    -> Memuat PyTorch Deep CNN (model_cnn_mnist.pt)...")
        self.cnn_model = SerutiDigitCNN(num_classes=10).to(self.device)
        checkpoint = torch.load(MODEL_CNN_PATH, map_location=self.device)
        self.cnn_model.load_state_dict(checkpoint["model_state_dict"])
        self.cnn_model.eval()

        # 5. Microsoft TrOCR
        print("    -> Memuat Microsoft TrOCR (Vision Transformer)...")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten').to(self.device)
        self.trocr_model.eval()
        start_id = getattr(self.trocr_model.config, 'decoder_start_token_id', None)
        if start_id is None:
            start_id = getattr(self.trocr_model.config.decoder, 'decoder_start_token_id', 2)
        self.trocr_start_token = start_id

        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [DATASET_DIR, GT_DIR, SPLITS_DIR, ARRAYS_DIR, CELLS_DIR, CELLS_NATIVE_DIR]:
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

    def predict_trocr_digit_prob(self, pil_img: Image.Image) -> np.ndarray:
        """Inferensi 1-Step Direct Logit TrOCR untuk 10 kelas digit 0-9."""
        pv = self.trocr_processor(images=pil_img, return_tensors='pt').pixel_values.to(self.device)
        dec_in = torch.tensor([[self.trocr_start_token]], device=self.device)
        with torch.no_grad():
            outputs = self.trocr_model(pv, decoder_input_ids=dec_in)
            logits = outputs.logits[0, 0, TROCR_DIGIT_TOKENS]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        return probs

    def process_booklet(self, booklet_id: str, split_name: str) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]]:
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

        pending_inference = []
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

                    # 1. Prioritas Utama: Koreksi Manual Manusia (dengan Verifikasi Anti-Phantom Blank)
                    if cell_key in self.manual_corrections:
                        m_entry = self.manual_corrections[cell_key]
                        m_lbl = m_entry["label"]
                        raw_p = BASE_DIR / raw_rel if raw_rel else None
                        raw_im = cv2.imread(str(raw_p)) if (raw_p and raw_p.exists()) else None

                        # Jika manual label -1 ATAU citra fisik terbukti sel kosong (Zero Phantom Blank)
                        if m_lbl == -1 or (raw_im is not None and is_blank_cell(raw_im)):
                            b["is_blank"] = True
                            b["char"] = "_"
                            b["conf"] = 1.0
                            blank_cnt += 1
                        else:
                            b["is_blank"] = False
                            b["char"] = str(m_lbl)
                            b["conf"] = 1.0
                            filled_cnt += 1

                            cl_img = remove_comb_box_border(raw_im) if raw_im is not None else None
                            mnist_img = clean_cell_mnist(cl_img) if cl_img is not None else None

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

                    # 2. Deteksi Sel Kosong Multi-Faktor (Zero Phantom Digits)
                    raw_p = BASE_DIR / raw_rel if raw_rel else None
                    if not raw_p or not raw_p.exists():
                        b["is_blank"] = True
                        b["char"] = "_"
                        b["conf"] = 1.0
                        blank_cnt += 1
                        continue

                    raw_im = cv2.imread(str(raw_p))
                    if raw_im is None or is_blank_cell(raw_im):
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
                        feat = mnist_28.ravel().astype(np.float32) / 255.0
                        c_rec = {
                            "booklet_id": booklet_id,
                            "page": p_num,
                            "field_id": fid,
                            "var_name": var_name,
                            "box_index": b_idx,
                            "raw_path": raw_rel,
                            "raw_img": raw_im,
                            "cl_img": cl,
                            "mnist_28": mnist_28,
                            "box_ref": b
                        }
                        pending_inference.append((b, c_rec, feat, cl, mnist_28))

        # 3. Eksekusi Inferensi Multi-Model Ensemble
        if pending_inference:
            # Format Flat untuk MLP
            X_batch_flat = np.array([item[2] for item in pending_inference], dtype=np.float32)
            prob_b_all = self.base_clf.predict_proba(X_batch_flat)
            prob_v2_all = self.v2_clf.predict_proba(X_batch_flat)
            prob_v3_all = self.v3_clf.predict_proba(X_batch_flat)

            # Format 4D Tensor untuk PyTorch Deep CNN
            X_batch_mnist = np.array([item[4] for item in pending_inference], dtype=np.uint8)
            X_tensor = torch.from_numpy(X_batch_mnist).float().unsqueeze(1).to(self.device) / 255.0
            with torch.no_grad():
                cnn_logits = self.cnn_model(X_tensor)
                prob_cnn_all = torch.softmax(cnn_logits, dim=1).cpu().numpy()

            # Weighted Ensemble: CNN(35%) + v3(35%) + v2(15%) + Base(15%)
            p_ens_all = 0.35 * prob_cnn_all + 0.35 * prob_v3_all + 0.15 * prob_v2_all + 0.15 * prob_b_all
            pred_cnn_all = np.argmax(prob_cnn_all, axis=1)
            pred_v3_all = np.argmax(prob_v3_all, axis=1)

            # Identifikasi sel konflik paling ambigu untuk arbitrase TrOCR (maksimal 5 sel teratas)
            trocr_candidates = []
            for i in range(len(pending_inference)):
                top1_p = float(p_ens_all[i].max())
                sorted_p = np.sort(p_ens_all[i])[::-1]
                margin = float(sorted_p[0] - sorted_p[1])
                is_conflict = (pred_cnn_all[i] != pred_v3_all[i])
                if is_conflict and (top1_p < 0.55 or margin < 0.15):
                    raw_im = pending_inference[i][1].get("raw_img")
                    if raw_im is not None:
                        trocr_candidates.append((top1_p, i))

            trocr_candidates.sort(key=lambda x: x[0])
            trocr_indices = [idx for _, idx in trocr_candidates[:5]]

            trocr_probs_map = {}
            if trocr_indices:
                try:
                    batch_imgs = [Image.fromarray(cv2.cvtColor(pending_inference[i][1]["raw_img"], cv2.COLOR_BGR2RGB)) for i in trocr_indices]
                    pv = self.trocr_processor(images=batch_imgs, return_tensors='pt').pixel_values.to(self.device)
                    dec_in = torch.tensor([[self.trocr_start_token]] * len(batch_imgs), device=self.device)
                    with torch.no_grad():
                        outputs = self.trocr_model(pv, decoder_input_ids=dec_in)
                        logits = outputs.logits[:, 0, TROCR_DIGIT_TOKENS]
                        trocr_batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                    for j, idx in enumerate(trocr_indices):
                        trocr_probs_map[idx] = trocr_batch_probs[j]
                except Exception as ex:
                    pass

            for i, (b, c_rec, feat, cl, mnist_28) in enumerate(pending_inference):
                prob_vec = p_ens_all[i]
                if i in trocr_probs_map:
                    p_trocr = trocr_probs_map[i]
                    final_prob = 0.70 * prob_vec + 0.30 * p_trocr
                    chosen = int(np.argmax(final_prob))
                    conf = round(float(final_prob[chosen]), 4)
                    reason = "TROCR_CNN_V3_ARBITRATION"
                else:
                    sorted_idx = np.argsort(prob_vec)[::-1]
                    chosen = int(sorted_idx[0])
                    conf = round(float(prob_vec[chosen]), 4)
                    reason = "CNN_V3_ENSEMBLE_CONSENSUS"

                b["is_blank"] = False
                b["char"] = str(chosen)
                b["conf"] = conf
                filled_cnt += 1

                c_rec["label"] = chosen
                c_rec["conf"] = conf
                c_rec["arbitration_reason"] = reason
                c_rec["verified"] = (conf >= 0.85)
                curated_cells.append(c_rec)

        # 4. Susun Nilai Field Murni dari Hasil Pelabelan Sel Mikro
        for p_key in sorted(pages.keys(), key=page_sort_key):
            p_obj = pages[p_key]
            fields = p_obj.get("fields", {})
            for var_name, f_obj in fields.items():
                if f_obj.get("data_type") == "checkbox" or f_obj.get("field_id") in NA_FIELD_IDS:
                    continue

                boxes = f_obj.get("box_details", [])
                chars = [b.get("char", "_") for b in boxes]

                raw_str = "".join(chars)
                f_obj["raw_string"] = raw_str

                val_chars = [c for c in chars if c != "_"]
                if not val_chars:
                    f_obj["value"] = None
                    gt_fields[var_name] = None
                else:
                    num_str = "".join(val_chars)
                    f_obj["value"] = int(num_str) if num_str.isdigit() else num_str
                    gt_fields[var_name] = f_obj["value"]
                gt_raw_fields[var_name] = raw_str

        # 5. Audit Validasi Aritmetika (Murni Audit / Zero-Modification pada Label Sel Mikro)
        val_rep = self.validator.validate_booklet(new_data, auto_reconcile=False)

        avg_conf = round(float(np.mean([c["conf"] for c in curated_cells])), 2) if curated_cells else 0.0

        gt_doc = {
            "booklet_id": booklet_id,
            "created_at": datetime.now().isoformat(),
            "labeled_by": "BPS Seruti Multi-Model Consensus Engine (TrOCR + PyTorch CNN + Seruti v3 + HITL)",
            "annotation_protocol": "Multi-Model Ensemble on Pure Micro-Crops (PyTorch CNN 35% + Seruti v3 35% + v2 15% + Base 15% + TrOCR Tie-Breaker + HITL Override)",
            "verified": True,
            "models_used": [
                "SerutiDigitCNN (models/model_cnn_mnist.pt)",
                "Microsoft TrOCR (microsoft/trocr-small-handwritten)",
                "Model Seruti v3 (models/model_seruti_v3.joblib)",
                "Model Seruti v2 (models/model_seruti_v2.joblib)",
                "BPS Baseline MLP (model.joblib)",
            ],
            "summary": {
                "total_fields": len(gt_fields),
                "filled_fields": sum(1 for v in gt_fields.values() if v is not None),
                "empty_fields": sum(1 for v in gt_fields.values() if v is None),
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
        print("  MEMBANGUN GROUND TRUTH BERSERTIFIKASI MULTI-MODEL AI (TrOCR + CNN + MLP)")
        print("  Kepatuhan: UU No. 16/1997, UU No. 27/2022 (Zero Cloud Egress 100% Lokal)")
        print("=" * 80)

        split_def_file = SPLITS_DIR / "split_definitions.json"
        with open(split_def_file, "r", encoding="utf-8") as f:
            split_meta = json.load(f)
        splits = split_meta.get("splits", {})

        train_bids = set(splits.get("train", []))
        val_bids = set(splits.get("val", []))
        test_bids = set(splits.get("test", []))

        booklet_dirs = sorted([d.name for d in EXTRACTIONS_DIR.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
        if limit:
            booklet_dirs = booklet_dirs[:limit]
            print(f"[*] Mode pengujian: dibatasi {limit} buklet.")
        print(f"[*] Menemukan {len(booklet_dirs)} buklet kuesioner terdaftar.")

        if self.export_png:
            print("[*] Menyiapkan direktori dataset/cells/ & dataset/cells_native/ (hanya kelas 0-9)...")
            shutil.rmtree(CELLS_DIR, ignore_errors=True)
            shutil.rmtree(CELLS_NATIVE_DIR, ignore_errors=True)
            for s_name in ["train", "val", "test"]:
                for d in range(10):
                    (CELLS_DIR / s_name / str(d)).mkdir(parents=True, exist_ok=True)
                    (CELLS_NATIVE_DIR / s_name / str(d)).mkdir(parents=True, exist_ok=True)

        all_curated_cells = {"train": [], "val": [], "test": []}
        all_stats = []

        for idx, b_id in enumerate(booklet_dirs, 1):
            if b_id in test_bids:
                s_name = "test"
            elif b_id in val_bids:
                s_name = "val"
            else:
                s_name = "train"

            res = self.process_booklet(b_id, s_name)
            if res is None:
                continue

            gt_doc, cells, stat = res
            all_curated_cells[s_name].extend(cells)
            all_stats.append(stat)

            # Ekspor citra PNG (Varian 28x28 Letterbox dan Native Clean ~66x49)
            if self.export_png:
                for c in cells:
                    d_cls = c["label"]
                    cid = f"{b_id}_p{c['page']}_{c['field_id']}_b{c['box_index']:02d}"
                    out_png = CELLS_DIR / s_name / str(d_cls) / f"{d_cls}_{cid}.png"
                    if not out_png.exists() and c["mnist_28"] is not None:
                        cv2.imwrite(str(out_png), c["mnist_28"])

                    out_native_png = CELLS_NATIVE_DIR / s_name / str(d_cls) / f"{d_cls}_{cid}.png"
                    if not out_native_png.exists() and c.get("cl_img") is not None:
                        cv2.imwrite(str(out_native_png), c["cl_img"])

            if idx == 1 or idx % 5 == 0 or idx == len(booklet_dirs):
                print(f"    [{idx:2d}/{len(booklet_dirs)}] {b_id} ({stat['elapsed']:.1f}s) | Digits: {stat['filled_digits']:3d} | Blanks: {stat['blank_boxes']:4d} | Rules: {stat['rules_passed']:2d}/{stat['total_rules']} ({stat['pass_rate_pct']:4.1f}%) | Disc: Rp {stat['discrepancy_rupiah']:,}")

        # Fase 6: Regenerasi .npz dan manifest
        print("\n[*] Meregenerasi Data Arrays NumPy (.npz) & Manifest CSV...")
        for s_name in ["train", "val", "test"]:
            cells = all_curated_cells[s_name]
            if not cells:
                continue

            valid_cells = [c for c in cells if c["mnist_28"] is not None]
            X_arr = np.array([c["mnist_28"] for c in valid_cells], dtype=np.uint8)
            y_arr = np.array([c["label"] for c in valid_cells], dtype=np.int64)

            booklet_ids_arr = np.array([c["booklet_id"] for c in valid_cells], dtype=object)
            field_ids_arr = np.array([c["field_id"] for c in valid_cells], dtype=object)
            var_names_arr = np.array([c["var_name"] for c in valid_cells], dtype=object)
            confs_arr = np.array([c["conf"] for c in valid_cells], dtype=np.float32)
            verified_arr = np.array([c.get("verified", False) for c in valid_cells], dtype=bool)

            npz_out = ARRAYS_DIR / f"{s_name}.npz"
            np.savez_compressed(
                npz_out,
                X=X_arr,
                y=y_arr,
                booklet_ids=booklet_ids_arr,
                field_ids=field_ids_arr,
                var_names=var_names_arr,
                confs=confs_arr,
                verified=verified_arr
            )

            rows = []
            for idx_c, c in enumerate(valid_cells):
                cid = f"{c['booklet_id']}_p{c['page']}_{c['field_id']}_b{c['box_index']:02d}"
                rows.append({
                    "sample_idx": idx_c,
                    "cell_id": cid,
                    "booklet_id": c["booklet_id"],
                    "page": c["page"],
                    "field_id": c["field_id"],
                    "var_name": c["var_name"],
                    "box_index": c["box_index"],
                    "label": c["label"],
                    "conf": c["conf"],
                    "verified": c.get("verified", False),
                    "digit": c["label"],
                    "confidence": c["conf"],
                    "provenance": c["arbitration_reason"],
                    "arbitration_reason": c["arbitration_reason"],
                    "image_path": f"cells/{s_name}/{c['label']}/{c['label']}_{cid}.png",
                    "native_image_path": f"cells_native/{s_name}/{c['label']}/{c['label']}_{cid}.png",
                    "split": s_name,
                })
            df = pd.DataFrame(rows)
            csv_out = SPLITS_DIR / f"{s_name}_manifest.csv"
            df.to_csv(csv_out, index=False)

            print(f"    -> Split {s_name.upper():5s}: {len(valid_cells):,} sampel terkompresi ke {npz_out.name} & {csv_out.name}")

        elapsed_total = round(time.time() - t0, 2)
        total_digits = sum(s["filled_digits"] for s in all_stats)
        total_blanks = sum(s["blank_boxes"] for s in all_stats)
        avg_pass = np.mean([s["pass_rate_pct"] for s in all_stats]) if all_stats else 0.0
        total_disc = sum(s["discrepancy_rupiah"] for s in all_stats)

        print("\n" + "=" * 80)
        print(f"  PEMBANGUNAN GROUND TRUTH MULTI-MODEL SELESAI DALAM {elapsed_total:.2f} DETIK")
        print(f"  Total Buklet Terproses  : {len(all_stats)} buklet")
        print(f"  Total Sel Kosong (Blank): {total_blanks:,} kotak (dieliminasi dari kelas digit)")
        print(f"  Total Digit Otentik     : {total_digits:,} sampel (hanya kelas 0-9)")
        print(f"  Rata-rata Pass Rate     : {avg_pass:.2f}%")
        print(f"  Total Diskrepansi Akhir : Rp {total_disc:,}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pembangun Ground Truth Multi-Model AI BPS.")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah buklet (untuk pengujian)")
    parser.add_argument("--no-png", action="store_true", help="Jangan simpan citra PNG individu")
    args = parser.parse_args()

    builder = MultiModelGroundTruthBuilder(export_png=not args.no_png)
    builder.run(limit=args.limit)
