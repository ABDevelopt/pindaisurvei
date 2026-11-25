"""
tests/test_dataset_and_benchmark.py
===================================
Unit & Integration Test Suite untuk Dataset Latih & Evaluasi Ground Truth
serta Mesin Tolok Ukur (Benchmark) Kuesioner BPS Seruti Inti.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATA_ARRAYS_DIR = DATASET_DIR / "data_arrays"
SPLITS_DIR = DATASET_DIR / "splits"
GT_DIR = DATASET_DIR / "ground_truth"
EVAL_DIR = DATASET_DIR / "eval"
MODELS_DIR = BASE_DIR / "models"


def test_dataset_directory_structure():
    """Memastikan seluruh direktori dataset terstruktur dengan benar."""
    assert DATASET_DIR.exists(), "Direktori dataset/ harus ada"
    assert DATA_ARRAYS_DIR.exists(), "Direktori data_arrays/ harus ada"
    assert SPLITS_DIR.exists(), "Direktori splits/ harus ada"
    assert GT_DIR.exists(), "Direktori ground_truth/ harus ada"
    assert EVAL_DIR.exists(), "Direktori eval/ harus ada"


def test_data_arrays_existence_and_shapes():
    """Memverifikasi berkas .npz, dimensi (N, 28, 28), dan tipe data uint8."""
    for split_name in ["train", "val", "test"]:
        npz_path = DATA_ARRAYS_DIR / f"{split_name}.npz"
        assert npz_path.exists(), f"Berkas {npz_path.name} harus ada"

        data = np.load(npz_path)
        assert "X" in data, f"Key 'X' harus ada di {split_name}.npz"
        assert "y" in data, f"Key 'y' harus ada di {split_name}.npz"
        assert "booklet_ids" in data, f"Key 'booklet_ids' harus ada di {split_name}.npz"
        assert "field_ids" in data, f"Key 'field_ids' harus ada di {split_name}.npz"
        assert "confs" in data, f"Key 'confs' harus ada di {split_name}.npz"
        assert "verified" in data, f"Key 'verified' harus ada di {split_name}.npz"

        X = data["X"]
        y = data["y"]
        assert len(X) == len(y), f"Jumlah X dan y harus identik pada {split_name}"
        assert X.ndim == 3, f"X harus berdimensi 3 (N, 28, 28), ditemukan {X.ndim}"
        assert X.shape[1:] == (28, 28), f"Resolusi citra harus (28, 28), ditemukan {X.shape[1:]}"
        assert X.dtype == np.uint8, f"Tipe data X harus uint8, ditemukan {X.dtype}"
        assert y.dtype == np.int64, f"Tipe data y harus int64, ditemukan {y.dtype}"

        # Periksa rentang kelas digit 0-9
        unique_labels = set(np.unique(y).tolist())
        assert unique_labels.issubset(set(range(10))), f"Label harus dalam rentang 0-9, ditemukan {unique_labels}"
        assert len(unique_labels) == 10, f"Seluruh 10 kelas (0-9) harus terwakili pada {split_name}"


def test_zero_data_leakage_between_splits():
    """Memastikan tidak ada tumpang tindih buklet antara train, val, dan test."""
    split_def_file = SPLITS_DIR / "split_definitions.json"
    assert split_def_file.exists(), "Berkas split_definitions.json harus ada"

    with open(split_def_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    splits = meta.get("splits", {})
    train_b = set(splits.get("train", []))
    val_b = set(splits.get("val", []))
    test_b = set(splits.get("test", []))

    assert len(train_b) > 0, "Train set tidak boleh kosong"
    assert len(val_b) > 0, "Val set tidak boleh kosong"
    assert len(test_b) > 0, "Test set tidak boleh kosong"

    # Verifikasi disjointness (Zero Data Leakage)
    assert train_b.isdisjoint(val_b), "Ada kebocoran data: Buklet Train dan Val beririsan!"
    assert train_b.isdisjoint(test_b), "Ada kebocoran data: Buklet Train dan Test beririsan!"
    assert val_b.isdisjoint(test_b), "Ada kebocoran data: Buklet Val dan Test beririsan!"

    total_unique = len(train_b) + len(val_b) + len(test_b)
    assert total_unique == 76, f"Total buklet harus 76, ditemukan {total_unique}"


def test_ground_truth_json_schema():
    """Memverifikasi skema Ground Truth JSON per buklet."""
    gt_files = list(GT_DIR.glob("*.json"))
    assert len(gt_files) == 76, f"Harus ada 76 berkas ground truth, ditemukan {len(gt_files)}"

    # Periksa sampel buklet
    sample_gt_path = GT_DIR / "booklet_01.json"
    assert sample_gt_path.exists(), "booklet_01.json harus ada di ground_truth/"

    with open(sample_gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    assert "booklet_id" in gt
    assert "summary" in gt
    assert "fields" in gt
    assert "raw_strings" in gt
    assert "cells_count" in gt

    s = gt["summary"]
    assert s["total_fields"] >= 300, "Total field kuesioner harus lengkap (>=300)"
    assert s["total_digits"] > 0, "Total digit terisi harus > 0"
    assert s["average_confidence"] > 90.0, "Rata-rata konfidensi harus tinggi"


def test_manifest_csv_files():
    """Memverifikasi berkas manifest CSV pada folder splits."""
    for sname in ["train", "val", "test"]:
        csv_file = SPLITS_DIR / f"{sname}_manifest.csv"
        assert csv_file.exists(), f"Berkas {csv_file.name} harus ada"
        with open(csv_file, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            assert "sample_idx" in header
            assert "booklet_id" in header
            assert "label" in header
            assert "conf" in header
            assert "image_path" in header


def test_benchmark_report_validity():
    """Memverifikasi laporan benchmark evaluasi model."""
    rep_path = EVAL_DIR / "benchmark_report.json"
    assert rep_path.exists(), "benchmark_report.json harus dihasilkan"

    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert "cell_level_benchmark" in rep
    assert "consistency_benchmark" in rep
    assert "bps_production_compliance" in rep

    cell_m = rep["cell_level_benchmark"]
    assert cell_m["total_test_samples"] == 6066, "Jumlah sampel test set harus 6,066"
    assert cell_m["overall_accuracy_pct"] >= 90.0, "Akurasi model baseline harus >= 90%"
    assert "class_metrics" in cell_m
    assert "confidence_sweep" in cell_m
    assert len(cell_m["class_metrics"]) == 10, "Harus ada 10 kelas digit"


def test_trained_model_v2_inference():
    """Memverifikasi bahwa model terlatih model_seruti_v2.joblib dapat dimuat dan memprediksi."""
    model_path = MODELS_DIR / "model_seruti_v2.joblib"
    assert model_path.exists(), "Berkas model_seruti_v2.joblib harus ada"

    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Model harus memiliki method predict"
    assert hasattr(model, "predict_proba"), "Model harus memiliki method predict_proba"

    # Test dummy inference dengan input 784 fitur
    dummy_input = np.zeros((2, 784), dtype=np.float32)
    preds = model.predict(dummy_input)
    probs = model.predict_proba(dummy_input)

    assert len(preds) == 2, "Harus menghasilkan 2 prediksi"
    assert probs.shape == (2, 10), "Probabilitas harus shape (2, 10)"
    assert set(preds).issubset(set(range(10))), "Hasil prediksi harus dalam kelas 0-9"
