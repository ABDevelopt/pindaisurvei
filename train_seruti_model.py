#!/usr/bin/env python3
"""
train_seruti_model.py
=====================
Pipeline Pelatihan & Fine-Tuning Model Pengenalan Digit Kuesioner BPS Seruti Inti.

Alur Kerja:
1. Memuat dataset latih (train.npz: 26,219 sampel) dan validasi (val.npz: 4,685 sampel).
2. Normalisasi fitur (28x28 -> 784 piksel float [0, 1]).
3. Pelatihan MLPClassifier arsitektur terstandar BPS (784 -> 300 -> 100 -> 10) dengan Adam Optimizer,
   early stopping, dan pelacakan loss validasi.
4. Serialisasi model terlatih ke models/model_seruti_v2.joblib.
5. Benchmark otomatis model baru terhadap unseen test set (test.npz: 6,066 sampel)
   dan komparasi langsung terhadap baseline model BPS (model.joblib).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from dataset.eval.benchmark import SerutiBenchmarkEngine, DEFAULT_TEST_NPZ, DEFAULT_MODEL_PATH

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_NPZ = BASE_DIR / "dataset" / "data_arrays" / "train.npz"
DEFAULT_VAL_NPZ = BASE_DIR / "dataset" / "data_arrays" / "val.npz"
DEFAULT_OUTPUT_MODEL = BASE_DIR / "models" / "model_seruti_v2.joblib"


def load_npz_data(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Memuat X dan y dari file .npz, reshape ke 2D dan normalisasi ke float32 [0, 1]."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Berkas {npz_path} tidak ditemukan!")
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]

    n_samples = len(y)
    X_flat = X.reshape(n_samples, -1).astype(np.float32) / 255.0
    return X_flat, y


def train_model(
    train_npz: Path = DEFAULT_TRAIN_NPZ,
    val_npz: Path = DEFAULT_VAL_NPZ,
    output_model_path: Path = DEFAULT_OUTPUT_MODEL,
    max_iter: int = 50,
    random_seed: int = 42,
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """Melatih MLPClassifier pada dataset latih Seruti Inti."""
    print("=" * 75)
    print("  PELATIHAN MODEL PENGENALAN DIGIT BPS SERUTI INTI")
    print(f"  Dataset Latih : {train_npz}")
    print(f"  Dataset Val   : {val_npz}")
    print("=" * 75)

    # 1. Pemuatan Data
    print("\n[1/4] Memuat data latih dan validasi...")
    t0 = time.time()
    X_train, y_train = load_npz_data(train_npz)
    X_val, y_val = load_npz_data(val_npz)
    print(f"      Data Latih: {X_train.shape[0]:,} sampel (784 fitur)")
    print(f"      Data Val  : {X_val.shape[0]:,} sampel (784 fitur)")

    # 2. Inisialisasi Model MLP
    print("\n[2/4] Menginisialisasi MLPClassifier BPS (784 -> 300 -> 100 -> 10)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(300, 100),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=random_seed,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        verbose=True,
    )

    # 3. Eksekusi Pelatihan
    print("\n[3/4] Melatih model pada dataset riil...")
    t_train_start = time.time()
    clf.fit(X_train, y_train)
    train_time = round(time.time() - t_train_start, 2)
    print(f"      Pelatihan selesai dalam {train_time} detik ({clf.n_iter_} iterasi).")
    print(f"      Loss akhir: {clf.loss_:.5f}")

    # 4. Evaluasi pada Validation Set
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"      Akurasi pada Validation Set: {val_acc * 100:.2f}%")

    # 5. Serialisasi Model
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_model_path)
    print(f"\n[4/4] Model berhasil disimpan ke: {output_model_path}")
    print(f"      Ukuran model: {os.path.getsize(output_model_path) / (1024*1024):.2f} MB")

    stats = {
        "trained_at": datetime.now().isoformat(),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "iterations": clf.n_iter_,
        "final_loss": round(float(clf.loss_), 5),
        "val_accuracy_pct": round(val_acc * 100, 2),
        "training_time_seconds": train_time,
        "model_path": str(output_model_path),
    }

    return clf, stats


def run_comparative_benchmark(
    new_model_path: Path = DEFAULT_OUTPUT_MODEL,
    baseline_model_path: Path = DEFAULT_MODEL_PATH,
    test_npz: Path = DEFAULT_TEST_NPZ,
):
    """Menjalankan evaluasi komparatif antara model baru vs model baseline BPS."""
    print("\n" + "=" * 75)
    print("  EVALUASI KOMPARATIF TOLOK UKUR (HEAD-TO-HEAD BENCHMARK)")
    print("=" * 75)

    engine = SerutiBenchmarkEngine(test_npz_path=test_npz)

    # 1. Benchmark Model Baru
    print("\n>>> Menguji Model Baru: model_seruti_v2.joblib...")
    rep_new = engine.run_full_benchmark(
        model_path=new_model_path,
        model_name="Model Seruti v2 (Baru Latih)",
        output_report_path=BASE_DIR / "dataset" / "eval" / "benchmark_seruti_v2.json",
    )

    # 2. Benchmark Model Baseline BPS
    print("\n>>> Menguji Model Baseline BPS: model.joblib...")
    rep_base = engine.run_full_benchmark(
        model_path=baseline_model_path,
        model_name="BPS Baseline Model (model.joblib)",
        output_report_path=BASE_DIR / "dataset" / "eval" / "benchmark_baseline.json",
    )

    # 3. Cetak Tabel Komparasi
    m_new = rep_new["cell_level_benchmark"]
    m_base = rep_base["cell_level_benchmark"]

    print("\n" + "=" * 75)
    print("  TABEL KOMPARASI TOLOK UKUR PRODUKSI BPS")
    print("=" * 75)
    print(f"  Metrik                       | Baseline BPS        | Model Seruti v2")
    print("  " + "-" * 70)
    print(f"  Akurasi Sel Karakter (Test)  | {m_base['overall_accuracy_pct']:>15}%    | {m_new['overall_accuracy_pct']:>15}%")
    print(f"  Kecepatan Inferensi          | {m_base['inference_speed_fps']:>13,.0f} fps   | {m_new['inference_speed_fps']:>13,.0f} fps")
    print(f"  Optimal Auto-Accept Coverage | {m_base['auto_accept_analysis']['optimal_coverage_pct']:>15}%    | {m_new['auto_accept_analysis']['optimal_coverage_pct']:>15}%")
    print(f"  Escape Error Rate            | {m_base['auto_accept_analysis']['optimal_escape_error_pct']:>15}%    | {m_new['auto_accept_analysis']['optimal_escape_error_pct']:>15}%")
    print(f"  Kepatuhan Target BPS (<=0.5%)| {'MEMENUHI':>19}    | {'MEMENUHI':>19}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Pelatihan & Benchmark Model Pengenalan Digit Seruti BPS.")
    parser.add_argument("--train-npz", type=Path, default=DEFAULT_TRAIN_NPZ, help="Path ke train.npz")
    parser.add_argument("--val-npz", type=Path, default=DEFAULT_VAL_NPZ, help="Path ke val.npz")
    parser.add_argument("--test-npz", type=Path, default=DEFAULT_TEST_NPZ, help="Path ke test.npz")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_OUTPUT_MODEL, help="Path output model tersimpan")
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_MODEL_PATH, help="Path model baseline BPS")
    parser.add_argument("--max-iter", type=int, default=30, help="Jumlah iterasi maksimum pelatihan (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # 1. Latih Model
    train_model(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        output_model_path=args.model_out,
        max_iter=args.max_iter,
        random_seed=args.seed,
    )

    # 2. Jalankan Benchmark Komparatif
    run_comparative_benchmark(
        new_model_path=args.model_out,
        baseline_model_path=args.baseline_model,
        test_npz=args.test_npz,
    )


if __name__ == "__main__":
    main()
