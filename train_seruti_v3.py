#!/usr/bin/env python3
"""
train_seruti_v3.py
==================
Pipeline Pelatihan Model Seruti v3:
1. Menggabungkan MNIST Foundation (60.000 sampel seimbang) + Dataset Riil Seruti (26.219 sampel).
2. Augmentasi Geometri & Pensil (Affine jitter, shear, stroke dilation/erosion).
3. Penyeimbangan Kelas (Class Balancing / Minority Oversampling) untuk digit 3, 4, 7, 9.
4. Pelatihan MLPClassifier (300, 100) hingga konvergen penuh.
5. Benchmark otomatis terhadap unseen test set (test.npz: 6.066 sampel) dan komparasi v1, v2, v3.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from dataset.eval.benchmark import SerutiBenchmarkEngine, DEFAULT_TEST_NPZ, DEFAULT_MODEL_PATH

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_NPZ = BASE_DIR / "dataset" / "data_arrays" / "train.npz"
DEFAULT_VAL_NPZ = BASE_DIR / "dataset" / "data_arrays" / "val.npz"
DEFAULT_MNIST_NPZ = Path(r"d:\01_Kedinasan_BPS\Pengenalan Kuesioner (OCR)\mnist.npz")
DEFAULT_OUTPUT_MODEL = BASE_DIR / "models" / "model_seruti_v3.joblib"


def augment_digit_image(im_28: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Melakukan augmentasi geometris affin dan ketebalan pensil pada citra 28x28."""
    # Jitter pergeseran dx, dy antara -2 s/d +2 piksel
    dx = rng.integers(-2, 3)
    dy = rng.integers(-2, 3)
    shear = rng.uniform(-0.15, 0.15)
    
    # Matriks Affine: [[1, shear, dx], [0, 1, dy]]
    M = np.float32([[1.0, shear, dx], [0.0, 1.0, dy]])
    aug = cv2.warpAffine(im_28, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Augmentasi ketebalan goresan pensil (dilasi / erosi)
    r = rng.random()
    if r < 0.35:
        aug = cv2.dilate(aug, np.ones((2, 2), np.uint8))
    elif r < 0.50:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        eroded = cv2.erode(aug, kernel)
        if eroded.sum() > 0.05 * aug.sum():
            aug = eroded

    return aug


def load_mnist_data(mnist_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Memuat data latih dan validasi MNIST (60.000 sampel)."""
    if not mnist_path.exists():
        raise FileNotFoundError(f"Berkas MNIST {mnist_path} tidak ditemukan!")
    
    m = np.load(mnist_path)
    X_tr = m["x_tr"].astype(np.float32)
    y_tr = m["y_tr"].astype(np.int64)
    X_va = m["x_va"].astype(np.float32)
    y_va = m["y_va"].astype(np.int64)
    
    X_mnist = np.vstack([X_tr, X_va])
    y_mnist = np.concatenate([y_tr, y_va])
    return X_mnist, y_mnist


def load_seruti_data(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Memuat data kuesioner Seruti (X uint8 28x28, y int)."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Berkas {npz_path} tidak ditemukan!")
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    return X, y


def build_augmented_training_data(
    mnist_path: Path = DEFAULT_MNIST_NPZ,
    seruti_train_path: Path = DEFAULT_TRAIN_NPZ,
    target_samples_per_class: int = 10000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Membangun dataset gabungan MNIST Foundation + Seruti Riil Ter-Augmentasi & Seimbang.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Pemuatan MNIST (60.000 sampel, ~6.000 per kelas)
    print("      -> Memuat MNIST Foundation...")
    X_mnist, y_mnist = load_mnist_data(mnist_path)
    print(f"         MNIST loaded: {len(y_mnist):,} sampel.")

    # 2. Pemuatan Seruti Riil (26.219 sampel)
    print("      -> Memuat Dataset Seruti Riil...")
    X_seruti, y_seruti = load_seruti_data(seruti_train_path)
    
    # Filter hanya label digit valid (0-9), abaikan crop rejected / bukan angka (-1)
    valid_mask = (y_seruti >= 0) & (y_seruti <= 9)
    if not np.all(valid_mask):
        num_rejected = int(np.sum(~valid_mask))
        print(f"         Mengabaikan {num_rejected:,} sel non-digit (bukan angka / bad crop) dari data latih.")
        X_seruti = X_seruti[valid_mask]
        y_seruti = y_seruti[valid_mask]
        
    print(f"         Seruti valid loaded: {len(y_seruti):,} sampel.")

    # Normalisasi Seruti ke float32 [0.0, 1.0]
    X_seruti_norm = (X_seruti.astype(np.float32) / 255.0).reshape(-1, 28, 28)

    combined_X = []
    combined_y = []

    # Tambahkan semua MNIST
    combined_X.append(X_mnist)
    combined_y.append(y_mnist)

    # Tambahkan Seruti asli
    combined_X.append(X_seruti_norm.reshape(-1, 784))
    combined_y.append(y_seruti)

    # 3. Penyeimbangan & Augmentasi Khusus Seruti
    # Khusus digit dengan sampel sedikit (kelas minoritas: 3, 4, 7, 9, dll.)
    # kita gandakan dan augmentasi agar bobot pensil lokal kuat dan seimbang
    print("      -> Melakukan augmentasi geometris & pensil untuk penyeimbangan kelas...")
    aug_X = []
    aug_y = []

    for digit in range(10):
        digit_indices = np.where(y_seruti == digit)[0]
        n_orig = len(digit_indices)
        
        # Hitung berapa banyak augmentasi yang dibutuhkan
        # Target: setiap digit memiliki setidaknya ~5.000 representasi pensil riil Seruti
        if n_orig < 5000:
            n_needed = 5000 - n_orig
            # Ambil acak dari sampel yang ada
            sample_choices = rng.choice(digit_indices, size=n_needed, replace=True)
            for idx in sample_choices:
                im = X_seruti_norm[idx]
                im_aug = augment_digit_image(im, rng)
                aug_X.append(im_aug.ravel())
                aug_y.append(digit)

    if aug_X:
        aug_X_arr = np.array(aug_X, dtype=np.float32)
        aug_y_arr = np.array(aug_y, dtype=np.int64)
        combined_X.append(aug_X_arr)
        combined_y.append(aug_y_arr)
        print(f"         Augmentasi tambahan dibuat: {len(aug_y_arr):,} sampel.")

    X_final = np.vstack(combined_X)
    y_final = np.concatenate(combined_y)

    # Acak urutan (shuffle)
    perm = rng.permutation(len(y_final))
    X_final = X_final[perm]
    y_final = y_final[perm]

    print(f"      -> Total Data Latih Gabungan v3: {len(y_final):,} sampel (784 fitur).")
    return X_final, y_final


def train_v3(
    mnist_path: Path = DEFAULT_MNIST_NPZ,
    train_npz: Path = DEFAULT_TRAIN_NPZ,
    val_npz: Path = DEFAULT_VAL_NPZ,
    output_model_path: Path = DEFAULT_OUTPUT_MODEL,
    max_iter: int = 60,
    random_seed: int = 42,
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """Melatih Model Seruti v3 dan menyimpan ke disk."""
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  PELATIHAN MODEL SERUTI v3 (MNIST FOUNDATION + BALANCED REAL PEN & AUGMENTATION)")
    print(f"  Target Output : {output_model_path}")
    print("=" * 80)

    # 1. Pemuatan & Pembangunan Data Latih
    print("\n[1/4] Membangun dataset gabungan seimbang...")
    t0 = time.time()
    X_train, y_train = build_augmented_training_data(
        mnist_path=mnist_path,
        seruti_train_path=train_npz,
        seed=random_seed,
    )
    
    # Data Val Seruti murni untuk early stopping & verifikasi
    X_val_raw, y_val = load_seruti_data(val_npz)
    X_val = (X_val_raw.astype(np.float32) / 255.0).reshape(-1, 784)
    print(f"      Data Validasi Seruti: {len(y_val):,} sampel (murni unseen).")

    # 2. Konfigurasi MLP Model
    print("\n[2/4] Menginisialisasi MLPClassifier Terstandar BPS (300, 100)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(300, 100),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=random_seed,
        early_stopping=True,
        n_iter_no_change=8,
        validation_fraction=0.1,
        verbose=True,
    )

    # 3. Eksekusi Pelatihan
    print("\n[3/4] Melatih Model v3...")
    t_train_start = time.time()
    clf.fit(X_train, y_train)
    train_duration = round(time.time() - t_train_start, 2)
    print(f"\n      Pelatihan tuntas dalam {train_duration} detik ({clf.n_iter_} iterasi).")
    print(f"      Loss akhir: {clf.loss_:.5f}")

    # 4. Evaluasi Validasi Seruti
    val_preds = clf.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_preds)) * 100.0
    print(f"\n[4/4] Akurasi pada Validation Set Seruti: {val_acc:.2f}%")

    # Simpan Model ke Disk
    joblib.dump(clf, output_model_path, compress=3)
    model_size_mb = round(output_model_path.stat().st_size / (1024 * 1024), 2)
    print(f"      Model berhasil disimpan di: {output_model_path} ({model_size_mb} MB)")

    meta = {
        "model_name": "Model Seruti v3 (MNIST Foundation + Balanced Real Pen)",
        "output_path": str(output_model_path),
        "total_train_samples": len(y_train),
        "train_duration_seconds": train_duration,
        "iterations": int(clf.n_iter_),
        "final_loss": float(clf.loss_),
        "val_accuracy_pct": val_acc,
        "model_size_mb": model_size_mb,
    }
    return clf, meta


def run_benchmark_comparison(model_v3_path: Path):
    """Menjalankan benchmark evaluasi standar BPS pada model v3 dan membandingkannya."""
    print("\n" + "=" * 80)
    print("  MENJALANKAN BENCHMARK EVALUASI STANDAR BPS PADA TEST SET (6.066 SAMPEL)")
    print("=" * 80)

    v3_json_path = BASE_DIR / "dataset" / "eval" / "benchmark_seruti_v3.json"
    engine = SerutiBenchmarkEngine(
        test_npz_path=DEFAULT_TEST_NPZ,
    )
    report_v3 = engine.run_full_benchmark(
        model_path=model_v3_path,
        model_name="Model Seruti v3 (MNIST Foundation + Balanced Real Pen)",
        output_report_path=v3_json_path,
    )
    print(f"\n[+] Laporan Benchmark v3 tersimpan di: {v3_json_path}")

    # Cetak ringkasan komparasi 3 model
    baseline_json_path = BASE_DIR / "dataset" / "eval" / "benchmark_baseline.json"
    v2_json_path = BASE_DIR / "dataset" / "eval" / "benchmark_seruti_v2.json"

    b_acc = "N/A"
    v2_acc = "N/A"
    if baseline_json_path.exists():
        with open(baseline_json_path, "r") as f:
            b_acc = f"{json.load(f)['cell_level_benchmark']['overall_accuracy_pct']:.2f}%"
    if v2_json_path.exists():
        with open(v2_json_path, "r") as f:
            v2_acc = f"{json.load(f)['cell_level_benchmark']['overall_accuracy_pct']:.2f}%"

    v3_acc = f"{report_v3['cell_level_benchmark']['overall_accuracy_pct']:.2f}%"
    v3_fps = f"{report_v3['cell_level_benchmark']['inference_speed_fps']:,.0f} fps"

    print("\n" + "-" * 80)
    print("  RINGKASAN TOLOK UKUR AKURASI (TEST SET UNSEEN: 6.066 SAMPEL)")
    print("-" * 80)
    print(f"  1. BPS Baseline Model (model.joblib) : {b_acc}  *(Pseudo-bias evaluasi mandiri)*")
    print(f"  2. Model Seruti v2 (Cold Start)      : {v2_acc}")
    print(f"  3. Model Seruti v3 (MNIST + Real Pen): {v3_acc}  (Speed: {v3_fps})")
    print("-" * 80)

    # Cetak breakdown per kelas v3
    print("\n  Performa Per Digit Model v3:")
    print("  Digit | Support | Akurasi  | Presisi  | Recall   | F1-Score")
    print("  ------+---------+----------+----------+----------+---------")
    cm = report_v3["cell_level_benchmark"]["class_metrics"]
    for d in range(10):
        m = cm[str(d)]
        print(f"    {d}   |  {m['support']:5d}  | {m['accuracy']*100:6.2f}%  | {m['precision']*100:6.2f}%  | {m['recall']*100:6.2f}%  | {m['f1_score']:6.4f}")
    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih & Benchmark Model Seruti v3")
    parser.add_argument("--max-iter", type=int, default=60, help="Maksimum iterasi MLP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-only", action="store_true", help="Hanya jalankan benchmark pada model yang sudah ada")
    args = parser.parse_args()

    if not args.eval_only or not DEFAULT_OUTPUT_MODEL.exists():
        clf, meta = train_v3(max_iter=args.max_iter, random_seed=args.seed)
    
    run_benchmark_comparison(DEFAULT_OUTPUT_MODEL)
