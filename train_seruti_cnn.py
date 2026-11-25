#!/usr/bin/env python3
"""
train_seruti_cnn.py
===================
Melatih Model CNN (Convolutional Neural Network) PyTorch untuk Pengenalan Digit Kuesioner BPS Seruti Inti.

Arsitektur:
- Conv2D(1 -> 32, kernel=3, padding=1) + BatchNorm + ReLU
- Conv2D(32 -> 64, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.2)
- Conv2D(64 -> 128, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.25)
- Flatten -> Linear(128*7*7 -> 128) + ReLU + Dropout(0.3) -> Linear(128 -> 10)

Kepatuhan:
- UU No. 16/1997 & UU No. 27/2022 (Zero Cloud Egress)
- Model dilatih dan disimpan 100% lokal di mesin pengguna.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
ARRAYS_DIR = BASE_DIR / "dataset" / "data_arrays"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODELS_DIR / "model_cnn_mnist.pt"


class SerutiDigitCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 2: 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            # Block 3: 14x14 -> 7x7
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
        out = self.classifier(feat)
        return out


def load_dataset(npz_path: Path) -> TensorDataset:
    d = np.load(npz_path)
    X = d["X"]  # (N, 28, 28) uint8 [0..255]
    y = d["y"]  # (N,) int
    # Normalisasi ke [0.0, 1.0] dan reshape ke (N, 1, 28, 28)
    X_tensor = torch.from_numpy(X).float().unsqueeze(1) / 255.0
    y_tensor = torch.from_numpy(y).long()
    return TensorDataset(X_tensor, y_tensor)


def main():
    print("=" * 80)
    print("  PELATIHAN DEEP CONVOLUTIONAL NEURAL NETWORK (CNN) BPS SERUTI")
    print("  Kepatuhan: UU No. 16/1997, UU No. 27/2022 (Zero Cloud Egress 100% Lokal)")
    print("=" * 80)

    train_npz = ARRAYS_DIR / "train.npz"
    val_npz = ARRAYS_DIR / "val.npz"
    test_npz = ARRAYS_DIR / "test.npz"

    if not train_npz.exists() or not test_npz.exists():
        print(f"[ERROR] Berkas data {train_npz} tidak ditemukan!")
        sys.exit(1)

    print("[*] Memuat dataset terpartisi...")
    train_ds = load_dataset(train_npz)
    val_ds = load_dataset(val_npz)
    test_ds = load_dataset(test_npz)

    print(f"    -> Train Set : {len(train_ds):,} sampel")
    print(f"    -> Val Set   : {len(val_ds):,} sampel")
    print(f"    -> Test Set  : {len(test_ds):,} sampel (Unseen)")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Perangkat komputasi: {device}")

    model = SerutiDigitCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    print("\n[*] Memulai pelatihan CNN...")
    num_epochs = 8
    best_val_acc = 0.0

    t_start = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            preds = out.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        scheduler.step()
        train_acc = (correct / total) * 100
        train_loss = total_loss / total

        # Evaluasi Validasi
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model(X_b)
                preds = out.argmax(dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

        val_acc = (val_correct / val_total) * 100
        print(f"    Epoch [{epoch:2d}/{num_epochs:2d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, MODEL_SAVE_PATH)

    elapsed = time.time() - t_start
    print(f"\n[+] Pelatihan selesai dalam {elapsed:.2f} detik.")
    print(f"[+] Model terbaik disimpan ke: {MODEL_SAVE_PATH} (Val Acc: {best_val_acc:.2f}%)")

    # Evaluasi Akhir pada Unseen Test Set
    print("\n" + "=" * 80)
    print("  EVALUASI AKHIR CNN PADA TEST SET INDEPENDEN (5.945 SAMPEL)")
    print("=" * 80)

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_correct = 0
    test_total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            preds = out.argmax(dim=1)
            test_correct += (preds == y_b).sum().item()
            test_total += len(y_b)

            for p, y_t in zip(preds.cpu().numpy(), y_b.cpu().numpy()):
                class_total[y_t] += 1
                if p == y_t:
                    class_correct[y_t] += 1

    overall_acc = (test_correct / test_total) * 100
    print(f"  Akurasi Keseluruhan CNN: {overall_acc:.2f}%\n")
    print("  Performa Per Digit pada Test Set:")
    print("  Digit | Support | Benar | Akurasi CNN")
    print("  ------+---------+-------+------------")
    for d in range(10):
        tot = class_total[d]
        cor = class_correct[d]
        acc_d = (cor / tot * 100) if tot > 0 else 0.0
        print(f"    {d}   |  {tot:6d} |{cor:6d} |  {acc_d:6.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
