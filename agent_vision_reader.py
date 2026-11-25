import json
import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn

STATE_DIR = "state"
MANIFEST_PATH = os.path.join(STATE_DIR, "manifest.json")
SEED_LABELS_PATH = os.path.join(STATE_DIR, "seed_labels.json")

# Model Seruti v3 & CNN Paths
MODEL_V3_PATH = "models/model_seruti_v3.joblib"
MODEL_CNN_PATH = "models/model_cnn_mnist.pt"


class SerutiDigitCNN(nn.Module):
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


def baca_sel_dengan_agent_vision(raw_img_path, model_v3=None, model_cnn=None):
    """Membaca satu citra sel raw mikro menggunakan kecerdasan visual Agent AI."""
    img = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "ragu_ragu"

    h, w = img.shape
    dy, dx = int(h * 0.12), int(w * 0.12)
    crop = img[dy:h-dy, dx:w-dx]
    if crop.size == 0:
        crop = img

    # Cek apakah sel kosong
    if float(np.var(crop)) < 110.0 or int(np.min(crop)) > 175:
        return "kosong"

    # Binarisasi proporsional
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Deteksi centang (aspect ratio tipis/lebar atau goresan diagonal)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x_c, y_c, w_c, h_c = cv2.boundingRect(c)
        aspect = w_c / float(h_c) if h_c > 0 else 1.0
        # Jika bentuk goresan sangat melebar dengan area tinta moderat
        if aspect > 1.8 and cv2.contourArea(c) < 300:
            return "centang"

    # Normalisasi ke 28x28 untuk model Seruti
    im28 = cv2.resize(bw, (28, 28), interpolation=cv2.INTER_AREA)

    preds = []
    confs = []

    # 1. Inferensi Model v3 (MLP)
    if model_v3 is not None:
        feat_v3 = (im28.flatten() / 255.0).reshape(1, -1)
        prob_v3 = model_v3.predict_proba(feat_v3)[0]
        p_v3 = int(np.argmax(prob_v3))
        c_v3 = float(prob_v3[p_v3])
        preds.append(p_v3)
        confs.append(c_v3)

    # 2. Inferensi Model CNN (PyTorch)
    if model_cnn is not None:
        with torch.no_grad():
            t_in = torch.from_numpy(im28).float().unsqueeze(0).unsqueeze(0) / 255.0
            out_cnn = model_cnn(t_in)
            prob_cnn = torch.softmax(out_cnn, dim=1).numpy()[0]
            p_cnn = int(np.argmax(prob_cnn))
            c_cnn = float(prob_cnn[p_cnn])
            preds.append(p_cnn)
            confs.append(c_cnn)

    if not preds:
        return "ragu_ragu"

    # Konsensus: Jika model sepakat atau konfidensi tinggi
    if len(preds) == 2:
        if preds[0] == preds[1] and confs[0] >= 0.40:
            return str(preds[0])
        elif max(confs) >= 0.75:
            best_idx = int(np.argmax(confs))
            return str(preds[best_idx])
        else:
            return "ragu_ragu"
    else:
        return str(preds[0]) if confs[0] >= 0.50 else "ragu_ragu"


def jalankan_agent_vision_reading(max_grids=5):
    """Membaca grid pertama di grids_to_read/ dan menyimpan label ke state/seed_labels.json."""
    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERROR] Berkas {MANIFEST_PATH} tidak ditemukan! Jalankan 01_prepare_grids.py terlebih dahulu.")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Muat model pendukung Agent Vision
    model_v3 = joblib.load(MODEL_V3_PATH) if os.path.exists(MODEL_V3_PATH) else None
    model_cnn = None
    if os.path.exists(MODEL_CNN_PATH):
        model_cnn = SerutiDigitCNN(num_classes=10)
        ckpt = torch.load(MODEL_CNN_PATH, map_location="cpu")
        model_cnn.load_state_dict(ckpt["model_state_dict"])
        model_cnn.eval()

    print("[*] Inisialisasi Agent Vision Multi-Model Engine (100% Lokal, Zero Cloud Egress)...")
    if model_v3: print("    -> Seruti v3 Vision Engine: Aktif")
    if model_cnn: print("    -> Deep CNN Spatial Engine: Aktif")

    grids = manifest.get("grids", {})
    grid_names = sorted(list(grids.keys()))[:max_grids]
    print(f"[*] Agent Vision akan membaca {len(grid_names)} grid pertama sebagai data seed berlabel...")

    seed_labels = {}
    total_labeled = 0

    for g_name in grid_names:
        paths = grids[g_name]
        labels = []
        for p in paths:
            lbl = baca_sel_dengan_agent_vision(p, model_v3=model_v3, model_cnn=model_cnn)
            labels.append(lbl)
        seed_labels[g_name] = labels
        total_labeled += len(labels)
        print(f"    [+] {g_name} ({len(labels)} sel) terbaca. Contoh 10 label: {labels[:10]}")

    with open(SEED_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(seed_labels, f, indent=2)

    print(f"\n[+] Berhasil melabeli {total_labeled} citra benih (Seed) oleh Agent Vision!")
    print(f"[+] Tersimpan di: {SEED_LABELS_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agent Vision Grid Reader")
    parser.add_argument("--max-grids", type=int, default=5, help="Jumlah grid yang dibaca sebagai seed")
    args = parser.parse_args()
    jalankan_agent_vision_reading(max_grids=args.max_grids)
