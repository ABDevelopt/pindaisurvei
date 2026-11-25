#!/usr/bin/env python3
"""
build_seruti_dataset.py
=======================
CLI Runner untuk Membangun Dataset Latih dan Evaluasi Ground Truth
Kuesioner BPS Susenas Seruti Inti (VSERUTI26.INTI).

Pemakaian:
  python build_seruti_dataset.py
  python build_seruti_dataset.py --no-png
  python build_seruti_dataset.py --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
from pathlib import Path
from seruti_pipeline.dataset_builder import SerutiDatasetBuilder, DEFAULT_EXTRACTIONS_DIR, DEFAULT_CROPS_DIR, DEFAULT_DATASET_DIR

def main():
    parser = argparse.ArgumentParser(description="Pembangun Dataset Latih dan Evaluasi Ground Truth Seruti Inti BPS.")
    parser.add_argument("--extractions-dir", type=Path, default=DEFAULT_EXTRACTIONS_DIR, help="Direktori hasil ekstraksi buklet")
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR, help="Direktori crops hierarkis buklet")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Direktori target output dataset")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Rasio data latih (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Rasio data validasi (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Rasio data uji benchmark (default: 0.15)")
    parser.add_argument("--no-png", action="store_true", help="Nonaktifkan ekspor citra PNG individu (hanya simpan .npz)")
    parser.add_argument("--no-clean-box-border", action="store_false", dest="clean_box_border", default=True, help="Nonaktifkan pembersihan garis kotak fisik comb-box")
    parser.add_argument("--no-raw-margin", action="store_false", dest="raw_margin", default=True, help="Jangan gunakan margin asli crop (gunakan clean_file 28x28 lama)")
    parser.add_argument("--target-npz-dim", type=int, default=26, help="Target dimensi proporsional letterbox untuk array .npz (default: 26)")
    parser.add_argument("--no-clean-cells", action="store_false", dest="clean_existing_cells", default=True, help="Jangan bersihkan folder cells/ lama sebelum ekspor")
    parser.add_argument("--seed", type=int, default=42, help="Seed deterministik (default: 42)")

    args = parser.parse_args()

    builder = SerutiDatasetBuilder(
        extractions_dir=args.extractions_dir,
        crops_dir=args.crops_dir,
        dataset_dir=args.dataset_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        export_png_cells=not args.no_png,
        clean_box_border=args.clean_box_border,
        raw_margin=args.raw_margin,
        target_npz_dim=args.target_npz_dim,
        clean_existing_cells=args.clean_existing_cells,
        random_seed=args.seed
    )

    builder.process_and_export_dataset()

if __name__ == "__main__":
    main()
