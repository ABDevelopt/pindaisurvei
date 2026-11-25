#!/usr/bin/env python3
"""
benchmark_all_ai_gt.py
======================
Menjalankan tolok ukur komparasi 3 model terhadap Ground Truth Otentik Bersertifikasi AI.
"""

from pathlib import Path
from dataset.eval.benchmark import SerutiBenchmarkEngine

BASE_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 80)
    print("  EVALUASI TOLOK UKUR PADA GROUND TRUTH BERSERTIFIKASI AI (6.221 SAMPEL)")
    print("=" * 80)

    engine = SerutiBenchmarkEngine()

    print("\n[1/3] Mengevaluasi BPS Baseline Model (model.joblib)...")
    r_base = engine.run_full_benchmark(
        model_path=Path(r"d:\01_Kedinasan_BPS\Pengenalan Kuesioner (OCR)\model.joblib"),
        model_name="BPS Baseline MLP (model.joblib)",
        output_report_path=BASE_DIR / "dataset" / "eval" / "benchmark_baseline_ai_gt.json"
    )

    print("\n[2/3] Mengevaluasi Model Seruti v2 (Cold Start)...")
    r_v2 = engine.run_full_benchmark(
        model_path=BASE_DIR / "models" / "model_seruti_v2.joblib",
        model_name="Model Seruti v2 (Cold Start)",
        output_report_path=BASE_DIR / "dataset" / "eval" / "benchmark_v2_ai_gt.json"
    )

    print("\n[3/3] Mengevaluasi Model Seruti v3 (MNIST Foundation + Real Pen)...")
    r_v3 = engine.run_full_benchmark(
        model_path=BASE_DIR / "models" / "model_seruti_v3.joblib",
        model_name="Model Seruti v3 (MNIST Foundation + Real Pen)",
        output_report_path=BASE_DIR / "dataset" / "eval" / "benchmark_v3_ai_gt.json"
    )

    b_acc = r_base["cell_level_benchmark"]["overall_accuracy_pct"]
    v2_acc = r_v2["cell_level_benchmark"]["overall_accuracy_pct"]
    v3_acc = r_v3["cell_level_benchmark"]["overall_accuracy_pct"]

    print("\n" + "=" * 80)
    print("  HASIL AKHIR TOLOK UKUR AKURASI OBJEKTIF (GROUND TRUTH OTENTIK AI)")
    print("=" * 80)
    print(f"  1. BPS Baseline Model (model.joblib) : {b_acc:.2f}%  (Akurasi Riil Tanpa Pseudo-Bias!)")
    print(f"  2. Model Seruti v2 (Cold Start)      : {v2_acc:.2f}%")
    print(f"  3. Model Seruti v3 (MNIST + Real Pen): {v3_acc:.2f}%")
    print("=" * 80)

    print("\nPerforma Per Digit pada Ground Truth Otentik AI:")
    print("Digit | Baseline | Seruti v2 | Seruti v3 | Selisih v3 vs Base")
    print("------+----------+-----------+-----------+--------------------")
    for d in range(10):
        acc_b = r_base["cell_level_benchmark"]["class_metrics"][str(d)]["accuracy"] * 100
        acc_2 = r_v2["cell_level_benchmark"]["class_metrics"][str(d)]["accuracy"] * 100
        acc_3 = r_v3["cell_level_benchmark"]["class_metrics"][str(d)]["accuracy"] * 100
        diff = acc_3 - acc_b
        diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
        print(f"  {d}   |  {acc_b:6.2f}% |   {acc_2:6.2f}% |   {acc_3:6.2f}% |       {diff_str}")
    print("=" * 80)

if __name__ == "__main__":
    main()
