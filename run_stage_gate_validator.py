#!/usr/bin/env python3
"""
run_stage_gate_validator.py
===========================
CLI Runner untuk Stage-Gate Quality Validator Seruti Scanner.

Memeriksa kelayakan setiap buklet melalui 4 Gerbang Kualitas:
- Gate 1: Preprocessing & Affine Alignment Invariant (< 5.0% aspect deviation)
- Gate 2: Zero-Loss Digit Count Invariant (N_crops == N_template == 2.073 persis 100%)
- Gate 3: Substantive Snapping Quality (Memisahkan kuesioner, triwulan, no halaman)
- Gate 4: Production Decision & Routing (PROCEED_TO_OCR, FLAG_FOR_HITL_REVIEW, REJECT_INCOMPLETE)
"""

import sys
import argparse
import time
from pathlib import Path

# Pastikan root workspace terdaftar di sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from seruti_pipeline.stage_gate_validator import StageGateValidator

def main():
    parser = argparse.ArgumentParser(description="Stage-Gate Quality Validator — Seruti Scanner")
    parser.add_argument("--booklet", type=str, default=None, help="ID buklet spesifik (misal booklet_01)")
    parser.add_argument("--all", action="store_true", help="Validasi seluruh 76 buklet")
    parser.add_argument("--limit", type=int, default=None, help="Batas jumlah buklet yang divalidasi")
    parser.add_argument("--deep-snapping", action="store_true", help="Hitung ulang snapping fisik secara mendalam per citra")
    args = parser.parse_args()

    validator = StageGateValidator(workspace_root=BASE_DIR)

    print("=" * 80)
    print("  SERUTI INTI 2026 — STAGE-GATE QUALITY VALIDATOR")
    print("  Prinsip Mutu : Zero-Loss Invariant & Independent Role Segregation")
    print("  Target Sel   : 2.065 Kuesioner + 1 Triwulan + 7 Nomor Halaman = 2.073 Total")
    print("=" * 80)

    if args.booklet and not args.all:
        print(f"\n[*] Memvalidasi buklet tunggal: {args.booklet} ...")
        res = validator.validate_booklet(args.booklet, check_snapping=args.deep_snapping)
        
        g1 = res["gates"]["gate_1_alignment"]["status"]
        g2 = res["gates"]["gate_2_digit_count"]["status"]
        g3 = res["gates"]["gate_3_snapping_quality"]["status"]
        d = res["decision"]

        print(f"\nHasil Evaluasi Stage-Gate [{res['booklet_id']}]:")
        print(f"  - Gate 1 (Affine Alignment)      : [{g1}] (Deviasi maks: {res['gates']['gate_1_alignment']['max_aspect_dev_pct']}%)")
        print(f"  - Gate 2 (Digit Count Invariant) : [{g2}] ({res['gates']['gate_2_digit_count']['total_crops_actual']}/{res['gates']['gate_2_digit_count']['total_crops_expected']} kotak, {res['gates']['gate_2_digit_count']['completion_pct']}%)")
        
        g2_b = res['gates']['gate_2_digit_count']['breakdown']
        print(f"      * Isian Kuesioner Responden  : {g2_b['questionnaire']['actual']}/{g2_b['questionnaire']['expected']}")
        print(f"      * Triwulan Administratif     : {g2_b['triwulan']['actual']}/{g2_b['triwulan']['expected']}")
        print(f"      * Nomor Halaman Footer       : {g2_b['page_number']['actual']}/{g2_b['page_number']['expected']}")

        g3_m = res['gates']['gate_3_snapping_quality']['metrics_by_role']
        print(f"  - Gate 3 (Snapping Substantif)   : [{g3}] (Kuesioner: {g3_m['questionnaire_substantive']['pct']}%)")
        print(f"      * Kuesioner Snapped Presisi  : {g3_m['questionnaire_substantive']['snapped']}/{g3_m['questionnaire_substantive']['total']}")
        print(f"      * Triwulan Snapped           : {g3_m['triwulan_administrative']['snapped']}/{g3_m['triwulan_administrative']['total']}")
        print(f"      * Nomor Halaman Snapped      : {g3_m['page_number_footer']['snapped']}/{g3_m['page_number_footer']['total']}")

        print(f"\n>>> KEPUTUSAN STAGE-GATE (GATE 4)  : [{d}]")
        print(f"    Rute Operasional               : {res['gates']['gate_4_decision']['routing']}")
        if res["reasons"]:
            print(f"    Catatan / Alasan               :")
            for r in res["reasons"]:
                print(f"      - {r}")

    else:
        print("\n[*] Menjalankan audit Stage-Gate pada seluruh buklet...")
        summary = validator.validate_all_booklets(check_snapping_deep=args.deep_snapping, limit=args.limit)

        print("\n" + "=" * 80)
        print("  RINGKASAN AUDIT STAGE-GATE SELURUH BUKLET")
        print("=" * 80)
        print(f"  Total Buklet Diaudit : {summary['total_booklets']}")
        print(f"  Waktu Eksekusi       : {summary['elapsed_seconds']}s")
        print(f"  Distribusi Keputusan :")
        print(f"    [PROCEED_TO_OCR]       : {summary['decisions'].get('PROCEED_TO_OCR', 0)} buklet ({summary['pass_rate_pct']}%)")
        print(f"    [FLAG_FOR_HITL_REVIEW] : {summary['decisions'].get('FLAG_FOR_HITL_REVIEW', 0)} buklet (Low-contrast scanner, affine fallback aman)")
        print(f"    [REJECT_INCOMPLETE]    : {summary['decisions'].get('REJECT_INCOMPLETE', 0)} buklet (Gagal Gate 1 atau Gate 2)")
        print("=" * 80)

        # Tampilkan rincian buklet yang perlu review jika ada
        flagged = [b for b in summary["booklets"] if b["decision"] != "PROCEED_TO_OCR"]
        if flagged:
            print(f"\nBuklet yang memerlukan perhatian khusus ({len(flagged)} buklet):")
            for b in flagged:
                print(f"  * {b['booklet_id']}: [{b['decision']}]")
                for r in b.get("reasons", []):
                    print(f"      - {r}")
        else:
            print("\nSemua buklet lolos 100% tanpa ada penolakan!")

        print(f"\nLaporan resmi tersimpan di: output/reports/stage_gate_audit_report.json")

if __name__ == "__main__":
    main()
