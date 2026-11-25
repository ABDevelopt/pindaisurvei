#!/usr/bin/env python3
"""
run_seruti_validator.py
=======================
Auditor & Mesin Validasi Konsistensi Aritmetika Kuesioner BPS Susenas Seruti Inti (VSERUTI26.INTI).

Kepatuhan Mutlak Regulasi:
- UU No. 16 Tahun 1997 tentang Statistik
- UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP)
- Zero PII Cloud Egress: Output terminal dan chat hanya menampilkan statistik agregat
- Zero-Jitter Invariant: Seluruh kalkulasi finansial menggunakan 64-bit integer presisi 100%

Penggunaan CLI:
    python run_seruti_validator.py --booklet booklet_01
    python run_seruti_validator.py --all
    python run_seruti_validator.py --all --auto-reconcile
    python run_seruti_validator.py --input output/extractions/booklet_09/booklet_09_extracted.json --format both
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Tambahkan root workspace ke sys.path jika belum ada
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from seruti_pipeline.validator import (
    SerutiArithmeticValidator,
    ValidationReport,
    RuleValidationResult,
)


# =============================================================================
# FORMATTER TABULAR ASCII & CSV (ZERO PII COMPLIANT)
# =============================================================================

def format_rupiah(val: Optional[int]) -> str:
    """Format integer rupiah dengan pemisah ribuan, atau tanda hubung jika None."""
    if val is None:
        return "-"
    return f"{val:,}"


def format_tabular_ascii(report: ValidationReport) -> str:
    """
    Menghasilkan laporan tabel ASCII terformat rapi untuk audit kuesioner.
    Zero PII compliant: tidak ada nama, NIK, alamat responden yang ditampilkan.
    """
    meta = report.audit_metadata
    summ = report.summary
    booklet_id = meta.get("booklet_id", "unknown")
    mode_str = "AUTO-RECONCILE (Koreksi Otomatis)" if meta.get("execution_mode") == "auto_reconcile" else "AUDIT ONLY (Tanpa Mutasi Data)"
    ts = meta.get("timestamp", datetime.now().isoformat())

    lines = []
    lines.append("=" * 120)
    lines.append("  BADAN PUSAT STATISTIK - LAPORAN AUDIT KONSISTENSI ARITMETIKA & VALIDASI SEMANTIK")
    lines.append("  Instrumen: Kuesioner Susenas Seruti Inti (VSERUTI26.INTI)")
    lines.append(f"  Buklet: {booklet_id} | Mode: {mode_str} | Waktu Audit: {ts}")
    lines.append("  Kepatuhan: UU No. 16/1997 & UU PDP No. 27/2022 (Zero PII Cloud Egress & Zero-Jitter Invariant)")
    lines.append("=" * 120)
    lines.append(
        f"{'No':<4} {'Rule ID':<35} {'Hal':<4} {'Tipe':<14} {'Target Value':<14} {'Expected':<14} {'Diff (Rp)':<12} {'Status':<10} {'Rekomendasi / Catatan'}"
    )
    lines.append("-" * 120)

    for idx, v in enumerate(report.validations, start=1):
        r_id = v.rule_id
        pg = str(v.page_num)
        r_type = v.rule_type
        t_val = format_rupiah(v.target_value)
        e_val = format_rupiah(v.expected_value)

        diff_str = "-"
        if v.difference is not None:
            if v.difference > 0:
                diff_str = f"+{v.difference:,}"
            elif v.difference < 0:
                diff_str = f"{v.difference:,}"
            else:
                diff_str = "0"

        stat_display = v.status
        if v.reconciled:
            stat_display = "PASS (R)"

        note = "-"
        if v.reconciled and v.reconciliation_note:
            note = f"[Reconciled] {v.reconciliation_note[:45]}"
        elif v.candidate_resolutions:
            best = v.candidate_resolutions[0]
            if best.has_dag_conflict:
                note = f"[Konflik DAG] {best.match_type} ditolak"
            else:
                note = f"Rec: {best.match_type} (score: {best.plausibility_score:.2f})"
        elif v.status != "PASS":
            note = "Review manual pemeriksa"

        lines.append(
            f"{idx:02d}  {r_id:<35} {pg:<4} {r_type:<14} {t_val:<14} {e_val:<14} {diff_str:<12} {stat_display:<10} {note}"
        )

    lines.append("-" * 120)
    lines.append("RINGKASAN AGREGAT:")
    lines.append(f"  - Total Aturan Semantik     : {summ.total_rules}")
    lines.append(f"  - Lulus Validasi (PASS)     : {summ.rules_passed} ({summ.pass_rate_pct:.2f}%)")
    lines.append(f"  - Diskrepansi (MISMATCH)    : {summ.rules_mismatched}")
    lines.append(f"  - Berhasil Direkonsiliasi   : {summ.rules_reconciled}")
    lines.append(f"  - Total Selisih Finansial   : Rp {summ.total_discrepancy_rupiah:,}")
    lines.append("  - Status Privasi & Integritas: 100% LOKAL (Zero PII Cloud Egress, Presisi Integer 64-bit)")
    lines.append("=" * 120)
    return "\n".join(lines)


def export_csv_report(report: ValidationReport, csv_path: Path) -> None:
    """Mengekspor rincian audit ke format CSV terstruktur."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rule_id",
            "page_num",
            "rule_type",
            "target_field",
            "target_value",
            "expected_value",
            "difference",
            "status",
            "reconciled",
            "reconciliation_note",
            "candidate_count",
            "description"
        ])
        for v in report.validations:
            writer.writerow([
                v.rule_id,
                v.page_num,
                v.rule_type,
                v.target_field,
                v.target_value if v.target_value is not None else "",
                v.expected_value if v.expected_value is not None else "",
                v.difference if v.difference is not None else "",
                v.status,
                "YES" if v.reconciled else "NO",
                v.reconciliation_note or "",
                len(v.candidate_resolutions),
                v.description
            ])


# =============================================================================
# MANAJEMEN INPUT & BATCH DISPATCHER
# =============================================================================

def find_target_extractions(
    input_arg: Optional[str],
    booklet_arg: Optional[str],
    process_all: bool
) -> List[Path]:
    """Menentukan daftar berkas `_extracted.json` yang akan diaudit."""
    default_dir = WORKSPACE_ROOT / "output" / "extractions"

    # Kasus 1: Input spesifik diberikan (bisa berupa berkas atau folder)
    if input_arg:
        p = Path(input_arg)
        if not p.is_absolute():
            p = WORKSPACE_ROOT / p
        if p.is_file():
            return [p]
        if p.is_dir():
            files = sorted(list(p.glob("*/*_extracted.json")) + list(p.glob("*_extracted.json")))
            if not files:
                files = sorted([f for f in p.glob("*.json") if not f.name.endswith("_meta.json") and not f.name.endswith("_report.json")])
            if files:
                return files

    # Kasus 2: Berkas spesifik booklet_XX
    if booklet_arg and booklet_arg.lower() != "all":
        b_id = booklet_arg.strip()
        candidate1 = default_dir / b_id / f"{b_id}_extracted.json"
        if candidate1.exists():
            return [candidate1]
        candidate2 = default_dir / f"{b_id}_extracted.json"
        if candidate2.exists():
            return [candidate2]

    # Kasus 3: Seluruh buklet (--all atau --booklet all atau default jika tidak ada arg)
    if process_all or (booklet_arg and booklet_arg.lower() == "all") or (not input_arg and not booklet_arg):
        if default_dir.exists():
            files = sorted(list(default_dir.glob("*/*_extracted.json")) + list(default_dir.glob("*_extracted.json")))
            return files

    return []


# =============================================================================
# CLI RUNNER ENTRYPOINT
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mesin Validasi & Smart Reconciliation Konsistensi Aritmetika Kuesioner BPS Seruti Inti.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path berkas _extracted.json tunggal ATAU direktori ekstraksi (default: output/extractions)."
    )
    parser.add_argument(
        "--booklet", "-b",
        type=str,
        default=None,
        help="ID buklet target (misal: booklet_01, atau 'all')."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Proses seluruh berkas hasil ekstraksi yang ditemukan di direktori output/extractions."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/reports/validation",
        help="Direktori penyimpanan laporan audit konsistensi aritmetika."
    )
    parser.add_argument(
        "--auto-reconcile",
        action="store_true",
        help="MODE REKONSILIASI: Terapkan rekomendasi perbaikan digit ambigu ke data kuesioner. "
             "Default dinonaktifkan (Audit Mode: hanya mencatat temuan tanpa mutasi data)."
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "tabular", "both"],
        default="both",
        help="Format laporan yang dihasilkan (json, tabular ASCII/CSV, atau both)."
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.85,
        help="Ambang batas confidence karakter untuk penandaan kotak ber-confidence rendah."
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Toleransi selisih aritmetika dalam Rupiah (Zero-Jitter default: 0)."
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Hanya cetak ringkasan agregat di terminal, tanpa mencetak tabel detail (Zero PII compliant)."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Tampilkan rincian per relasi yang mengalami diskrepansi pada log terminal."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    start_time = time.time()

    # Tentukan berkas ekstraksi
    target_files = find_target_extractions(args.input, args.booklet, args.all)
    if not target_files:
        print(f"[ERROR] Tidak ada berkas _extracted.json yang ditemukan untuk diproses.", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = WORKSPACE_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    validator = SerutiArithmeticValidator(
        conf_threshold=args.conf_threshold,
        tolerance=args.tolerance
    )

    mode_label = "AUTO-RECONCILE" if args.auto_reconcile else "AUDIT ONLY"

    print("=" * 80)
    print("  BPS SERUTI INTI - MESIN AUDIT & REKONSILIASI KONSISTENSI ARITMETIKA")
    print(f"  Mode Operasi  : {mode_label}")
    print(f"  Berkas Target : {len(target_files)} berkas")
    print(f"  Direktori Out : {out_dir}")
    print(f"  Toleransi Rp  : {args.tolerance} Rupiah (Zero-Jitter Invariant)")
    print("=" * 80)

    batch_summary = {
        "processed_at": datetime.now().isoformat(),
        "total_booklets": len(target_files),
        "execution_mode": "auto_reconcile" if args.auto_reconcile else "audit_only",
        "booklets": []
    }

    success_count = 0
    total_discrepancy_all = 0

    for idx, fpath in enumerate(target_files, start=1):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                extracted_data = json.load(f)

            b_id = extracted_data.get("summary", {}).get("booklet_id")
            if not b_id:
                b_id = fpath.stem.replace("_extracted", "")

            # Jalankan validasi konsistensi aritmetika
            report = validator.validate_booklet(
                extracted_data=extracted_data,
                auto_reconcile=args.auto_reconcile
            )

            # Ekspor JSON Report
            if args.format in ["json", "both"]:
                json_out_path = out_dir / f"{b_id}_validation.json"
                with open(json_out_path, "w", encoding="utf-8") as jf:
                    json.dump(report.to_dict(), jf, indent=2, ensure_ascii=False)

                # Simpan juga langsung ke folder ekstraksi booklet
                if fpath.parent.is_dir() and fpath.parent != out_dir:
                    try:
                        ext_val_path = fpath.parent / f"{b_id}_validation.json"
                        with open(ext_val_path, "w", encoding="utf-8") as ejf:
                            json.dump(report.to_dict(), ejf, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

                # Jika mode auto-reconcile, simpan juga berkas kuesioner yang telah direkonsiliasi
                if args.auto_reconcile and report.reconciled_data:
                    reconciled_out_path = out_dir / f"{b_id}_reconciled.json"
                    with open(reconciled_out_path, "w", encoding="utf-8") as rf:
                        json.dump(report.reconciled_data, rf, indent=2, ensure_ascii=False)

            # Ekspor Tabular ASCII & CSV Report
            if args.format in ["tabular", "both"]:
                txt_out_path = out_dir / f"{b_id}_validation_report.txt"
                with open(txt_out_path, "w", encoding="utf-8") as tf:
                    tf.write(format_tabular_ascii(report))

                csv_out_path = out_dir / f"{b_id}_validation_summary.csv"
                export_csv_report(report, csv_out_path)

            # Catat metrik agregat batch
            summ = report.summary
            total_discrepancy_all += summ.total_discrepancy_rupiah
            batch_summary["booklets"].append({
                "booklet_id": b_id,
                "file_path": str(fpath),
                "total_rules": summ.total_rules,
                "passed": summ.rules_passed,
                "mismatched": summ.rules_mismatched,
                "reconciled": summ.rules_reconciled,
                "pass_rate_pct": summ.pass_rate_pct,
                "total_discrepancy_rupiah": summ.total_discrepancy_rupiah
            })

            success_count += 1

            # Sinkronisasi status validasi ke basis data SQLite si_periksa.db
            try:
                from db.db_init import get_connection
                conn = get_connection()
                db_val_status = "PASS" if summ.rules_mismatched == 0 else "WARN"
                conn.execute(
                    "UPDATE booklets SET val_status=?, updated_at=datetime('now','localtime') WHERE id=?",
                    (db_val_status, b_id)
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

            # Output per buklet (Zero PII compliant: hanya metrik numerik)
            print(
                f"[{idx:02d}/{len(target_files):02d}] {b_id:<12} | "
                f"Lulus: {summ.rules_passed:02d}/{summ.total_rules:02d} ({summ.pass_rate_pct:5.1f}%) | "
                f"Diskrepansi: {summ.rules_mismatched:02d} | "
                f"Rekonsiliasi: {summ.rules_reconciled:02d} | "
                f"Selisih: Rp {summ.total_discrepancy_rupiah:,}"
            )

            # Jika verbose dan tidak summary-only, cetak tabel ASCII untuk buklet tunggal
            if not args.summary_only and (args.verbose or len(target_files) == 1):
                print("\n" + format_tabular_ascii(report) + "\n")

        except Exception as e:
            print(f"[{idx:02d}/{len(target_files):02d}] Gagal memproses {fpath.name}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    batch_summary["total_elapsed_seconds"] = round(elapsed, 2)
    batch_summary["successful_booklets"] = success_count

    # Simpan ringkasan batch
    batch_summary_path = out_dir / "validation_batch_summary.json"
    with open(batch_summary_path, "w", encoding="utf-8") as bf:
        json.dump(batch_summary, bf, indent=2)

    # Simpan juga ke direktori extractions untuk sinkronisasi lintas-modul
    try:
        alt_sum_path = WORKSPACE_ROOT / "output" / "extractions" / "validation_batch_summary.json"
        alt_sum_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alt_sum_path, "w", encoding="utf-8") as abf:
            json.dump(batch_summary, abf, indent=2)
    except Exception:
        pass

    # Cetak ringkasan agregat akhir (Zero PII Cloud Egress compliant)
    print("=" * 80)
    print("  RINGKASAN AGREGAT AUDIT ARITMETIKA SELESAI")
    print(f"  Buklet Berhasil Diproses : {success_count} / {len(target_files)}")
    print(f"  Total Waktu Eksekusi     : {elapsed:.2f} detik")
    print(f"  Total Selisih Finansial  : Rp {total_discrepancy_all:,}")
    print(f"  Laporan Tersimpan di     : {out_dir}")
    print("  Kepatuhan UU PDP & UU 16 : 100% LOKAL (Zero PII Cloud Egress)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
