"""
tests/test_cli_and_reports_adversarial.py
=========================================
Adversarial Verification Suite for M2 CLI Runner & Generated Audit Reports.
Tested by Challenger 2 (Empirical Challenger).

Verifies:
1. CLI flag permutations and exit code contracts.
2. JSON Schema compliance across all validation reports.
3. ASCII tabular and CSV report structure, column alignment, and headers.
4. Extraction source immutability (100% SHA-256 invariant).
5. Auto-reconcile output generation and reconciliation marking.
6. Zero PII compliance (no NIK, respondent names, addresses, phone numbers).
7. Strict 64-bit integer arithmetic (Zero-Jitter Invariant).
"""

import json
import csv
import re
import hashlib
import subprocess
import sys
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(filepath: Path) -> str:
    """Compute exact SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestCLIExecutionPermutations:
    """Test CLI execution across required and adversarial flag permutations."""

    def test_cli_help_flag(self):
        """Verify that --help exits with code 0 and provides full options."""
        res = subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--help"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True
        )
        assert res.returncode == 0
        assert "usage: run_seruti_validator.py" in res.stdout
        assert "--auto-reconcile" in res.stdout
        assert "--format" in res.stdout
        assert "--summary-only" in res.stdout

    def test_cli_booklet_01_audit_mode(self):
        """Verify audit mode execution on booklet_01."""
        res = subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--booklet", "booklet_01"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True
        )
        assert res.returncode == 0
        assert "AUDIT ONLY" in res.stdout
        assert "booklet_01" in res.stdout
        assert "17/21" in res.stdout
        assert "Rp 18,000,006" in res.stdout

    def test_cli_booklet_09_auto_reconcile(self):
        """Verify auto-reconcile mode execution on booklet_09."""
        res = subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--booklet", "booklet_09", "--auto-reconcile"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True
        )
        assert res.returncode == 0
        assert "AUTO-RECONCILE" in res.stdout
        assert "booklet_09" in res.stdout
        assert "Rekonsiliasi:" in res.stdout

    def test_cli_all_summary_only(self):
        """Verify --all --summary-only executes cleanly without printing ASCII table."""
        res = subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--all", "--summary-only"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True
        )
        assert res.returncode == 0
        assert "booklet_01" in res.stdout
        assert "booklet_09" in res.stdout
        assert "RINGKASAN AGREGAT AUDIT ARITMETIKA SELESAI" in res.stdout
        assert "Buklet Berhasil Diproses :" in res.stdout
        # ASCII detailed table should NOT be printed when --summary-only is set
        assert "BADAN PUSAT STATISTIK - LAPORAN AUDIT" not in res.stdout

    def test_cli_all_format_json(self):
        """Verify --all --format json generates json and batch summary."""
        res = subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--all", "--format", "json"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True
        )
        assert res.returncode == 0
        rep_dir = WORKSPACE_ROOT / "output" / "reports" / "validation"
        assert (rep_dir / "booklet_01_validation.json").exists()
        assert (rep_dir / "booklet_09_validation.json").exists()
        assert (rep_dir / "validation_batch_summary.json").exists()

    def test_cli_invalid_format_rejection(self):
        """Adversarial: invalid choice for format must fail with non-zero exit code."""
        res = subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--format", "xml"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True
        )
        assert res.returncode != 0
        assert "invalid choice" in res.stderr

    def test_cli_nonexistent_booklet(self):
        """Adversarial: non-existent booklet should fail cleanly with exit code 1."""
        res = subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--booklet", "booklet_9999"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True
        )
        assert res.returncode == 1
        assert "[ERROR]" in res.stderr or "[ERROR]" in res.stdout


class TestExtractionSourceImmutability:
    """Verify that source extraction files remain 100% immutable across all runs."""

    def test_source_files_remain_unmodified(self):
        extract_dir = WORKSPACE_ROOT / "output" / "extractions"
        b01_src = extract_dir / "booklet_01" / "booklet_01_extracted.json"
        b09_src = extract_dir / "booklet_09" / "booklet_09_extracted.json"

        assert b01_src.exists()
        assert b09_src.exists()

        hash_b01_before = compute_sha256(b01_src)
        hash_b09_before = compute_sha256(b09_src)

        # Run audit-only mode
        subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--all"],
            cwd=str(WORKSPACE_ROOT),
            check=True,
            capture_output=True
        )
        assert compute_sha256(b01_src) == hash_b01_before
        assert compute_sha256(b09_src) == hash_b09_before

        # Run auto-reconcile mode
        subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--all", "--auto-reconcile"],
            cwd=str(WORKSPACE_ROOT),
            check=True,
            capture_output=True
        )
        # Raw extractions must NEVER be mutated in place
        assert compute_sha256(b01_src) == hash_b01_before
        assert compute_sha256(b09_src) == hash_b09_before


class TestReportSchemaAndContent:
    """Audit schema, structure, and values of generated reports."""

    @pytest.fixture(autouse=True)
    def ensure_reports_generated(self):
        subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--all", "--auto-reconcile", "--format", "both"],
            cwd=str(WORKSPACE_ROOT),
            check=True,
            capture_output=True
        )

    def test_validation_json_schema(self):
        rep_dir = WORKSPACE_ROOT / "output" / "reports" / "validation"
        for b_id in ["booklet_01", "booklet_09"]:
            json_file = rep_dir / f"{b_id}_validation.json"
            assert json_file.exists()
            data = json.loads(json_file.read_text(encoding="utf-8"))

            # 1. Top level keys
            assert set(data.keys()) == {"audit_metadata", "summary", "validations"}

            # 2. Audit metadata keys
            meta = data["audit_metadata"]
            for req_key in ["booklet_id", "timestamp", "validator_version", "execution_mode", "conf_threshold", "tolerance_rupiah", "zero_pii_compliant"]:
                assert req_key in meta
            assert meta["booklet_id"] == b_id
            assert meta["zero_pii_compliant"] is True

            # 3. Summary keys
            summ = data["summary"]
            for req_key in ["total_rules", "rules_passed", "rules_mismatched", "rules_reconciled", "pass_rate_pct", "total_discrepancy_rupiah"]:
                assert req_key in summ
            assert summ["total_rules"] == 21
            assert isinstance(summ["total_discrepancy_rupiah"], int)

            # 4. Validations list
            validations = data["validations"]
            assert len(validations) == 21
            for v in validations:
                for req_key in [
                    "rule_id", "rule_type", "page_num", "description",
                    "target_field", "target_value", "expected_value",
                    "difference", "status", "reconciled", "reconciliation_note",
                    "affected_fields", "candidate_resolutions", "applied_resolution"
                ]:
                    assert req_key in v, f"Missing key {req_key} in validation item {v.get('rule_id')}"

                # Strict integer check (Zero-Jitter Invariant)
                if v["target_value"] is not None:
                    assert isinstance(v["target_value"], int)
                if v["expected_value"] is not None:
                    assert isinstance(v["expected_value"], int)
                if v["difference"] is not None:
                    assert isinstance(v["difference"], int)

    def test_reconciled_booklet_json(self):
        """Verify that auto-reconcile produces valid reconciled booklet files with corrected values."""
        rep_dir = WORKSPACE_ROOT / "output" / "reports" / "validation"
        for b_id in ["booklet_01", "booklet_09"]:
            rec_file = rep_dir / f"{b_id}_reconciled.json"
            assert rec_file.exists()
            rec_data = json.loads(rec_file.read_text(encoding="utf-8"))
            assert "pages" in rec_data
            assert "summary" in rec_data

        # Booklet 09 specifically had REL_20 reconciled: 350006 -> 350800
        b09_rec = json.loads((rep_dir / "booklet_09_reconciled.json").read_text(encoding="utf-8"))
        f_rec = b09_rec["pages"]["hal_7"]["fields"]["b6_r607_total_aset_keuangan"]
        assert f_rec["value"] == 350800


    def test_ascii_tabular_report_structure(self):
        """Verify ASCII tabular report headers, alignments, and columns."""
        rep_dir = WORKSPACE_ROOT / "output" / "reports" / "validation"
        for b_id in ["booklet_01", "booklet_09"]:
            txt_file = rep_dir / f"{b_id}_validation_report.txt"
            assert txt_file.exists()
            content = txt_file.read_text(encoding="utf-8")

            # Check header
            assert "BADAN PUSAT STATISTIK - LAPORAN AUDIT KONSISTENSI ARITMETIKA" in content
            assert f"Buklet: {b_id}" in content
            assert "No   Rule ID                             Hal  Tipe           Target Value   Expected       Diff (Rp)    Status     Rekomendasi / Catatan" in content
            assert "RINGKASAN AGREGAT:" in content
            assert "Total Aturan Semantik     : 21" in content
            assert "Status Privasi & Integritas: 100% LOKAL (Zero PII Cloud Egress, Presisi Integer 64-bit)" in content

            # Check line count
            lines = content.strip().splitlines()
            assert len(lines) >= 33  # 21 rules + headers + summary

    def test_csv_summary_structure(self):
        """Verify CSV report format and row count."""
        rep_dir = WORKSPACE_ROOT / "output" / "reports" / "validation"
        for b_id in ["booklet_01", "booklet_09"]:
            csv_file = rep_dir / f"{b_id}_validation_summary.csv"
            assert csv_file.exists()
            with open(csv_file, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == 22  # 1 header + 21 rules
                header = rows[0]
                assert header == [
                    "rule_id", "page_num", "rule_type", "target_field",
                    "target_value", "expected_value", "difference", "status",
                    "reconciled", "reconciliation_note", "candidate_count", "description"
                ]

    def test_batch_summary_schema(self):
        """Verify validation_batch_summary.json schema."""
        rep_dir = WORKSPACE_ROOT / "output" / "reports" / "validation"
        batch_file = rep_dir / "validation_batch_summary.json"
        assert batch_file.exists()
        b_data = json.loads(batch_file.read_text(encoding="utf-8"))
        assert "processed_at" in b_data
        assert "total_booklets" in b_data
        assert "execution_mode" in b_data
        assert "booklets" in b_data
        assert "total_elapsed_seconds" in b_data
        assert "successful_booklets" in b_data
        assert b_data["total_booklets"] >= 2
        assert b_data["successful_booklets"] == b_data["total_booklets"]


class TestZeroPIICompliance:
    """Adversarial scanning for personal identifiable information (PII)."""

    def test_terminal_stdout_zero_pii(self):
        """Verify CLI terminal output emits zero PII."""
        res = subprocess.run(
            [sys.executable, "run_seruti_validator.py", "--all", "--auto-reconcile", "--verbose"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        combined_output = res.stdout + res.stderr

        # 1. Indonesian NIK pattern (16 consecutive digits)
        nik_matches = re.findall(r"\b\d{16}\b", combined_output)
        assert len(nik_matches) == 0, f"Found NIK pattern in stdout: {nik_matches}"

        # 2. Indonesian phone number patterns (08xx-xxxx-xxxx or +628xx)
        phone_matches = re.findall(r"\b(08\d{8,11}|\+628\d{8,11})\b", combined_output)
        assert len(phone_matches) == 0, f"Found phone number in stdout: {phone_matches}"

        # 3. PII keyword triggers (context-aware patterns)
        forbidden_keywords = [
            "nama kepala rumah tangga", "nama krt", "nama responden",
            "alamat responden"
        ]
        lowered = combined_output.lower()
        for kw in forbidden_keywords:
            assert kw not in lowered, f"Forbidden PII keyword found in stdout: '{kw}'"

        # Check address patterns (RT/RW, Jalan, Dusun) with boundary/digit context
        address_patterns = [
            r"\brt\s*\d+", r"\brw\s*\d+", r"\bjalan\s+[a-zA-Z]", r"\bdusun\s+[a-zA-Z]"
        ]
        for pat in address_patterns:
            addr_matches = re.findall(pat, lowered)
            assert len(addr_matches) == 0, f"Found address pattern '{pat}' in stdout: {addr_matches}"


    def test_report_files_zero_pii(self):
        """Verify all generated reports in output/reports/validation/ emit zero PII."""
        rep_dir = WORKSPACE_ROOT / "output" / "reports" / "validation"
        report_files = [
            rep_dir / "booklet_01_validation.json",
            rep_dir / "booklet_09_validation.json",
            rep_dir / "booklet_01_validation_report.txt",
            rep_dir / "booklet_09_validation_report.txt",
            rep_dir / "booklet_01_validation_summary.csv",
            rep_dir / "booklet_09_validation_summary.csv",
            rep_dir / "validation_batch_summary.json"
        ]

        forbidden_keywords = [
            "nama kepala rumah tangga", "nama krt", "nama responden",
            "alamat responden"
        ]

        for r_file in report_files:
            text = r_file.read_text(encoding="utf-8")
            # 1. NIK pattern
            nik_matches = re.findall(r"\b\d{16}\b", text)
            assert len(nik_matches) == 0, f"Found NIK pattern in {r_file.name}: {nik_matches}"

            # 2. Phone pattern
            phone_matches = re.findall(r"\b(08\d{8,11}|\+628\d{8,11})\b", text)
            assert len(phone_matches) == 0, f"Found phone in {r_file.name}: {phone_matches}"

            # 3. Keywords
            lowered = text.lower()
            for kw in forbidden_keywords:
                assert kw not in lowered, f"Forbidden keyword '{kw}' found in {r_file.name}"
