"""
tests/test_validator_adversarial.py
===================================
Adversarial Challenge Test Harness for seruti_pipeline/validator.py (Milestone 2).

Conducted by: Challenger 1 (M2 Stress & DAG Conflict Challenger)
Compliance: Zero-Jitter Invariant, UU No. 16/1997, UU No. 27/2022 (Zero PII Cloud Egress)

Challenge Dimensions:
1. Extreme Numerical Values: 100 billion & 1 trillion Rupiah exact 64-bit math, zero-jitter.
2. Adversarial DAG Conflicts:
   - Child breaks Parent (REL_05 candidate breaks passing REL_09)
   - Parent breaks Child (REL_09 candidate breaks passing REL_05)
   - Inequality violation (REL_10 candidate breaks passing REL_11 OOP <= Total)
   - Fallback to non-conflicting candidate
3. Payload Graceful Error Handling:
   - Completely empty dictionary
   - None inputs, None summary, None roster_art_count
   - None pages, None hal_x, None fields
   - None box_details in discrepancy field
4. Ambiguous Resolution Tie-Breaking:
   - Conflicting candidates with equal plausibility scores must NOT be auto-applied.
"""

import copy
import pytest
from seruti_pipeline.validator import (
    SerutiArithmeticValidator,
    ResolutionCandidate,
    RuleValidationResult,
    ValidationReport,
    CONFUSION_PAIRS
)
from seruti_pipeline.semantic_relations import SERUTI_SEMANTIC_RELATIONS


# =============================================================================
# 1. EXTREME NUMERICAL VALUES & ZERO-JITTER STRESS
# =============================================================================

def test_extreme_numerical_values_64bit_exact_math():
    """
    Stress-test extreme financial values (100 Billion Rupiah & beyond).
    Verifies 64-bit integer precision and zero-jitter invariant.
    """
    validator = SerutiArithmeticValidator(tolerance=0)

    # 100 Billion Rupiah test case on REL_18 (Total Mutasi Aset Tambah)
    amount_100_billion = 100_000_000_000
    comp1 = 60_000_000_000
    comp2 = 40_000_000_000

    payload = {
        "summary": {"booklet_id": "booklet_extreme_100b", "roster_art_count": 1},
        "pages": {
            "hal_6": {
                "fields": {
                    "r505_1_tanah_bangunan_tambah": {
                        "value": comp1,
                        "raw_string": str(comp1)
                    },
                    "r505_2_kendaraan_tambah": {
                        "value": comp2,
                        "raw_string": str(comp2)
                    },
                    "r505_7_total_mutasi_tambah": {
                        "value": amount_100_billion,
                        "raw_string": str(amount_100_billion)
                    }
                }
            }
        }
    }

    report = validator.validate_booklet(payload, auto_reconcile=False)
    v18 = next(v for v in report.validations if v.rule_id == "REL_18_R505_TOTAL_MUTASI_ASET_TAMBAH")

    assert v18.status == "PASS"
    assert isinstance(v18.target_value, int)
    assert isinstance(v18.expected_value, int)
    assert isinstance(v18.difference, int)
    assert v18.target_value == 100_000_000_000
    assert v18.expected_value == 100_000_000_000
    assert v18.difference == 0


def test_extreme_numerical_values_discrepancy_and_string_separators():
    """
    Test 1 Trillion Rupiah with string separators (dots/underscores/commas)
    and verify exact integer discrepancy without floating-point jitter.
    """
    validator = SerutiArithmeticValidator(tolerance=0)

    target_str = "1.000.000.000.000"  # 1 Trillion
    expected_sum = 900_000_000_000      # 900 Billion
    expected_diff = 100_000_000_000     # 100 Billion diff

    payload = {
        "summary": {"booklet_id": "booklet_extreme_1t", "roster_art_count": 1},
        "pages": {
            "hal_7": {
                "fields": {
                    "b6_r601_uang_tunai": {"value": 500_000_000_000},
                    "b6_r602a_tabungan_bank": {"value": 400_000_000_000},
                    # Target formatted with Indonesian dot thousands separators
                    "b6_r607_total_aset_keuangan": {
                        "value": target_str,
                        "raw_string": target_str
                    }
                }
            }
        }
    }

    report = validator.validate_booklet(payload, auto_reconcile=False)
    v20 = next(v for v in report.validations if v.rule_id == "REL_20_R607_TOTAL_ASET_KEUANGAN")

    assert v20.status == "MISMATCH"
    assert isinstance(v20.target_value, int)
    assert v20.target_value == 1_000_000_000_000
    assert v20.expected_value == 900_000_000_000
    assert v20.difference == expected_diff
    assert isinstance(report.summary.total_discrepancy_rupiah, int)
    assert report.summary.total_discrepancy_rupiah >= expected_diff


# =============================================================================
# 2. ADVERSARIAL DAG CONFLICT STRESS TESTS
# =============================================================================

def test_adversarial_dag_conflict_child_breaks_parent():
    """
    Adversarial Attack Scenario:
    REL_05 (child) has a mismatch. A locally plausible candidate modifies
    target b4_r452_kesehatan_total from 500,000 to 600,000 (visual confusion 5 <-> 6).
    HOWEVER, REL_09 (parent: b4_r463 = b4_r452 + b4_r456) is ALREADY PASSING
    with 1,500,000 = 500,000 + 1,000,000.
    Modifying b4_r452 to 600,000 would break REL_09 (600k + 1000k = 1600k != 1500k).

    Check: check_dag_conflict MUST mark candidate has_dag_conflict=True,
    and auto-reconcile MUST NOT apply it.
    """
    validator = SerutiArithmeticValidator(tolerance=0)

    payload = {
        "summary": {"booklet_id": "test_dag_child_breaks_parent", "roster_art_count": 1},
        "pages": {
            "hal_4": {
                "fields": {
                    "b4_r453_obat_total": {"value": 100000},
                    "b4_r454_rawat_jalan_total": {"value": 200000},
                    "b4_r455_rawat_inap_total": {"value": 300000},
                    # Sum of components is 600,000.
                    # Target read is 500,000 with box 0 = '5' (confusion pair with '6')
                    "b4_r452_kesehatan_total": {
                        "value": 500000,
                        "raw_string": "500000",
                        "box_details": [
                            {"char": "5", "conf": 0.8, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                        ]
                    },
                    "b4_r456_pendidikan_total": {"value": 1000000},
                    # REL_09 target is 1,500,000. PASSES initially with 500k + 1000k!
                    "b4_r463_total_kes_pend_total": {"value": 1500000}
                }
            }
        }
    }

    # 1. Audit mode
    report_audit = validator.validate_booklet(payload, auto_reconcile=False)
    v05 = next(v for v in report_audit.validations if v.rule_id == "REL_05_KESEHATAN_SUBTOTAL_TOTAL")
    v09 = next(v for v in report_audit.validations if v.rule_id == "REL_09_TOTAL_KESEHATAN_PENDIDIKAN_TOTAL")

    assert v05.status == "MISMATCH"
    assert v09.status == "PASS"

    target_cand = next((c for c in v05.candidate_resolutions if c.location == "TARGET"), None)
    assert target_cand is not None
    assert target_cand.proposed_value == 600000
    assert target_cand.has_dag_conflict is True, "Target candidate MUST be flagged with DAG conflict!"
    assert "REL_09_TOTAL_KESEHATAN_PENDIDIKAN_TOTAL" in target_cand.dag_conflict_reason

    # 2. Auto-reconcile mode: candidate must NOT be applied
    report_auto = validator.validate_booklet(payload, auto_reconcile=True)
    v05_auto = next(v for v in report_auto.validations if v.rule_id == "REL_05_KESEHATAN_SUBTOTAL_TOTAL")
    v09_auto = next(v for v in report_auto.validations if v.rule_id == "REL_09_TOTAL_KESEHATAN_PENDIDIKAN_TOTAL")

    # Target candidate must be blocked
    if v05_auto.applied_resolution:
        assert v05_auto.applied_resolution["proposed_value"] != 600000 or v05_auto.applied_resolution["location"] != "TARGET"
    assert v09_auto.status == "PASS", "Parent relation REL_09 must remain PASS and not be broken!"


def test_adversarial_dag_conflict_parent_breaks_child():
    """
    Adversarial Attack Scenario:
    REL_09 (parent) has a mismatch. A candidate proposes modifying source field
    b4_r452_kesehatan_total from 500,000 to 600,000.
    HOWEVER, REL_05 (child) is ALREADY PASSING with 500,000 = 100k + 150k + 250k.
    Modifying b4_r452 would break REL_05.

    Check: check_dag_conflict MUST mark candidate has_dag_conflict=True,
    and auto-reconcile MUST NOT apply it.
    """
    validator = SerutiArithmeticValidator(tolerance=0)

    payload = {
        "summary": {"booklet_id": "test_dag_parent_breaks_child", "roster_art_count": 1},
        "pages": {
            "hal_4": {
                "fields": {
                    "b4_r453_obat_total": {"value": 100000},
                    "b4_r454_rawat_jalan_total": {"value": 150000},
                    "b4_r455_rawat_inap_total": {"value": 250000},
                    # REL_05 passes: 100k + 150k + 250k = 500k
                    "b4_r452_kesehatan_total": {
                        "value": 500000,
                        "raw_string": "500000",
                        "box_details": [
                            {"char": "5", "conf": 0.8, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                        ]
                    },
                    "b4_r456_pendidikan_total": {"value": 1000000},
                    # REL_09 target is 1,600,000, but sum is 500k + 1000k = 1,500,000 (diff = 100,000)
                    "b4_r463_total_kes_pend_total": {"value": 1600000}
                }
            }
        }
    }

    report = validator.validate_booklet(payload, auto_reconcile=False)
    v09 = next(v for v in report.validations if v.rule_id == "REL_09_TOTAL_KESEHATAN_PENDIDIKAN_TOTAL")

    source_cand = next((c for c in v09.candidate_resolutions if c.field_name == "b4_r452_kesehatan_total"), None)
    assert source_cand is not None
    assert source_cand.proposed_value == 600000
    assert source_cand.has_dag_conflict is True, "Source candidate modifying b4_r452 MUST have DAG conflict!"
    assert "REL_05_KESEHATAN_SUBTOTAL_TOTAL" in source_cand.dag_conflict_reason


def test_adversarial_dag_conflict_inequality_rel11():
    """
    Adversarial Attack Scenario:
    REL_10 (Total Kes & Pend OOP) has mismatch. Target is 100,000, expected sum is 700,000.
    A candidate proposes modifying b4_r463_total_kes_pend_oop to 700,000.
    HOWEVER, REL_11 checks OOP <= Total.
    Total (b4_r463_total_kes_pend_total) is 500,000.
    Initially 100,000 <= 500,000 (PASS).
    If OOP is modified to 700,000, 700,000 <= 500,000 would FAIL REL_11.

    Check: check_dag_conflict MUST catch the inequality violation.
    """
    validator = SerutiArithmeticValidator(tolerance=0)

    payload = {
        "summary": {"booklet_id": "test_dag_inequality", "roster_art_count": 1},
        "pages": {
            "hal_4": {
                "fields": {
                    "b4_r452_kesehatan_oop": {"value": 400000},
                    "b4_r456_pendidikan_oop": {"value": 300000},
                    # Target 100,000, box 0 = '1' (confusion pair with '7')
                    "b4_r463_total_kes_pend_oop": {
                        "value": 100000,
                        "raw_string": "100000",
                        "box_details": [
                            {"char": "1", "conf": 0.8, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                            {"char": "0", "conf": 1.0, "is_blank": False},
                        ]
                    },
                    "b4_r463_total_kes_pend_total": {"value": 500000}
                }
            }
        }
    }

    report_auto = validator.validate_booklet(payload, auto_reconcile=True)
    v10 = next(v for v in report_auto.validations if v.rule_id == "REL_10_TOTAL_KESEHATAN_PENDIDIKAN_OOP")
    v11 = next(v for v in report_auto.validations if v.rule_id == "REL_11_OOP_LE_TOTAL_CHECK")

    target_cand = next((c for c in v10.candidate_resolutions if c.location == "TARGET"), None)
    assert target_cand is not None
    assert target_cand.proposed_value == 700000
    assert target_cand.has_dag_conflict is True
    assert "REL_11_OOP_LE_TOTAL_CHECK" in target_cand.dag_conflict_reason

    # Auto-reconcile must NOT apply this candidate
    assert v10.reconciled is False
    assert v10.status == "MISMATCH"
    assert v11.status == "PASS"


# =============================================================================
# 3. EMPTY, MISSING, AND NULL PAYLOADS ERROR HANDLING
# =============================================================================

def test_graceful_handling_empty_dict():
    """Validates that a completely empty extraction dict does not crash."""
    validator = SerutiArithmeticValidator()
    report = validator.validate_booklet({})
    assert isinstance(report, ValidationReport)
    assert report.summary.total_rules == 21
    assert len(report.validations) == 21


@pytest.mark.parametrize("corrupted_payload,case_name", [
    (None, "None input"),
    ({"summary": None}, "summary None"),
    ({"summary": {"roster_art_count": None}}, "roster_art_count None"),
    ({"pages": None}, "pages None"),
    ({"pages": {"hal_1": None}}, "hal_1 None"),
    ({"pages": {"hal_3": {"fields": None}}}, "fields None"),
    (
        {
            "summary": {"booklet_id": "test_box_details_none", "roster_art_count": 1},
            "pages": {
                "hal_5": {
                    "fields": {
                        "r501_art_01_uang": {"value": 1000},
                        "r501_art_01_barang": {"value": 1000},
                        "r501_art_01_total": {"value": 3000, "box_details": None, "raw_string": "3000"}
                    }
                }
            }
        },
        "box_details None in mismatch field"
    ),
])
def test_graceful_handling_null_and_corrupt_payloads(corrupted_payload, case_name):
    """
    Stress-tests corrupted, null, and malformed extraction payloads.
    The validator MUST gracefully handle these without unhandled exceptions
    (e.g. AttributeError, TypeError) crashing the process.
    """
    validator = SerutiArithmeticValidator()
    # This call should not raise an unhandled exception
    report = validator.validate_booklet(corrupted_payload, auto_reconcile=True)
    assert isinstance(report, ValidationReport)
    assert report.summary.total_rules == 21


# =============================================================================
# 4. AMBIGUOUS CANDIDATES WITH EQUAL PLAUSIBILITY SCORES
# =============================================================================

def test_ambiguous_equal_score_candidates_not_auto_applied():
    """
    Stress-Test Ambiguity Guard:
    When multiple valid candidates (without DAG conflicts) have EQUAL plausibility scores
    and propose conflicting modifications (e.g. modify target vs modify source),
    the validator MUST NOT arbitrarily auto-apply one candidate.
    It must preserve data safety and require human review.
    """
    validator = SerutiArithmeticValidator()

    # In REL_12 (r501_art_01_total = r501_art_01_uang + r501_art_01_barang)
    # Source uang = 5,000,000 (box 0 = '5', conf = 0.8)
    # Source barang = 0
    # Expected sum = 5,000,000
    # Target = 6,000,000 (box 0 = '6', conf = 0.8)
    # Candidate 1 (Target): swap '6' -> '5' (visual confusion pair, conf 0.8 -> score 0.69)
    # Candidate 2 (Source): swap '5' -> '6' (visual confusion pair, conf 0.8 -> score 0.69)
    # Both are non-DAG conflicting, with identical plausibility scores.
    payload = {
        "summary": {"booklet_id": "test_ambiguous_tie", "roster_art_count": 1},
        "pages": {
            "hal_5": {
                "page_num": 5,
                "fields": {
                    "r501_art_01_uang": {
                        "value": 5000000,
                        "raw_string": "5000000",
                        "box_details": [{"char": "5", "conf": 0.8, "is_blank": False}] + [{"char": "0", "conf": 1.0, "is_blank": False}] * 6
                    },
                    "r501_art_01_barang": {
                        "value": 0,
                        "raw_string": "0000000",
                        "box_details": [{"char": "0", "conf": 1.0, "is_blank": False}] * 7
                    },
                    "r501_art_01_total": {
                        "value": 6000000,
                        "raw_string": "6000000",
                        "box_details": [{"char": "6", "conf": 0.8, "is_blank": False}] + [{"char": "0", "conf": 1.0, "is_blank": False}] * 6
                    }
                }
            }
        }
    }

    report = validator.validate_booklet(payload, auto_reconcile=True)
    v12 = next(v for v in report.validations if v.rule_id == "REL_12_R501_TOTAL_UPAH_PER_ART")

    # Verify that equal-score candidates were generated
    valid_candidates = [c for c in v12.candidate_resolutions if not c.has_dag_conflict]
    assert len(valid_candidates) >= 2
    assert abs(valid_candidates[0].plausibility_score - valid_candidates[1].plausibility_score) < 1e-4

    # CRITICAL INVARIANT: Ambiguous resolutions must NOT be auto-applied!
    assert v12.reconciled is False, (
        f"AMBIGUOUS RESOLUTION FAILED: Two candidates had identical score {valid_candidates[0].plausibility_score}, "
        f"but {v12.applied_resolution} was auto-applied arbitrarily!"
    )
    assert v12.status == "MISMATCH"
