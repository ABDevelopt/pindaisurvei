"""
tests/test_validator.py
=======================
Rangkaian Pengujian Komprehensif untuk Mesin Validasi Konsistensi Aritmetika
dan Smart Reconciliation Kuesioner BPS Susenas Seruti Inti (VSERUTI26.INTI).

Cakupan Pengujian:
1. Evaluasi ke-21 relasi semantik terhadap data sintetis sempurna (100% PASS).
2. Evaluasi terhadap data ekstraksi riil (booklet_01 dan booklet_09).
3. Uji deteksi rekonsiliasi cerdas swap 5 <-> 6 pada REL_18 (booklet_01).
4. Uji deteksi rekonsiliasi cerdas target resolution pada REL_20 (booklet_09).
5. Uji penolakan konflik DAG (Cross-Relation DAG Conflict Guard).
6. Uji perbandingan mode audit-only vs auto-reconcile.
7. Uji imputasi subtotal kosong (Omitted Subtotal Imputation).
8. Uji integritas aritmetika 64-bit integer (Zero-Jitter Invariant).
9. Uji resolusi alias bidireksional.
10. Uji eksekusi CLI runner (run_seruti_validator.py).
"""

import copy
import json
import subprocess
import sys
from pathlib import Path
import pytest

from seruti_pipeline.semantic_relations import (
    SERUTI_SEMANTIC_RELATIONS,
    SemanticRelation,
)
from seruti_pipeline.validator import (
    SerutiArithmeticValidator,
    ResolutionCandidate,
    RuleValidationResult,
    ValidationReport,
    CONFUSION_PAIRS,
    FIELD_ALIASES,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# FIXTURES: DATA SINTETIS BERSIH DENGAN 21 RELASI LENGKAP
# =============================================================================

@pytest.fixture
def clean_synthetic_extraction() -> dict:
    """Membuat struktur kuesioner sintetis di mana ke-21 relasi semantik terpenuhi 100%."""
    data = {
        "summary": {
            "booklet_id": "booklet_synthetic_clean",
            "roster_art_count": 4,
            "total_fields": 250,
            "processed_at": "2026-11-14T20:00:00Z"
        },
        "pages": {
            "hal_1": {
                "page_num": 1,
                "fields": {
                    "b1_r108_nus": {"value": 4, "raw_string": "_____04"}
                }
            },
            "hal_2": {
                "page_num": 2,
                "fields": {
                    "b3_art_01_hub_krt": {"value": 1, "raw_string": "1"}
                }
            },
            "hal_3": {
                "page_num": 3,
                "fields": {
                    # 9 komoditas sebulan (REL_03)
                    "b4_r415_pembersih_sebulan": {"value": 100000, "raw_string": "___100000"},
                    "b4_r416_jasa_domestik_sebulan": {"value": 150000, "raw_string": "___150000"},
                    "b4_r417_transport_sebulan": {"value": 200000, "raw_string": "___200000"},
                    "b4_r419_bbm_sebulan": {"value": 300000, "raw_string": "___300000"},
                    "b4_r426_komunikasi_sebulan": {"value": 120000, "raw_string": "___120000"},
                    "b4_r428_pulsa_internet_sebulan": {"value": 250000, "raw_string": "___250000"},
                    "b4_r437_jasa_lainnya_sebulan": {"value": 80000, "raw_string": "____80000"},
                    "b4_r438_perawatan_diri_sebulan": {"value": 110000, "raw_string": "___110000"},
                    "b4_r439_salon_sebulan": {"value": 90000, "raw_string": "____90000"},
                    # Target REL_03: sum(9 komoditas) = 1,400,000
                    "b4_r451_total_bukan_makanan_sebulan": {"value": 1400000, "raw_string": "__1400000"},

                    # 29 komoditas 3 bulan (REL_04) - menggunakan kombinasi nama canonical dan alias
                    "b4_r417_transport_3bln": {"value": 100000},
                    "b4_r418_kendaraan_3bln": {"value": 50000},
                    "b4_r420_servis_kendaraan_3bln": {"value": 50000},
                    "b4_r421_sewa_tol_parkir_3bln": {"value": 50000},
                    "b4_r422_angkutan_darat_3bln": {"value": 50000},
                    "b4_r423_angkutan_air_3bln": {"value": 50000},
                    "b4_r424_angkutan_udara_3bln": {"value": 50000},
                    "b4_r425_ongkir_3bln": {"value": 50000},               # alias dari jasa_pengiriman_3bln
                    "b4_r427_alat_komunikasi_3bln": {"value": 50000},       # alias dari pos_paket_3bln
                    "b4_r430_rekreasi_3bln": {"value": 50000},              # alias dari r429
                    "b4_r431_alat_olahraga_3bln": {"value": 50000},         # alias dari r430
                    "b4_r432_tiket_wisata_3bln": {"value": 50000},          # alias dari r431
                    "b4_r434_buku_tulis_3bln": {"value": 50000},            # alias dari r432
                    "b4_r435_liburan_umroh_3bln": {"value": 50000},         # alias dari r433
                    "b4_r434_pakaian_3bln": {"value": 50000},
                    "b4_r435_alas_kaki_3bln": {"value": 50000},
                    "b4_r436_hotel_3bln": {"value": 50000},                 # alias dari tekstil_3bln
                    "b4_r437_jasa_lainnya_3bln": {"value": 50000},
                    "b4_r440_perhiasan_tas_3bln": {"value": 50000},         # alias dari perlengkapan_pribadi
                    "b4_r441_admin_bank_3bln": {"value": 50000},
                    "b4_r442_jasa_dokumen_3bln": {"value": 50000},
                    "b4_r443_pajak_premi_3bln": {"value": 50000},           # alias dari gadai_asuransi
                    "b4_r444_pajak_3bln": {"value": 50000},
                    "b4_r445_premi_asuransi_3bln": {"value": 50000},
                    "b4_r446_pesta_3bln": {"value": 50000},                 # alias dari denda_3bln
                    "b4_r447_perkawinan_3bln": {"value": 50000},            # alias dari bunga_konsumsi
                    "b4_r448_ultah_khitanan_3bln": {"value": 50000},        # alias dari sumbangan
                    "b4_r449_pemakaman_3bln": {"value": 50000},             # alias dari zakat_infaq
                    "b4_r450_pesta_lainnya_3bln": {"value": 50000},         # alias dari pesta_3bln
                    # Target REL_04: 100k + 28*50k = 1,500,000
                    "b4_r451_total_bukan_makanan_3bln": {"value": 1500000, "raw_string": "__1500000"}
                }
            },
            "hal_4": {
                "page_num": 4,
                "fields": {
                    # Kesehatan Total & OOP (REL_05, REL_06)
                    "b4_r453_obat_total": {"value": 100000},
                    "b4_r453_obat_oop": {"value": 80000},
                    "b4_r454_rawat_jalan_total": {"value": 150000},
                    "b4_r454_rawat_jalan_oop": {"value": 100000},
                    "b4_r455_rawat_inap_total": {"value": 250000},
                    "b4_r455_rawat_inap_oop": {"value": 120000},
                    # Target REL_05: 100k + 150k + 250k = 500,000
                    "b4_r452_kesehatan_total": {"value": 500000},
                    # Target REL_06: 80k + 100k + 120k = 300,000
                    "b4_r452_kesehatan_oop": {"value": 300000},

                    # Pendidikan Total & OOP (REL_07, REL_08)
                    "b4_r457_paud_total": {"value": 50000},
                    "b4_r457_paud_oop": {"value": 40000},
                    "b4_r458_sd_total": {"value": 100000},
                    "b4_r458_sd_oop": {"value": 80000},
                    "b4_r459_smp_total": {"value": 150000},
                    "b4_r459_smp_oop": {"value": 100000},
                    "b4_r460_sma_total": {"value": 200000},
                    "b4_r460_sma_oop": {"value": 150000},
                    "b4_r461_pt_total": {"value": 400000},
                    "b4_r461_pt_oop": {"value": 300000},
                    "b4_r462_kursus_total": {"value": 100000},
                    "b4_r462_kursus_oop": {"value": 80000},
                    # Target REL_07: 50+100+150+200+400+100 = 1,000,000
                    "b4_r456_pendidikan_total": {"value": 1000000},
                    # Target REL_08: 40+80+100+150+300+80 = 750,000
                    "b4_r456_pendidikan_oop": {"value": 750000},

                    # Grand Total Kesehatan + Pendidikan (REL_09, REL_10, REL_11)
                    # REL_09: 500k + 1000k = 1,500,000
                    "b4_r463_total_kes_pend_total": {"value": 1500000},
                    # REL_10: 300k + 750k = 1,050,000
                    # REL_11: OOP (1,050,000) <= Total (1,500,000)
                    "b4_r463_total_kes_pend_oop": {"value": 1050000}
                }
            },
            "hal_5": {
                "page_num": 5,
                "fields": {
                    # Upah ART 01 (REL_12)
                    "r501_art_01_uang": {"value": 4000000},
                    "r501_art_01_barang": {"value": 500000},
                    "r501_art_01_total": {"value": 4500000},

                    # Upah ART 02 s.d 05 untuk Subtotal Upah Seluruh ART (REL_13)
                    "r501_art_02_uang": {"value": 2000000},
                    "r501_art_03_uang": {"value": 1500000},
                    "r501_art_04_uang": {"value": 0},
                    "r501_art_05_uang": {"value": 0},
                    # REL_13: 4M + 2M + 1.5M = 7,500,000
                    "r501_subtotal_upah_uang": {"value": 7500000},

                    # Rincian 503 Pendapatan & Biaya Kepemilikan (REL_14, REL_15)
                    "r503_1_bunga_diterima": {"value": 200000},
                    "r503_2_dividen_diterima": {"value": 300000},
                    "r503_3_sewa_tanah_diterima": {"value": 500000},
                    # REL_14: 200k + 300k + 500k = 1,000,000
                    "r503_4_total_diterima": {"value": 1000000},

                    "r503_1_bunga_dibayar": {"value": 150000},
                    "r503_3_sewa_tanah_dibayar": {"value": 100000},
                    # REL_15: 150k + 100k = 250,000
                    "r503_4_total_dibayar": {"value": 250000}
                }
            },
            "hal_6": {
                "page_num": 6,
                "fields": {
                    # Transfer Berjalan Diterima & Diberikan (REL_16, REL_17)
                    "r504_1b_terima_barang": {"value": 500000},
                    "r504_3_swasta_terima_barang": {"value": 250000},
                    # REL_16: 500k + 250k = 750,000
                    "r504_5_total_terima_barang": {"value": 750000},

                    "r504_2_remitansi_beri_uang": {"value": 400000},
                    "r504_3_swasta_beri_uang": {"value": 200000},
                    # REL_17: 400k + 200k = 600,000
                    "r504_5_total_beri_uang": {"value": 600000},

                    # Mutasi Aset Tambah & Kurang (REL_18, REL_19)
                    "r505_1_tanah_bangunan_tambah": {"value": 50000000},
                    "r505_2_kendaraan_tambah": {"value": 25000000},
                    # REL_18: 50M + 25M = 75,000,000
                    "r505_7_total_mutasi_tambah": {"value": 75000000},

                    "r505_1_tanah_bangunan_kurang": {"value": 30000000},
                    "r505_2_kendaraan_kurang": {"value": 10000000},
                    # REL_19: 30M + 10M = 40,000,000
                    "r505_7_total_mutasi_kurang": {"value": 40000000}
                }
            },
            "hal_7": {
                "page_num": 7,
                "fields": {
                    # Aset Keuangan (REL_20)
                    "b6_r601_uang_tunai": {"value": 1000000},
                    "b6_r602a_tabungan_bank": {"value": 5000000},
                    "b6_r602b_deposito": {"value": 10000000},
                    "b6_r602c_giro": {"value": 0},
                    "b6_r603_saham": {"value": 2000000},
                    "b6_r604a_reksadana": {"value": 1000000},
                    "b6_r604b_obligasi": {"value": 0},
                    "b6_r604c_aset_kripto": {"value": 500000},
                    "b6_r605a_emas_murni": {"value": 3000000},
                    "b6_r605b_valuta_asing": {"value": 500000},
                    "b6_r606_aset_lainnya": {"value": 0},
                    # REL_20: sum = 23,000,000
                    "b6_r607_total_aset_keuangan": {"value": 23000000},

                    # Kewajiban Utang (REL_21)
                    "b6_r608a_kpr": {"value": 15000000},
                    "b6_r608b_kredit_kendaraan": {"value": 5000000},
                    "b6_r608c_kredit_lain": {"value": 0},
                    "b6_r609_pinjaman_bank": {"value": 2000000},
                    "b6_r610_pinjaman_nonbank": {"value": 0},
                    "b6_r611a_utang_perorangan": {"value": 500000},
                    "b6_r611b_utang_pinjol": {"value": 0},
                    # REL_21: sum = 22,500,000
                    "b6_r611_total_kewajiban": {"value": 22500000}
                }
            }
        }
    }
    return data


# =============================================================================
# TEST SUITE 1: EVALUASI SELURUH 21 RELASI SEMANTIK (CLEAN SYNTHETIC)
# =============================================================================

def test_all_21_relations_synthetic_clean(clean_synthetic_extraction):
    """Memastikan ke-21 relasi semantik dievaluasi dan 100% PASS pada data bersih."""
    validator = SerutiArithmeticValidator(tolerance=0)
    report = validator.validate_booklet(clean_synthetic_extraction, auto_reconcile=False)

    assert report.summary.total_rules == 21
    assert report.summary.rules_passed == 21
    assert report.summary.rules_mismatched == 0
    assert report.summary.pass_rate_pct == 100.0
    assert report.summary.total_discrepancy_rupiah == 0

    # Pastikan setiap relasi dari SERUTI_SEMANTIC_RELATIONS ada di report
    all_rule_ids = {r.relation_id for r in SERUTI_SEMANTIC_RELATIONS}
    tested_rule_ids = {v.rule_id for v in report.validations}
    assert tested_rule_ids == all_rule_ids

    # Verifikasi status individual
    for v in report.validations:
        assert v.status == "PASS"
        assert v.difference == 0 or v.difference is None


# =============================================================================
# TEST SUITE 2: EVALUASI TERHADAP EXTRAKSI RIIL (BOOKLET 01 & 09)
# =============================================================================

def test_evaluation_real_extractions():
    """Memastikan validator dapat memproses ekstraksi riil tanpa error."""
    validator = SerutiArithmeticValidator(tolerance=0)

    # Booklet 01
    p1 = WORKSPACE_ROOT / "output" / "extractions" / "booklet_01" / "booklet_01_extracted.json"
    if p1.exists():
        with open(p1, "r", encoding="utf-8") as f:
            d1 = json.load(f)
        rep1 = validator.validate_booklet(d1, auto_reconcile=False)
        assert rep1.summary.total_rules == 21
        assert rep1.summary.rules_passed >= 14  # Baseline audit: 14 lulus awal
        assert len(rep1.validations) == 21

    # Booklet 09
    p9 = WORKSPACE_ROOT / "output" / "extractions" / "booklet_09" / "booklet_09_extracted.json"
    if p9.exists():
        with open(p9, "r", encoding="utf-8") as f:
            d9 = json.load(f)
        rep9 = validator.validate_booklet(d9, auto_reconcile=False)
        assert rep9.summary.total_rules == 21
        assert rep9.summary.rules_passed >= 12  # Baseline audit: 12 lulus awal pada 7 halaman riil
        assert len(rep9.validations) == 21


# =============================================================================
# TEST SUITE 3: SMART RECONCILIATION SWAP 5 <-> 6 PADA REL_18 (BOOKLET 01)
# =============================================================================

def test_smart_reconciliation_rel18_booklet01_confusion_swap():
    """
    Menguji bahwa Smart Reconciliation berhasil mengidentifikasi swap 5 <-> 6
    pada REL_18_R505_TOTAL_MUTASI_ASET_TAMBAH di booklet_01.
    """
    p1 = WORKSPACE_ROOT / "output" / "extractions" / "booklet_01" / "booklet_01_extracted.json"
    assert p1.exists(), f"File {p1} wajib tersedia."

    with open(p1, "r", encoding="utf-8") as f:
        d1 = json.load(f)

    validator = SerutiArithmeticValidator()
    report = validator.validate_booklet(d1, auto_reconcile=False)

    v18 = next((v for v in report.validations if v.rule_id == "REL_18_R505_TOTAL_MUTASI_ASET_TAMBAH"), None)
    assert v18 is not None
    assert v18.status == "MISMATCH"
    assert v18.difference == -10000000  # 50,000,000 - 60,000,000

    # Periksa keberadaan kandidat resolusi
    assert len(v18.candidate_resolutions) >= 2

    # Kandidat 1: Target box 2 swap '5' -> '6' (Visual Confusion Pair)
    target_cand = next((c for c in v18.candidate_resolutions if c.location == "TARGET"), None)
    assert target_cand is not None
    assert target_cand.current_char == "5"
    assert target_cand.proposed_char == "6"
    assert target_cand.match_type in ["VISUAL_CONFUSION_PAIR", "TARGET_HARMONIZATION_MULTI_BOX"]
    assert ("5", "6") in CONFUSION_PAIRS
    assert not target_cand.has_dag_conflict

    # Kandidat 2: Source box 2 swap '6' -> '5' (Visual Confusion Pair)
    source_cand = next((c for c in v18.candidate_resolutions if c.location == "SOURCE"), None)
    assert source_cand is not None
    assert source_cand.field_name == "r505_1_tanah_bangunan_tambah"
    assert source_cand.current_char == "6"
    assert source_cand.proposed_char == "5"
    assert source_cand.match_type == "VISUAL_CONFUSION_PAIR"
    assert ("6", "5") in CONFUSION_PAIRS
    assert not source_cand.has_dag_conflict

    # Uji dalam mode auto-reconcile
    report_rec = validator.validate_booklet(d1, auto_reconcile=True)
    v18_rec = next((v for v in report_rec.validations if v.rule_id == "REL_18_R505_TOTAL_MUTASI_ASET_TAMBAH"), None)
    assert v18_rec.status == "PASS"
    assert v18_rec.reconciled is True
    assert v18_rec.difference == 0


# =============================================================================
# TEST SUITE 4: SMART RECONCILIATION TARGET RESOLUTION PADA REL_20 (BOOKLET 09)
# =============================================================================

def test_smart_reconciliation_rel20_booklet09_target_resolution():
    """
    Menguji bahwa Smart Reconciliation berhasil mengidentifikasi perbaikan target
    pada REL_20_R607_TOTAL_ASET_KEUANGAN di booklet_09 (350,006 -> 350,800).
    """
    p9 = WORKSPACE_ROOT / "output" / "extractions" / "booklet_09" / "booklet_09_extracted.json"
    assert p9.exists(), f"File {p9} wajib tersedia."

    with open(p9, "r", encoding="utf-8") as f:
        d9 = json.load(f)

    validator = SerutiArithmeticValidator()
    report = validator.validate_booklet(d9, auto_reconcile=False)

    v20 = next((v for v in report.validations if v.rule_id == "REL_20_R607_TOTAL_ASET_KEUANGAN"), None)
    assert v20 is not None
    assert v20.status == "MISMATCH"
    assert v20.difference == -794  # 350,006 - 350,800

    # Periksa kandidat target resolution
    assert len(v20.candidate_resolutions) >= 1
    best_cand = v20.candidate_resolutions[0]
    assert best_cand.location == "TARGET"
    assert best_cand.field_name == "b6_r607_total_aset_keuangan"
    assert best_cand.proposed_value == 350800
    assert best_cand.match_type == "TARGET_HARMONIZATION_MULTI_BOX"
    assert best_cand.plausibility_score >= 0.70
    assert not best_cand.has_dag_conflict

    # Uji dalam mode auto-reconcile
    report_rec = validator.validate_booklet(d9, auto_reconcile=True)
    v20_rec = next((v for v in report_rec.validations if v.rule_id == "REL_20_R607_TOTAL_ASET_KEUANGAN"), None)
    assert v20_rec.status == "PASS"
    assert v20_rec.reconciled is True
    assert v20_rec.target_value == 350800
    assert v20_rec.difference == 0
    assert report_rec.summary.rules_passed >= 14
    assert report_rec.summary.rules_reconciled >= 1


# =============================================================================
# TEST SUITE 5: PROTEKSI DAG ANTI-KONFLIK (CROSS-RELATION CONFLICT GUARD)
# =============================================================================

def test_dag_conflict_rejection(clean_synthetic_extraction):
    """
    Menguji bahwa simulasi kandidat rekonsiliasi yang merusak relasi lain
    yang sebelumnya berstatus PASS akan DITOLAK (has_dag_conflict = True).
    """
    validator = SerutiArithmeticValidator()
    data = copy.deepcopy(clean_synthetic_extraction)

    # Buat anomali pada REL_09: ubah target b4_r463_total_kes_pend_total menjadi 1,500,500 (+500)
    # Sumber: b4_r452_kesehatan_total (500k) + b4_r456_pendidikan_total (1000k) = 1,500,000
    # Relasi hulu REL_05_KESEHATAN_SUBTOTAL_TOTAL berstatus PASS (500k == 500k).
    data["pages"]["hal_4"]["fields"]["b4_r463_total_kes_pend_total"]["value"] = 1500500

    # Buat objek kandidat palsu yang mencoba menyelesaikan REL_09 dengan mengubah
    # source 'b4_r452_kesehatan_total' dari 500,000 menjadi 500,500
    flat_map = validator.build_flat_variable_map(data)
    initial_evals = {}
    for rel in SERUTI_SEMANTIC_RELATIONS:
        initial_evals[rel.relation_id] = validator.evaluate_single_relation(rel, flat_map)

    assert initial_evals["REL_05_KESEHATAN_SUBTOTAL_TOTAL"].status == "PASS"
    assert initial_evals["REL_09_TOTAL_KESEHATAN_PENDIDIKAN_TOTAL"].status == "MISMATCH"

    bad_candidate = ResolutionCandidate(
        location="SOURCE",
        field_name="b4_r452_kesehatan_total",
        box_index=4,
        current_char="0",
        proposed_char="5",
        current_value=500000,
        proposed_value=500500,
        match_type="ARITHMETIC_ONLY",
        plausibility_score=0.50
    )

    rel_09 = next(r for r in SERUTI_SEMANTIC_RELATIONS if r.relation_id == "REL_09_TOTAL_KESEHATAN_PENDIDIKAN_TOTAL")
    has_conflict, reason = validator.check_dag_conflict(bad_candidate, rel_09, flat_map, initial_evals)

    # Kandidat WAJIB memiliki konflik DAG karena merusak REL_05!
    assert has_conflict is True
    assert "REL_05_KESEHATAN_SUBTOTAL_TOTAL" in reason


# =============================================================================
# TEST SUITE 6: PERBANDINGAN MODE AUDIT-ONLY VS AUTO-RECONCILE
# =============================================================================

def test_auto_reconcile_vs_audit_only_mode(clean_synthetic_extraction):
    """
    Menguji bahwa mode audit-only tidak memutasi data mentah,
    sedangkan mode auto-reconcile mengaplikasikan perbaikan yang sah.
    """
    validator = SerutiArithmeticValidator()

    # Buat ketidaksesuaian yang dapat direkonsiliasi (swap 5 <-> 6 pada REL_18)
    data_audit = copy.deepcopy(clean_synthetic_extraction)
    data_audit["pages"]["hal_6"]["fields"]["r505_7_total_mutasi_tambah"]["value"] = 76000000
    data_audit["pages"]["hal_6"]["fields"]["r505_7_total_mutasi_tambah"]["raw_string"] = "___76000000"
    data_audit["pages"]["hal_6"]["fields"]["r505_7_total_mutasi_tambah"]["box_details"] = [
        {"box_index": 0, "char": "_", "is_blank": True},
        {"box_index": 1, "char": "_", "is_blank": True},
        {"box_index": 2, "char": "_", "is_blank": True},
        {"box_index": 3, "char": "7", "conf": 1.0, "is_blank": False},
        {"box_index": 4, "char": "6", "conf": 0.98, "alt_char": "5", "is_blank": False},  # 6 -> 5
        {"box_index": 5, "char": "0", "conf": 1.0, "is_blank": False},
        {"box_index": 6, "char": "0", "conf": 1.0, "is_blank": False},
        {"box_index": 7, "char": "0", "conf": 1.0, "is_blank": False},
        {"box_index": 8, "char": "0", "conf": 1.0, "is_blank": False},
        {"box_index": 9, "char": "0", "conf": 1.0, "is_blank": False},
        {"box_index": 10, "char": "0", "conf": 1.0, "is_blank": False},
    ]

    # Eksekusi mode AUDIT
    rep_audit = validator.validate_booklet(data_audit, auto_reconcile=False)
    v18_audit = next(v for v in rep_audit.validations if v.rule_id == "REL_18_R505_TOTAL_MUTASI_ASET_TAMBAH")
    assert v18_audit.status == "MISMATCH"
    assert v18_audit.reconciled is False
    assert v18_audit.applied_resolution is None
    # Data asli tidak termutasi
    assert data_audit["pages"]["hal_6"]["fields"]["r505_7_total_mutasi_tambah"]["value"] == 76000000

    # Eksekusi mode AUTO-RECONCILE
    data_rec = copy.deepcopy(data_audit)
    rep_rec = validator.validate_booklet(data_rec, auto_reconcile=True)
    v18_rec = next(v for v in rep_rec.validations if v.rule_id == "REL_18_R505_TOTAL_MUTASI_ASET_TAMBAH")
    assert v18_rec.status == "PASS"
    assert v18_rec.reconciled is True
    assert v18_rec.applied_resolution is not None
    assert v18_rec.target_value == 75000000


# =============================================================================
# TEST SUITE 7: IMPUTASI SUBTOTAL KOSONG (OMITTED SUBTOTAL IMPUTATION)
# =============================================================================

def test_omitted_subtotal_imputation(clean_synthetic_extraction):
    """
    Menguji bahwa subtotal kosong yang komponennya terisi akan diidentifikasi
    sebagai UNFILLED_SUBTOTAL dan diimputasi pada mode auto-reconcile.
    """
    validator = SerutiArithmeticValidator()
    data = copy.deepcopy(clean_synthetic_extraction)

    # Kosongkan subtotal R501_TOTAL_UPAH_PER_ART (target: r501_art_01_total)
    data["pages"]["hal_5"]["fields"]["r501_art_01_total"] = {
        "value": 0,
        "raw_string": "_________0",
        "box_details": [{"box_index": i, "char": "_" if i < 9 else "0", "is_blank": (i < 9)} for i in range(10)]
    }

    # Audit mode
    rep_audit = validator.validate_booklet(data, auto_reconcile=False)
    v12_audit = next(v for v in rep_audit.validations if v.rule_id == "REL_12_R501_TOTAL_UPAH_PER_ART")
    assert v12_audit.status == "UNFILLED_SUBTOTAL"
    assert len(v12_audit.candidate_resolutions) >= 1
    cand = v12_audit.candidate_resolutions[0]
    assert cand.match_type == "OMITTED_SUBTOTAL_IMPUTATION"
    assert cand.proposed_value == 4500000  # 4M uang + 500k barang

    # Auto-reconcile mode
    rep_rec = validator.validate_booklet(data, auto_reconcile=True)
    v12_rec = next(v for v in rep_rec.validations if v.rule_id == "REL_12_R501_TOTAL_UPAH_PER_ART")
    assert v12_rec.status == "PASS"
    assert v12_rec.reconciled is True
    assert v12_rec.target_value == 4500000


# =============================================================================
# TEST SUITE 8: INTEGRITAS ARITMETIKA 64-BIT INTEGER (ZERO-JITTER INVARIANT)
# =============================================================================

def test_strict_64bit_integer_math_zero_jitter(clean_synthetic_extraction):
    """
    Memastikan tidak ada floating-point rounding error pada nilai finansial besar
    (miliaran rupiah) yang dapat memicu jitter angka.
    """
    validator = SerutiArithmeticValidator(tolerance=0)
    data = copy.deepcopy(clean_synthetic_extraction)

    # Tetapkan nilai transaksi besar (Rp 55.450.750.325)
    val_a = 30250500125
    val_b = 25200250200
    expected_sum = val_a + val_b

    data["pages"]["hal_6"]["fields"]["r505_1_tanah_bangunan_tambah"]["value"] = val_a
    data["pages"]["hal_6"]["fields"]["r505_2_kendaraan_tambah"]["value"] = val_b
    data["pages"]["hal_6"]["fields"]["r505_7_total_mutasi_tambah"]["value"] = expected_sum

    report = validator.validate_booklet(data, auto_reconcile=False)
    v18 = next(v for v in report.validations if v.rule_id == "REL_18_R505_TOTAL_MUTASI_ASET_TAMBAH")

    assert isinstance(v18.target_value, int)
    assert isinstance(v18.expected_value, int)
    assert v18.target_value == expected_sum
    assert v18.difference == 0
    assert v18.status == "PASS"


# =============================================================================
# TEST SUITE 9: RESOLUSI ALIAS BIDIREKSIONAL
# =============================================================================

def test_bidirectional_field_aliasing():
    """Menguji resolusi variabel dua arah antara penamaan Hal 3 grid dan semantic relations."""
    flat_map = {
        "b4_r425_ongkir_3bln": {"value": 75000},
        "b4_r436_hotel_3bln": {"value": 120000}
    }

    # Cari menggunakan nama di semantic_relations.py
    obj1 = SerutiArithmeticValidator.resolve_field_obj(flat_map, "b4_r425_jasa_pengiriman_3bln")
    assert obj1 is not None
    assert obj1["value"] == 75000

    obj2 = SerutiArithmeticValidator.resolve_field_obj(flat_map, "b4_r436_tekstil_3bln")
    assert obj2 is not None
    assert obj2["value"] == 120000

    # Cari balik menggunakan alias
    flat_map_reverse = {
        "b4_r425_jasa_pengiriman_3bln": {"value": 90000}
    }
    obj_rev = SerutiArithmeticValidator.resolve_field_obj(flat_map_reverse, "b4_r425_ongkir_3bln")
    assert obj_rev is not None
    assert obj_rev["value"] == 90000


# =============================================================================
# TEST SUITE 10: INTEGRASI CLI RUNNER (RUN_SERUTI_VALIDATOR.PY)
# =============================================================================

def test_cli_runner_execution(tmp_path):
    """Menguji eksekusi skrip CLI run_seruti_validator.py secara end-to-end."""
    cli_path = WORKSPACE_ROOT / "run_seruti_validator.py"
    assert cli_path.exists()

    out_test_dir = tmp_path / "val_reports"

    cmd = [
        sys.executable,
        str(cli_path),
        "--booklet", "booklet_09",
        "--output", str(out_test_dir),
        "--format", "both",
        "--auto-reconcile",
        "--summary-only"
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
    assert res.returncode == 0
    assert "RINGKASAN AGREGAT AUDIT ARITMETIKA SELESAI" in res.stdout
    assert "booklet_09" in res.stdout

    # Verifikasi berkas laporan terbentuk
    assert (out_test_dir / "booklet_09_validation.json").exists()
    assert (out_test_dir / "booklet_09_validation_report.txt").exists()
    assert (out_test_dir / "booklet_09_validation_summary.csv").exists()
    assert (out_test_dir / "booklet_09_reconciled.json").exists()
    assert (out_test_dir / "validation_batch_summary.json").exists()
