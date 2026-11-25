"""
seruti_pipeline/validator.py
============================
Mesin Validasi Konsistensi Aritmetika dan Rekonsiliasi Cerdas (Smart Reconciliation)
untuk Kuesioner BPS Susenas Seruti Inti (VSERUTI26.INTI).

Kepatuhan Mutlak:
- UU No. 16 Tahun 1997 tentang Statistik
- UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP)
- Zero PII Cloud Egress: Tidak mengekspos data pribadi individual
- Zero-Jitter Invariant: Seluruh kalkulasi finansial Rupiah menggunakan 64-bit integer
  tanpa pembulatan floating-point sembarangan.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import copy
from datetime import datetime

from seruti_pipeline.semantic_relations import (
    SemanticRelation,
    SERUTI_SEMANTIC_RELATIONS,
)


# =============================================================================
# KONSTANTA MATRIKS AMBIGUITAS & ALIAS BIDIREKSIONAL
# =============================================================================

# Matriks ambiguitas visual tulisan tangan (Visual Confusion Pairs) resmi BPS
CONFUSION_PAIRS: Set[Tuple[str, str]] = {
    ("5", "6"), ("6", "5"),
    ("3", "8"), ("8", "3"),
    ("0", "6"), ("6", "0"),
    ("1", "7"), ("7", "1"),
    ("2", "7"), ("7", "2"),
    ("4", "9"), ("9", "4"),
    ("0", "8"), ("8", "0"),
    ("1", "9"), ("9", "1"),
    ("3", "5"), ("5", "3"),
    ("0", "9"), ("9", "0"),
}

CONFUSION_MAP: Dict[str, List[str]] = {
    "0": ["6", "8", "9"],
    "1": ["7", "4", "9"],
    "2": ["7", "3"],
    "3": ["8", "5", "9"],
    "4": ["9", "1"],
    "5": ["6", "3"],
    "6": ["0", "5", "8"],
    "7": ["1", "2"],
    "8": ["3", "0", "6"],
    "9": ["4", "1", "7", "0"],
}

# Pemetaan Alias Bidireksional (Hal 3 Grid, Hal 4, Hal 5, Hal 6, Hal 7 vs semantic_relations.py)
FIELD_ALIASES: Dict[str, List[str]] = {
    # Hal 3
    "b4_r425_jasa_pengiriman_3bln": ["b4_r425_ongkir_3bln"],
    "b4_r427_pos_paket_3bln": ["b4_r427_alat_komunikasi_3bln"],
    "b4_r429_rekreasi_3bln": ["b4_r430_rekreasi_3bln"],
    "b4_r430_hiburan_3bln": ["b4_r431_alat_olahraga_3bln"],
    "b4_r431_olahraga_3bln": ["b4_r432_tiket_wisata_3bln"],
    "b4_r432_buku_majalah_3bln": ["b4_r434_buku_tulis_3bln"],
    "b4_r433_paket_wisata_3bln": ["b4_r435_liburan_umroh_3bln"],
    "b4_r436_tekstil_3bln": ["b4_r436_hotel_3bln"],
    "b4_r440_perlengkapan_pribadi_3bln": ["b4_r440_perhiasan_tas_3bln"],
    "b4_r443_gadai_asuransi_3bln": ["b4_r443_pajak_premi_3bln"],
    "b4_r446_denda_3bln": ["b4_r446_pesta_3bln"],
    "b4_r447_bunga_konsumsi_3bln": ["b4_r447_perkawinan_3bln"],
    "b4_r448_sumbangan_3bln": ["b4_r448_ultah_khitanan_3bln"],
    "b4_r449_zakat_infaq_3bln": ["b4_r449_pemakaman_3bln"],
    "b4_r450_pesta_3bln": ["b4_r450_pesta_lainnya_3bln"],

    # Hal 4
    "b4_r452_kesehatan_total": ["b4_r452_total", "p4_F002"],
    "b4_r452_kesehatan_oop": ["b4_r452_oop", "p4_F003"],
    "b4_r453_obat_total": ["b4_r453_total", "p4_F000"],
    "b4_r453_obat_oop": ["b4_r453_oop", "p4_F001"],
    "b4_r454_rawat_jalan_total": ["b4_r454_total", "p4_F004"],
    "b4_r454_rawat_jalan_oop": ["b4_r454_oop", "p4_F005"],
    "b4_r455_rawat_inap_total": ["b4_r455_total", "p4_F006"],
    "b4_r455_rawat_inap_oop": ["b4_r455_oop", "p4_F007"],
    "b4_r456_pendidikan_total": ["b4_r456_total", "p4_F008"],
    "b4_r456_pendidikan_oop": ["b4_r456_oop", "p4_F009"],
    "b4_r457_paud_total": ["b4_r457_total", "p4_F010"],
    "b4_r457_paud_oop": ["b4_r457_oop", "p4_F011"],
    "b4_r458_sd_total": ["b4_r458_total", "p4_F012"],
    "b4_r458_sd_oop": ["b4_r458_oop", "p4_F013"],
    "b4_r459_smp_total": ["b4_r459_total", "p4_F014"],
    "b4_r459_smp_oop": ["b4_r459_oop", "p4_F015"],
    "b4_r460_sma_total": ["b4_r460_total", "p4_F016"],
    "b4_r460_sma_oop": ["b4_r460_oop", "p4_F017"],
    "b4_r461_pt_total": ["b4_r461_total", "p4_F018"],
    "b4_r461_pt_oop": ["b4_r461_oop", "p4_F019"],
    "b4_r462_kursus_total": ["b4_r462_total", "p4_F020"],
    "b4_r462_kursus_oop": ["b4_r462_oop", "p4_F021"],
    "b4_r463_total_kes_pend_total": ["b4_r463_total", "p4_F022"],
    "b4_r463_total_kes_pend_oop": ["b4_r463_oop", "p4_F023"],

    # Hal 5
    "r501_art_01_uang": ["b5_r501_1_uang", "p5_F003"],
    "r501_art_01_barang": ["b5_r501_1_barang", "p5_F004"],
    "r501_art_01_total": ["b5_r501_1_thr", "p5_F005"],
    "r501_art_02_uang": ["b5_r501_2_uang", "p5_F009"],
    "r501_art_02_barang": ["b5_r501_2_barang", "p5_F012"],
    "r501_art_02_total": ["b5_r501_2_thr", "p5_F015"],
    "r501_art_03_uang": ["b5_r501_3_uang", "p5_F023"],
    "r501_art_03_barang": ["b5_r501_3_barang", "p5_F026"],
    "r501_art_03_total": ["b5_r501_3_thr", "p5_F029"],
    "r501_art_04_uang": ["b5_r501_4_uang", "p5_F037"],
    "r501_art_04_barang": ["b5_r501_4_barang", "p5_F040"],
    "r501_art_04_total": ["b5_r501_4_thr", "p5_F043"],
    "r501_art_05_uang": ["b5_r501_5_uang", "p5_F048"],
    "r501_art_05_barang": ["b5_r501_5_barang", "p5_F049"],
    "r501_art_05_total": ["b5_r501_5_thr", "p5_F050"],
    "r501_subtotal_upah_uang": ["b5_r501_jml_uang", "p5_F053"],
    "r503_1_bunga_diterima": ["b5_r503_1_terima", "p5_F092"],
    "r503_1_bunga_dibayar": ["b5_r503_1_bayar", "p5_F093"],
    "r503_2_dividen_diterima": ["b5_r503_2_terima", "p5_F094"],
    "r503_3_sewa_tanah_diterima": ["b5_r503_3_terima", "p5_F096"],
    "r503_3_sewa_tanah_dibayar": ["b5_r503_4_bayar", "p5_F098"],
    "r503_4_total_diterima": ["b5_r503_jml_terima", "p5_F099"],
    "r503_4_total_dibayar": ["b5_r503_jml_bayar", "p5_F100"],

    # Hal 6
    "r504_1b_terima_barang": ["b6_r504_1b_terima_barang", "p6_F002"],
    "r504_2_remitansi_beri_uang": ["b6_r504_2_bayar_uang", "p6_F007"],
    "r504_3_swasta_terima_barang": ["b6_r504_3_terima_barang", "p6_F010"],
    "r504_3_swasta_beri_uang": ["b6_r504_3_bayar_uang", "p6_F011"],
    "r504_5_total_terima_barang": ["b6_r504_jml_terima_barang", "p6_F014"],
    "r504_5_total_beri_uang": ["b6_r504_jml_bayar_uang", "p6_F015"],
    "r505_1_tanah_bangunan_tambah": ["p6_F040", "b6_r505_6_btt", "b6_r505_1_btt", "p6_F018", "b6_r505_6_lahan", "p6_F043"],
    "r505_2_kendaraan_tambah": ["b6_r505_2_btt", "p6_F023"],
    "r505_7_total_mutasi_tambah": ["b6_r505_7_btt", "p6_F044"],
    "r505_1_tanah_bangunan_kurang": ["b6_r505_4_btt", "p6_F030"],
    "r505_2_kendaraan_kurang": ["b6_r505_5_btt", "p6_F035"],
    "r505_7_total_mutasi_kurang": ["b6_r505_7_bnb", "p6_F045"],

    # Hal 7
    "b6_r607_total_aset_keuangan": ["b6_r600_total_aset", "p7_F020"],
    "b6_r611_total_kewajiban": ["b6_r600_total_kewajiban", "p7_F021"]
}


# =============================================================================
# STRUKTUR DATA HASIL VALIDASI & REKONSILIASI
# =============================================================================

@dataclass
class ResolutionCandidate:
    """Kandidat resolusi rekonsiliasi diskrepansi aritmetika."""
    location: str                   # 'TARGET' atau 'SOURCE'
    field_name: str                 # Nama variabel yang disesuaikan
    box_index: Optional[int]        # Indeks kotak (0-indexed) atau None
    current_char: Optional[str]     # Karakter awal terbaca
    proposed_char: Optional[str]    # Karakter usulan perbaikan
    current_value: Any              # Nilai sebelum koreksi
    proposed_value: Any             # Nilai usulan sesudah koreksi
    match_type: str                 # Tipe deteksi (ALT_CHAR_MATCH, VISUAL_CONFUSION_PAIR, dll)
    conf: float = 1.0               # Confidence model OCR untuk karakter asal
    alt_char: Optional[str] = None  # Karakter alternatif kedua dari OCR
    alt_conf: float = 0.0           # Confidence alternatif kedua
    plausibility_score: float = 0.0 # Skor kelayakan S_rec in [0.0, 1.0]
    reason: str = ""                # Penjelasan naratif perbaikan
    has_dag_conflict: bool = False  # Apakah perbaikan ini merusak relasi lain
    dag_conflict_reason: Optional[str] = None
    differing_boxes: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "field": self.field_name,
            "box_index": self.box_index,
            "current_char": self.current_char,
            "proposed_char": self.proposed_char,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "match_type": self.match_type,
            "conf": round(float(self.conf), 4),
            "alt_char": self.alt_char,
            "alt_conf": round(float(self.alt_conf), 4),
            "plausibility_score": round(float(self.plausibility_score), 4),
            "reason": self.reason,
            "has_dag_conflict": self.has_dag_conflict,
            "dag_conflict_reason": self.dag_conflict_reason,
            "differing_boxes": self.differing_boxes
        }


@dataclass
class RuleValidationResult:
    """Hasil evaluasi 1 aturan relasi semantik."""
    rule_id: str
    rule_type: str
    page_num: int
    description: str
    target_field: str
    target_value: Optional[int]
    expected_value: Optional[int]
    difference: Optional[int]
    status: str                     # 'PASS', 'MISMATCH', 'UNFILLED_SUBTOTAL', 'TARGET_EMPTY'
    reconciled: bool = False
    reconciliation_note: Optional[str] = None
    affected_fields: List[str] = field(default_factory=list)
    candidate_resolutions: List[ResolutionCandidate] = field(default_factory=list)
    applied_resolution: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "page_num": self.page_num,
            "description": self.description,
            "target_field": self.target_field,
            "target_value": self.target_value,
            "expected_value": self.expected_value,
            "difference": self.difference,
            "status": self.status,
            "reconciled": self.reconciled,
            "reconciliation_note": self.reconciliation_note,
            "affected_fields": self.affected_fields,
            "candidate_resolutions": [c.to_dict() for c in self.candidate_resolutions],
            "applied_resolution": self.applied_resolution
        }


@dataclass
class ValidationSummary:
    """Ringkasan agregat audit aritmetika per buklet."""
    total_rules: int = 21
    rules_passed: int = 0
    rules_mismatched: int = 0
    rules_reconciled: int = 0
    pass_rate_pct: float = 0.0
    total_discrepancy_rupiah: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rules": self.total_rules,
            "rules_passed": self.rules_passed,
            "rules_mismatched": self.rules_mismatched,
            "rules_reconciled": self.rules_reconciled,
            "pass_rate_pct": round(float(self.pass_rate_pct), 2),
            "total_discrepancy_rupiah": int(self.total_discrepancy_rupiah)
        }


@dataclass
class ValidationReport:
    """Laporan audit konsistensi aritmetika lengkap."""
    audit_metadata: Dict[str, Any]
    summary: ValidationSummary
    validations: List[RuleValidationResult]
    reconciled_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_metadata": self.audit_metadata,
            "summary": self.summary.to_dict(),
            "validations": [v.to_dict() for v in self.validations]
        }


# =============================================================================
# KELAS UTAMA: SerutiArithmeticValidator
# =============================================================================

class SerutiArithmeticValidator:
    """
    Mesin Validasi dan Rekonsiliasi Konsistensi Aritmetika Kuesioner BPS Seruti Inti.
    
    Fitur Utama:
    1. Evaluasi ke-21 relasi semantik resmi BPS.
    2. Aritmetika integer 64-bit presisi tinggi (Zero-Jitter Invariant).
    3. Resolusi alias variabel bidireksional.
    4. Smart Reconciliation Multi-Hypothesis:
       - Perturbasi delta terpandu pada target & source.
       - Priors confusion matrix tulisan tangan BPS.
       - Imputasi subtotal yang belum diisi (unfilled subtotal).
       - Harmonisasi target multi-box.
    5. Proteksi DAG anti-konflik (Cross-relation DAG conflict guard).
    6. Pemisahan mode audit (audit-only) vs mode auto-reconcile.
    """

    def __init__(self, conf_threshold: float = 0.85, tolerance: int = 0):
        self.conf_threshold: float = float(conf_threshold)
        self.tolerance: int = int(tolerance)
        self._dag_dependencies: Dict[str, Set[str]] = self._build_dag_dependency_graph()

    # -------------------------------------------------------------------------
    # PEMETAAN STRUKTUR VARIABEL & RESOLUSI ALIAS
    # -------------------------------------------------------------------------

    @staticmethod
    def build_flat_variable_map(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Membangun kamus flat dari seluruh variabel kuesioner yang tersebar di `pages.hal_<p>.fields`.
        Kamus ini memudahkan akses O(1) ke seluruh field objek.
        """
        flat_map: Dict[str, Any] = {}
        if not isinstance(extracted_data, dict):
            return flat_map

        pages = extracted_data.get("pages") or {}
        if isinstance(pages, dict):
            for hal_key, hal_obj in pages.items():
                if not isinstance(hal_obj, dict):
                    continue
                fields = hal_obj.get("fields") or {}
                if isinstance(fields, dict):
                    for var_name, field_obj in fields.items():
                        flat_map[var_name] = field_obj

        # Simpan metadata global jika ada
        summary = extracted_data.get("summary")
        if isinstance(summary, dict):
            flat_map["__summary__"] = summary
        return flat_map

    @classmethod
    def resolve_field_obj(cls, flat_map: Dict[str, Any], field_name: str) -> Optional[Dict[str, Any]]:
        """
        Mencari field object dengan dukungan resolusi alias dua arah (bidirectional).
        """
        # 1. Pencarian langsung
        if field_name in flat_map:
            return flat_map[field_name]

        # 2. Pencarian forward alias (canonical -> aliases)
        for alias in FIELD_ALIASES.get(field_name, []):
            if alias in flat_map:
                return flat_map[alias]

        # 3. Pencarian reverse alias (alias -> canonical)
        for canon, aliases in FIELD_ALIASES.items():
            if field_name in aliases and canon in flat_map:
                return flat_map[canon]

        return None

    @staticmethod
    def get_field_integer_value(field_obj: Any) -> Optional[int]:
        """
        Mengekstrak nilai integer (64-bit) dari field object secara aman dan presisi.
        """
        if field_obj is None:
            return None

        if isinstance(field_obj, int):
            return field_obj

        if isinstance(field_obj, float):
            return int(round(field_obj))

        if isinstance(field_obj, str):
            clean = field_obj.replace(".", "").replace(",", "").replace("_", "").strip()
            if not clean:
                return None
            try:
                return int(clean)
            except ValueError:
                return None

        if isinstance(field_obj, dict):
            # Prioritas 1: nilai 'value' terurai
            val = field_obj.get("value")
            if val is not None:
                if isinstance(val, int):
                    return val
                if isinstance(val, float):
                    return int(round(val))
                if isinstance(val, str):
                    clean = val.replace(".", "").replace(",", "").replace("_", "").strip()
                    if clean:
                        try:
                            return int(clean)
                        except ValueError:
                            pass

            # Prioritas 2: konstruksi dari raw_string
            raw_str = field_obj.get("raw_string")
            if isinstance(raw_str, str):
                clean = raw_str.replace(".", "").replace(",", "").replace("_", "").strip()
                if clean:
                    try:
                        return int(clean)
                    except ValueError:
                        return None

        return None

    @staticmethod
    def is_field_all_blank(field_obj: Optional[Dict[str, Any]]) -> bool:
        """
        Memeriksa apakah seluruh kotak pada field bernilai kosong (blank / unfilled).
        """
        if not field_obj or not isinstance(field_obj, dict):
            return True

        boxes = field_obj.get("box_details") or []
        if boxes and isinstance(boxes, list):
            return all(isinstance(b, dict) and b.get("is_blank", True) for b in boxes)

        raw = field_obj.get("raw_string")
        if isinstance(raw, str):
            clean = raw.replace("_", "").replace(" ", "").strip()
            return len(clean) == 0
        return True

    @classmethod
    def is_field_empty_or_zero(cls, field_obj: Optional[Dict[str, Any]]) -> bool:
        """
        Memeriksa apakah field kosong atau hanya berisi default angka nol (unfilled / omitted subtotal).
        Pada kuesioner BPS, subtotal yang tidak diisi seringkali berupa blank dengan kotak terakhir '0'.
        """
        if not field_obj or not isinstance(field_obj, dict):
            return True

        val = field_obj.get("value")
        if val in [None, 0]:
            raw = field_obj.get("raw_string")
            if isinstance(raw, str):
                clean = raw.replace("_", "").replace(" ", "").strip()
                if clean in ["", "0"]:
                    return True
            elif raw is None and val is None:
                return True
            boxes = field_obj.get("box_details") or []
            if boxes and isinstance(boxes, list):
                non_blanks = [b for b in boxes if isinstance(b, dict) and not b.get("is_blank", True)]
                if len(non_blanks) == 0:
                    return True
                if len(non_blanks) == 1 and non_blanks[0].get("char") == "0":
                    return True
        return False

    # -------------------------------------------------------------------------
    # PEMBANGUNAN DAG DEPENDENCY GRAPH
    # -------------------------------------------------------------------------

    def _build_dag_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Membuat Directed Acyclic Graph keterkaitan variabel antar-aturan semantik.
        Memetakan nama field (target maupun source) ke himpunan relation_id yang menggunakannya.
        """
        dag: Dict[str, Set[str]] = {}
        for rel in SERUTI_SEMANTIC_RELATIONS:
            rid = rel.relation_id
            # Daftarkan target
            all_target_names = [rel.target_field] + FIELD_ALIASES.get(rel.target_field, [])
            for t_name in all_target_names:
                dag.setdefault(t_name, set()).add(rid)

            # Daftarkan sumber
            for s in rel.source_fields:
                all_source_names = [s] + FIELD_ALIASES.get(s, [])
                for s_name in all_source_names:
                    dag.setdefault(s_name, set()).add(rid)
        return dag

    # -------------------------------------------------------------------------
    # EVALUASI SATU ATURAN RELASI
    # -------------------------------------------------------------------------

    def evaluate_single_relation(
        self,
        rel: SemanticRelation,
        flat_map: Dict[str, Any],
        roster_art_count: int = 0
    ) -> RuleValidationResult:
        """
        Mengevaluasi 1 aturan relasi semantik secara murni matematis (integer 64-bit).
        """
        rule_id = rel.relation_id
        rule_type = rel.rule_type
        target_name = rel.target_field
        target_obj = self.resolve_field_obj(flat_map, target_name)
        target_val = self.get_field_integer_value(target_obj)

        affected = [target_name] + [s for s in rel.source_fields]

        # ---------------------------------------------------------------------
        # Aturan Khusus 1: Jumlah ART (REL_01)
        # ---------------------------------------------------------------------
        if rule_id == "REL_01_JUMLAH_ART":
            source_sum = int(roster_art_count)
            pass_status = (target_val == source_sum) if target_val is not None else (roster_art_count > 0)
            diff = (target_val - source_sum) if target_val is not None else None
            return RuleValidationResult(
                rule_id=rule_id,
                rule_type=rule_type,
                page_num=rel.page_num,
                description=rel.description,
                target_field=target_name,
                target_value=target_val,
                expected_value=source_sum,
                difference=diff,
                status="PASS" if pass_status else "MISMATCH",
                affected_fields=affected
            )

        # ---------------------------------------------------------------------
        # Aturan Khusus 2: Hubungan KRT ART 01 wajib Kode 1 (REL_02)
        # ---------------------------------------------------------------------
        if rule_id == "REL_02_KRT_NOMOR_URUT_01":
            krt_val = target_obj.get("value") if isinstance(target_obj, dict) else target_obj
            str_val = str(krt_val).strip() if krt_val is not None else ""
            pass_status = str_val in ["1", "01"]
            return RuleValidationResult(
                rule_id=rule_id,
                rule_type=rule_type,
                page_num=rel.page_num,
                description=rel.description,
                target_field=target_name,
                target_value=int(str_val) if str_val.isdigit() else (krt_val if isinstance(krt_val, int) else None),
                expected_value=1,
                difference=0 if pass_status else 1,
                status="PASS" if pass_status else "MISMATCH",
                affected_fields=affected
            )

        # ---------------------------------------------------------------------
        # Aturan Bounded: Inequality (REL_11: OOP <= Total)
        # ---------------------------------------------------------------------
        if rule_type == "inequality":
            src_name = rel.source_fields[0] if rel.source_fields else ""
            src_obj = self.resolve_field_obj(flat_map, src_name)
            src_val = self.get_field_integer_value(src_obj) or 0
            t_val = target_val or 0

            if rel.operator == "<=":
                pass_status = (t_val <= src_val + self.tolerance)
                diff = max(0, t_val - src_val)
            elif rel.operator == ">=":
                pass_status = (t_val >= src_val - self.tolerance)
                diff = max(0, src_val - t_val)
            else:
                diff = abs(t_val - src_val)
                pass_status = (diff <= self.tolerance)

            return RuleValidationResult(
                rule_id=rule_id,
                rule_type=rule_type,
                page_num=rel.page_num,
                description=rel.description,
                target_field=target_name,
                target_value=t_val,
                expected_value=src_val,
                difference=diff,
                status="PASS" if pass_status else "MISMATCH",
                affected_fields=affected
            )

        # ---------------------------------------------------------------------
        # Aturan Penjumlahan Vertikal / Horizontal (Sum Subtotal / Total)
        # ---------------------------------------------------------------------
        source_nums: List[int] = []
        for s in rel.source_fields:
            s_obj = self.resolve_field_obj(flat_map, s)
            s_val = self.get_field_integer_value(s_obj)
            source_nums.append(s_val if s_val is not None else 0)

        computed_sum = sum(source_nums)

        if target_val is not None:
            diff = target_val - computed_sum
            pass_status = (abs(diff) <= self.tolerance)
            if pass_status:
                status_str = "PASS"
            elif target_val == 0 and computed_sum > 0 and self.is_field_empty_or_zero(target_obj):
                status_str = "UNFILLED_SUBTOTAL"
            else:
                status_str = "MISMATCH"
        else:
            diff = -computed_sum
            if computed_sum == 0:
                status_str = "PASS"
                target_val = 0
            else:
                status_str = "TARGET_EMPTY"

        return RuleValidationResult(
            rule_id=rule_id,
            rule_type=rule_type,
            page_num=rel.page_num,
            description=rel.description,
            target_field=target_name,
            target_value=target_val,
            expected_value=computed_sum,
            difference=diff,
            status=status_str,
            affected_fields=affected
        )

    # -------------------------------------------------------------------------
    # SMART RECONCILIATION ENGINE
    # -------------------------------------------------------------------------

    def calculate_plausibility_score(
        self,
        is_alt_match: bool,
        is_confusable: bool,
        conf: float,
        alt_conf: float,
        is_low_conf: bool
    ) -> float:
        """
        Menghitung skor kelayakan plausibilitas S_rec in [0.0, 1.0].
        S_rec = w_alt * I_alt + w_vis * I_vis + w_conf * (1.0 - conf) + w_sec * alt_conf
        """
        w_alt = 0.40
        w_vis = 0.30
        w_conf = 0.20
        w_sec = 0.10

        score = 0.0
        if is_alt_match:
            score += w_alt
        if is_confusable:
            score += w_vis

        # Normalisasi invers confidence (semakin rendah confidence awal, semakin wajar dikoreksi)
        score += w_conf * max(0.0, min(1.0, 1.0 - conf))
        score += w_sec * max(0.0, min(1.0, alt_conf))

        # Baseline plausibilitas untuk koreksi aritmetika valid
        base = 0.35 if (is_alt_match or is_confusable or is_low_conf) else 0.10
        return min(1.0, max(0.0, score + base))

    def reconcile_discrepancy(
        self,
        rel: SemanticRelation,
        init_res: RuleValidationResult,
        flat_map: Dict[str, Any],
        current_evaluations: Dict[str, RuleValidationResult]
    ) -> List[ResolutionCandidate]:
        """
        Mendeteksi multi-hipotesis kandidat resolusi cerdas untuk diskrepansi aritmetika.
        """
        candidates: List[ResolutionCandidate] = []
        rule_id = rel.relation_id
        target_name = rel.target_field
        target_obj = self.resolve_field_obj(flat_map, target_name)
        target_val = init_res.target_value
        expected_val = init_res.expected_value or 0

        # ---------------------------------------------------------------------
        # Hipotesis Khusus REL_02: Kaidah Mutlak KRT ART 01
        # ---------------------------------------------------------------------
        if rule_id == "REL_02_KRT_NOMOR_URUT_01" and init_res.status != "PASS":
            boxes = (target_obj.get("box_details") or []) if isinstance(target_obj, dict) else []
            b0 = boxes[0] if (boxes and isinstance(boxes[0], dict)) else {}
            c_char = b0.get("char", str(target_val))
            conf = b0.get("conf", 1.0)
            alt = b0.get("alt_char")
            alt_conf = b0.get("alt_conf", 0.0)

            is_alt = (alt == "1")
            is_conf = (c_char, "1") in CONFUSION_PAIRS
            score = self.calculate_plausibility_score(is_alt, is_conf, conf, alt_conf, conf < self.conf_threshold)

            cand = ResolutionCandidate(
                location="TARGET",
                field_name=target_name,
                box_index=0 if boxes else None,
                current_char=c_char,
                proposed_char="1",
                current_value=target_val,
                proposed_value=1,
                match_type="ALT_CHAR_MATCH" if is_alt else "VISUAL_CONFUSION_PAIR",
                conf=conf,
                alt_char=alt,
                alt_conf=alt_conf,
                plausibility_score=max(score, 0.95),
                reason="Prediksi ambigu; kaidah mutlak KRT ART 01 mengoreksi ke Kode 1 (Kepala Rumah Tangga)."
            )
            cand.has_dag_conflict, cand.dag_conflict_reason = self.check_dag_conflict(
                cand, rel, flat_map, current_evaluations
            )
            candidates.append(cand)
            return candidates

        # ---------------------------------------------------------------------
        # Hipotesis 1: Imputasi Subtotal yang Dikosongkan (Unfilled Subtotal)
        # ---------------------------------------------------------------------
        if (init_res.status in ["UNFILLED_SUBTOTAL", "TARGET_EMPTY"]) or (
            target_val == 0 and expected_val > 0 and self.is_field_empty_or_zero(target_obj)
        ):
            cand = ResolutionCandidate(
                location="TARGET",
                field_name=target_name,
                box_index=None,
                current_char=None,
                proposed_char=str(expected_val),
                current_value=0,
                proposed_value=expected_val,
                match_type="OMITTED_SUBTOTAL_IMPUTATION",
                conf=1.0,
                alt_char=None,
                alt_conf=0.0,
                plausibility_score=0.95,
                reason=f"Subtotal belum dijumlahkan oleh responden; diimputasi otomatis dari komponen: Rp {expected_val:,}."
            )
            cand.has_dag_conflict, cand.dag_conflict_reason = self.check_dag_conflict(
                cand, rel, flat_map, current_evaluations
            )
            candidates.append(cand)
            return candidates

        if target_val is None or init_res.difference is None or init_res.difference == 0:
            return candidates

        diff = int(init_res.difference)  # diff = target_val - expected_val

        # ---------------------------------------------------------------------
        # Hipotesis 2: Perturbasi 1-Digit pada Kotak TARGET
        # Y_new = Y_read + delta_val ==> kita ingin Y_new = expected_val
        # delta_val = expected_val - target_val = -diff
        # ---------------------------------------------------------------------
        if isinstance(target_obj, dict):
            boxes = target_obj.get("box_details") or []
            raw_str = target_obj.get("raw_string") or ""
            raw_len = len(raw_str)

            for b_idx, b in enumerate(boxes):
                if not isinstance(b, dict) or b.get("is_blank", False):
                    continue
                cur_char = b.get("char")
                if not cur_char or not str(cur_char).isdigit():
                    continue

                conf = float(b.get("conf", 1.0))
                alt_char = b.get("alt_char")
                alt_conf = float(b.get("alt_conf", 0.0))

                power = 10 ** (raw_len - 1 - b_idx)
                if diff % power == 0:
                    delta_digit = - (diff // power)
                    cur_d = int(cur_char)
                    req_d = cur_d + delta_digit

                    if 0 <= req_d <= 9:
                        req_str = str(req_d)
                        is_alt = (req_str == alt_char)
                        is_conf = (cur_char, req_str) in CONFUSION_PAIRS
                        is_low = conf < self.conf_threshold

                        if is_alt:
                            m_type = "ALT_CHAR_MATCH"
                        elif is_conf:
                            m_type = "VISUAL_CONFUSION_PAIR"
                        elif is_low:
                            m_type = "LOW_CONFIDENCE"
                        else:
                            m_type = "ARITHMETIC_ONLY"

                        score = self.calculate_plausibility_score(is_alt, is_conf, conf, alt_conf, is_low)

                        cand = ResolutionCandidate(
                            location="TARGET",
                            field_name=target_name,
                            box_index=b_idx,
                            current_char=cur_char,
                            proposed_char=req_str,
                            current_value=target_val,
                            proposed_value=expected_val,
                            match_type=m_type,
                            conf=conf,
                            alt_char=alt_char,
                            alt_conf=alt_conf,
                            plausibility_score=score,
                            reason=f"Selisih Rp {diff:,} diselesaikan dengan mengganti kotak {b_idx} target ('{cur_char}' -> '{req_str}'; {m_type}) sehingga total menjadi Rp {expected_val:,}."
                        )
                        cand.has_dag_conflict, cand.dag_conflict_reason = self.check_dag_conflict(
                            cand, rel, flat_map, current_evaluations
                        )
                        candidates.append(cand)

        # ---------------------------------------------------------------------
        # Hipotesis 3: Perturbasi 1-Digit pada Kotak KOMPONEN SUMBER (Source Fields)
        # S_new = S_old + delta_val ==> kita ingin sum(sources) = target_val
        # delta_val = diff (karena diff = target_val - expected_val)
        # ---------------------------------------------------------------------
        for s_name in rel.source_fields:
            s_obj = self.resolve_field_obj(flat_map, s_name)
            if not isinstance(s_obj, dict):
                continue

            s_val = self.get_field_integer_value(s_obj) or 0
            boxes = s_obj.get("box_details") or []
            raw_str = s_obj.get("raw_string") or ""
            raw_len = len(raw_str)

            for b_idx, b in enumerate(boxes):
                if not isinstance(b, dict) or b.get("is_blank", False):
                    continue
                cur_char = b.get("char")
                if not cur_char or not str(cur_char).isdigit():
                    continue

                conf = float(b.get("conf", 1.0))
                alt_char = b.get("alt_char")
                alt_conf = float(b.get("alt_conf", 0.0))

                power = 10 ** (raw_len - 1 - b_idx)
                if diff % power == 0:
                    delta_digit = diff // power
                    cur_d = int(cur_char)
                    req_d = cur_d + delta_digit

                    if 0 <= req_d <= 9:
                        req_str = str(req_d)
                        is_alt = (req_str == alt_char)
                        is_conf = (cur_char, req_str) in CONFUSION_PAIRS
                        is_low = conf < self.conf_threshold

                        if is_alt:
                            m_type = "ALT_CHAR_MATCH"
                        elif is_conf:
                            m_type = "VISUAL_CONFUSION_PAIR"
                        elif is_low:
                            m_type = "LOW_CONFIDENCE"
                        else:
                            m_type = "ARITHMETIC_ONLY"

                        new_s_val = s_val + diff
                        score = self.calculate_plausibility_score(is_alt, is_conf, conf, alt_conf, is_low)

                        cand = ResolutionCandidate(
                            location="SOURCE",
                            field_name=s_name,
                            box_index=b_idx,
                            current_char=cur_char,
                            proposed_char=req_str,
                            current_value=s_val,
                            proposed_value=new_s_val,
                            match_type=m_type,
                            conf=conf,
                            alt_char=alt_char,
                            alt_conf=alt_conf,
                            plausibility_score=score,
                            reason=f"Selisih Rp {diff:,} diselesaikan dengan mengganti kotak {b_idx} pada rincian {s_name} ('{cur_char}' -> '{req_str}'; {m_type}) sehingga nilai menjadi Rp {new_s_val:,}."
                        )
                        cand.has_dag_conflict, cand.dag_conflict_reason = self.check_dag_conflict(
                            cand, rel, flat_map, current_evaluations
                        )
                        candidates.append(cand)

        # ---------------------------------------------------------------------
        # Hipotesis 4: Harmonisasi Target Multi-Box / Confusion (misal REL_20 booklet_09)
        # Jika perturbasi 1-digit tidak menemukan kandidat tanpa konflik,
        # periksa apakah target dapat diselaraskan dengan expected_val
        # di mana seluruh digit yang berbeda konsisten dengan CONFUSION_PAIRS.
        # ---------------------------------------------------------------------
        if expected_val > 0 and isinstance(target_obj, dict):
            raw_t = list(target_obj.get("raw_string") or "")
            boxes_t = target_obj.get("box_details") or []
            raw_len = len(raw_t)

            if raw_len > 0 and isinstance(boxes_t, list) and len(boxes_t) == raw_len:
                exp_str = str(expected_val).rjust(raw_len, "_")
                diff_boxes = []
                all_match_priors = True

                for idx, (ct, ce) in enumerate(zip(raw_t, exp_str)):
                    if ct != ce:
                        b = boxes_t[idx]
                        if not isinstance(b, dict):
                            all_match_priors = False
                            break
                        is_conf = (ct, ce) in CONFUSION_PAIRS
                        is_alt = (ce == b.get("alt_char"))
                        is_low = float(b.get("conf", 1.0)) < self.conf_threshold

                        if is_conf or is_alt or is_low:
                            diff_boxes.append({
                                "box_index": idx,
                                "current_char": ct,
                                "proposed_char": ce,
                                "is_confusable": is_conf,
                                "is_alt_match": is_alt,
                                "conf": float(b.get("conf", 1.0))
                            })
                        else:
                            all_match_priors = False
                            break

                # Harmonisasi multi-box untuk 2 s.d. 3 kotak berbeda yang semuanya valid confusion/alt
                if all_match_priors and 2 <= len(diff_boxes) <= 3:
                    cand = ResolutionCandidate(
                        location="TARGET",
                        field_name=target_name,
                        box_index=diff_boxes[0]["box_index"] if len(diff_boxes) == 1 else None,
                        current_char=diff_boxes[0]["current_char"] if len(diff_boxes) == 1 else None,
                        proposed_char=diff_boxes[0]["proposed_char"] if len(diff_boxes) == 1 else None,
                        current_value=target_val,
                        proposed_value=expected_val,
                        match_type="TARGET_HARMONIZATION_MULTI_BOX",
                        conf=min(b["conf"] for b in diff_boxes),
                        alt_char=None,
                        alt_conf=0.0,
                        plausibility_score=0.82,
                        reason=f"Harmonisasi target total dengan komponen (Rp {expected_val:,}); {len(diff_boxes)} digit ambigu selaras dengan matriks visual BPS.",
                        differing_boxes=diff_boxes
                    )
                    cand.has_dag_conflict, cand.dag_conflict_reason = self.check_dag_conflict(
                        cand, rel, flat_map, current_evaluations
                    )
                    candidates.append(cand)

        # Urutkan kandidat berdasarkan skor kelayakan tertinggi
        candidates.sort(key=lambda c: (not c.has_dag_conflict, c.plausibility_score), reverse=True)
        return candidates

    # -------------------------------------------------------------------------
    # PROTEKSI DAG ANTI-KONFLIK (CROSS-RELATION DAG CONFLICT GUARD)
    # -------------------------------------------------------------------------

    def check_dag_conflict(
        self,
        candidate: ResolutionCandidate,
        current_rel: SemanticRelation,
        flat_map: Dict[str, Any],
        current_evaluations: Dict[str, RuleValidationResult]
    ) -> Tuple[bool, Optional[str]]:
        """
        Simulasi substitusi nilai untuk mendeteksi apakah perubahan ini merusak
        relasi semantik lain yang sebelumnya telah berstatus PASS (anti false positive).
        """
        f_name = candidate.field_name
        prop_val = candidate.proposed_value
        field_obj = self.resolve_field_obj(flat_map, f_name)

        if not isinstance(field_obj, dict):
            return False, None

        orig_val = field_obj.get("value")

        # Ambil relasi yang terdampak oleh variabel ini
        affected_rids = self._dag_dependencies.get(f_name, set())

        has_conflict = False
        conflict_reason = None

        try:
            # Simulasi perubahan nilai secara lokal
            field_obj["value"] = prop_val

            for rid in affected_rids:
                if rid == current_rel.relation_id:
                    continue  # Relasi asal yang sedang diperbaiki tidak dihitung konflik

                prev_res = current_evaluations.get(rid)
                if not prev_res or prev_res.status != "PASS":
                    continue  # Hanya guard terhadap relasi yang sebelumnya SUDAH LULUS (PASS)

                rel_obj = next((r for r in SERUTI_SEMANTIC_RELATIONS if r.relation_id == rid), None)
                if not rel_obj:
                    continue

                test_res = self.evaluate_single_relation(rel_obj, flat_map)
                if test_res.status != "PASS":
                    has_conflict = True
                    conflict_reason = (
                        f"Perubahan {f_name}={prop_val:,} merusak relasi {rid} "
                        f"yang sebelumnya PASS (selisih baru: Rp {test_res.difference:,})."
                    )
                    break
        finally:
            # Kembalikan nilai asli
            field_obj["value"] = orig_val

        return has_conflict, conflict_reason

    # -------------------------------------------------------------------------
    # VALIDASI & REKONSILIASI PENUH BUKLET
    # -------------------------------------------------------------------------

    def validate_booklet(
        self,
        extracted_data: Dict[str, Any],
        auto_reconcile: bool = False
    ) -> ValidationReport:
        """
        Mengeksekusi validasi konsistensi aritmetika terhadap seluruh 21 aturan semantik.
        
        Parameter:
            extracted_data: dictionary kuesioner dari `_extracted.json`
            auto_reconcile: flag boolean. Jika True, otomatis menerapkan perbaikan
                            terbaik yang tidak memiliki konflik DAG ke memori data.
        """
        if not isinstance(extracted_data, dict):
            extracted_data = {}
        flat_map = self.build_flat_variable_map(extracted_data)
        summary_meta = extracted_data.get("summary") or {}
        if not isinstance(summary_meta, dict):
            summary_meta = {}
        booklet_id = summary_meta.get("booklet_id") or "booklet_unknown"
        roster_raw = summary_meta.get("roster_art_count")
        try:
            roster_art_count = int(roster_raw) if roster_raw is not None else 0
        except (ValueError, TypeError):
            roster_art_count = 0
        if roster_art_count < 0:
            roster_art_count = 0

        # Evaluasi awal ke-21 relasi
        initial_evaluations: Dict[str, RuleValidationResult] = {}
        for rel in SERUTI_SEMANTIC_RELATIONS:
            res = self.evaluate_single_relation(rel, flat_map, roster_art_count)
            initial_evaluations[rel.relation_id] = res

        final_validations: List[RuleValidationResult] = []
        rules_reconciled = 0

        # Iterasi pemeriksaan dan smart reconciliation
        for rel in SERUTI_SEMANTIC_RELATIONS:
            cur_res = initial_evaluations[rel.relation_id]

            if cur_res.status != "PASS":
                candidates = self.reconcile_discrepancy(rel, cur_res, flat_map, initial_evaluations)
                cur_res.candidate_resolutions = candidates

                if auto_reconcile:
                    # Pilih kandidat terbaik tanpa konflik DAG
                    valid_candidates = [c for c in candidates if not c.has_dag_conflict]
                    if valid_candidates:
                        best = valid_candidates[0]

                        # Ambiguity Guard: Jika terdapat >= 2 kandidat teratas dengan skor seimbang (tie)
                        # yang mengusulkan perubahan berbeda, JANGAN auto-reconcile!
                        is_ambiguous_tie = False
                        if len(valid_candidates) > 1:
                            second = valid_candidates[1]
                            if abs(best.plausibility_score - second.plausibility_score) < 1e-4:
                                # Periksa apakah mengusulkan mutasi berbeda (beda field atau beda nilai)
                                if best.field_name != second.field_name or best.proposed_value != second.proposed_value:
                                    is_ambiguous_tie = True

                        if is_ambiguous_tie:
                            cur_res.reconciled = False
                            cur_res.reconciliation_note = (
                                f"Ambigu: Ditemukan {len(valid_candidates)} kandidat resolusi dengan skor seimbang "
                                f"(S={best.plausibility_score:.2f}). Rekonsiliasi otomatis ditangguhkan untuk verifikasi manual."
                            )
                        elif best.plausibility_score >= 0.50:
                            target_obj = self.resolve_field_obj(flat_map, best.field_name)
                            if isinstance(target_obj, dict):
                                target_obj["value"] = best.proposed_value

                                # Perbarui raw_string dan box jika ada box_index
                                if best.box_index is not None and best.proposed_char:
                                    raw = list(target_obj.get("raw_string") or "")
                                    if best.box_index < len(raw):
                                        raw[best.box_index] = best.proposed_char
                                        target_obj["raw_string"] = "".join(raw)
                                    boxes = target_obj.get("box_details") or []
                                    if isinstance(boxes, list) and best.box_index < len(boxes):
                                        if isinstance(boxes[best.box_index], dict):
                                            boxes[best.box_index]["char"] = best.proposed_char

                            # Evaluasi ulang status relasi
                            re_eval = self.evaluate_single_relation(rel, flat_map, roster_art_count)
                            cur_res.status = "PASS" if re_eval.status == "PASS" else cur_res.status
                            cur_res.reconciled = True
                            cur_res.reconciliation_note = best.reason
                            cur_res.applied_resolution = best.to_dict()
                            cur_res.target_value = re_eval.target_value
                            cur_res.difference = re_eval.difference
                            rules_reconciled += 1

            final_validations.append(cur_res)

        # Hitung statistik agregat akhir
        rules_passed = sum(1 for v in final_validations if v.status == "PASS")
        rules_mismatched = sum(1 for v in final_validations if v.status != "PASS")
        pass_rate = (rules_passed / len(final_validations)) * 100.0 if final_validations else 0.0
        total_disc = sum(abs(v.difference) for v in final_validations if v.difference and v.status != "PASS")

        summary = ValidationSummary(
            total_rules=len(final_validations),
            rules_passed=rules_passed,
            rules_mismatched=rules_mismatched,
            rules_reconciled=rules_reconciled,
            pass_rate_pct=pass_rate,
            total_discrepancy_rupiah=total_disc
        )

        audit_meta = {
            "booklet_id": booklet_id,
            "timestamp": datetime.now().isoformat(),
            "validator_version": "1.0.0",
            "execution_mode": "auto_reconcile" if auto_reconcile else "audit_only",
            "conf_threshold": self.conf_threshold,
            "tolerance_rupiah": self.tolerance,
            "zero_pii_compliant": True
        }

        reconciled_export = copy.deepcopy(extracted_data) if auto_reconcile else None

        return ValidationReport(
            audit_metadata=audit_meta,
            summary=summary,
            validations=final_validations,
            reconciled_data=reconciled_export
        )
