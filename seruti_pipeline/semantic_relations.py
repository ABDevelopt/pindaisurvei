"""
seruti_pipeline/semantic_relations.py
======================================
Definisi Hubungan Semantik, Formula Aritmetika (Subtotal/Total),
dan Aturan Konsistensi Silang (Cross-Consistency Rules)
Kuesioner BPS Susenas Seruti Inti (VSERUTI26.INTI).

Modul ini memetakan:
1. Hubungan Aritmetika Penjumlahan (Subtotal & Total).
2. Hubungan Konsistensi Silang (Cross-Block / Cross-Page Validations).
3. Aturan Batas Logis (Bounded Checks, misal OOP <= Total).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class SemanticRelation:
    """Representasi hubungan matematis atau logis antar-elemen kuesioner."""
    relation_id: str
    rule_type: str                   # 'sum_vertical', 'sum_horizontal', 'cross_table_match', 'inequality'
    page_num: int                    # Halaman tempat target berada (atau 0 jika cross-page)
    target_field: str                # Field penampung hasil (misal: 'b6_r607_total_aset_keuangan')
    source_fields: List[str]         # Daftar field komponen
    description: str                 # Deskripsi semantik resmi BPS
    tolerance: float = 0.0           # Toleransi pembulatan (biasanya 0)
    operator: str = "="              # '=', '<=', '>='

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "rule_type": self.rule_type,
            "page_num": self.page_num,
            "target_field": self.target_field,
            "source_fields": self.source_fields,
            "operator": self.operator,
            "description": self.description,
            "tolerance": self.tolerance
        }


# =============================================================================
# KATALOG 21 ATURAN SEMANTIK RESMI SERUTI INTI (VSERUTI26.INTI)
# =============================================================================

SERUTI_SEMANTIC_RELATIONS: List[SemanticRelation] = [
    # -------------------------------------------------------------------------
    # 1. KONSISTENSI SILANG: WILAYAH (HAL 1) <-> ROSTER ART (HAL 2)
    # -------------------------------------------------------------------------
    SemanticRelation(
        relation_id="REL_01_JUMLAH_ART",
        rule_type="cross_table_match",
        page_num=0,
        target_field="b1_r108_nus",
        source_fields=["p2_roster_art_count"],
        description="Jumlah anggota rumah tangga yang tercatat di Blok III harus konsisten dan terisi lengkap."
    ),
    SemanticRelation(
        relation_id="REL_02_KRT_NOMOR_URUT_01",
        rule_type="cross_table_match",
        page_num=2,
        target_field="b3_art_01_hub_krt",
        source_fields=["const_1"],
        description="ART nomor urut 01 pada Blok III wajib berstatus Kepala Rumah Tangga (Kode 1)."
    ),

    # -------------------------------------------------------------------------
    # 2. PENGELUARAN BUKAN MAKANAN (HAL 3)
    # -------------------------------------------------------------------------
    SemanticRelation(
        relation_id="REL_03_TOTAL_BUKAN_MAKANAN_SEBULAN",
        rule_type="sum_vertical",
        page_num=3,
        target_field="b4_r451_total_bukan_makanan_sebulan",
        source_fields=[
            "b4_r415_pembersih_sebulan",
            "b4_r416_jasa_domestik_sebulan",
            "b4_r417_transport_sebulan",
            "b4_r419_bbm_sebulan",
            "b4_r426_komunikasi_sebulan",
            "b4_r428_pulsa_internet_sebulan",
            "b4_r437_jasa_lainnya_sebulan",
            "b4_r438_perawatan_diri_sebulan",
            "b4_r439_salon_sebulan"
        ],
        description="Rincian 451 (Total Bukan Makanan Sebulan) adalah penjumlahan seluruh komoditas bukan makanan sebulan."
    ),
    SemanticRelation(
        relation_id="REL_04_TOTAL_BUKAN_MAKANAN_3BLN",
        rule_type="sum_vertical",
        page_num=3,
        target_field="b4_r451_total_bukan_makanan_3bln",
        source_fields=[
            "b4_r417_transport_3bln",
            "b4_r418_kendaraan_3bln",
            "b4_r420_servis_kendaraan_3bln",
            "b4_r421_sewa_tol_parkir_3bln",
            "b4_r422_angkutan_darat_3bln",
            "b4_r423_angkutan_air_3bln",
            "b4_r424_angkutan_udara_3bln",
            "b4_r425_jasa_pengiriman_3bln",
            "b4_r427_pos_paket_3bln",
            "b4_r429_rekreasi_3bln",
            "b4_r430_hiburan_3bln",
            "b4_r431_olahraga_3bln",
            "b4_r432_buku_majalah_3bln",
            "b4_r433_paket_wisata_3bln",
            "b4_r434_pakaian_3bln",
            "b4_r435_alas_kaki_3bln",
            "b4_r436_tekstil_3bln",
            "b4_r437_jasa_lainnya_3bln",
            "b4_r440_perlengkapan_pribadi_3bln",
            "b4_r441_admin_bank_3bln",
            "b4_r442_jasa_dokumen_3bln",
            "b4_r443_gadai_asuransi_3bln",
            "b4_r444_pajak_3bln",
            "b4_r445_premi_asuransi_3bln",
            "b4_r446_denda_3bln",
            "b4_r447_bunga_konsumsi_3bln",
            "b4_r448_sumbangan_3bln",
            "b4_r449_zakat_infaq_3bln",
            "b4_r450_pesta_3bln"
        ],
        description="Rincian 451 (Total Bukan Makanan Tiga Bulan) adalah penjumlahan seluruh komoditas bukan makanan 3 bulan."
    ),

    # -------------------------------------------------------------------------
    # 3. KESEHATAN DAN PENDIDIKAN (HAL 4)
    # -------------------------------------------------------------------------
    SemanticRelation(
        relation_id="REL_05_KESEHATAN_SUBTOTAL_TOTAL",
        rule_type="sum_vertical",
        page_num=4,
        target_field="b4_r452_kesehatan_total",
        source_fields=["b4_r453_obat_total", "b4_r454_rawat_jalan_total", "b4_r455_rawat_inap_total"],
        description="Rincian 452 (Kesehatan Total) adalah jumlah dari Rincian 453 (Obat) + 454 (Rawat Jalan) + 455 (Rawat Inap)."
    ),
    SemanticRelation(
        relation_id="REL_06_KESEHATAN_SUBTOTAL_OOP",
        rule_type="sum_vertical",
        page_num=4,
        target_field="b4_r452_kesehatan_oop",
        source_fields=["b4_r453_obat_oop", "b4_r454_rawat_jalan_oop", "b4_r455_rawat_inap_oop"],
        description="Rincian 452 (Kesehatan OOP) adalah jumlah dari Rincian 453 OOP + 454 OOP + 455 OOP."
    ),
    SemanticRelation(
        relation_id="REL_07_PENDIDIKAN_SUBTOTAL_TOTAL",
        rule_type="sum_vertical",
        page_num=4,
        target_field="b4_r456_pendidikan_total",
        source_fields=[
            "b4_r457_paud_total",
            "b4_r458_sd_total",
            "b4_r459_smp_total",
            "b4_r460_sma_total",
            "b4_r461_pt_total",
            "b4_r462_kursus_total"
        ],
        description="Rincian 456 (Pendidikan Total) adalah jumlah dari Rincian 457 s.d. 462 (PAUD, SD, SMP, SMA, PT, Kursus)."
    ),
    SemanticRelation(
        relation_id="REL_08_PENDIDIKAN_SUBTOTAL_OOP",
        rule_type="sum_vertical",
        page_num=4,
        target_field="b4_r456_pendidikan_oop",
        source_fields=[
            "b4_r457_paud_oop",
            "b4_r458_sd_oop",
            "b4_r459_smp_oop",
            "b4_r460_sma_oop",
            "b4_r461_pt_oop",
            "b4_r462_kursus_oop"
        ],
        description="Rincian 456 (Pendidikan OOP) adalah jumlah dari Rincian 457 s.d. 462 kolom OOP."
    ),
    SemanticRelation(
        relation_id="REL_09_TOTAL_KESEHATAN_PENDIDIKAN_TOTAL",
        rule_type="sum_vertical",
        page_num=4,
        target_field="b4_r463_total_kes_pend_total",
        source_fields=["b4_r452_kesehatan_total", "b4_r456_pendidikan_total"],
        description="Rincian 463 (Total Kesehatan & Pendidikan) adalah penjumlahan Rincian 452 (Kesehatan) + Rincian 456 (Pendidikan)."
    ),
    SemanticRelation(
        relation_id="REL_10_TOTAL_KESEHATAN_PENDIDIKAN_OOP",
        rule_type="sum_vertical",
        page_num=4,
        target_field="b4_r463_total_kes_pend_oop",
        source_fields=["b4_r452_kesehatan_oop", "b4_r456_pendidikan_oop"],
        description="Rincian 463 OOP adalah penjumlahan Rincian 452 OOP + Rincian 456 OOP."
    ),
    SemanticRelation(
        relation_id="REL_11_OOP_LE_TOTAL_CHECK",
        rule_type="inequality",
        page_num=4,
        target_field="b4_r463_total_kes_pend_oop",
        source_fields=["b4_r463_total_kes_pend_total"],
        operator="<=",
        description="Biaya Out of Pocket (OOP) tidak boleh melampaui Total Pengeluaran (OOP <= Total)."
    ),

    # -------------------------------------------------------------------------
    # 4. PENDAPATAN UPAH/GAJI & USAHA (HAL 5)
    # -------------------------------------------------------------------------
    SemanticRelation(
        relation_id="REL_12_R501_TOTAL_UPAH_PER_ART",
        rule_type="sum_horizontal",
        page_num=5,
        target_field="r501_art_01_total",
        source_fields=["r501_art_01_uang", "r501_art_01_barang"],
        description="Total upah per ART adalah penjumlahan upah uang dan upah barang."
    ),
    SemanticRelation(
        relation_id="REL_13_R501_JUMLAH_UPAH_SELURUH_ART",
        rule_type="sum_vertical",
        page_num=5,
        target_field="r501_subtotal_upah_uang",
        source_fields=[
            "r501_art_01_uang",
            "r501_art_02_uang",
            "r501_art_03_uang",
            "r501_art_04_uang",
            "r501_art_05_uang"
        ],
        description="Baris JUMLAH Upah Uang R501 adalah penjumlahan seluruh upah uang ART."
    ),
    SemanticRelation(
        relation_id="REL_14_R503_TOTAL_PENDAPATAN_KEPEMILIKAN",
        rule_type="sum_vertical",
        page_num=5,
        target_field="r503_4_total_diterima",
        source_fields=["r503_1_bunga_diterima", "r503_2_dividen_diterima", "r503_3_sewa_tanah_diterima"],
        description="Total Pendapatan Kepemilikan (Diterima) adalah penjumlahan Rincian 503.1 + 503.2 + 503.3."
    ),
    SemanticRelation(
        relation_id="REL_15_R503_TOTAL_BIAYA_KEPEMILIKAN",
        rule_type="sum_vertical",
        page_num=5,
        target_field="r503_4_total_dibayar",
        source_fields=["r503_1_bunga_dibayar", "r503_3_sewa_tanah_dibayar"],
        description="Total Biaya Kepemilikan (Dibayar) adalah penjumlahan Rincian 503.1 + 503.3 kolom dibayar."
    ),

    # -------------------------------------------------------------------------
    # 5. TRANSFER BERJALAN & MUTASI ASET (HAL 6)
    # -------------------------------------------------------------------------
    SemanticRelation(
        relation_id="REL_16_R504_TOTAL_TRANSFER_DITERIMA_BARANG",
        rule_type="sum_vertical",
        page_num=6,
        target_field="r504_5_total_terima_barang",
        source_fields=["r504_1b_terima_barang", "r504_3_swasta_terima_barang"],
        description="Total Transfer Berjalan Diterima (Barang) adalah penjumlahan Rincian 504.1b + 504.3."
    ),
    SemanticRelation(
        relation_id="REL_17_R504_TOTAL_TRANSFER_DIBERIKAN_UANG",
        rule_type="sum_vertical",
        page_num=6,
        target_field="r504_5_total_beri_uang",
        source_fields=["r504_2_remitansi_beri_uang", "r504_3_swasta_beri_uang"],
        description="Total Transfer Berjalan Diberikan (Uang) adalah penjumlahan Rincian 504.2 + 504.3."
    ),
    SemanticRelation(
        relation_id="REL_18_R505_TOTAL_MUTASI_ASET_TAMBAH",
        rule_type="sum_vertical",
        page_num=6,
        target_field="r505_7_total_mutasi_tambah",
        source_fields=["r505_1_tanah_bangunan_tambah", "r505_2_kendaraan_tambah"],
        description="Total Mutasi Aset (Penambahan/Pembelian) adalah penjumlahan baris 1 s.d. 6 Rincian 505."
    ),
    SemanticRelation(
        relation_id="REL_19_R505_TOTAL_MUTASI_ASET_KURANG",
        rule_type="sum_vertical",
        page_num=6,
        target_field="r505_7_total_mutasi_kurang",
        source_fields=["r505_1_tanah_bangunan_kurang", "r505_2_kendaraan_kurang"],
        description="Total Mutasi Aset (Pengurangan/Penjualan) adalah penjumlahan baris 1 s.d. 6 Rincian 505."
    ),

    # -------------------------------------------------------------------------
    # 6. ASET KEUANGAN & KEWAJIBAN (HAL 7)
    # -------------------------------------------------------------------------
    SemanticRelation(
        relation_id="REL_20_R607_TOTAL_ASET_KEUANGAN",
        rule_type="sum_vertical",
        page_num=7,
        target_field="b6_r607_total_aset_keuangan",
        source_fields=[
            "b6_r601_uang_tunai",
            "b6_r602a_tabungan_bank",
            "b6_r602b_deposito",
            "b6_r602c_giro",
            "b6_r603_saham",
            "b6_r604a_reksadana",
            "b6_r604b_obligasi",
            "b6_r604c_aset_kripto",
            "b6_r605a_emas_murni",
            "b6_r605b_valuta_asing",
            "b6_r606_aset_lainnya"
        ],
        description="Rincian 607 (Total Aset Keuangan) adalah penjumlahan saldo Rincian 601 s.d. 606."
    ),
    SemanticRelation(
        relation_id="REL_21_R611_TOTAL_KEWAJIBAN_UTANG",
        rule_type="sum_vertical",
        page_num=7,
        target_field="b6_r611_total_kewajiban",
        source_fields=[
            "b6_r608a_kpr",
            "b6_r608b_kredit_kendaraan",
            "b6_r608c_kredit_lain",
            "b6_r609_pinjaman_bank",
            "b6_r610_pinjaman_nonbank",
            "b6_r611a_utang_perorangan",
            "b6_r611b_utang_pinjol"
        ],
        description="Rincian 611 (Total Kewajiban/Pinjaman) adalah penjumlahan saldo Rincian 608a s.d. 611b."
    ),
]


def get_relations_for_page(page_num: int) -> List[Dict[str, Any]]:
    """Mendapatkan daftar aturan relasi semantik yang relevan untuk nomor halaman tertentu."""
    rels = [r.to_dict() for r in SERUTI_SEMANTIC_RELATIONS if r.page_num == page_num or r.page_num == 0]
    return rels


def get_all_relations() -> List[Dict[str, Any]]:
    """Mendapatkan seluruh katalog aturan relasi semantik kuesioner Seruti Inti."""
    return [r.to_dict() for r in SERUTI_SEMANTIC_RELATIONS]
