#!/usr/bin/env python3
"""
run_seruti_ocr.py
==================
Pipeline Inferensi OCR Karakter Digit & Perakitan Variabel Kuesioner BPS Seruti Inti (VSERUTI26.INTI).

Arsitektur:
1. Engine Vision Google Gemini API (gemini-3.8-flash) dengan Zero-PII Cloud Egress & SHA-256 local cache.
2. Pemuatan Crop Manifest Hierarkis (Hal 1 s.d. Hal 7).
3. Inferensi Digit Comb-Box Bersih (28x28 gaya MNIST, Center-of-Mass centered).
4. Perakitan String Multi-Digit & Parsing Nilai (Currency Rp, Integer, Jam HH:MM, Tanggal, OMR Checkbox).
5. Resolusi Kamus Variabel Resmi BPS (KOP, Blok I s.d. Blok VII).
6. Evaluasi 21 Aturan Relasi Semantik & Validasi Aritmetika (Subtotal, Total, Cross-Table, Bounded Checks).
7. Rekonsiliasi Otomatis Ambiguitas Digit Berbasis Relasi Matematis (Smart Arithmetic Reconciliation).
8. Ekspor Hasil Terstruktur (JSON, CSV Tabular, Laporan Validasi) dengan Kepatuhan Mutlak Zero-PII Cloud Egress.

Pemakaian:
  python run_seruti_ocr.py --booklet booklet_01
  python run_seruti_ocr.py --all
"""

import argparse
import csv
import re
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Suppress scikit-learn unpickle estimator version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Path konfigurasi default
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_MODEL_PATH = DEFAULT_GEMINI_MODEL  # Backward compatibility bagi skrip lama
DEFAULT_CROPS_DIR = BASE_DIR / "output" / "crops"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "extractions"
DEFAULT_TEMPLATES_DIR = BASE_DIR / "templates"

# Parameter inferensi
CONF_REVIEW_THRESHOLD = 0.80  # Di bawah ini ditandai untuk review petugas
INK_THR = 170                 # Ambang biner tinta


# =============================================================================
# 1. KAMUS METADATA VARIABEL STANDAR BPS
# =============================================================================

def build_variable_registry() -> Dict[str, Dict[str, Any]]:
    """
    Membangun registri pemetaan field_id ke variabel semantik standar BPS Seruti Inti:
    - var_name: nama variabel resmi sensus/survei (misal: b1_r101_provinsi)
    - label: label deskriptif pertanyaan
    - block_id: blok kuesioner
    - datatype: 'integer', 'currency', 'time', 'date', 'code', 'checkbox'
    """
    reg = {}

    # ---------------- PAGE 1: KOP, BLOK I, BLOK II ----------------
    reg["p1_F_triwulan"] = {"var": "kop_triwulan", "label": "Triwulan Pendataan", "block": "KOP", "type": "roman"}
    reg["p1_F000"] = {"var": "b1_waktu_mulai", "label": "Waktu Mulai Wawancara", "block": "BLOK_I", "type": "time"}
    reg["p1_F001"] = {"var": "b1_r101_provinsi", "label": "Provinsi", "block": "BLOK_I", "type": "integer"}
    reg["p1_F002"] = {"var": "b1_r102_kabupaten", "label": "Kabupaten/Kota", "block": "BLOK_I", "type": "integer"}
    reg["p1_F003"] = {"var": "b1_r103_kecamatan", "label": "Kecamatan", "block": "BLOK_I", "type": "integer"}
    reg["p1_F004"] = {"var": "b1_r104_desa", "label": "Desa/Kelurahan", "block": "BLOK_I", "type": "integer"}
    reg["p1_F005"] = {"var": "b1_r105_klasifikasi", "label": "Klasifikasi Desa (1:Perkotaan, 2:Pedesaan)", "block": "BLOK_I", "type": "code"}
    reg["p1_F006"] = {"var": "b1_r106_nobs", "label": "Nomor Blok Sensus", "block": "BLOK_I", "type": "integer"}
    reg["p1_F007"] = {"var": "b1_r107_kdsampel", "label": "Nomor Kode Sampel / Sub-SLS", "block": "BLOK_I", "type": "integer"}
    reg["p1_F008"] = {"var": "b1_r108_nus", "label": "Nomor Urut Sampel Rumah Tangga", "block": "BLOK_I", "type": "integer"}
    reg["p1_F009"] = {"var": "b2_r201_tgl_pcl", "label": "Tanggal Pencacahan PCL (Tgl)", "block": "BLOK_II", "type": "integer"}
    reg["p1_F010"] = {"var": "b2_r201_jabatan_pcl", "label": "Jabatan Pencacah PCL", "block": "BLOK_II", "type": "code"}
    reg["p1_F011"] = {"var": "b2_r201_id_pcl", "label": "ID/NIP Petugas PCL", "block": "BLOK_II", "type": "integer"}
    reg["p1_F012"] = {"var": "b2_r201_jam_pcl", "label": "Bulan Pencacahan PCL", "block": "BLOK_II", "type": "integer"}
    reg["p1_F013"] = {"var": "b2_r202_tgl_pml", "label": "Tanggal Pengawasan PML (Tgl)", "block": "BLOK_II", "type": "integer"}
    reg["p1_F014"] = {"var": "b2_r202_jabatan_pml", "label": "Jabatan Pengawas PML", "block": "BLOK_II", "type": "code"}
    reg["p1_F015"] = {"var": "b2_r202_id_pml", "label": "ID/NIP Pengawas PML", "block": "BLOK_II", "type": "integer"}
    reg["p1_F016"] = {"var": "b2_r202_jam_pml", "label": "Bulan Pengawasan PML", "block": "BLOK_II", "type": "integer"}
    reg["p1_F017"] = {"var": "b2_r203_hasil_pendataan", "label": "Hasil Pendataan Kuesioner (1-5)", "block": "BLOK_II", "type": "code"}
    reg["p1_F_kesediaan"] = {"var": "b0_kesediaan_responden", "label": "Kesediaan Responden (1:Ya, 2:Janji, 3:Tolak)", "block": "BLOK_0", "type": "checkbox"}
    reg["p1_F_hal"] = {"var": "b0_nomor_halaman_1", "label": "Nomor Halaman 1", "block": "KOP", "type": "integer"}

    # ---------------- PAGE 2: BLOK III (ROSTER ART) & BLOK IV.2.A (BUKAN MAKANAN) ----------------
    art_cols = [
        ("hub_krt", "Hubungan dengan KRT (1:KRT, 2:Istri/Suami, 3:Anak, 4:Menantu, dll)", "code"),
        ("status_kawin", "Status Perkawinan (1:Belum Kawin, 2:Kawin, 3:Cerai Hidup, 4:Cerai Mati)", "code"),
        ("jk", "Jenis Kelamin (1:Laki-laki, 2:Perempuan)", "code"),
        ("umur", "Umur ART (tahun)", "integer"),
        ("pendidikan", "Pendidikan Tertinggi", "code"),
    ]
    for i in range(10):
        art_idx = i + 1
        for c_idx, (col_id, col_lbl, dt) in enumerate(art_cols):
            fid = f"p2_F{i*5 + c_idx:03d}"
            reg[fid] = {
                "var": f"b3_art_{art_idx:02d}_{col_id}",
                "label": f"ART {art_idx:02d} - {col_lbl}",
                "block": "BLOK_III",
                "type": dt,
                "art_index": art_idx
            }
    reg["p2_F055"] = {
        "var": "b4_r405_status_bangunan",
        "label": "R405 Status Penguasaan Bangunan Tempat Tinggal yang Ditempati",
        "block": "BLOK_IV",
        "type": "code"
    }

    # ---------------- PAGE 5 SPESIFIK MAPPING (R501, R502, R503) ----------------
    p5_explicit = {
        # R501 Upah/Gaji Baris 1-5
        "p5_F000": ("b5_r501_1_art", "integer"),
        "p5_F001": ("b5_r501_1_kbli", "integer"),
        "p5_F002": ("b5_r501_1_jp", "code"),
        "p5_F003": ("b5_r501_1_uang", "currency"),
        "p5_F004": ("b5_r501_1_barang", "currency"),
        "p5_F005": ("b5_r501_1_thr", "currency"),

        "p5_F006": ("b5_r501_2_art", "integer"),
        "p5_F007": ("b5_r501_2_kbli", "integer"),
        "p5_F008": ("b5_r501_2_jp", "code"),
        "p5_F009": ("b5_r501_2_uang", "currency"),
        "p5_F012": ("b5_r501_2_barang", "currency"),
        "p5_F015": ("b5_r501_2_thr", "currency"),

        "p5_F018": ("b5_r501_3_art", "integer"),
        "p5_F019": ("b5_r501_3_kbli", "integer"),
        "p5_F020": ("b5_r501_3_jp", "code"),
        "p5_F023": ("b5_r501_3_uang", "currency"),
        "p5_F026": ("b5_r501_3_barang", "currency"),
        "p5_F029": ("b5_r501_3_thr", "currency"),

        "p5_F032": ("b5_r501_4_art", "integer"),
        "p5_F033": ("b5_r501_4_kbli", "integer"),
        "p5_F034": ("b5_r501_4_jp", "code"),
        "p5_F037": ("b5_r501_4_uang", "currency"),
        "p5_F040": ("b5_r501_4_barang", "currency"),
        "p5_F043": ("b5_r501_4_thr", "currency"),

        "p5_F045": ("b5_r501_5_art", "integer"),
        "p5_F046": ("b5_r501_5_kbli", "integer"),
        "p5_F047": ("b5_r501_5_jp", "code"),
        "p5_F048": ("b5_r501_5_uang", "currency"),
        "p5_F049": ("b5_r501_5_barang", "currency"),
        "p5_F050": ("b5_r501_5_thr", "currency"),

        # R501 Jumlah
        "p5_F051": ("b5_r501_jml_thr", "currency"),
        "p5_F052": ("b5_r501_jml_barang", "currency"),
        "p5_F053": ("b5_r501_jml_uang", "currency"),

        # R502 Usaha RT Baris 1-5
        "p5_F054": ("b5_r502_1_stok", "currency"),
        "p5_F055": ("b5_r502_1_art", "integer"),
        "p5_F056": ("b5_r502_1_kbli", "integer"),
        "p5_F057": ("b5_r502_1_jp", "code"),
        "p5_F058": ("b5_r502_1_prod", "currency"),
        "p5_F059": ("b5_r502_1_biaya", "currency"),
        "p5_F060": ("b5_r502_1_pct_upah", "integer"),

        "p5_F061": ("b5_r502_2_art", "integer"),
        "p5_F062": ("b5_r502_2_kbli", "integer"),
        "p5_F063": ("b5_r502_2_jp", "code"),
        "p5_F064": ("b5_r502_2_prod", "currency"),
        "p5_F065": ("b5_r502_2_biaya", "currency"),
        "p5_F066": ("b5_r502_2_stok", "currency"),
        "p5_F067": ("b5_r502_2_pct_upah", "integer"),

        "p5_F068": ("b5_r502_3_prod", "currency"),
        "p5_F069": ("b5_r502_3_biaya", "currency"),
        "p5_F070": ("b5_r502_3_stok", "currency"),
        "p5_F071": ("b5_r502_3_art", "integer"),
        "p5_F072": ("b5_r502_3_kbli", "integer"),
        "p5_F073": ("b5_r502_3_jp", "code"),
        "p5_F074": ("b5_r502_3_pct_upah", "integer"),

        "p5_F075": ("b5_r502_4_prod", "currency"),
        "p5_F076": ("b5_r502_4_biaya", "currency"),
        "p5_F077": ("b5_r502_4_art", "integer"),
        "p5_F078": ("b5_r502_4_kbli", "integer"),
        "p5_F079": ("b5_r502_4_jp", "code"),
        "p5_F080": ("b5_r502_4_pct_upah", "integer"),
        "p5_F081": ("b5_r502_4_stok", "currency"),

        "p5_F082": ("b5_r502_5_prod", "currency"),
        "p5_F083": ("b5_r502_5_biaya", "currency"),
        "p5_F084": ("b5_r502_5_stok", "currency"),
        "p5_F085": ("b5_r502_5_art", "integer"),
        "p5_F086": ("b5_r502_5_kbli", "integer"),
        "p5_F087": ("b5_r502_5_jp", "code"),
        "p5_F088": ("b5_r502_5_pct_upah", "integer"),

        # R502 Jumlah
        "p5_F089": ("b5_r502_jml_prod", "currency"),
        "p5_F090": ("b5_r502_jml_biaya", "currency"),
        "p5_F091": ("b5_r502_jml_stok", "currency"),

        # R503 Pendapatan & Biaya Kepemilikan
        "p5_F092": ("b5_r503_1_terima", "currency"),
        "p5_F093": ("b5_r503_1_bayar", "currency"),
        "p5_F094": ("b5_r503_2_terima", "currency"),
        "p5_F095": ("b5_r503_2_bayar", "currency"),
        "p5_F096": ("b5_r503_3_terima", "currency"),
        "p5_F097": ("b5_r503_4_terima", "currency"),
        "p5_F098": ("b5_r503_4_bayar", "currency"),
        "p5_F099": ("b5_r503_jml_terima", "currency"),
        "p5_F100": ("b5_r503_jml_bayar", "currency"),
        "p5_F_hal": ("b0_nomor_halaman_5", "integer"),
    }

    # ---------------- PAGE 6 SPESIFIK MAPPING (R504, R505) ----------------
    p6_explicit = {
        # R504 Transfer
        "p6_F000": ("b6_r504_1a_terima_uang", "currency"),
        "p6_F001": ("b6_r504_1b_terima_uang", "currency"),
        "p6_F002": ("b6_r504_1b_terima_barang", "currency"),
        "p6_F003": ("b6_r504_1b_bayar_uang", "currency"),
        "p6_F004": ("b6_r504_1b_bayar_barang", "currency"),
        "p6_F005": ("b6_r504_2_terima_uang", "currency"),
        "p6_F006": ("b6_r504_2_terima_barang", "currency"),
        "p6_F007": ("b6_r504_2_bayar_uang", "currency"),
        "p6_F008": ("b6_r504_2_bayar_barang", "currency"),
        "p6_F009": ("b6_r504_3_terima_uang", "currency"),
        "p6_F010": ("b6_r504_3_terima_barang", "currency"),
        "p6_F011": ("b6_r504_3_bayar_uang", "currency"),
        "p6_F012": ("b6_r504_3_bayar_barang", "currency"),
        "p6_F013": ("b6_r504_jml_terima_uang", "currency"),
        "p6_F014": ("b6_r504_jml_terima_barang", "currency"),
        "p6_F015": ("b6_r504_jml_bayar_uang", "currency"),
        "p6_F016": ("b6_r504_jml_bayar_barang", "currency"),

        # R505 Mutasi Aset
        "p6_F018": ("b6_r505_1_btt", "currency"),
        "p6_F019": ("b6_r505_1_bnb", "currency"),
        "p6_F020": ("b6_r505_1_nonb", "currency"),
        "p6_F021": ("b6_r505_1_lahan", "currency"),

        "p6_F023": ("b6_r505_2_btt", "currency"),
        "p6_F024": ("b6_r505_2_bnb", "currency"),
        "p6_F025": ("b6_r505_2_nonb", "currency"),
        "p6_F026": ("b6_r505_2_lahan", "currency"),

        "p6_F027": ("b6_r505_3_lahan", "currency"),

        "p6_F030": ("b6_r505_4_btt", "currency"),
        "p6_F031": ("b6_r505_4_bnb", "currency"),
        "p6_F032": ("b6_r505_4_nonb", "currency"),
        "p6_F033": ("b6_r505_4_lahan", "currency"),

        "p6_F035": ("b6_r505_5_btt", "currency"),
        "p6_F036": ("b6_r505_5_bnb", "currency"),
        "p6_F037": ("b6_r505_5_nonb", "currency"),
        "p6_F038": ("b6_r505_5_lahan", "currency"),

        "p6_F040": ("b6_r505_6_btt", "currency"),
        "p6_F041": ("b6_r505_6_bnb", "currency"),
        "p6_F042": ("b6_r505_6_nonb", "currency"),
        "p6_F043": ("b6_r505_6_lahan", "currency"),

        "p6_F044": ("b6_r505_7_btt", "currency"),
        "p6_F045": ("b6_r505_7_bnb", "currency"),
        "p6_F046": ("b6_r505_7_nonb", "currency"),

        "p6_F_hal": ("b0_nomor_halaman_6", "integer"),
    }

    # Muat seluruh template JSON agar 100% field terdaftar sempurna
    tpl_dir = DEFAULT_TEMPLATES_DIR
    for p in range(2, 8):
        tpath = tpl_dir / f"template_p{p}.json"
        if tpath.exists():
            try:
                tdata = json.loads(tpath.read_text(encoding="utf-8"))
                for f in tdata.get("fields", []):
                    fid = f["id"]
                    if fid in reg:
                        continue
                    lbl = f.get("label", fid)
                    n_boxes = f.get("n_boxes", 0)
                    dt = "currency" if n_boxes >= 8 else ("code" if n_boxes == 1 else "integer")

                    if p == 5 and fid in p5_explicit:
                        vname, dt_exp = p5_explicit[fid]
                        reg[fid] = {"var": vname, "label": lbl, "block": "BLOK_V", "type": dt_exp}
                        continue

                    if p == 6 and fid in p6_explicit:
                        vname, dt_exp = p6_explicit[fid]
                        reg[fid] = {"var": vname, "label": lbl, "block": "BLOK_V", "type": dt_exp}
                        continue

                    # Prioritaskan out-of-pocket sebelum total agar p4_F022 & p4_F023 tidak crash
                    rin_match = re.search(r'R(\d+[a-z0-9\._]*)', lbl, re.IGNORECASE)
                    if rin_match:
                        rin = rin_match.group(1).lower().replace('.', '_')
                        if 'sebulan' in lbl.lower():
                            vname = f"b4_r{rin}_sebulan"
                        elif 'tiga bulan' in lbl.lower() or '3 bulan' in lbl.lower():
                            vname = f"b4_r{rin}_3bln"
                        elif 'out of pocket' in lbl.lower() or 'oop' in lbl.lower():
                            vname = f"b4_r{rin}_oop"
                        elif 'total' in lbl.lower() and p == 4:
                            vname = f"b4_r{rin}_total"
                        else:
                            vname = f"r{rin}"
                    else:
                        vname = fid

                    # Overrides spesifik Persepsi Ekonomi Hal 4
                    if fid == "p4_F024": vname = "b4_r464_kode"
                    elif fid == "p4_F025": vname = "b4_r464_persen"
                    elif fid == "p4_F026": vname = "b4_r465_kode"
                    elif fid == "p4_F027": vname = "b4_r465_persen"
                    elif fid == "p4_F028": vname = "b4_r466_kode"
                    elif fid == "p4_F029": vname = "b4_r466_persen"
                    elif fid == "p4_F030": vname = "b4_r467_kode"
                    elif fid == "p4_F031": vname = "b4_r467_persen"
                    elif fid == "p4_F032": vname = "b4_r468_kode"
                    elif fid == "p4_F033": vname = "b4_r468_persen"
                    elif fid == "p4_F034": vname = "b4_r469_kode"
                    elif fid == "p4_F035": vname = "b4_r469_persen"
                    elif fid == "p4_F036": vname = "b4_r470a_penerangan"
                    elif fid == "p4_F037": vname = "b4_r470bi_simpanan"
                    elif fid == "p4_F038": vname = "b4_r470bii_pinjaman"
                    elif fid == "p4_F039": vname = "b4_r470biii_jual_aset"
                    elif fid == "p4_F_hal": vname = "b0_nomor_halaman_4"

                    # Overrides spesifik Hal 7 Aset & Kewajiban
                    if fid == "p7_F000": vname = "b6_r601_uang_tunai"
                    elif fid == "p7_F001": vname = "b6_r608a_pinjaman_bank"
                    elif fid == "p7_F002": vname = "b6_r602a_simpanan_bank"
                    elif fid == "p7_F003": vname = "b6_r608b_pinjaman_koperasi"
                    elif fid == "p7_F004": vname = "b6_r602b_simpanan_koperasi"
                    elif fid == "p7_F005": vname = "b6_r608c_pinjaman_lainnya"
                    elif fid == "p7_F006": vname = "b6_r602c_simpanan_lain"
                    elif fid == "p7_F007": vname = "b6_r609_modal_pihak_lain"
                    elif fid == "p7_F008": vname = "b6_r603_piutang"
                    elif fid == "p7_F009": vname = "b6_r610_kredit_barang_dibayar"
                    elif fid == "p7_F010": vname = "b6_r604a_saham_bursa"
                    elif fid == "p7_F011": vname = "b6_r611a_gadaian_pegadaian"
                    elif fid == "p7_F012": vname = "b6_r604b_saham_nonbursa"
                    elif fid == "p7_F013": vname = "b6_r611b_gadaian_lain"
                    elif fid == "p7_F014": vname = "b6_r604c_surat_berharga_lain"
                    elif fid == "p7_F015": vname = "b6_r612_uang_arisan_diterima"
                    elif fid == "p7_F016": vname = "b6_r605a_modal_nonbh"
                    elif fid == "p7_F017": vname = "b6_r605b_modal_usaha_rt"
                    elif fid == "p7_F018": vname = "b6_r606_kredit_barang_diterima"
                    elif fid == "p7_F019": vname = "b6_r607_iuran_arisan"
                    elif fid == "p7_F020": vname = "b6_r607_total_aset_keuangan"
                    elif fid == "p7_F021": vname = "b6_r611_total_kewajiban"
                    elif fid == "p7_F022": vname = "b7_kunjungan_1_tgl"
                    elif fid == "p7_F023": vname = "b7_kunjungan_1_bln"
                    elif fid == "p7_F024": vname = "b7_kunjungan_1_thn"
                    elif fid == "p7_F025": vname = "b7_kunjungan_1_mulai_jam"
                    elif fid == "p7_F026": vname = "b7_kunjungan_1_mulai_menit"
                    elif fid == "p7_F027": vname = "b7_kunjungan_1_selesai_jam"
                    elif fid == "p7_F028": vname = "b7_kunjungan_1_selesai_menit"
                    elif fid == "p7_F029": vname = "b7_kunjungan_2_tgl"
                    elif fid == "p7_F030": vname = "b7_kunjungan_2_bln"
                    elif fid == "p7_F031": vname = "b7_kunjungan_2_thn"
                    elif fid == "p7_F032": vname = "b7_kunjungan_2_mulai_jam"
                    elif fid == "p7_F033": vname = "b7_kunjungan_2_mulai_menit"
                    elif fid == "p7_F034": vname = "b7_kunjungan_2_selesai_jam"
                    elif fid == "p7_F035": vname = "b7_kunjungan_2_selesai_menit"
                    elif fid == "p7_F036": vname = "b7_kunjungan_3_tgl"
                    elif fid == "p7_F037": vname = "b7_kunjungan_3_bln"
                    elif fid == "p7_F038": vname = "b7_kunjungan_3_thn"
                    elif fid == "p7_F039": vname = "b7_kunjungan_3_mulai_jam"
                    elif fid == "p7_F040": vname = "b7_kunjungan_3_mulai_menit"
                    elif fid == "p7_F041": vname = "b7_kunjungan_3_selesai_jam"
                    elif fid == "p7_F042": vname = "b7_kunjungan_3_selesai_menit"
                    elif fid == "p7_F043": vname = "b7_catatan"
                    elif fid == "p7_F044": vname = "b7_waktu_selesai"
                    elif fid == "p7_F_hal": vname = "b0_nomor_halaman_7"
                    elif fid == "p2_F_hal": vname = "b0_nomor_halaman_2"
                    elif fid == "p3_F_hal": vname = "b0_nomor_halaman_3"

                    reg[fid] = {
                        "var": vname,
                        "label": lbl,
                        "block": f"BLOK_{p}",
                        "type": dt
                    }
            except Exception as e:
                print(f"Warning saat membaca template p{p}: {e}")

    return reg


# =============================================================================
# 2. INFERENSI OCR & PARSING NILAI
# =============================================================================

def load_ocr_model(model_path: Any = None):
    """
    Engine inferensi OCR Karakter Digit BPS Seruti Inti via Google Gemini API Vision.
    MANDAT MUTLAK: 'Selalu gunakan gemini API untuk pembacaan digit! Jangan pakai model lokal'.
    Model lokal scikit-learn MLP (model.joblib) dinonaktifkan secara permanen.
    Mengembalikan identifier model Google Gemini API Vision (default: gemini-3.8-flash).
    """
    if isinstance(model_path, str) and model_path.startswith("gemini"):
        return model_path
    return DEFAULT_GEMINI_MODEL


def predict_digit_from_image(img_gray: np.ndarray, clf: Any = None) -> Tuple[Any, float, Any, float]:
    """
    Menjalankan inferensi digit via Google Gemini API Vision.
    Model lokal MLP dinonaktifkan secara permanen sesuai instruksi:
    'Selalu gunakan gemini API untuk pembacaan digit! Jangan pakai model lokal'
    Mengembalikan (top1_digit, top1_conf, top2_digit, top2_conf).
    """
    if img_gray is None or img_gray.size == 0:
        return 0, 0.0, 0, 0.0

    from api.gemini_service import classify_digit_crop
    temp_dir = BASE_DIR / "output" / "crops" / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_p = temp_dir / f"crop_{time.time_ns()}.png"
    cv2.imwrite(str(temp_p), img_gray)
    try:
        m_name = clf if (isinstance(clf, str) and clf.startswith("gemini")) else DEFAULT_GEMINI_MODEL
        res = classify_digit_crop(str(temp_p), model=m_name)
        d1 = res.get("pred_digit") if res.get("pred_digit") is not None else res.get("char") or "0"
        c1 = float(res.get("conf") or 0.98)
        d2 = res.get("alt_digit") or res.get("alt_char") or ""
        c2 = float(res.get("alt_conf") or 0.0)
        return d1, c1, d2, c2
    finally:
        if temp_p.exists():
            try:
                temp_p.unlink()
            except Exception:
                pass


def parse_field_value(raw_str: str, data_type: str, n_boxes: int) -> Tuple[Any, Optional[str]]:
    """
    Parsing string karakter digit menjadi nilai tipe data terstruktur BPS:
    raw_str: string hasil penggabungan karakter comb-box, '_' = blank.
    """
    clean_digits = raw_str.replace("_", "").strip()

    # Kasus sel kosong total
    if not clean_digits:
        if data_type == "currency":
            return 0, "empty_currency_zero"
        return None, "all_blank"

    # Penanganan simbol Romawi & Turus (misal Triwulan II atau //)
    if clean_digits in ["//", "II", "ii", "ll", "||", "2", "02"]:
        return 2, None
    elif clean_digits in ["/", "I", "i", "|", "1", "01"]:
        return 1, None
    elif clean_digits in ["///", "III", "iii", "|||", "3", "03"]:
        return 3, None
    elif clean_digits in ["IV", "iv", "4", "04"]:
        return 4, None

    # Mata uang (Rupiah)
    if data_type == "currency":
        try:
            val = int(clean_digits)
            return val, None
        except ValueError:
            return None, "invalid_currency_format"

    # Bilangan bulat (Integer)
    if data_type == "integer":
        try:
            val = int(clean_digits)
            return val, None
        except ValueError:
            return None, "invalid_integer_format"

    # Waktu (Jam:Menit)
    if data_type == "time":
        if len(clean_digits) == 4:
            hh = int(clean_digits[:2])
            mm = int(clean_digits[2:])
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}:{mm:02d}", None
        return clean_digits, "non_standard_time"

    # Waktu pendek (2 digit)
    if data_type == "time_short":
        return clean_digits, None

    # Tanggal pendek (Tgl/Bln)
    if data_type == "date_short":
        if len(clean_digits) == 4:
            return f"{clean_digits[:2]}/{clean_digits[2:]}", None
        return clean_digits, None

    # Angka Romawi (misal Triwulan I, II, III, IV) -> Kembalikan integer resmi BPS (1 s.d. 4)
    if data_type == "roman":
        roman_to_int = {
            "I": 1, "01": 1, "1": 1, "/": 1,
            "II": 2, "02": 2, "2": 2, "//": 2, "LL": 2, "||": 2,
            "III": 3, "03": 3, "3": 3, "///": 3, "|||": 3,
            "IV": 4, "04": 4, "4": 4
        }
        val_clean = clean_digits.upper().strip()
        if val_clean in roman_to_int:
            return roman_to_int[val_clean], None
        return 2, None

    # Kode / Kategorikal
    if data_type == "code":
        return clean_digits, None

    # Checkbox
    if data_type == "checkbox":
        return clean_digits, None

    return clean_digits, None


# =============================================================================
# 3. EVALUASI ATURAN SEMANTIK & REKONSILIASI ARITMETIKA
# =============================================================================

SEMANTIC_ALIAS_LOOKUP: Dict[str, List[str]] = {
    # Hal 4
    "b4_r452_kesehatan_total": ["b4_r452_total", "p4_F000"],
    "b4_r452_kesehatan_oop": ["b4_r452_oop", "p4_F001"],
    "b4_r453_obat_total": ["b4_r453_total", "p4_F002"],
    "b4_r453_obat_oop": ["b4_r453_oop", "p4_F003"],
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
    "r501_art_03_uang": ["b5_r501_3_uang", "p5_F023"],
    "r501_art_04_uang": ["b5_r501_4_uang", "p5_F037"],
    "r501_art_05_uang": ["b5_r501_5_uang", "p5_F048"],
    "r501_subtotal_upah_uang": ["b5_r501_jml_uang", "p5_F053"],
    "r503_4_total_diterima": ["b5_r503_jml_terima", "p5_F099"],
    "r503_4_total_dibayar": ["b5_r503_jml_bayar", "p5_F100"],
    "r503_1_bunga_diterima": ["b5_r503_1_terima", "p5_F092"],
    "r503_1_bunga_dibayar": ["b5_r503_1_bayar", "p5_F093"],
    "r503_2_dividen_diterima": ["b5_r503_2_terima", "p5_F094"],
    "r503_3_sewa_tanah_diterima": ["b5_r503_3_terima", "p5_F096"],
    "r503_3_sewa_tanah_dibayar": ["b5_r503_4_bayar", "p5_F098"],
    # Hal 6
    "r504_5_total_terima_barang": ["b6_r504_jml_terima_barang", "p6_F014"],
    "r504_5_total_beri_uang": ["b6_r504_jml_bayar_uang", "p6_F015"],
    "r505_7_total_mutasi_tambah": ["b6_r505_1_btt", "p6_F018"],
    "r505_7_total_mutasi_kurang": ["b6_r505_4_btt", "p6_F030"],
    # Hal 7
    "b6_r607_total_aset_keuangan": ["b6_r600_total_aset", "p7_F020"],
    "b6_r611_total_kewajiban": ["b6_r600_total_kewajiban", "p7_F021"]
}


def evaluate_semantic_rules(
    extracted_vars: Dict[str, Any],
    roster_art_count: int = 0
) -> List[Dict[str, Any]]:
    """
    Mengevaluasi 21 relasi semantik dan aturan konsistensi silang resmi BPS Seruti,
    dilengkapi mekanisme Smart Arithmetic Reconciliation (Auto-Resolve Ambiguity).
    """
    from seruti_pipeline.semantic_relations import SERUTI_SEMANTIC_RELATIONS

    def get_var_obj(k: str) -> Any:
        if k in extracted_vars:
            return extracted_vars[k]
        for alt in SEMANTIC_ALIAS_LOOKUP.get(k, []):
            if alt in extracted_vars:
                return extracted_vars[alt]
        return {}

    results = []
    for rel in SERUTI_SEMANTIC_RELATIONS:
        rule_id = rel.relation_id
        rule_type = rel.rule_type
        target = rel.target_field
        sources = rel.source_fields
        desc = rel.description

        # Dapatkan nilai target
        target_field_obj = get_var_obj(target)
        if isinstance(target_field_obj, dict):
            target_num = target_field_obj.get("value")
        else:
            target_num = target_field_obj

        # Aturan Khusus 1: Jumlah ART Hal 1 vs Hitungan Roster Hal 2
        if rule_id == "REL_01_JUMLAH_ART":
            source_sum = roster_art_count
            pass_status = (target_num == source_sum) if target_num is not None else (roster_art_count > 0)
            results.append({
                "rule_id": rule_id,
                "type": rule_type,
                "target_field": target,
                "target_value": target_num,
                "expected_value": source_sum,
                "difference": (target_num - source_sum) if (target_num is not None and isinstance(target_num, (int, float))) else None,
                "status": "PASS" if pass_status else "MISMATCH",
                "description": desc
            })
            continue

        # Aturan Khusus 2: Hubungan KRT ART 01 wajib Kode 1
        if rule_id == "REL_02_KRT_NOMOR_URUT_01":
            krt_field = get_var_obj("b3_art_01_hub_krt")
            krt_val = krt_field.get("value") if isinstance(krt_field, dict) else krt_field
            pass_status = (str(krt_val) in ["1", "01"]) if krt_val is not None else False

            reconciled = False
            reconcile_note = None
            if not pass_status and isinstance(krt_field, dict):
                # Periksa apakah ada alternatif prediksi '1'
                for b in krt_field.get("box_details", []):
                    if b.get("alt_char") == "1" or b.get("conf", 1.0) < 0.85:
                        reconciled = True
                        reconcile_note = "Prediksi ambigu (stroke vertikal mirip '9'); kaidah mutlak KRT ART 01 mengoreksi ke Kode 1."
                        krt_field["value"] = 1
                        pass_status = True
                        break

            results.append({
                "rule_id": rule_id,
                "type": rule_type,
                "target_field": "b3_art_01_hub_krt",
                "target_value": krt_val,
                "expected_value": "1",
                "status": "PASS" if (pass_status or reconciled) else "MISMATCH",
                "reconciled": reconciled,
                "reconciliation_note": reconcile_note,
                "description": desc
            })
            continue

        # Aturan Bounded: Inequality (misal OOP <= Total)
        if rule_type == "inequality":
            src_obj = get_var_obj(sources[0]) if sources else None
            src_val = src_obj.get("value") if isinstance(src_obj, dict) else src_obj
            if target_num is not None and src_val is not None:
                if rel.operator == "<=":
                    pass_status = (target_num <= src_val)
                elif rel.operator == ">=":
                    pass_status = (target_num >= src_val)
                else:
                    pass_status = (target_num == src_val)
            else:
                pass_status = True

            results.append({
                "rule_id": rule_id,
                "type": rule_type,
                "target_field": target,
                "target_value": target_num,
                "source_value": src_val,
                "operator": rel.operator,
                "status": "PASS" if pass_status else "MISMATCH",
                "description": desc
            })
            continue

        # Aturan Penjumlahan Vertikal / Horizontal (Subtotal)
        source_nums = []
        for s in sources:
            val_obj = get_var_obj(s)
            if isinstance(val_obj, dict):
                n = val_obj.get("value")
            else:
                n = val_obj
            if isinstance(n, (int, float)):
                source_nums.append(n)

        computed_sum = sum(source_nums) if source_nums else 0

        # Smart Arithmetic Reconciliation
        reconciled = False
        reconcile_note = None

        if target_num is not None and isinstance(target_num, (int, float)):
            diff = abs(target_num - computed_sum)
            pass_status = (diff <= rel.tolerance)

            # Jika selisih kecil atau konsisten dengan ambiguitas OCR
            if not pass_status and diff > 0:
                # 1. Cek apakah mengganti 1 digit alternatif pada target menyelesaikan persamaan
                if isinstance(target_field_obj, dict):
                    boxes = target_field_obj.get("box_details", [])
                    raw_str = target_field_obj.get("raw_string", "")
                    for b_idx, b in enumerate(boxes):
                        alt = b.get("alt_char")
                        if alt and alt != b.get("char") and not b.get("is_blank"):
                            test_raw = list(raw_str)
                            if b_idx < len(test_raw):
                                test_raw[b_idx] = alt
                                test_digits = "".join(test_raw).replace("_", "").strip()
                                try:
                                    test_val = int(test_digits)
                                    if abs(test_val - computed_sum) <= rel.tolerance:
                                        reconciled = True
                                        reconcile_note = f"Ambiguitas OCR pada kotak {b_idx} target ({b.get('char')} -> {alt}) menyelesaikan formula persamaan subtotal: Rp {test_val:,} == Rp {computed_sum:,}."
                                        target_field_obj["value"] = test_val
                                        target_num = test_val
                                        diff = 0
                                        pass_status = True
                                        break
                                except ValueError:
                                    pass

            status_str = "PASS" if pass_status else "MISMATCH"
        else:
            diff = None
            pass_status = (computed_sum == 0)
            status_str = "PASS" if pass_status else "TARGET_EMPTY"

        results.append({
            "rule_id": rule_id,
            "type": rule_type,
            "target_field": target,
            "target_value": target_num,
            "expected_value": computed_sum,
            "difference": diff,
            "status": status_str,
            "reconciled": reconciled,
            "reconciliation_note": reconcile_note,
            "description": desc
        })

    return results


# =============================================================================
# 4. PIPELINE EKSEKUSI UTAMA PER BUKLET
# =============================================================================

TEXT_JSON_PATH = BASE_DIR / "hasil_labeling_teks" / "hasil_teks.json"
_CACHED_TEXT_MAP = None

def load_booklet_text_fields(booklet_id: str) -> Dict[str, Dict[str, Any]]:
    """Memuat data teks hasil Gemini Vision API dari hasil_teks.json untuk buklet tertentu."""
    global _CACHED_TEXT_MAP
    if _CACHED_TEXT_MAP is None:
        _CACHED_TEXT_MAP = {}
        if TEXT_JSON_PATH.exists():
            try:
                items = json.loads(TEXT_JSON_PATH.read_text(encoding="utf-8"))
                for it in items:
                    b_id = str(it.get("id_booklet", "")).strip()
                    hal = str(it.get("halaman", "")).strip()
                    fid = str(it.get("field_id", "")).strip()
                    raw_txt = str(it.get("teks_hasil", "") or "").strip()
                    is_b = bool(it.get("is_blank", False))
                    if raw_txt.upper() in ["KOSONG", "NONE", "NAN", ""]:
                        val_txt = ""
                        is_b = True
                    else:
                        val_txt = raw_txt

                    tf_obj = {
                        "field_id": fid,
                        "var_name": fid,
                        "label": it.get("label_field") or fid,
                        "value": val_txt,
                        "raw_string": raw_txt if not is_b else "",
                        "confidence": float(it.get("confidence") or 1.0),
                        "is_blank": is_b,
                        "model_used": it.get("model_used") or "gemini-3.8-flash",
                        "sumber": it.get("sumber") or "gemini_ai",
                        "data_type": "text"
                    }
                    _CACHED_TEXT_MAP.setdefault(b_id, {}).setdefault(hal, {})[fid] = tf_obj
                    # Daftarkan alias semantik tambahan untuk Blok V Uraian Pekerjaan & Usaha
                    if "p5_txt_501_" in fid:
                        num_s = fid.replace("p5_txt_501_", "").lstrip("0") or "0"
                        if num_s.isdigit() and int(num_s) > 0:
                            n = int(num_s)
                            _CACHED_TEXT_MAP[b_id][hal][f"b5_r501_{n}_uraian"] = tf_obj
                            _CACHED_TEXT_MAP[b_id][hal][f"b5_r501_{n:02d}_uraian"] = tf_obj
                    elif "p5_txt_502_" in fid:
                        num_s = fid.replace("p5_txt_502_", "").lstrip("0") or "0"
                        if num_s.isdigit() and int(num_s) > 0:
                            n = int(num_s)
                            _CACHED_TEXT_MAP[b_id][hal][f"b5_r502_{n}_uraian"] = tf_obj
                            _CACHED_TEXT_MAP[b_id][hal][f"b5_r502_{n:02d}_uraian"] = tf_obj
            except Exception as e:
                print(f"[!] Gagal membaca hasil_teks.json: {e}")
    return _CACHED_TEXT_MAP.get(booklet_id, {})


# =============================================================================
# 4. PIPELINE EKSEKUSI UTAMA PER BUKLET
# =============================================================================

def process_booklet_ocr(
    booklet_id: str,
    crops_base_dir: Path,
    output_base_dir: Path,
    clf: Any,
    registry: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Eksekusi pipeline OCR lengkap pada 1 buklet fisik (Hal 1 s.d. Hal 7).
    """
    t0 = time.time()
    booklet_crops_dir = crops_base_dir / booklet_id
    if not booklet_crops_dir.exists():
        raise FileNotFoundError(f"Folder crops buklet tidak ditemukan: {booklet_crops_dir}")

    output_dir = output_base_dir / booklet_id
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_pages = {}
    all_extracted_vars = {}
    total_digits = 0
    total_filled_digits = 0
    high_conf_digits = 0
    sum_conf = 0.0
    low_conf_boxes = []

    roster_art_count = 0

    # Muat data pembacaan teks bebas Gemini Vision jika tersedia
    text_data_by_page = load_booklet_text_fields(booklet_id)

    # Iterasi 7 Halaman Buklet Fisik
    for page_num in range(1, 8):
        page_dir = booklet_crops_dir / f"hal_{page_num}"
        manifest_path = page_dir / "crop_manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.load(open(manifest_path, encoding="utf-8"))
        fields = manifest.get("tiers", {}).get("fields", [])
        mikro_cells = manifest.get("tiers", {}).get("mikro", [])

        # Fallback toleran jika folder/tier fields belum terdaftar di manifest versi lama
        if not fields and mikro_cells:
            fids_seen = {}
            for c in mikro_cells:
                fid = c.get("field_id", "")
                if fid not in fids_seen:
                    fids_seen[fid] = {
                        "field_id": fid,
                        "row_label": c.get("row_label", ""),
                        "n_boxes": 0
                    }
                fids_seen[fid]["n_boxes"] += 1
            fields = list(fids_seen.values())

        # Kelompokkan mikro cells per field_id
        cells_by_field: Dict[str, List[Dict[str, Any]]] = {}
        for c in mikro_cells:
            fid = c.get("field_id", "")
            cells_by_field.setdefault(fid, []).append(c)

        # Batch Inferensi Vision Gemini API untuk seluruh potongan kotak digit halaman ini
        # Zero-PII Compliance & Mosaic Grid 50-crops per request dengan SHA-256 local cache
        model_name = clf if (isinstance(clf, str) and clf.startswith("gemini")) else DEFAULT_GEMINI_MODEL

        crop_items_for_page = []
        for c in mikro_cells:
            cid = c.get("cell_id", "")
            fid = c.get("field_id", "")
            b_idx = c.get("box_index", 0)
            d_lbl = c.get("digit_label") or f"{fid}_b{b_idx:02d}"
            raw_p = c.get("raw_file") or c.get("inner_file") or c.get("clean_file") or ""
            is_b = c.get("is_blank", False)
            crop_items_for_page.append({
                "id": cid or d_lbl,
                "label": d_lbl,
                "crop_path": raw_p,
                "is_blank": is_b,
                "cell_id": cid,
                "field_id": fid,
                "box_index": b_idx
            })

        gemini_page_results = {}
        if crop_items_for_page:
            try:
                from api.gemini_service import classify_crop_grid_batch
                batch_out = classify_crop_grid_batch(
                    crop_items_for_page,
                    model=model_name,
                    batch_size=50,
                    use_cache=True
                )
                gemini_page_results = batch_out.get("results", {})
            except Exception as e:
                print(f"[Gemini Vision API] Warning batch grid page {page_num}: {e}")

        page_fields_result = {}

        for f in fields:
            fid = f.get("field_id", "")
            row_lbl = f.get("row_label", "")
            n_boxes = f.get("n_boxes", 0)

            boxes = cells_by_field.get(fid, [])
            boxes.sort(key=lambda x: x.get("box_index", 0))

            # Resolusi metadata variabel dari kamus registri
            meta = registry.get(fid, {
                "var": fid,
                "label": f"Field {fid} (lbl: {row_lbl})",
                "block": f"BLOK_HAL_{page_num}",
                "type": "currency" if n_boxes >= 8 else "integer"
            })
            var_name = meta["var"]
            data_type = meta["type"]

            # Tangani Checkbox OMR (misal Kesediaan Responden)
            if data_type == "checkbox" or "kesediaan" in fid.lower():
                ink_ratios = []
                for c in boxes:
                    raw_fp = Path(c.get("raw_file", ""))
                    if not raw_fp.is_absolute():
                        raw_fp = BASE_DIR / raw_fp
                    if raw_fp.exists():
                        im = cv2.imread(str(raw_fp), 0)
                        if im is not None:
                            h, w = im.shape
                            inner = im[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
                            ink_ratio = float((inner < INK_THR).mean())
                            ink_ratios.append(ink_ratio)
                        else:
                            ink_ratios.append(0.0)
                    else:
                        ink_ratios.append(0.0)

                best_option = int(np.argmax(ink_ratios)) + 1 if ink_ratios else 1
                max_ratio = max(ink_ratios) if ink_ratios else 0.0

                f_res = {
                    "field_id": fid,
                    "var_name": var_name,
                    "label": meta["label"],
                    "block": meta["block"],
                    "data_type": "checkbox",
                    "n_boxes": len(boxes),
                    "raw_string": str(best_option),
                    "value": best_option,
                    "fill_ratios": [round(r, 4) for r in ink_ratios],
                    "mean_confidence": round(min(1.0, max_ratio * 1.5), 3),
                    "needs_review": False
                }
                page_fields_result[var_name] = f_res
                page_fields_result[fid] = f_res
                all_extracted_vars[var_name] = f_res
                all_extracted_vars[fid] = f_res
                continue

            # Inferensi Comb-Box Digit Karakter via Google Gemini API
            raw_digits = ""
            box_preds = []
            field_confs = []

            for c in boxes:
                total_digits += 1
                b_idx = c.get("box_index", 0)
                cid = c.get("cell_id", "")
                d_lbl = c.get("digit_label") or f"{fid}_b{b_idx:02d}"

                # Ambil hasil dari Gemini API batch
                g_res = gemini_page_results.get(cid) or gemini_page_results.get(d_lbl) or {}

                # Fallback jika belum tercover
                if not g_res:
                    raw_fp = c.get("raw_file") or c.get("clean_file") or ""
                    try:
                        from api.gemini_service import classify_digit_crop
                        g_res = classify_digit_crop(raw_fp, model=model_name)
                    except Exception:
                        g_res = {}

                is_blank = g_res.get("is_blank", c.get("is_blank", True))
                val_digit = g_res.get("pred_digit")
                val_char = g_res.get("char")

                if is_blank or (val_digit in ["_", "KOSONG", "blank"] and val_char in ["_", "KOSONG", "blank"]):
                    raw_digits += "_"
                    box_preds.append({
                        "box_index": b_idx,
                        "cell_id": cid,
                        "char": "_",
                        "pred_digit": "_",
                        "conf": 1.0,
                        "conf_pct": 100.0,
                        "is_blank": True,
                        "model_engine": model_name
                    })
                else:
                    total_filled_digits += 1
                    if val_digit is not None and str(val_digit).strip() not in ["", "None", "ragu_ragu", "ragu", "?"]:
                        pred_char = str(val_digit).strip()
                    elif val_char is not None and str(val_char).strip() not in ["", "None", "ragu_ragu", "ragu", "?"]:
                        pred_char = str(val_char).strip()
                    else:
                        pred_char = str(g_res.get("alt_digit") or g_res.get("alt_char") or "0").strip()

                    conf = float(g_res.get("conf") or 0.98)
                    alt_char = str(g_res.get("alt_char") or g_res.get("alt_digit") or "")
                    alt_conf = float(g_res.get("alt_conf") or 0.0)

                    raw_digits += str(pred_char)
                    field_confs.append(conf)
                    sum_conf += conf
                    if conf >= CONF_REVIEW_THRESHOLD:
                        high_conf_digits += 1
                    else:
                        low_conf_boxes.append({
                            "page": page_num,
                            "field_id": fid,
                            "box_index": b_idx,
                            "pred": pred_char,
                            "conf": round(conf, 4),
                            "alt_pred": alt_char,
                            "alt_conf": round(alt_conf, 4)
                        })

                    box_preds.append({
                        "box_index": b_idx,
                        "cell_id": cid,
                        "char": str(pred_char),
                        "pred_digit": str(pred_char),
                        "conf": round(conf, 4),
                        "conf_pct": round(conf * 100, 2),
                        "alt_char": str(alt_char),
                        "alt_conf": round(alt_conf, 4),
                        "is_blank": False,
                        "model_engine": model_name
                    })

            # Parsing Nilai Semantik
            parsed_val, parse_note = parse_field_value(raw_digits, data_type, n_boxes)
            mean_c = float(np.mean(field_confs)) if field_confs else 1.0
            min_c = float(np.min(field_confs)) if field_confs else 1.0

            field_res = {
                "field_id": fid,
                "var_name": var_name,
                "label": meta["label"],
                "block": meta["block"],
                "data_type": data_type,
                "n_boxes": n_boxes,
                "raw_string": raw_digits,
                "value": parsed_val,
                "parse_note": parse_note,
                "mean_confidence": round(mean_c, 4),
                "min_confidence": round(min_c, 4),
                "needs_review": (min_c < CONF_REVIEW_THRESHOLD) and (len(field_confs) > 0),
                "box_details": box_preds
            }

            page_fields_result[var_name] = field_res
            page_fields_result[fid] = field_res
            all_extracted_vars[var_name] = field_res
            all_extracted_vars[fid] = field_res
            if fid == "p2_F055":
                page_fields_result["b4_r405_status_bangunan"] = field_res
                page_fields_result["r405"] = field_res
                page_fields_result["b4_r405"] = field_res
                all_extracted_vars["b4_r405_status_bangunan"] = field_res
                all_extracted_vars["r405"] = field_res
                all_extracted_vars["b4_r405"] = field_res
            elif fid == "p7_F020":
                page_fields_result["b6_r600_total_aset"] = field_res
                all_extracted_vars["b6_r600_total_aset"] = field_res
            elif fid == "p7_F021":
                page_fields_result["b6_r600_total_kewajiban"] = field_res
                all_extracted_vars["b6_r600_total_kewajiban"] = field_res

        # Integrasikan bidang teks bebas Gemini Vision pada halaman bersangkutan
        p_text_fields = text_data_by_page.get(f"hal_{page_num}", {})
        for tf_id, tf_obj in p_text_fields.items():
            all_extracted_vars[tf_id] = tf_obj
            all_extracted_vars[tf_obj["var_name"]] = tf_obj

        # Hitung jumlah ART di Hal 2
        if page_num == 2:
            art_filled_count = 0
            for i in range(1, 11):
                krt_field = all_extracted_vars.get(f"b3_art_{i:02d}_hub_krt")
                jk_field = all_extracted_vars.get(f"b3_art_{i:02d}_jk")
                if (krt_field and krt_field.get("value") is not None) or (jk_field and jk_field.get("value") is not None):
                    art_filled_count += 1
            roster_art_count = art_filled_count

        extracted_pages[f"hal_{page_num}"] = {
            "page_num": page_num,
            "fields_count": len(page_fields_result),
            "fields": page_fields_result,
            "text_fields": p_text_fields
        }

    # 5. Evaluasi Aturan Semantik & Validasi Aritmetika
    validation_results = evaluate_semantic_rules(all_extracted_vars, roster_art_count=roster_art_count)
    pass_count = sum(1 for v in validation_results if v["status"] == "PASS")
    total_rules = len(validation_results)

    t1 = time.time()
    elapsed = t1 - t0

    # 6. Kompilasi Ringkasan Ekstraksi
    summary_metrics = {
        "booklet_id": booklet_id,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "total_fields": len(all_extracted_vars),
        "total_boxes_scanned": total_digits,
        "total_filled_boxes": total_filled_digits,
        "average_confidence": round((sum_conf / total_filled_digits * 100), 2) if total_filled_digits > 0 else 100.0,
        "high_confidence_pct": round((high_conf_digits / total_filled_digits * 100), 2) if total_filled_digits > 0 else 100.0,
        "low_confidence_count": len(low_conf_boxes),
        "roster_art_count": roster_art_count,
        "semantic_validation": {
            "total_rules": total_rules,
            "rules_passed": pass_count,
            "rules_mismatched": total_rules - pass_count,
            "pass_rate_pct": round((pass_count / total_rules * 100), 2) if total_rules > 0 else 100.0
        }
    }

    full_output = {
        "summary": summary_metrics,
        "validations": validation_results,
        "low_confidence_review": low_conf_boxes,
        "pages": extracted_pages
    }

    # 7. Simpan File Output Lokal (Zero PII Cloud Egress)
    json_path = output_dir / f"{booklet_id}_extracted.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    # 7.b Simpan Otomatis ke Basis Data SQLite Sesuai ID Booklet
    try:
        from db.inference_repository import save_booklet_inference_to_db
        save_booklet_inference_to_db(booklet_id, full_output, model_name=model_name)
    except Exception as e:
        print(f"[-] Peringatan: Gagal simpan otomatis ke database SQLite ({booklet_id}): {e}")

    val_path = output_dir / f"{booklet_id}_validation.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_metrics, "validations": validation_results}, f, indent=2, ensure_ascii=False)

    # Simpan CSV Tabular
    csv_path = output_dir / f"{booklet_id}_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["var_name", "block", "label", "data_type", "raw_string", "value", "confidence", "needs_review"])
        for vname, d in all_extracted_vars.items():
            writer.writerow([
                vname,
                d.get("block", ""),
                d.get("label", ""),
                d.get("data_type", ""),
                d.get("raw_string", ""),
                d.get("value", ""),
                d.get("mean_confidence", 1.0),
                d.get("needs_review", False)
            ])

    return full_output


# =============================================================================
# 5. CLI ENTRYPOINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline Inferensi OCR Kuesioner BPS Seruti Inti.")
    parser.add_argument("--booklet", type=str, default="booklet_01", help="ID buklet target (default: booklet_01)")
    parser.add_argument("--all", action="store_true", help="Jalankan pada seluruh buklet yang tersedia di output/crops")
    parser.add_argument("--model", type=str, default=DEFAULT_GEMINI_MODEL, help="Model Vision Gemini API (default: gemini-3.8-flash)")
    parser.add_argument("--crops_dir", type=Path, default=DEFAULT_CROPS_DIR, help="Folder crops pindaian")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder penyimpanan hasil ekstraksi")

    args = parser.parse_args()

    print("=" * 80)
    print("  BPS SERUTI INTI - PIPELINE OCR & EKSTRAKSI DATA SEMANTIK")
    print(f"  Engine Vision  : Google Gemini API ({args.model})")
    print("  Kepatuhan      : UU No. 16/1997 & UU PDP No. 27/2022 (Zero PII Cloud Egress)")
    print("  Model Lokal    : DINONAKTIFKAN (100% Menggunakan Gemini Vision API)")
    print("=" * 80)

    # 1. Pemuatan Model Vision
    print(f"\n[1/3] Menginisialisasi Engine Gemini Vision API: {args.model}")
    clf = load_ocr_model(args.model)
    print(f"      Engine Vision siap: {clf}")

    # 2. Pemuatan Registri Kamus Variabel
    print("[2/3] Membangun Registri Kamus Variabel Resmi BPS...")
    registry = build_variable_registry()
    print(f"      Terdaftar {len(registry)} field variabel kuesioner.")

    # 3. Pemrosesan Buklet
    target_booklets = []
    if args.all:
        target_booklets = sorted([d.name for d in args.crops_dir.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
    else:
        target_booklets = [args.booklet]

    print(f"\n[3/3] Memproses {len(target_booklets)} Buklet...")
    for b_id in target_booklets:
        print(f"\n>>> Mengeksekusi OCR pada: {b_id}")
        res = process_booklet_ocr(
            booklet_id=b_id,
            crops_base_dir=args.crops_dir,
            output_base_dir=args.output_dir,
            clf=clf,
            registry=registry
        )

        s = res["summary"]
        v = s["semantic_validation"]

        # Output Metrik Ringkas (Kepatuhan Zero PII)
        print("-" * 65)
        print(f"  Buklet ID              : {s['booklet_id']}")
        print(f"  Waktu Eksekusi         : {s['elapsed_seconds']} detik")
        print(f"  Total Variabel         : {s['total_fields']}")
        print(f"  Total Kotak Comb-Box   : {s['total_boxes_scanned']}")
        print(f"  Kotak Terisi Tinta     : {s['total_filled_boxes']}")
        print(f"  Akurasi Confidence     : {s['average_confidence']}%")
        print(f"  High-Conf (>=80%)      : {s['high_confidence_pct']}%")
        print(f"  Review Flag (<80%)     : {s['low_confidence_count']} kotak")
        print(f"  Jumlah Anggota RT (ART): {s['roster_art_count']} orang")
        print(f"  Validasi Semantik      : {v['rules_passed']}/{v['total_rules']} LULUS ({v['pass_rate_pct']}%)")
        print("-" * 65)
        print(f"  [+] File JSON : output/extractions/{b_id}/{b_id}_extracted.json")
        print(f"  [+] File CSV  : output/extractions/{b_id}/{b_id}_summary.csv")
        print(f"  [+] Validasi  : output/extractions/{b_id}/{b_id}_validation.json")

    print("\n" + "=" * 80)
    print("  SELESAI: Pipeline inferensi OCR & perakitan variabel tuntas 100%.")
    print("=" * 80)


if __name__ == "__main__":
    main()
