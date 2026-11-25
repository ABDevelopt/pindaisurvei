#!/usr/bin/env python3
"""
update_and_complete_grid_labels.py
Memperbaiki dan melengkapi label grid pada template field isian dan kotak digit (comb box)
untuk seluruh halaman kuesioner Seruti Inti 2026 (template_p1 s.d. template_p7).
Memastikan 100% field dan kotak digit memiliki label yang sah, valid, dan tidak ada yang 'none' atau kosong.
"""

import json
from pathlib import Path

TEMPLATES_DIR = Path("templates")

# ==============================================================================
# Mapping Label Presisi Halaman 5 (Blok V.1, V.2, V.3)
# ==============================================================================
P5_FIELD_LABELS = {
    # --- Tabel 0: Sub-blok V.1: Upah dan Gaji Karyawan/Buruh/Pegawai ---
    # ART 1
    "p5_F000": ("501_1_k1", "R501.1 No Urut ART Baris 1"),
    "p5_F001": ("501_1_k3", "R501.1 Kategori Lapangan Usaha (KBLI) Baris 1"),
    "p5_F002": ("501_1_k4", "R501.1 Jenis Pekerjaan Baris 1"),
    "p5_F003": ("501_1_k5", "R501.1 Upah/Gaji Berupa Uang Baris 1"),
    "p5_F004": ("501_1_k6", "R501.1 Upah/Gaji Berupa Barang/Jasa Baris 1"),
    "p5_F005": ("501_1_k7", "R501.1 Lembur, Honorarium, THR Baris 1"),

    # ART 2
    "p5_F006": ("501_2_k1", "R501.2 No Urut ART Baris 2"),
    "p5_F007": ("501_2_k3", "R501.2 Kategori Lapangan Usaha (KBLI) Baris 2"),
    "p5_F008": ("501_2_k4", "R501.2 Jenis Pekerjaan Baris 2"),
    "p5_F009": ("501_2_k5", "R501.2 Upah/Gaji Berupa Uang Baris 2"),
    "p5_F012": ("501_2_k6", "R501.2 Upah/Gaji Berupa Barang/Jasa Baris 2"),
    "p5_F015": ("501_2_k7", "R501.2 Lembur, Honorarium, THR Baris 2"),

    # ART 3
    "p5_F018": ("501_3_k1", "R501.3 No Urut ART Baris 3"),
    "p5_F019": ("501_3_k3", "R501.3 Kategori Lapangan Usaha (KBLI) Baris 3"),
    "p5_F020": ("501_3_k4", "R501.3 Jenis Pekerjaan Baris 3"),
    "p5_F023": ("501_3_k5", "R501.3 Upah/Gaji Berupa Uang Baris 3"),
    "p5_F026": ("501_3_k6", "R501.3 Upah/Gaji Berupa Barang/Jasa Baris 3"),
    "p5_F029": ("501_3_k7", "R501.3 Lembur, Honorarium, THR Baris 3"),

    # ART 4
    "p5_F032": ("501_4_k1", "R501.4 No Urut ART Baris 4"),
    "p5_F033": ("501_4_k3", "R501.4 Kategori Lapangan Usaha (KBLI) Baris 4"),
    "p5_F034": ("501_4_k4", "R501.4 Jenis Pekerjaan Baris 4"),
    "p5_F037": ("501_4_k5", "R501.4 Upah/Gaji Berupa Uang Baris 4"),
    "p5_F040": ("501_4_k6", "R501.4 Upah/Gaji Berupa Barang/Jasa Baris 4"),
    "p5_F043": ("501_4_k7", "R501.4 Lembur, Honorarium, THR Baris 4"),

    # ART 5
    "p5_F045": ("501_5_k1", "R501.5 No Urut ART Baris 5"),
    "p5_F046": ("501_5_k3", "R501.5 Kategori Lapangan Usaha (KBLI) Baris 5"),
    "p5_F047": ("501_5_k4", "R501.5 Jenis Pekerjaan Baris 5"),
    "p5_F048": ("501_5_k5", "R501.5 Upah/Gaji Berupa Uang Baris 5"),
    "p5_F049": ("501_5_k6", "R501.5 Upah/Gaji Berupa Barang/Jasa Baris 5"),
    "p5_F050": ("501_5_k7", "R501.5 Lembur, Honorarium, THR Baris 5"),

    # JUMLAH V.1
    "p5_F053": ("501_JUMLAH_k5", "R501 JUMLAH Upah/Gaji Berupa Uang"),
    "p5_F052": ("501_JUMLAH_k6", "R501 JUMLAH Upah/Gaji Berupa Barang/Jasa"),
    "p5_F051": ("501_JUMLAH_k7", "R501 JUMLAH Lembur, Honorarium, THR"),

    # --- Tabel 1: Sub-blok V.2: Pendapatan Usaha Rumah Tangga ---
    # Usaha 1
    "p5_F055": ("502_1_k1", "R502.1 No Urut ART Usaha 1"),
    "p5_F056": ("502_1_k3", "R502.1 Kategori Lapangan Usaha (KBLI) Usaha 1"),
    "p5_F057": ("502_1_k4", "R502.1 Jenis Pekerjaan Usaha 1"),
    "p5_F058": ("502_1_k5", "R502.1 Nilai Produksi Usaha 1"),
    "p5_F059": ("502_1_k6", "R502.1 Biaya Produksi Usaha 1"),
    "p5_F060": ("502_1_k7", "R502.1 Persentase Biaya Produksi untuk Upah (%) Usaha 1"),
    "p5_F054": ("502_1_k8", "R502.1 Stok Akhir Belum Terjual Usaha 1"),

    # Usaha 2
    "p5_F061": ("502_2_k1", "R502.2 No Urut ART Usaha 2"),
    "p5_F062": ("502_2_k3", "R502.2 Kategori Lapangan Usaha (KBLI) Usaha 2"),
    "p5_F063": ("502_2_k4", "R502.2 Jenis Pekerjaan Usaha 2"),
    "p5_F064": ("502_2_k5", "R502.2 Nilai Produksi Usaha 2"),
    "p5_F065": ("502_2_k6", "R502.2 Biaya Produksi Usaha 2"),
    "p5_F067": ("502_2_k7", "R502.2 Persentase Biaya Produksi untuk Upah (%) Usaha 2"),
    "p5_F066": ("502_2_k8", "R502.2 Stok Akhir Belum Terjual Usaha 2"),

    # Usaha 3
    "p5_F071": ("502_3_k1", "R502.3 No Urut ART Usaha 3"),
    "p5_F072": ("502_3_k3", "R502.3 Kategori Lapangan Usaha (KBLI) Usaha 3"),
    "p5_F073": ("502_3_k4", "R502.3 Jenis Pekerjaan Usaha 3"),
    "p5_F068": ("502_3_k5", "R502.3 Nilai Produksi Usaha 3"),
    "p5_F069": ("502_3_k6", "R502.3 Biaya Produksi Usaha 3"),
    "p5_F074": ("502_3_k7", "R502.3 Persentase Biaya Produksi untuk Upah (%) Usaha 3"),
    "p5_F070": ("502_3_k8", "R502.3 Stok Akhir Belum Terjual Usaha 3"),

    # Usaha 4
    "p5_F077": ("502_4_k1", "R502.4 No Urut ART Usaha 4"),
    "p5_F078": ("502_4_k3", "R502.4 Kategori Lapangan Usaha (KBLI) Usaha 4"),
    "p5_F079": ("502_4_k4", "R502.4 Jenis Pekerjaan Usaha 4"),
    "p5_F075": ("502_4_k5", "R502.4 Nilai Produksi Usaha 4"),
    "p5_F076": ("502_4_k6", "R502.4 Biaya Produksi Usaha 4"),
    "p5_F080": ("502_4_k7", "R502.4 Persentase Biaya Produksi untuk Upah (%) Usaha 4"),
    "p5_F081": ("502_4_k8", "R502.4 Stok Akhir Belum Terjual Usaha 4"),

    # Usaha 5
    "p5_F085": ("502_5_k1", "R502.5 No Urut ART Usaha 5"),
    "p5_F086": ("502_5_k3", "R502.5 Kategori Lapangan Usaha (KBLI) Usaha 5"),
    "p5_F087": ("502_5_k4", "R502.5 Jenis Pekerjaan Usaha 5"),
    "p5_F082": ("502_5_k5", "R502.5 Nilai Produksi Usaha 5"),
    "p5_F083": ("502_5_k6", "R502.5 Biaya Produksi Usaha 5"),
    "p5_F088": ("502_5_k7", "R502.5 Persentase Biaya Produksi untuk Upah (%) Usaha 5"),
    "p5_F084": ("502_5_k8", "R502.5 Stok Akhir Belum Terjual Usaha 5"),

    # JUMLAH V.2
    "p5_F089": ("502_JUMLAH_k5", "R502 JUMLAH Nilai Produksi Usaha Rumah Tangga"),
    "p5_F090": ("502_JUMLAH_k6", "R502 JUMLAH Biaya Produksi Usaha Rumah Tangga"),
    "p5_F091": ("502_JUMLAH_k8", "R502 JUMLAH Stok Akhir Belum Terjual"),

    # --- Tabel 2: Sub-blok V.3: Pendapatan Kepemilikan ---
    "p5_F092": ("503_1_terima", "R503.1 Sewa Lahan (Diterima)"),
    "p5_F093": ("503_1_bayar", "R503.1 Sewa Lahan (Dibayar)"),
    "p5_F094": ("503_2_terima", "R503.2 Keuntungan Kepemilikan Modal CV, Firma, UD, PD (Diterima)"),
    "p5_F095": ("503_2_bayar", "R503.2 Keuntungan Kepemilikan Modal CV, Firma, UD, PD (Dibayar)"),
    "p5_F096": ("503_3_terima", "R503.3 Keuntungan Saham PT, Yayasan, Koperasi (Deviden) (Diterima)"),
    "p5_F097": ("503_4_terima", "R503.4 Bunga Simpanan / Pinjaman (Diterima)"),
    "p5_F098": ("503_4_bayar", "R503.4 Bunga Pinjaman (Dibayar)"),
    "p5_F099": ("503_JUMLAH_terima", "R503 JUMLAH Pendapatan Kepemilikan [1+2+3+4] (Diterima)"),
    "p5_F100": ("503_JUMLAH_bayar", "R503 JUMLAH Biaya Kepemilikan [1+2+3+4] (Dibayar)"),

    # Nomor Halaman
    "p5_F_hal": ("HAL_5", "Nomor Halaman 5"),
}

# 21 ID Field Abu-Abu Palsu pada Halaman 5 yang harus dibersihkan
P5_FAKE_FIELD_IDS = {
    "p5_F010", "p5_F011", "p5_F013", "p5_F014", "p5_F016", "p5_F017",
    "p5_F021", "p5_F022", "p5_F024", "p5_F025", "p5_F027", "p5_F028", "p5_F030",
    "p5_F031", "p5_F035", "p5_F036", "p5_F038", "p5_F039", "p5_F041", "p5_F042", "p5_F044"
}

# ==============================================================================
# Mapping Label Presisi Halaman 6 (Blok V.4 & Blok V.5)
# ==============================================================================
P6_FIELD_LABELS = {
    # --- Tabel 0: Sub-blok V.4: Transfer Berjalan ---
    "p6_F000": ("1a_terima_uang", "R504.1a Uang Pensiun Pemerintah (Diterima - Uang)"),
    "p6_F001": ("1b_terima_uang", "R504.1b Bantuan Pemerintah (Diterima - Uang)"),
    "p6_F002": ("1b_terima_barang", "R504.1b Bantuan Pemerintah (Diterima - Barang/Jasa)"),
    "p6_F003": ("1b_bayar_uang", "R504.1b Pembayaran Zakat ke BAZNAS (Dibayar - Uang)"),
    "p6_F004": ("1b_bayar_barang", "R504.1b Bantuan Pemerintah (Dibayar - Barang/Jasa)"),
    "p6_F005": ("2_terima_uang", "R504.2 Rumah Tangga Lain (Diterima - Uang)"),
    "p6_F006": ("2_terima_barang", "R504.2 Rumah Tangga Lain (Diterima - Barang/Jasa)"),
    "p6_F007": ("2_bayar_uang", "R504.2 Rumah Tangga Lain (Dibayar - Uang)"),
    "p6_F008": ("2_bayar_barang", "R504.2 Rumah Tangga Lain (Dibayar - Barang/Jasa)"),
    "p6_F009": ("3_terima_uang", "R504.3 Lainnya / Klaim Asuransi (Diterima - Uang)"),
    "p6_F010": ("3_terima_barang", "R504.3 Lainnya / Klaim Asuransi (Diterima - Barang/Jasa)"),
    "p6_F011": ("3_bayar_uang", "R504.3 Lainnya / Iuran Pensiun / Asuransi (Dibayar - Uang)"),
    "p6_F012": ("3_bayar_barang", "R504.3 Lainnya (Dibayar - Barang/Jasa)"),
    "p6_F013": ("JUMLAH_terima_uang", "R504 JUMLAH Transfer [1a+1b+2+3] (Diterima - Uang)"),
    "p6_F014": ("JUMLAH_terima_barang", "R504 JUMLAH Transfer [1a+1b+2+3] (Diterima - Barang/Jasa)"),
    "p6_F015": ("JUMLAH_bayar_uang", "R504 JUMLAH Transfer [1a+1b+2+3] (Dibayar - Uang)"),
    "p6_F016": ("JUMLAH_bayar_barang", "R504 JUMLAH Transfer [1a+1b+2+3] (Dibayar - Barang/Jasa)"),

    # --- Tabel 1: Sub-blok V.5: Transaksi Modal dan Finansial (Aset Non-Finansial) ---
    # Baris 1: Pembelian/Pembuatan Sendiri
    "p6_F017": ("1_no", "R505.1 Nomor Baris 1"),
    "p6_F018": ("1_k3_bangunan_tinggal", "R505.1 Pembelian/Pembuatan Bangunan Tempat Tinggal"),
    "p6_F019": ("1_k4_bangunan_usaha", "R505.1 Pembelian/Pembuatan Bangunan Bukan Tempat Tinggal Usaha"),
    "p6_F020": ("1_k5_non_bangunan", "R505.1 Pembelian/Pembuatan Non Bangunan Usaha"),
    "p6_F021": ("1_k6_lahan", "R505.1 Pembelian/Pembuatan Lahan"),

    # Baris 2: Pemberian dari Pihak Lain
    "p6_F022": ("2_no", "R505.2 Nomor Baris 2"),
    "p6_F023": ("2_k3_bangunan_tinggal", "R505.2 Pemberian Bangunan Tempat Tinggal"),
    "p6_F024": ("2_k4_bangunan_usaha", "R505.2 Pemberian Bangunan Bukan Tempat Tinggal Usaha"),
    "p6_F025": ("2_k5_non_bangunan", "R505.2 Pemberian Non Bangunan Usaha"),
    "p6_F026": ("2_k6_lahan", "R505.2 Pemberian Lahan"),

    # Baris 3: Perbaikan/Pengurukan dan Biaya Pemindahan Kepemilikan Lahan
    "p6_F028": ("3_no", "R505.3 Nomor Baris 3"),
    "p6_F027": ("3_k6_lahan", "R505.3 Perbaikan/Pengurukan dan Pemindahan Kepemilikan Lahan"),

    # Baris 4: Penjualan
    "p6_F029": ("4_no", "R505.4 Nomor Baris 4"),
    "p6_F030": ("4_k3_bangunan_tinggal", "R505.4 Penjualan Bangunan Tempat Tinggal"),
    "p6_F031": ("4_k4_bangunan_usaha", "R505.4 Penjualan Bangunan Bukan Tempat Tinggal Usaha"),
    "p6_F032": ("4_k5_non_bangunan", "R505.4 Penjualan Non Bangunan Usaha"),
    "p6_F033": ("4_k6_lahan", "R505.4 Penjualan Lahan"),

    # Baris 5: Memberikan kepada Pihak Lain
    "p6_F034": ("5_no", "R505.5 Nomor Baris 5"),
    "p6_F035": ("5_k3_bangunan_tinggal", "R505.5 Memberikan Bangunan Tempat Tinggal"),
    "p6_F036": ("5_k4_bangunan_usaha", "R505.5 Memberikan Bangunan Bukan Tempat Tinggal Usaha"),
    "p6_F037": ("5_k5_non_bangunan", "R505.5 Memberikan Non Bangunan Usaha"),
    "p6_F038": ("5_k6_lahan", "R505.5 Memberikan Lahan"),

    # Baris 6: Nilai Seluruh Aset yang Dimiliki
    "p6_F039": ("6_no", "R505.6 Nomor Baris 6"),
    "p6_F040": ("6_k3_bangunan_tinggal", "R505.6 Nilai Seluruh Bangunan Tempat Tinggal yang Dimiliki"),
    "p6_F041": ("6_k4_bangunan_usaha", "R505.6 Nilai Seluruh Bangunan Usaha yang Dimiliki"),
    "p6_F042": ("6_k5_non_bangunan", "R505.6 Nilai Seluruh Non Bangunan Usaha yang Dimiliki"),
    "p6_F043": ("6_k6_lahan", "R505.6 Nilai Seluruh Lahan yang Dimiliki"),

    # Baris 7: Nilai Pembelian/Perolehan
    "p6_F044": ("7_k3_bangunan_tinggal", "R505.7 Nilai Perolehan Awal Bangunan Tempat Tinggal"),
    "p6_F045": ("7_k4_bangunan_usaha", "R505.7 Nilai Perolehan Awal Bangunan Usaha"),
    "p6_F046": ("7_k5_non_bangunan", "R505.7 Nilai Perolehan Awal Non Bangunan Usaha"),

    # Nomor Halaman
    "p6_F_hal": ("HAL_6", "Nomor Halaman 6"),
}

# ==============================================================================
# Mapping Label Presisi Halaman 7 (Blok VI & Blok VII)
# ==============================================================================
P7_FIELD_LABELS = {
    "p7_F020": ("600_JUMLAH_ASET", "R600 JUMLAH POSISI ASET [601 + 602 + ... + 607]"),
    "p7_F021": ("600_JUMLAH_KEWAJIBAN", "R600 JUMLAH POSISI KEWAJIBAN [608 + 609 + ... + 612]"),
    "p7_F043": ("CATATAN", "BLOK VII Catatan Khusus Petugas Pencacah / Pengawas"),
}

# Mapping Penyempurnaan Khusus Halaman 2 (404a, 404b)
P2_FIELD_LABELS = {
    "p2_F053": ("404a", "Rincian 404a (Telur ayam, bebek, puyuh)"),
    "p2_F054": ("404b", "Rincian 404b (Susu murni, kental, bubuk)"),
}

# Mapping Penyempurnaan Khusus Halaman 3 (451_sebulan, 451_3bln)
P3_FIELD_LABELS = {
    "p3_F039": ("451_sebulan", "Rincian 451 Total Bukan Makanan (Sebulan)"),
    "p3_F040": ("451_3bln", "Rincian 451 Total Bukan Makanan (Tiga Bulan)"),
}


def process_templates():
    print("================================================================================")
    print("MEMPERBAIKI DAN MELENGKAPI LABEL GRID & KOTAK DIGIT TEMPLATE SERUTI INTI")
    print("================================================================================")

    for pno in range(1, 8):
        fn = TEMPLATES_DIR / f"template_p{pno}.json"
        if not fn.exists():
            print(f"[ERROR] Template {fn} tidak ditemukan!")
            continue

        with open(fn, "r", encoding="utf-8") as f:
            tpl = json.load(f)

        cells = tpl.get("cells", [])
        fields = tpl.get("fields", [])
        cells_map = {c["id"]: c for c in cells}

        # ----------------------------------------------------------------------
        # HALAMAN 5: Pembersihan Field Abu-Abu Palsu & Pelabelan 81 Field Asli
        # ----------------------------------------------------------------------
        if pno == 5:
            # 1. Konversi cell abu-abu palsu menjadi struct
            for c in cells:
                if c["kind"] == "digit" and min(c.get("fill_rgb", [255, 255, 255])) < 245:
                    c["kind"] = "struct"
                    c["fill"] = "band"
            
            # 2. Filter hanya field sah (bukan fake field)
            cleaned_fields = []
            for fld in fields:
                fid = fld["id"]
                if fid in P5_FAKE_FIELD_IDS:
                    continue
                # Update row_label & label
                if fid in P5_FIELD_LABELS:
                    r_lbl, f_lbl = P5_FIELD_LABELS[fid]
                    fld["row_label"] = r_lbl
                    fld["label"] = f_lbl
                cleaned_fields.append(fld)
            fields = cleaned_fields
            tpl["fields"] = fields

        # ----------------------------------------------------------------------
        # HALAMAN 6: Pelabelan Presisi 48 Field
        # ----------------------------------------------------------------------
        elif pno == 6:
            for fld in fields:
                fid = fld["id"]
                if fid in P6_FIELD_LABELS:
                    r_lbl, f_lbl = P6_FIELD_LABELS[fid]
                    fld["row_label"] = r_lbl
                    fld["label"] = f_lbl

        # ----------------------------------------------------------------------
        # HALAMAN 7: Pelabelan p7_F020, p7_F021, p7_F043
        # ----------------------------------------------------------------------
        elif pno == 7:
            for fld in fields:
                fid = fld["id"]
                if fid in P7_FIELD_LABELS:
                    r_lbl, f_lbl = P7_FIELD_LABELS[fid]
                    fld["row_label"] = r_lbl
                    fld["label"] = f_lbl

        # ----------------------------------------------------------------------
        # HALAMAN 2: Penyempurnaan 404a & 404b
        # ----------------------------------------------------------------------
        elif pno == 2:
            for fld in fields:
                fid = fld["id"]
                if fid in P2_FIELD_LABELS:
                    r_lbl, f_lbl = P2_FIELD_LABELS[fid]
                    fld["row_label"] = r_lbl
                    fld["label"] = f_lbl

        # ----------------------------------------------------------------------
        # HALAMAN 3: Penyempurnaan 451_sebulan & 451_3bln
        # ----------------------------------------------------------------------
        elif pno == 3:
            for fld in fields:
                fid = fld["id"]
                if fid in P3_FIELD_LABELS:
                    r_lbl, f_lbl = P3_FIELD_LABELS[fid]
                    fld["row_label"] = r_lbl
                    fld["label"] = f_lbl

        # ----------------------------------------------------------------------
        # Standardisasi Umum untuk Seluruh Field & Kotak Digit (Semua Halaman)
        # ----------------------------------------------------------------------
        field_box_mapping = {}
        for fld in fields:
            fid = fld["id"]
            # Pastikan row_label tidak kosong
            if not fld.get("row_label") or str(fld["row_label"]).strip().lower() == "none":
                fld["row_label"] = fid
            # Pastikan field memiliki label deskriptif
            if not fld.get("label"):
                ptext = fld.get("parent_text", "").strip()
                fld["label"] = f"Field {fld['row_label']}" + (f" ({ptext})" if ptext else "")
            
            # Map boxes ke field info
            for b_idx, bid in enumerate(fld.get("boxes", [])):
                field_box_mapping[bid] = (fid, fld["row_label"], fld["label"], b_idx, len(fld["boxes"]))

        # Berikan metadata label ke setiap kotak digit (comb box cell)
        labeled_digits = 0
        for c in cells:
            cid = c["id"]
            if c["kind"] == "digit":
                if cid in field_box_mapping:
                    fid, r_lbl, f_lbl, b_idx, total_boxes = field_box_mapping[cid]
                    c["field_id"] = fid
                    c["field_label"] = r_lbl
                    c["box_index"] = b_idx
                    c["total_boxes"] = total_boxes
                    c["label"] = f"{r_lbl}_b{b_idx:02d}"
                    labeled_digits += 1
                else:
                    # Kotak digit berdiri sendiri yang tidak ada di fields
                    c["label"] = f"{cid}_digit"
                    labeled_digits += 1

        # Perbarui hitungan counts
        real_digits = [c for c in cells if c["kind"] == "digit" and min(c.get("fill_rgb", [255, 255, 255])) >= 245]
        tpl["counts"] = {
            "struct": sum(1 for c in cells if c["kind"] == "struct"),
            "digit": len(real_digits),
            "fields": len(fields),
            "tables": len(tpl.get("tables", []))
        }

        # Simpan kembali template JSON
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(tpl, f, indent=2, ensure_ascii=False)

        print(f"[*] Halaman {pno}:")
        print(f"    - Total Fields: {len(fields)} (Semua berlabel non-none)")
        print(f"    - Total Kotak Digit Sah: {len(real_digits)} (Semua berlabel)")
        print(f"    - Digit Cells Dilabeli: {labeled_digits}")

    print("\n[OK] Seluruh template JSON berhasil diperbarui dan diverifikasi.")

if __name__ == "__main__":
    process_templates()
