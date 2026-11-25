#!/usr/bin/env python3
"""
tools/standardize_all_grid_labels.py
====================================
Menyeragamkan seluruh label field isian dan kotak digit (comb box) pada
kuesioner BPS Seruti Inti 2026 untuk SELURUH HALAMAN 1 sampai 7.

Format Standar Seragam:
{NomorRincian/Tabel/Blok}_{NoUrut/Subrincian}_{Kolom/Periode}
Contoh:
- Hal 1: 101_prov, 102_kab, ..., 201_pencacah_tgl, 201_pencacah_nip, dst.
- Hal 2: 301_01_k3 (Hub KRT), 301_01_k4 (Status Kawin), 301_01_k5 (JK), 301_01_k6 (Umur), 301_01_k7 (Ijazah) s.d. 301_10_k7
         401_sebulan s.d. 414_sebulan, 413_sebulan & 413_3bln
- Hal 3: 415_sebulan s.d. 451_3bln (dengan pembeda _sebulan dan _3bln)
- Hal 4: 452_sebulan & 452_3bln s.d. 463_3bln, 464_kode & 464_alasan s.d. 469_alasan, 470a, 470b_simpanan, dst.
- Hal 5: 501_1_k1 s.d. 501_JUMLAH_k7, 502_1_k1 s.d. 502_JUMLAH_k8, 503_1_terima s.d. 503_JUMLAH_bayar
- Hal 6: 504_1a_terima_uang s.d. 504_JUMLAH_bayar_barang, 505_1_no, 505_1_k3_bangunan_tinggal s.d. 505_7_k5_non_bangunan
- Hal 7: 601_uang_tunai s.d. 607_arisan, 600_JUMLAH_POSISI_ASET, 608a_pinjaman_bank s.d. 612_arisan, 600_JUMLAH_POSISI_KEWAJIBAN,
         701_kunj1_tgl s.d. 703_kunj3_sel_men, 700_catatan, 700_waktu_selesai
"""

import json
from pathlib import Path

TEMPLATES_DIR = Path("templates")

# ==============================================================================
# 1. HALAMAN 1: Blok I (Pengenalan Tempat) & Blok II (Keterangan Petugas)
# ==============================================================================
P1_FIELD_LABELS = {
    "p1_F_triwulan": ("TRIWULAN", "Triwulan Pendataan"),
    "p1_F000": ("WAKTU_MULAI", "Waktu Mulai Wawancara (Jam : Menit)"),
    "p1_F001": ("101_prov", "R101 Provinsi"),
    "p1_F002": ("102_kab", "R102 Kabupaten/Kota"),
    "p1_F003": ("103_kec", "R103 Kecamatan"),
    "p1_F004": ("104_desa", "R104 Desa/Kelurahan"),
    "p1_F005": ("105_klasifikasi", "R105 Klasifikasi Desa/Kelurahan (1: Perkotaan, 2: Perdesaan)"),
    "p1_F006": ("106_sls", "R106 Nomor Blok Sensus / SLS"),
    "p1_F007": ("107_sub_sls", "R107 Nomor Sub-SLS"),
    "p1_F008": ("109_sampel", "R109 Nomor Urut Sampel Rumah Tangga"),
    "p1_F009": ("201_pencacah_tgl", "R201 Tanggal Pencacahan (Tgl)"),
    "p1_F010": ("201_pencacah_jabatan", "R201 Jabatan Pencacah (1: Staf Prov, 2: Staf Kab/Kota, 3: Mitra)"),
    "p1_F011": ("201_pencacah_nip", "R201 NIP / ID Mitra Pencacah"),
    "p1_F012": ("201_pencacah_bln", "R201 Bulan Pencacahan (Bln)"),
    "p1_F013": ("202_pengawas_tgl", "R202 Tanggal Pengawasan/Pemeriksaan (Tgl)"),
    "p1_F014": ("202_pengawas_jabatan", "R202 Jabatan Pengawas (1: Staf Prov, 2: Staf Kab/Kota, 3: Mitra)"),
    "p1_F015": ("202_pengawas_nip", "R202 NIP / ID Mitra Pengawas"),
    "p1_F016": ("202_pengawas_bln", "R202 Bulan Pengawasan/Pemeriksaan (Bln)"),
    "p1_F017": ("203_hasil_pencacahan", "R203 Hasil Pencacahan Rumah Tangga"),
    "p1_F_kesediaan": ("KESEDIAAN_RESPONDEN", "Kesediaan Responden Diwawancarai"),
    "p1_F_hal": ("HAL_1", "Nomor Halaman 1"),
}

# ==============================================================================
# 2. HALAMAN 2: Blok III (Roster ART) & Blok IV.1-IV.2.A (Bukan Makanan Bagian 1)
# ==============================================================================
P2_FIELD_LABELS = {
    # Roster 10 ART (Kolom k3: Hub KRT, k4: Status Kawin, k5: JK, k6: Umur, k7: Ijazah)
    "p2_F000": ("301_01_k3", "R301.01 Hubungan dengan KRT ART 01"),
    "p2_F001": ("301_01_k4", "R301.01 Status Perkawinan ART 01"),
    "p2_F002": ("301_01_k5", "R301.01 Jenis Kelamin ART 01"),
    "p2_F003": ("301_01_k6", "R301.01 Umur (tahun) ART 01"),
    "p2_F004": ("301_01_k7", "R301.01 Ijazah/STTB Tertinggi ART 01"),

    "p2_F005": ("301_02_k3", "R301.02 Hubungan dengan KRT ART 02"),
    "p2_F006": ("301_02_k4", "R301.02 Status Perkawinan ART 02"),
    "p2_F007": ("301_02_k5", "R301.02 Jenis Kelamin ART 02"),
    "p2_F008": ("301_02_k6", "R301.02 Umur (tahun) ART 02"),
    "p2_F009": ("301_02_k7", "R301.02 Ijazah/STTB Tertinggi ART 02"),

    "p2_F010": ("301_03_k3", "R301.03 Hubungan dengan KRT ART 03"),
    "p2_F011": ("301_03_k4", "R301.03 Status Perkawinan ART 03"),
    "p2_F012": ("301_03_k5", "R301.03 Jenis Kelamin ART 03"),
    "p2_F013": ("301_03_k6", "R301.03 Umur (tahun) ART 03"),
    "p2_F014": ("301_03_k7", "R301.03 Ijazah/STTB Tertinggi ART 03"),

    "p2_F015": ("301_04_k3", "R301.04 Hubungan dengan KRT ART 04"),
    "p2_F016": ("301_04_k4", "R301.04 Status Perkawinan ART 04"),
    "p2_F017": ("301_04_k5", "R301.04 Jenis Kelamin ART 04"),
    "p2_F018": ("301_04_k6", "R301.04 Umur (tahun) ART 04"),
    "p2_F019": ("301_04_k7", "R301.04 Ijazah/STTB Tertinggi ART 04"),

    "p2_F020": ("301_05_k3", "R301.05 Hubungan dengan KRT ART 05"),
    "p2_F021": ("301_05_k4", "R301.05 Status Perkawinan ART 05"),
    "p2_F022": ("301_05_k5", "R301.05 Jenis Kelamin ART 05"),
    "p2_F023": ("301_05_k6", "R301.05 Umur (tahun) ART 05"),
    "p2_F024": ("301_05_k7", "R301.05 Ijazah/STTB Tertinggi ART 05"),

    "p2_F025": ("301_06_k3", "R301.06 Hubungan dengan KRT ART 06"),
    "p2_F026": ("301_06_k4", "R301.06 Status Perkawinan ART 06"),
    "p2_F027": ("301_06_k5", "R301.06 Jenis Kelamin ART 06"),
    "p2_F028": ("301_06_k6", "R301.06 Umur (tahun) ART 06"),
    "p2_F029": ("301_06_k7", "R301.06 Ijazah/STTB Tertinggi ART 06"),

    "p2_F030": ("301_07_k3", "R301.07 Hubungan dengan KRT ART 07"),
    "p2_F031": ("301_07_k4", "R301.07 Status Perkawinan ART 07"),
    "p2_F032": ("301_07_k5", "R301.07 Jenis Kelamin ART 07"),
    "p2_F033": ("301_07_k6", "R301.07 Umur (tahun) ART 07"),
    "p2_F034": ("301_07_k7", "R301.07 Ijazah/STTB Tertinggi ART 07"),

    "p2_F035": ("301_08_k3", "R301.08 Hubungan dengan KRT ART 08"),
    "p2_F036": ("301_08_k4", "R301.08 Status Perkawinan ART 08"),
    "p2_F037": ("301_08_k5", "R301.08 Jenis Kelamin ART 08"),
    "p2_F038": ("301_08_k6", "R301.08 Umur (tahun) ART 08"),
    "p2_F039": ("301_08_k7", "R301.08 Ijazah/STTB Tertinggi ART 08"),

    "p2_F040": ("301_09_k3", "R301.09 Hubungan dengan KRT ART 09"),
    "p2_F041": ("301_09_k4", "R301.09 Status Perkawinan ART 09"),
    "p2_F042": ("301_09_k5", "R301.09 Jenis Kelamin ART 09"),
    "p2_F043": ("301_09_k6", "R301.09 Umur (tahun) ART 09"),
    "p2_F044": ("301_09_k7", "R301.09 Ijazah/STTB Tertinggi ART 09"),

    "p2_F045": ("301_10_k3", "R301.10 Hubungan dengan KRT ART 10"),
    "p2_F046": ("301_10_k4", "R301.10 Status Perkawinan ART 10"),
    "p2_F047": ("301_10_k5", "R301.10 Jenis Kelamin ART 10"),
    "p2_F048": ("301_10_k6", "R301.10 Umur (tahun) ART 10"),
    "p2_F049": ("301_10_k7", "R301.10 Ijazah/STTB Tertinggi ART 10"),

    # Blok IV.1 & Blok IV.2.A: Pengeluaran Bukan Makanan Selama Sebulan dan Tiga Bulan Terakhir
    # Kolom (3): Sebulan Terakhir (Rp) | Kolom (4): Tiga Bulan Terakhir (Rp)
    "p2_F050": ("401_3bln", "R401 Pakaian, alas kaki, dan tutup kepala [402 + 403] (Tiga Bulan)"),
    "p2_F051": ("402_3bln", "R402 Pakaian dan tutup kepala (kain, pakaian jadi, seragam, topi, laundry, dsb) (Tiga Bulan)"),
    "p2_F052": ("403_3bln", "R403 Alas kaki (sendal, sepatu, sol dan pemeliharaan) (Tiga Bulan)"),
    "p2_F053": ("404_sebulan", "R404 Perumahan, listrik, air, dan bahan bakar memasak (Sebulan)"),
    "p2_F054": ("404_3bln", "R404 Perumahan, listrik, air, dan bahan bakar memasak (Tiga Bulan)"),
    "p2_F055": ("405_status_bangunan", "R405 Status Penguasaan Bangunan Tempat Tinggal yang Ditempati"),
    "p2_F056": ("406a_sebulan", "R406a Perkiraan Sewa Rumah Milik Sendiri / Bebas Sewa (Sebulan)"),
    "p2_F057": ("406b_sebulan", "R406b Nilai Kontrak Rumah (Sebulan)"),
    "p2_F058": ("406c_sebulan", "R406c Nilai Sewa Rumah (Sebulan)"),
    "p2_F059": ("406d_sebulan", "R406d Perkiraan Sewa Rumah Dinas / Lainnya (Sebulan)"),
    "p2_F060": ("407_3bln", "R407 Pemeliharaan rumah dan perbaikan ringan (Tiga Bulan)"),
    "p2_F061": ("408_sebulan", "R408 Listrik PLN, non PLN, bahan bakar generator (Sebulan)"),
    "p2_F062": ("409_sebulan", "R409 Air PAM, pikulan, beli, dsb (Sebulan)"),
    "p2_F063": ("410_sebulan", "R410 Gas LPG, gas kota, biogas (Sebulan)"),
    "p2_F064": ("411_sebulan", "R411 Bahan bakar lainnya (minyak tanah, kayu bakar, arang) (Sebulan)"),
    "p2_F065": ("412_sebulan", "R412 Biaya pengumpulan/pemungutan sampah/limbah (Sebulan)"),
    "p2_F066": ("413_sebulan", "R413 Furnitur, peralatan, dan pemeliharaan rumah [414 + 415 + 416] (Sebulan)"),
    "p2_F067": ("413_3bln", "R413 Furnitur, peralatan, dan pemeliharaan rumah [414 + 415 + 416] (Tiga Bulan)"),
    "p2_F068": ("414_3bln", "R414 Furnitur, peralatan, dan barang pecah belah (Tiga Bulan)"),
    "p2_F_hal": ("HAL_2", "Nomor Halaman 2"),
}

# ==============================================================================
# 3. HALAMAN 3: Blok IV.2.A (Pengeluaran Bukan Makanan Bagian 2)
# ==============================================================================
P3_FIELD_LABELS = {
    # Kolom (3): Sebulan Terakhir (Rp) | Kolom (4): Tiga Bulan Terakhir (Rp)
    "p3_F000": ("415_sebulan", "R415 Bahan pembersih dan pemeliharaan rumah (deterjen, sabun cuci, karbol) (Sebulan)"),
    "p3_F001": ("416_sebulan", "R416 Biaya jasa pekerja domestik (pembantu ruta, supir, dsb) (Sebulan)"),
    "p3_F002": ("417_sebulan", "R417 Transportasi [418 + 419 + 420 + 421 + 422 + 423 + 424 + 425] (Sebulan)"),
    "p3_F003": ("417_3bln", "R417 Transportasi [418 + 419 + 420 + 421 + 422 + 423 + 424 + 425] (Tiga Bulan)"),
    "p3_F004": ("418_3bln", "R418 Pembelian kendaraan (sepeda motor, mobil, sepeda, perahu motor) (Tiga Bulan)"),
    "p3_F005": ("419_sebulan", "R419 Bahan bakar kendaraan (bensin, solar, dsb) (Sebulan)"),
    "p3_F006": ("420_3bln", "R420 Pemeliharaan, pelumas, suku cadang kendaraan (servis, ganti oli, ban) (Tiga Bulan)"),
    "p3_F007": ("421_3bln", "R421 Sewa kendaraan, tol, parkir, biaya SIM, kursus mengemudi (Tiga Bulan)"),
    "p3_F008": ("422_3bln", "R422 Biaya angkutan penumpang darat (bus, angkot, taksi, ojek, kereta) (Tiga Bulan)"),
    "p3_F009": ("423_3bln", "R423 Biaya angkutan penumpang air (ferry, perahu motor, kapal laut) (Tiga Bulan)"),
    "p3_F010": ("424_3bln", "R424 Biaya angkutan penumpang udara (Tiga Bulan)"),
    "p3_F011": ("425_3bln", "R425 Ongkos kirim dan pengiriman barang lainnya (ongkir olshop, kurir) (Tiga Bulan)"),
    "p3_F012": ("426_sebulan", "R426 Komunikasi [427 + 428 + 429] (Sebulan)"),
    "p3_F013": ("426_3bln", "R426 Komunikasi [427 + 428 + 429] (Tiga Bulan)"),
    "p3_F014": ("427_3bln", "R427 Alat komunikasi dan informasi (HP, laptop, printer, aksesoris) (Tiga Bulan)"),
    "p3_F015": ("428_sebulan", "R428 Rekening telepon, pulsa, paket data, biaya internet (Sebulan)"),
    "p3_F016": ("429_sebulan", "R429 Langganan TV kabel dan streaming film/musik (Sebulan)"),
    "p3_F017": ("430_3bln", "R430 Rekreasi, Olahraga, dan Budaya [431 + 432 + 433 + 434 + 435 + 436] (Tiga Bulan)"),
    "p3_F018": ("431_3bln", "R431 Mainan, barang rekreasi, alat musik, alat olahraga (kamera, gitar, bola) (Tiga Bulan)"),
    "p3_F019": ("432_3bln", "R432 Tiket wisata, bioskop, pertunjukan, biaya olahraga (gym, renang, klub) (Tiga Bulan)"),
    "p3_F020": ("433_3bln", "R433 Hewan dan tanaman peliharaan (kandang, akuarium, pupuk, pakan hewan) (Tiga Bulan)"),
    "p3_F021": ("434_3bln", "R434 Buku, majalah, surat kabar, alat tulis (buku pelajaran, novel, komik) (Tiga Bulan)"),
    "p3_F022": ("435_3bln", "R435 Paket liburan, umroh, haji, dan perjalanan rohani (Tiga Bulan)"),
    "p3_F023": ("436_3bln", "R436 Hotel dan akomodasi lainnya (hotel, villa, homestay) (Tiga Bulan)"),
    "p3_F024": ("437_sebulan", "R437 Barang dan Jasa Lainnya [438 + 439 + 440 + 441 + 442] (Sebulan)"),
    "p3_F025": ("437_3bln", "R437 Barang dan Jasa Lainnya [438 + 439 + 440 + 441 + 442] (Tiga Bulan)"),
    "p3_F026": ("438_sebulan", "R438 Produk perawatan diri, kosmetik, kecantikan (sabun, parfum, makeup) (Sebulan)"),
    "p3_F027": ("439_sebulan", "R439 Jasa perawatan diri dan kecantikan (salon, cukur rambut, dsb) (Sebulan)"),
    "p3_F028": ("440_3bln", "R440 Pembelian perhiasan, jam tangan, tas, koper, stroller (Tiga Bulan)"),
    "p3_F029": ("441_3bln", "R441 Biaya administrasi dan jasa lembaga keuangan (admin bank, transfer) (Tiga Bulan)"),
    "p3_F030": ("442_3bln", "R442 Biaya jasa lainnya (fotokopi, paspor/visa/kependudukan, penitipan anak) (Tiga Bulan)"),
    "p3_F031": ("443_3bln", "R443 Pajak dan Premi Asuransi [444 + 445] (Tiga Bulan)"),
    "p3_F032": ("444a_3bln", "R444a Pajak (PBB, STNK, PPh, dsb) (Tiga Bulan)"),
    "p3_F033": ("444b_3bln", "R444b Tilang, denda, dsb (Tiga Bulan)"),
    "p3_F034": ("445_3bln", "R445 Premi asuransi (kesehatan, kerugian, kecelakaan, dsb) (Tiga Bulan)"),
    "p3_F035": ("446_3bln", "R446 Keperluan Pesta dan Upacara/Kenduri [447 + 448 + 449 + 450] (Tiga Bulan)"),
    "p3_F036": ("447_3bln", "R447 Perkawinan (Tiga Bulan)"),
    "p3_F037": ("448_3bln", "R448 Ulang tahun dan khitanan (Tiga Bulan)"),
    "p3_F038": ("449_3bln", "R449 Pemakaman (Tiga Bulan)"),
    "p3_F039": ("450_3bln", "R450 Lainnya (Tiga Bulan)"),
    "p3_F040": ("451_sebulan", "R451 TOTAL PENGELUARAN BUKAN MAKANAN (Sebulan)"),
    "p3_F041": ("451_3bln", "R451 TOTAL PENGELUARAN BUKAN MAKANAN (Tiga Bulan)"),
    "p3_F_hal": ("HAL_3", "Nomor Halaman 3"),
}

# ==============================================================================
# 4. HALAMAN 4: Blok IV.2.B (Kesehatan & Pendidikan) & Blok IV.3 (Persepsi)
# ==============================================================================
P4_FIELD_LABELS = {
    # Blok IV.2.B: Pengeluaran Kesehatan dan Pendidikan Selama Tiga Bulan Terakhir
    # Kolom (3): Total (Rp) | Kolom (4): Out of Pocket / OOP (Rp)
    "p4_F000": ("452_total", "R452 Kesehatan [453 + 454 + 455] (Total)"),
    "p4_F001": ("452_oop", "R452 Kesehatan [453 + 454 + 455] (Out of Pocket)"),
    "p4_F002": ("453_total", "R453 Obat-obatan dan produk kesehatan (obat, vitamin, jamu, kacamata) (Total)"),
    "p4_F003": ("453_oop", "R453 Obat-obatan dan produk kesehatan (obat, vitamin, jamu, kacamata) (Out of Pocket)"),
    "p4_F004": ("454_total", "R454 Biaya rawat jalan dan laboratorium (konsultasi, lab, pengobatan) (Total)"),
    "p4_F005": ("454_oop", "R454 Biaya rawat jalan dan laboratorium (konsultasi, lab, pengobatan) (Out of Pocket)"),
    "p4_F006": ("455_total", "R455 Biaya rawat inap (kamar, konsultasi, obat inap) (Total)"),
    "p4_F007": ("455_oop", "R455 Biaya rawat inap (kamar, konsultasi, obat inap) (Out of Pocket)"),
    "p4_F008": ("456_total", "R456 Pendidikan [457 + 458 + 459 + 460 + 461 + 462] (Total)"),
    "p4_F009": ("456_oop", "R456 Pendidikan [457 + 458 + 459 + 460 + 461 + 462] (Out of Pocket)"),
    "p4_F010": ("457_total", "R457 Biaya pra sekolah (PAUD, TK) (Total)"),
    "p4_F011": ("457_oop", "R457 Biaya pra sekolah (PAUD, TK) (Out of Pocket)"),
    "p4_F012": ("458_total", "R458 Biaya sekolah dasar (SD) / sederajat (Total)"),
    "p4_F013": ("458_oop", "R458 Biaya sekolah dasar (SD) / sederajat (Out of Pocket)"),
    "p4_F014": ("459_total", "R459 Biaya sekolah menengah pertama (SMP) / sederajat (Total)"),
    "p4_F015": ("459_oop", "R459 Biaya sekolah menengah pertama (SMP) / sederajat (Out of Pocket)"),
    "p4_F016": ("460_total", "R460 Biaya sekolah menengah atas (SMA) / sederajat (Total)"),
    "p4_F017": ("460_oop", "R460 Biaya sekolah menengah atas (SMA) / sederajat (Out of Pocket)"),
    "p4_F018": ("461_total", "R461 Biaya pendidikan tinggi (Akademi, Universitas) (Total)"),
    "p4_F019": ("461_oop", "R461 Biaya pendidikan tinggi (Akademi, Universitas) (Out of Pocket)"),
    "p4_F020": ("462_total", "R462 Biaya kursus dan bimbingan belajar di luar sekolah (Total)"),
    "p4_F021": ("462_oop", "R462 Biaya kursus dan bimbingan belajar di luar sekolah (Out of Pocket)"),
    "p4_F022": ("463_total", "R463 TOTAL PENGELUARAN KESEHATAN DAN PENDIDIKAN [452 + 456] (Total)"),
    "p4_F023": ("463_oop", "R463 TOTAL PENGELUARAN KESEHATAN DAN PENDIDIKAN [452 + 456] (Out of Pocket)"),

    # Blok IV.3: Persepsi Terhadap Kondisi Ekonomi Rumah Tangga Triwulan Ini
    # Kolom (3): Perubahan Kode (1: Naik, 2: Tetap, 3: Turun, 5: Tidak relevan)
    # Kolom (4): Persentase Perubahan (%) jika kode 1 atau 3
    "p4_F024": ("464_kode", "R464 Kecenderungan Pengeluaran Konsumsi RT (Kode: 1 Naik, 2 Tetap, 3 Turun, 5 Tidak relevan)"),
    "p4_F025": ("464_persen", "R464 Persentase Perubahan Pengeluaran Konsumsi RT (%)"),
    "p4_F026": ("465_kode", "R465 Kecenderungan Konsumsi Bahan Makanan (Kode: 1, 2, 3, 5)"),
    "p4_F027": ("465_persen", "R465 Persentase Perubahan Konsumsi Bahan Makanan (%)"),
    "p4_F028": ("466_kode", "R466 Kecenderungan Makanan Jadi Restoran/Kaki Lima (Kode: 1, 2, 3, 5)"),
    "p4_F029": ("466_persen", "R466 Persentase Perubahan Makanan Jadi Restoran (%)"),
    "p4_F030": ("467_kode", "R467 Kecenderungan Belanja Online (Kode: 1, 2, 3, 5)"),
    "p4_F031": ("467_persen", "R467 Persentase Perubahan Belanja Online (%)"),
    "p4_F032": ("468_kode", "R468 Kecenderungan Pendapatan/Upah RT (Kode: 1, 2, 3, 5)"),
    "p4_F033": ("468_persen", "R468 Persentase Perubahan Pendapatan/Upah RT (%)"),
    "p4_F034": ("469_kode", "R469 Kecenderungan Keuntungan Usaha RT (Kode: 1, 2, 3, 5)"),
    "p4_F035": ("469_persen", "R469 Persentase Perubahan Keuntungan Usaha RT (%)"),

    # R470: Kecukupan Pendapatan & Sumber Pembiayaan
    "p4_F036": ("470a", "R470a Apakah Pendapatan Rumah Tangga Memenuhi Konsumsi (1: Ya, 5: Tidak)"),
    "p4_F037": ("470bi_simpanan", "R470bi Mengambil Sisa Simpanan Triwulan Sebelumnya (1: Ya, 5: Tidak)"),
    "p4_F038": ("470bii_pinjaman", "R470bii Melakukan Kredit / Pinjaman (1: Ya, 5: Tidak)"),
    "p4_F039": ("470biii_jual_aset", "R470biii Menjual Aset / Barang Tahan Lama (1: Ya, 5: Tidak)"),
    "p4_F_hal": ("HAL_4", "Nomor Halaman 4"),
}

# ==============================================================================
# 5. HALAMAN 5: Blok V.1 (Gaji), Blok V.2 (Usaha RT), Blok V.3 (Kepemilikan)
# ==============================================================================
P5_FIELD_LABELS = {
    # --- V.1 Upah & Gaji Karyawan ---
    "p5_F000": ("501_1_k1", "R501.1 No Urut ART Baris 1"),
    "p5_F001": ("501_1_k3", "R501.1 Kategori Lapangan Usaha (KBLI) Baris 1"),
    "p5_F002": ("501_1_k4", "R501.1 Jenis Pekerjaan Baris 1"),
    "p5_F003": ("501_1_k5", "R501.1 Upah/Gaji Berupa Uang Baris 1"),
    "p5_F004": ("501_1_k6", "R501.1 Upah/Gaji Berupa Barang/Jasa Baris 1"),
    "p5_F005": ("501_1_k7", "R501.1 Lembur, Honorarium, THR Baris 1"),

    "p5_F006": ("501_2_k1", "R501.2 No Urut ART Baris 2"),
    "p5_F007": ("501_2_k3", "R501.2 Kategori Lapangan Usaha (KBLI) Baris 2"),
    "p5_F008": ("501_2_k4", "R501.2 Jenis Pekerjaan Baris 2"),
    "p5_F009": ("501_2_k5", "R501.2 Upah/Gaji Berupa Uang Baris 2"),
    "p5_F012": ("501_2_k6", "R501.2 Upah/Gaji Berupa Barang/Jasa Baris 2"),
    "p5_F015": ("501_2_k7", "R501.2 Lembur, Honorarium, THR Baris 2"),

    "p5_F018": ("501_3_k1", "R501.3 No Urut ART Baris 3"),
    "p5_F019": ("501_3_k3", "R501.3 Kategori Lapangan Usaha (KBLI) Baris 3"),
    "p5_F020": ("501_3_k4", "R501.3 Jenis Pekerjaan Baris 3"),
    "p5_F023": ("501_3_k5", "R501.3 Upah/Gaji Berupa Uang Baris 3"),
    "p5_F026": ("501_3_k6", "R501.3 Upah/Gaji Berupa Barang/Jasa Baris 3"),
    "p5_F029": ("501_3_k7", "R501.3 Lembur, Honorarium, THR Baris 3"),

    "p5_F032": ("501_4_k1", "R501.4 No Urut ART Baris 4"),
    "p5_F033": ("501_4_k3", "R501.4 Kategori Lapangan Usaha (KBLI) Baris 4"),
    "p5_F034": ("501_4_k4", "R501.4 Jenis Pekerjaan Baris 4"),
    "p5_F037": ("501_4_k5", "R501.4 Upah/Gaji Berupa Uang Baris 4"),
    "p5_F040": ("501_4_k6", "R501.4 Upah/Gaji Berupa Barang/Jasa Baris 4"),
    "p5_F043": ("501_4_k7", "R501.4 Lembur, Honorarium, THR Baris 4"),

    "p5_F045": ("501_5_k1", "R501.5 No Urut ART Baris 5"),
    "p5_F046": ("501_5_k3", "R501.5 Kategori Lapangan Usaha (KBLI) Baris 5"),
    "p5_F047": ("501_5_k4", "R501.5 Jenis Pekerjaan Baris 5"),
    "p5_F048": ("501_5_k5", "R501.5 Upah/Gaji Berupa Uang Baris 5"),
    "p5_F049": ("501_5_k6", "R501.5 Upah/Gaji Berupa Barang/Jasa Baris 5"),
    "p5_F050": ("501_5_k7", "R501.5 Lembur, Honorarium, THR Baris 5"),

    "p5_F053": ("501_JUMLAH_k5", "R501 JUMLAH Upah/Gaji Berupa Uang"),
    "p5_F052": ("501_JUMLAH_k6", "R501 JUMLAH Upah/Gaji Berupa Barang/Jasa"),
    "p5_F051": ("501_JUMLAH_k7", "R501 JUMLAH Lembur, Honorarium, THR"),

    # --- V.2 Pendapatan Usaha Rumah Tangga ---
    "p5_F055": ("502_1_k1", "R502.1 No Urut ART Usaha 1"),
    "p5_F056": ("502_1_k3", "R502.1 Kategori Lapangan Usaha (KBLI) Usaha 1"),
    "p5_F057": ("502_1_k4", "R502.1 Jenis Pekerjaan Usaha 1"),
    "p5_F058": ("502_1_k5", "R502.1 Nilai Produksi Usaha 1"),
    "p5_F059": ("502_1_k6", "R502.1 Biaya Produksi Usaha 1"),
    "p5_F060": ("502_1_k7", "R502.1 Persentase Biaya Produksi untuk Upah (%) Usaha 1"),
    "p5_F054": ("502_1_k8", "R502.1 Stok Akhir Belum Terjual Usaha 1"),

    "p5_F061": ("502_2_k1", "R502.2 No Urut ART Usaha 2"),
    "p5_F062": ("502_2_k3", "R502.2 Kategori Lapangan Usaha (KBLI) Usaha 2"),
    "p5_F063": ("502_2_k4", "R502.2 Jenis Pekerjaan Usaha 2"),
    "p5_F064": ("502_2_k5", "R502.2 Nilai Produksi Usaha 2"),
    "p5_F065": ("502_2_k6", "R502.2 Biaya Produksi Usaha 2"),
    "p5_F067": ("502_2_k7", "R502.2 Persentase Biaya Produksi untuk Upah (%) Usaha 2"),
    "p5_F066": ("502_2_k8", "R502.2 Stok Akhir Belum Terjual Usaha 2"),

    "p5_F071": ("502_3_k1", "R502.3 No Urut ART Usaha 3"),
    "p5_F072": ("502_3_k3", "R502.3 Kategori Lapangan Usaha (KBLI) Usaha 3"),
    "p5_F073": ("502_3_k4", "R502.3 Jenis Pekerjaan Usaha 3"),
    "p5_F068": ("502_3_k5", "R502.3 Nilai Produksi Usaha 3"),
    "p5_F069": ("502_3_k6", "R502.3 Biaya Produksi Usaha 3"),
    "p5_F074": ("502_3_k7", "R502.3 Persentase Biaya Produksi untuk Upah (%) Usaha 3"),
    "p5_F070": ("502_3_k8", "R502.3 Stok Akhir Belum Terjual Usaha 3"),

    "p5_F077": ("502_4_k1", "R502.4 No Urut ART Usaha 4"),
    "p5_F078": ("502_4_k3", "R502.4 Kategori Lapangan Usaha (KBLI) Usaha 4"),
    "p5_F079": ("502_4_k4", "R502.4 Jenis Pekerjaan Usaha 4"),
    "p5_F075": ("502_4_k5", "R502.4 Nilai Produksi Usaha 4"),
    "p5_F076": ("502_4_k6", "R502.4 Biaya Produksi Usaha 4"),
    "p5_F080": ("502_4_k7", "R502.4 Persentase Biaya Produksi untuk Upah (%) Usaha 4"),
    "p5_F081": ("502_4_k8", "R502.4 Stok Akhir Belum Terjual Usaha 4"),

    "p5_F085": ("502_5_k1", "R502.5 No Urut ART Usaha 5"),
    "p5_F086": ("502_5_k3", "R502.5 Kategori Lapangan Usaha (KBLI) Usaha 5"),
    "p5_F087": ("502_5_k4", "R502.5 Jenis Pekerjaan Usaha 5"),
    "p5_F082": ("502_5_k5", "R502.5 Nilai Produksi Usaha 5"),
    "p5_F083": ("502_5_k6", "R502.5 Biaya Produksi Usaha 5"),
    "p5_F088": ("502_5_k7", "R502.5 Persentase Biaya Produksi untuk Upah (%) Usaha 5"),
    "p5_F084": ("502_5_k8", "R502.5 Stok Akhir Belum Terjual Usaha 5"),

    "p5_F089": ("502_JUMLAH_k5", "R502 JUMLAH Nilai Produksi Usaha Rumah Tangga"),
    "p5_F090": ("502_JUMLAH_k6", "R502 JUMLAH Biaya Produksi Usaha Rumah Tangga"),
    "p5_F091": ("502_JUMLAH_k8", "R502 JUMLAH Stok Akhir Belum Terjual"),

    # --- V.3 Pendapatan Kepemilikan ---
    "p5_F092": ("503_1_terima", "R503.1 Sewa Lahan (Diterima)"),
    "p5_F093": ("503_1_bayar", "R503.1 Sewa Lahan (Dibayar)"),
    "p5_F094": ("503_2_terima", "R503.2 Keuntungan Modal CV, Firma, UD (Diterima)"),
    "p5_F095": ("503_2_bayar", "R503.2 Keuntungan Modal CV, Firma, UD (Dibayar)"),
    "p5_F096": ("503_3_terima", "R503.3 Keuntungan Saham PT, Koperasi (Deviden) (Diterima)"),
    "p5_F097": ("503_4_terima", "R503.4 Bunga Simpanan / Pinjaman (Diterima)"),
    "p5_F098": ("503_4_bayar", "R503.4 Bunga Pinjaman (Dibayar)"),
    "p5_F099": ("503_JUMLAH_terima", "R503 JUMLAH Pendapatan Kepemilikan (Diterima)"),
    "p5_F100": ("503_JUMLAH_bayar", "R503 JUMLAH Biaya Kepemilikan (Dibayar)"),
    "p5_F_hal": ("HAL_5", "Nomor Halaman 5"),
}

P5_FAKE_FIELD_IDS = {
    "p5_F010", "p5_F011", "p5_F013", "p5_F014", "p5_F016", "p5_F017",
    "p5_F021", "p5_F022", "p5_F024", "p5_F025", "p5_F027", "p5_F028", "p5_F030",
    "p5_F031", "p5_F035", "p5_F036", "p5_F038", "p5_F039", "p5_F041", "p5_F042", "p5_F044"
}

# Sel & field nomor baris tabel 505 (label cetak kolom No, bukan kotak isian responden)
P6_ROW_NO_CELL_IDS = {"p6_c218", "p6_c260", "p6_c305", "p6_c318", "p6_c360", "p6_c403"}
P6_ROW_NO_FIELD_IDS = {"p6_F017", "p6_F022", "p6_F028", "p6_F029", "p6_F034", "p6_F039"}

# ==============================================================================
# 6. HALAMAN 6: Blok V.4 (Transfer Berjalan) & Blok V.5 (Aset Non-Finansial)
# ==============================================================================
P6_FIELD_LABELS = {
    # --- Tabel 504: Transfer Berjalan ---
    "p6_F000": ("504_1a_terima_uang", "R504.1a Uang Pensiun Pemerintah (Diterima - Uang)"),
    "p6_F001": ("504_1b_terima_uang", "R504.1b Bantuan Pemerintah & Zakat (Diterima - Uang)"),
    "p6_F002": ("504_1b_terima_barang", "R504.1b Bantuan Pemerintah (Diterima - Barang/Jasa)"),
    "p6_F003": ("504_1b_bayar_uang", "R504.1b Pembayaran Zakat ke BAZNAS (Dibayar - Uang)"),
    "p6_F004": ("504_1b_bayar_barang", "R504.1b Bantuan Pemerintah (Dibayar - Barang/Jasa)"),
    "p6_F005": ("504_2_terima_uang", "R504.2 Rumah Tangga Lain (Diterima - Uang)"),
    "p6_F006": ("504_2_terima_barang", "R504.2 Rumah Tangga Lain (Diterima - Barang/Jasa)"),
    "p6_F007": ("504_2_bayar_uang", "R504.2 Rumah Tangga Lain (Dibayar - Uang)"),
    "p6_F008": ("504_2_bayar_barang", "R504.2 Rumah Tangga Lain (Dibayar - Barang/Jasa)"),
    "p6_F009": ("504_3_terima_uang", "R504.3 Lainnya / Klaim Asuransi (Diterima - Uang)"),
    "p6_F010": ("504_3_terima_barang", "R504.3 Lainnya / Klaim Asuransi (Diterima - Barang/Jasa)"),
    "p6_F011": ("504_3_bayar_uang", "R504.3 Lainnya / Iuran Pensiun & Asuransi (Dibayar - Uang)"),
    "p6_F012": ("504_3_bayar_barang", "R504.3 Lainnya (Dibayar - Barang/Jasa)"),
    "p6_F013": ("504_JUMLAH_terima_uang", "R504 JUMLAH Transfer [1a+1b+2+3] (Diterima - Uang)"),
    "p6_F014": ("504_JUMLAH_terima_barang", "R504 JUMLAH Transfer [1a+1b+2+3] (Diterima - Barang/Jasa)"),
    "p6_F015": ("504_JUMLAH_bayar_uang", "R504 JUMLAH Transfer [1a+1b+2+3] (Dibayar - Uang)"),
    "p6_F016": ("504_JUMLAH_bayar_barang", "R504 JUMLAH Transfer [1a+1b+2+3] (Dibayar - Barang/Jasa)"),

    # --- Tabel 505: Transaksi Modal dan Finansial (Aset Non-Finansial) ---
    # Catatan: Kolom (1) Nomor baris (505_1 s.d. 505_6) adalah label baris cetak (bukan isian responden),
    # sehingga tidak memerlukan overlay kotak digit maupun field isian.
    # Baris 1: Pembelian/Pembuatan Sendiri
    "p6_F018": ("505_1_k3_bangunan_tinggal", "R505.1 Pembelian/Pembuatan Bangunan Tempat Tinggal"),
    "p6_F019": ("505_1_k4_bangunan_usaha", "R505.1 Pembelian/Pembuatan Bangunan Bukan Tempat Tinggal Usaha"),
    "p6_F020": ("505_1_k5_non_bangunan", "R505.1 Pembelian/Pembuatan Non Bangunan Usaha"),
    "p6_F021": ("505_1_k6_lahan", "R505.1 Pembelian/Pembuatan Lahan"),

    # Baris 2: Pemberian dari Pihak Lain
    "p6_F023": ("505_2_k3_bangunan_tinggal", "R505.2 Pemberian Bangunan Tempat Tinggal"),
    "p6_F024": ("505_2_k4_bangunan_usaha", "R505.2 Pemberian Bangunan Bukan Tempat Tinggal Usaha"),
    "p6_F025": ("505_2_k5_non_bangunan", "R505.2 Pemberian Non Bangunan Usaha"),
    "p6_F026": ("505_2_k6_lahan", "R505.2 Pemberian Lahan"),

    # Baris 3: Perbaikan/Pengurukan dan Pemindahan Kepemilikan Lahan
    "p6_F027": ("505_3_k6_lahan", "R505.3 Perbaikan/Pengurukan dan Pemindahan Kepemilikan Lahan"),

    # Baris 4: Penjualan
    "p6_F030": ("505_4_k3_bangunan_tinggal", "R505.4 Penjualan Bangunan Tempat Tinggal"),
    "p6_F031": ("505_4_k4_bangunan_usaha", "R505.4 Penjualan Bangunan Bukan Tempat Tinggal Usaha"),
    "p6_F032": ("505_4_k5_non_bangunan", "R505.4 Penjualan Non Bangunan Usaha"),
    "p6_F033": ("505_4_k6_lahan", "R505.4 Penjualan Lahan"),

    # Baris 5: Memberikan kepada Pihak Lain
    "p6_F035": ("505_5_k3_bangunan_tinggal", "R505.5 Memberikan Bangunan Tempat Tinggal"),
    "p6_F036": ("505_5_k4_bangunan_usaha", "R505.5 Memberikan Bangunan Bukan Tempat Tinggal Usaha"),
    "p6_F037": ("505_5_k5_non_bangunan", "R505.5 Memberikan Non Bangunan Usaha"),
    "p6_F038": ("505_5_k6_lahan", "R505.5 Memberikan Lahan"),

    # Baris 6: Nilai Seluruh Aset yang Dimiliki
    "p6_F040": ("505_6_k3_bangunan_tinggal", "R505.6 Nilai Seluruh Bangunan Tempat Tinggal yang Dimiliki"),
    "p6_F041": ("505_6_k4_bangunan_usaha", "R505.6 Nilai Seluruh Bangunan Bukan Tempat Tinggal Usaha yang Dimiliki"),
    "p6_F042": ("505_6_k5_non_bangunan", "R505.6 Nilai Seluruh Non Bangunan Usaha yang Dimiliki"),
    "p6_F043": ("505_6_k6_lahan", "R505.6 Nilai Seluruh Lahan yang Dimiliki"),

    # Baris 7: Nilai Pembelian/Perolehan Awal
    "p6_F044": ("505_7_k3_bangunan_tinggal", "R505.7 Nilai Perolehan Awal Bangunan Tempat Tinggal"),
    "p6_F045": ("505_7_k4_bangunan_usaha", "R505.7 Nilai Perolehan Awal Bangunan Bukan Tempat Tinggal Usaha"),
    "p6_F046": ("505_7_k5_non_bangunan", "R505.7 Nilai Perolehan Awal Non Bangunan Usaha"),
    "p6_F_hal": ("HAL_6", "Nomor Halaman 6"),
}

# ==============================================================================
# 7. HALAMAN 7: Blok VI (Posisi Aset & Kewajiban) & Blok VII (Kunjungan)
# ==============================================================================
P7_FIELD_LABELS = {
    # --- Blok VI Aset ---
    "p7_F000": ("601_uang_tunai", "R601 Uang Tunai (dompet, brankas, e-money/e-wallet)"),
    "p7_F002": ("602a_simpanan_bank", "R602a Simpanan di Bank & Lembaga Keuangan Lainnya"),
    "p7_F004": ("602b_simpanan_koperasi", "R602b Simpanan di Koperasi Simpan Pinjam / Credit Union"),
    "p7_F006": ("602c_simpanan_lain", "R602c Simpanan Lainnya (Sekolah, RT, dll)"),
    "p7_F008": ("603_piutang", "R603 Sisa Piutang yang Dapat Ditagih"),
    "p7_F010": ("604a_saham_bursa", "R604a Surat Berharga - Saham Terdaftar di Bursa Efek"),
    "p7_F012": ("604b_saham_non_bursa", "R604b Surat Berharga - Saham Tidak Terdaftar di Bursa Efek"),
    "p7_F014": ("604c_surat_berharga_lain", "R604c Surat Berharga Lainnya (SUN, Obligasi, Reksadana)"),
    "p7_F016": ("605a_penyertaan_non_bh", "R605a Nilai Penyertaan Modal di Perusahaan Non-BH (CV, Firma, UD)"),
    "p7_F017": ("605b_penyertaan_urt", "R605b Nilai Penyertaan Modal di Usaha RT Pihak Lain"),
    "p7_F018": ("606_kredit_barang", "R606 Sisa Kredit Barang yang Belum Diterima (jika memiliki URT)"),
    "p7_F019": ("607_arisan", "R607 Jumlah Iuran Arisan yang Telah Dibayar dari Arisan Aktif"),
    "p7_F020": ("600_JUMLAH_POSISI_ASET", "R600 JUMLAH POSISI ASET [601 + 602 + ... + 607]"),

    # --- Blok VI Kewajiban ---
    "p7_F001": ("608a_pinjaman_bank", "R608a Sisa Pinjaman Belum Dibayar - Bank & Lembaga Pembiayaan"),
    "p7_F003": ("608b_pinjaman_koperasi", "R608b Sisa Pinjaman Belum Dibayar - Koperasi Simpan Pinjam / Credit Union"),
    "p7_F005": ("608c_pinjaman_lain", "R608c Sisa Pinjaman Belum Dibayar - Lainnya (Rentenir, Pinjol)"),
    "p7_F007": ("609_modal_pihak_lain", "R609 Nilai Modal Usaha Milik Pihak Lain (jika memiliki URT)"),
    "p7_F009": ("610_utang_kredit_barang", "R610 Sisa Kredit Barang yang Belum Dibayar"),
    "p7_F011": ("611a_gadai_pegadaian", "R611a Sisa Nilai Gadaian Belum Dibayar - PT Pegadaian"),
    "p7_F013": ("611b_gadai_lain", "R611b Sisa Nilai Gadaian Belum Dibayar - Tempat Gadai Lainnya"),
    "p7_F015": ("612_arisan_diterima", "R612 Jumlah Uang Arisan yang Telah Diterima dari Arisan Aktif"),
    "p7_F021": ("600_JUMLAH_POSISI_KEWAJIBAN", "R600 JUMLAH POSISI KEWAJIBAN [608 + 609 + ... + 612]"),

    # --- Blok VII Kunjungan & Catatan ---
    "p7_F022": ("701_kunj1_tgl", "Kunjungan I: Tanggal Pencacahan (Tgl)"),
    "p7_F023": ("701_kunj1_bln", "Kunjungan I: Bulan Pencacahan (Bln)"),
    "p7_F024": ("701_kunj1_thn", "Kunjungan I: Tahun Pencacahan (Thn)"),
    "p7_F025": ("701_kunj1_mul_jam", "Kunjungan I: Waktu Mulai (Jam)"),
    "p7_F026": ("701_kunj1_mul_men", "Kunjungan I: Waktu Mulai (Menit)"),
    "p7_F027": ("701_kunj1_sel_jam", "Kunjungan I: Waktu Selesai (Jam)"),
    "p7_F028": ("701_kunj1_sel_men", "Kunjungan I: Waktu Selesai (Menit)"),

    "p7_F029": ("702_kunj2_tgl", "Kunjungan II: Tanggal Pencacahan (Tgl)"),
    "p7_F030": ("702_kunj2_bln", "Kunjungan II: Bulan Pencacahan (Bln)"),
    "p7_F031": ("702_kunj2_thn", "Kunjungan II: Tahun Pencacahan (Thn)"),
    "p7_F032": ("702_kunj2_mul_jam", "Kunjungan II: Waktu Mulai (Jam)"),
    "p7_F033": ("702_kunj2_mul_men", "Kunjungan II: Waktu Mulai (Menit)"),
    "p7_F034": ("702_kunj2_sel_jam", "Kunjungan II: Waktu Selesai (Jam)"),
    "p7_F035": ("702_kunj2_sel_men", "Kunjungan II: Waktu Selesai (Menit)"),

    "p7_F036": ("703_kunj3_tgl", "Kunjungan III: Tanggal Pencacahan (Tgl)"),
    "p7_F037": ("703_kunj3_bln", "Kunjungan III: Bulan Pencacahan (Bln)"),
    "p7_F038": ("703_kunj3_thn", "Kunjungan III: Tahun Pencacahan (Thn)"),
    "p7_F039": ("703_kunj3_mul_jam", "Kunjungan III: Waktu Mulai (Jam)"),
    "p7_F040": ("703_kunj3_mul_men", "Kunjungan III: Waktu Mulai (Menit)"),
    "p7_F041": ("703_kunj3_sel_jam", "Kunjungan III: Waktu Selesai (Jam)"),
    "p7_F042": ("703_kunj3_sel_men", "Kunjungan III: Waktu Selesai (Menit)"),

    "p7_F043": ("700_catatan", "BLOK VII Catatan Khusus Petugas Pencacah / Pengawas"),
    "p7_F044": ("700_waktu_selesai", "Waktu Selesai Wawancara (Jam : Menit)"),
    "p7_F_hal": ("HAL_7", "Nomor Halaman 7"),
}

ALL_PAGE_MAPPINGS = {
    1: P1_FIELD_LABELS,
    2: P2_FIELD_LABELS,
    3: P3_FIELD_LABELS,
    4: P4_FIELD_LABELS,
    5: P5_FIELD_LABELS,
    6: P6_FIELD_LABELS,
    7: P7_FIELD_LABELS,
}

def standardize_all_templates():
    print("================================================================================")
    print("STANDARDISASI PENUH & PENYELARASAN SELURUH LABEL GRID SERUTI INTI 2026 (HAL 1-7)")
    print("================================================================================")

    total_fields_all = 0
    total_digits_all = 0

    for pno in range(1, 8):
        fn = TEMPLATES_DIR / f"template_p{pno}.json"
        if not fn.exists():
            print(f"[ERROR] {fn} tidak ditemukan!")
            continue

        with open(fn, "r", encoding="utf-8") as f:
            tpl = json.load(f)

        cells = tpl.get("cells", [])
        fields = tpl.get("fields", [])
        mapping = ALL_PAGE_MAPPINGS.get(pno, {})

        # Khusus Hal 5: Bersihkan sel abu-abu palsu
        if pno == 5:
            for c in cells:
                if c["kind"] == "digit" and min(c.get("fill_rgb", [255, 255, 255])) < 245:
                    c["kind"] = "struct"
                    c["fill"] = "band"
            fields = [fld for fld in fields if fld["id"] not in P5_FAKE_FIELD_IDS]

        # Khusus Hal 6: Nomor baris tabel 505 bukan kotak digit isian, ubah ke struct dan hapus field-nya
        if pno == 6:
            for c in cells:
                if c["id"] in P6_ROW_NO_CELL_IDS:
                    c["kind"] = "struct"
                    c.pop("field_id", None)
                    c.pop("field_label", None)
                    c.pop("box_index", None)
                    c.pop("total_boxes", None)
                    c.pop("label", None)
                    c.pop("printed", None)
            fields = [fld for fld in fields if fld["id"] not in P6_ROW_NO_FIELD_IDS]
            for sb in tpl.get("sub_blocks", []):
                if sb.get("id") == "sub_505":
                    sb["fields"] = [fid for fid in sb.get("fields", []) if fid not in P6_ROW_NO_FIELD_IDS]
                    sb["fields_count"] = len(sb["fields"])
                    sb["digits_count"] = 240

        # Terapkan label standar seragam ke semua field
        for fld in fields:
            fid = fld["id"]
            if fid in mapping:
                r_lbl, f_lbl = mapping[fid]
                fld["row_label"] = r_lbl
                fld["label"] = f_lbl
            else:
                # Pastikan tidak kosong
                if not fld.get("row_label") or str(fld["row_label"]).strip().lower() == "none":
                    fld["row_label"] = fid
                if not fld.get("label"):
                    fld["label"] = f"Field {fld['row_label']}"

        tpl["fields"] = fields

        # Map kotak digit individual (comb box cells)
        field_box_mapping = {}
        for fld in fields:
            fid = fld["id"]
            r_lbl = fld["row_label"]
            f_lbl = fld["label"]
            for b_idx, bid in enumerate(fld.get("boxes", [])):
                field_box_mapping[bid] = (fid, r_lbl, f_lbl, b_idx, len(fld["boxes"]))

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
                    c["label"] = f"{cid}_digit"
                    labeled_digits += 1

        real_digits = [c for c in cells if c["kind"] == "digit" and min(c.get("fill_rgb", [255, 255, 255])) >= 245]
        tpl["counts"] = {
            "struct": sum(1 for c in cells if c["kind"] == "struct"),
            "digit": len(real_digits),
            "fields": len(fields),
            "tables": len(tpl.get("tables", []))
        }

        with open(fn, "w", encoding="utf-8") as f:
            json.dump(tpl, f, indent=2, ensure_ascii=False)

        total_fields_all += len(fields)
        total_digits_all += len(real_digits)

        print(f"[*] Halaman {pno}:")
        print(f"    - Total Fields  : {len(fields)} (Seluruhnya berlabel format standar)")
        print(f"    - Kotak Digit   : {len(real_digits)} sel comb-box")
        print(f"    - Contoh Label  : {[fld['row_label'] for fld in fields[:3]]}")

    print("\n--------------------------------------------------------------------------------")
    print(f"TOTAL KONSOLIDASI SELURUH HALAMAN (1-7):")
    print(f"- Total Fields Terstandardisasi: {total_fields_all}")
    print(f"- Total Kotak Digit Sah Terlabeli: {total_digits_all}")
    print("--------------------------------------------------------------------------------")

if __name__ == "__main__":
    standardize_all_templates()
