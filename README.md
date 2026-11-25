# PindaiSurvei v2.0 (Si-Periksa)
### Sistem Pembacaan, Digitalisasi & Validasi Dokumen Survei Multi-Instrumen
**Badan Pusat Statistik (BPS) Kabupaten Penajam Paser Utara**

---

## 📌 Ringkasan Proyek

**PindaiSurvei** (sebelumnya *Si-Periksa*) adalah platform cerdas untuk otomatisasi pembacaan, digitalisasi, verifikasi konsistensi, dan rekonsiliasi dokumen kuesioner survei berbasis kertas (PAPI) di lingkungan BPS.

Sistem ini dirancang khusus untuk menangani berbagai instrumen survei resmi BPS:
1. **VSERUTI26.INTI** — Survei Ekonomi Rumah Tangga Triwulanan (Kuesioner Inti)
2. **VSERUTI26.MAK** — Survei Ekonomi Rumah Tangga Triwulanan (Kuesioner Konsumsi/Modul Makanan)
3. **VSEN26.KP** — Survei Sosial Ekonomi Nasional (Susenas Modul Konsumsi/Pengeluaran)

---

## 🚀 Fitur Unggulan

- **Pipeline Ekstraksi Multi-Tahap (7-Stage Pipeline)**:
  - *Auto-Deskewing & Perspective Correction* berbasis deteksi sudut & regresi garis.
  - *Lattice Grid Alignment & Template Snapping* presisi sub-piksel.
  - *Comb-Box Segmentation* otomatis untuk digit angka dan kode kategoris.
- **Vision AI Inference (Google Gemini Vision `gemini-3.8-flash`)**:
  - Pembacaan digit comb-box cepat via batch mosaic grid dengan tingkat akurasi tinggi (>98%).
  - Ekstraksi teks tulisan tangan (*handwriting recognition*) untuk uraian lapangan usaha & jenis pekerjaan (Blok V R501/R502).
  - Cache inferensi lokal terenkripsi SHA-256 untuk efisiensi kuota.
- **Kuesioner Digital Resmi & Penampil Buku Kertas (Book View)**:
  - Format visual identik 100% dengan kuesioner kertas fisik (A4 Landscape & Portrait).
  - Fitur *Smart Cursor Zoom & Pan* yang responsif mengikuti gerakan kursor pengguna.
  - Tampilan perbandingan langsung (*side-by-side*) antara pindaian dokumen asli dan hasil digitalisasi.
- **Human-in-the-Loop (HITL) Verification**:
  - Antarmuka koreksi cepat untuk sel dengan nilai keyakinan (*confidence score*) di bawah ambang batas (*threshold*).
  - Fitur *Smart Auto-Accept* untuk digit konsisten dengan akurasi terverifikasi.
- **Validasi Konsistensi Silang (Cross-Validation Rules)**:
  - Aturan aritmetika dan relasi semantik antar-blok (Blok I s.d. Blok VII).
  - Audit trail komprehensif untuk pelacakan perbedaan nilai dan anomali.

---

## 🛡️ Kepatuhan & Keamanan Data (Zero-PII Cloud Egress)

Sistem ini mematuhi standar keamanan data ketat sesuai:
- **UU No. 16 Tahun 1997 tentang Statistik** (Kerahasiaan data individual responden).
- **UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP)**.

> **Zero PII to Cloud**: Hanya potongan (*crop*) sel comb-box digit tanpa identitas yang diproses untuk inferensi. Seluruh pindaian dokumen penuh, database SQLite mikro, dan arsip data responden **100% tersimpan secara offline di mesin lokal** dan tidak pernah diunggah ke repositori publik.

---

## 🛠️ Instalasi & Menjalankan Aplikasi

### 1. Prasyarat
- Python 3.10+ (disarankan Python 3.12 atau 3.13)
- Google Gemini API Key

### 2. Setup Lingkungan
```bash
# Clone repositori
git clone https://github.com/ABDevelopt/pindaisurvei.git
cd pindaisurvei

# Salin konfigurasi environment
cp .env.example .env
# Masukkan GEMINI_API_KEY Anda di dalam file .env
```

### 3. Menjalankan Server
```bash
# Mode Lokal (127.0.0.1:8000)
python app.py

# Mode Jaringan LAN (dapat diakses komputer lain dalam satu Wi-Fi/kantor)
python app.py --lan
```

Akses aplikasi melalui browser:
- **Dashboard / Review**: `http://localhost:8000/static/review.html`
- **Kuesioner Digital**: `http://localhost:8000/static/kuesioner_digital.html`
- **Comb-Box Grid**: `http://localhost:8000/static/grid.html`
- **Dokumentasi REST API**: `http://localhost:8000/api/docs`

---

## 📂 Struktur Direktori

```text
pindaisurvei/
├── api/                   # Router & endpoint FastAPI (surveys, booklets, pipeline, hitl, validation)
├── db/                    # Inisialisasi basis data SQLite & skema SQL
├── seruti_pipeline/       # Core engine OCR, deskewing, aligner, validator, handwriting reader
├── services/              # Logika bisnis digital document & label normalizer
├── static/                # Antarmuka web (kuesioner digital, review, grid, aset brand/logo)
├── templates/             # Metadata koordinat bounding box template kuesioner resmi (Hal 1-7)
├── tests/                 # Unit test suite & adversarial validation tests
├── tools/                 # Skrip diagnostik, inspeksi, dan utilitas pipeline
├── app.py                 # FastAPI application server entrypoint
└── README.md
```

---

## 📄 Lisensi

Hak Cipta © 2026 BPS Kabupaten Penajam Paser Utara. Dikembangkan untuk keperluan internal kedinasan statistik.
