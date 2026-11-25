#!/usr/bin/env python3
"""
apply_kesediaan_and_waktu_fixes.py
Mengimplementasikan:
1. Kalibrasi presisi tinggi Blok VI (y1 = 410.11 pt, memotong kelebihan 8.4 pt ruang kosong di bawah total).
2. Menambahkan cakupan Kesediaan Responden & Waktu Mulai Wawancara pada Halaman 1.
3. Menambahkan cakupan Waktu Selesai Wawancara pada Halaman 7.
4. Memberi label eksplisit row_label = 'WAKTU_MULAI' pada p1_F000 dan 'WAKTU_SELESAI' pada p7_F028.
"""
import json
from pathlib import Path

TEMPLATES_DIR = Path("templates")

# ==============================================================================
# 1. UPDATE TEMPLATE P1 (Halaman 1)
# ==============================================================================
print("=== Memperbarui Template P1 (Kesediaan Responden & Waktu Mulai Wawancara) ===")
p1_file = TEMPLATES_DIR / "template_p1.json"
with open(p1_file, "r", encoding="utf-8") as f:
    p1 = json.load(f)

# A. Labeli field p1_F000
for fld in p1.get("fields", []):
    if fld["id"] == "p1_F000":
        fld["row_label"] = "WAKTU_MULAI"
        fld["parent_text"] = "Waktu mulai wawancara"

# B. Tambahkan sel checkbox untuk 3 opsi kesediaan responden jika belum ada
existing_cell_ids = set(c["id"] for c in p1.get("cells", []))
kesediaan_cells = [
    {
        "id": "p1_c_kesediaan_1",
        "kind": "digit", # treat as checkbox/mark
        "bbox_pt": [80.0, 193.5, 92.0, 206.5],
        "bbox_n": [80.0/595.32, 193.5/841.92, 92.0/595.32, 206.5/841.92],
        "fill": "white",
        "fill_rgb": [255, 255, 255],
        "text": "1"
    },
    {
        "id": "p1_c_kesediaan_2",
        "kind": "digit",
        "bbox_pt": [80.0, 205.0, 92.0, 218.0],
        "bbox_n": [80.0/595.32, 205.0/841.92, 92.0/595.32, 218.0/841.92],
        "fill": "white",
        "fill_rgb": [255, 255, 255],
        "text": "2"
    },
    {
        "id": "p1_c_kesediaan_3",
        "kind": "digit",
        "bbox_pt": [80.0, 216.5, 92.0, 229.5],
        "bbox_n": [80.0/595.32, 216.5/841.92, 92.0/595.32, 229.5/841.92],
        "fill": "white",
        "fill_rgb": [255, 255, 255],
        "text": "3"
    }
]

for kc in kesediaan_cells:
    if kc["id"] not in existing_cell_ids:
        p1["cells"].append(kc)
        existing_cell_ids.add(kc["id"])

# C. Tambahkan field kesediaan responden jika belum ada
existing_field_ids = set(f["id"] for f in p1.get("fields", []))
if "p1_F_kesediaan" not in existing_field_ids:
    p1["fields"].append({
        "id": "p1_F_kesediaan",
        "row_label": "KESEDIAAN_RESPONDEN",
        "n_boxes": 3,
        "boxes": ["p1_c_kesediaan_1", "p1_c_kesediaan_2", "p1_c_kesediaan_3"],
        "parent": None,
        "parent_text": "Kesediaan responden diwawancarai (1: Ya, 2: Janji, 3: Tolak)",
        "printed": None,
        "bbox_pt": [80.0, 193.5, 92.0, 229.5]
    })

# D. Update Blocks dan Sub-Blocks Halaman 1
p1_blocks = [
    {
        "id": "blok_kesediaan",
        "code": "KESEDIAAN",
        "title": "KESEDIAAN RESPONDEN & WAKTU MULAI WAWANCARA",
        "bbox_pt": [14.16, 185.00, 582.00, 251.66],
        "sub_blocks": ["sub_kesediaan", "sub_waktu_mulai"]
    },
    {
        "id": "blok_1",
        "code": "BLOK I",
        "title": "KETERANGAN TEMPAT",
        "bbox_pt": [14.16, 251.66, 582.00, 542.47],
        "sub_blocks": ["sub_101_111"]
    },
    {
        "id": "blok_2",
        "code": "BLOK II",
        "title": "KETERANGAN PENDATAAN",
        "bbox_pt": [14.16, 560.23, 582.00, 809.50],
        "sub_blocks": ["sub_201_203"]
    }
]

p1_sub_blocks = [
    {
        "id": "sub_kesediaan",
        "block_id": "blok_kesediaan",
        "code": "KESEDIAAN",
        "name": "Kesediaan Responden Diwawancarai (3 Opsi Respon)",
        "type": "respondent_consent",
        "bbox_pt": [14.16, 185.00, 425.00, 251.66],
        "cells_count": 3,
        "fields_count": 1,
        "digits_count": 3,
        "fields": ["p1_F_kesediaan"]
    },
    {
        "id": "sub_waktu_mulai",
        "block_id": "blok_kesediaan",
        "code": "WAKTU_MULAI",
        "name": "Waktu Mulai Wawancara (Jam : Menit)",
        "type": "interview_time",
        "bbox_pt": [425.00, 185.00, 582.00, 251.66],
        "cells_count": 4,
        "fields_count": 1,
        "digits_count": 4,
        "fields": ["p1_F000"]
    },
    {
        "id": "sub_101_111",
        "block_id": "blok_1",
        "code": "R101-R111",
        "name": "Pengenalan Wilayah Administrasi & Sampel",
        "type": "admin_sample",
        "bbox_pt": [14.16, 251.66, 582.00, 542.47],
        "cells_count": 67,
        "fields_count": 8,
        "digits_count": 24,
        "fields": ["p1_F001", "p1_F002", "p1_F003", "p1_F004", "p1_F005", "p1_F006", "p1_F007", "p1_F008"]
    },
    {
        "id": "sub_201_203",
        "block_id": "blok_2",
        "code": "R201-R203",
        "name": "Keterangan Petugas PCL/PML & Hasil Pendataan",
        "type": "survey_officer",
        "bbox_pt": [14.16, 560.23, 582.00, 809.50],
        "cells_count": 47,
        "fields_count": 9,
        "digits_count": 21,
        "fields": ["p1_F009", "p1_F010", "p1_F011", "p1_F012", "p1_F013", "p1_F014", "p1_F015", "p1_F016", "p1_F017"]
    }
]

p1["blocks"] = p1_blocks
p1["sub_blocks"] = p1_sub_blocks

with open(p1_file, "w", encoding="utf-8") as f:
    json.dump(p1, f, indent=2, ensure_ascii=False)
print("  Template P1 berhasil diperbarui.")

# ==============================================================================
# 2. UPDATE TEMPLATE P7 (Halaman 7)
# ==============================================================================
print("\n=== Memperbarui Template P7 (Presisi Blok VI & Waktu Selesai Wawancara) ===")
p7_file = TEMPLATES_DIR / "template_p7.json"
with open(p7_file, "r", encoding="utf-8") as f:
    p7 = json.load(f)

# A. Labeli field p7_F028
for fld in p7.get("fields", []):
    if fld["id"] == "p7_F028":
        fld["row_label"] = "WAKTU_SELESAI"
        fld["parent_text"] = "Waktu selesai wawancara"

# B. Update Blocks Halaman 7:
#    - blok_6 y1 diubah dari 418.51 ke 410.11 (presisi garis bawah tabel)
#    - blok_7 y1 diperluas dari 568.32 ke 586.00 (mencakup waktu selesai wawancara)
p7_blocks = [
    {
        "id": "blok_6",
        "code": "BLOK VI",
        "title": "POSISI ASET DAN KEWAJIBAN KEUANGAN",
        "bbox_pt": [14.16, 30.96, 822.94, 410.11], # Presisi tinggi: 410.11 pt
        "sub_blocks": ["sub_601_607", "sub_608_612"]
    },
    {
        "id": "blok_7",
        "code": "BLOK VII",
        "title": "CATATAN / KUNJUNGAN PENCACAHAN & WAKTU SELESAI",
        "bbox_pt": [14.16, 424.39, 822.94, 586.00], # Mencakup kunjungan s.d. waktu selesai
        "sub_blocks": ["sub_kunjungan", "sub_waktu_selesai"]
    }
]

# C. Update Sub-Blocks Halaman 7:
p7_sub_blocks = [
    {
        "id": "sub_601_607",
        "block_id": "blok_6",
        "code": "R601-R607",
        "name": "Posisi Aset Keuangan Rumah Tangga",
        "type": "financial_assets",
        "bbox_pt": [14.16, 30.96, 426.29, 410.11], # Presisi tinggi: 410.11 pt
        "cells_count": 178,
        "fields_count": 13,
        "digits_count": 143,
        "fields": ["p7_F000", "p7_F001", "p7_F002", "p7_F003", "p7_F004", "p7_F005", "p7_F006", "p7_F007", "p7_F008", "p7_F009", "p7_F010", "p7_F011", "p7_F012"]
    },
    {
        "id": "sub_608_612",
        "block_id": "blok_6",
        "code": "R608-R612",
        "name": "Posisi Kewajiban Keuangan (Utang/Pinjaman)",
        "type": "financial_liabilities",
        "bbox_pt": [426.29, 30.96, 822.94, 410.11], # Presisi tinggi: 410.11 pt
        "cells_count": 125,
        "fields_count": 9,
        "digits_count": 99,
        "fields": ["p7_F013", "p7_F014", "p7_F015", "p7_F016", "p7_F017", "p7_F018", "p7_F019", "p7_F020", "p7_F021"]
    },
    {
        "id": "sub_kunjungan",
        "block_id": "blok_7",
        "code": "R701-R703",
        "name": "Jadwal Kunjungan Pencacahan, Waktu Wawancara, & Catatan",
        "type": "survey_visits",
        "bbox_pt": [14.16, 424.39, 822.94, 568.32],
        "cells_count": 24,
        "fields_count": 6,
        "digits_count": 16,
        "fields": ["p7_F022", "p7_F023", "p7_F024", "p7_F025", "p7_F026", "p7_F027"]
    },
    {
        "id": "sub_waktu_selesai",
        "block_id": "blok_7",
        "code": "WAKTU_SELESAI",
        "name": "Waktu Selesai Wawancara (Jam : Menit)",
        "type": "interview_time",
        "bbox_pt": [14.16, 568.32, 420.00, 586.00],
        "cells_count": 4,
        "fields_count": 1,
        "digits_count": 4,
        "fields": ["p7_F028"]
    }
]

p7["blocks"] = p7_blocks
p7["sub_blocks"] = p7_sub_blocks

with open(p7_file, "w", encoding="utf-8") as f:
    json.dump(p7, f, indent=2, ensure_ascii=False)
print("  Template P7 berhasil diperbarui.")
