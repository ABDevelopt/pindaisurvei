#!/usr/bin/env python3
"""
inject_subblock_metadata.py - Menyuntikkan arsitektur hierarki Sub-Blok ke template JSON Seruti Inti 2026
dengan presisi geometris sub-piksel garis vektor PDF dan batas antar-tabel resmi.

Hierarki:
  Page -> Blocks (Makro) -> Sub-Blocks (Modular) -> Tables -> Cells & Fields (Mikro)
"""
import json
from pathlib import Path

TEMPLATES_DIR = Path("templates")

PAGE_HIERARCHY = {
    1: {
        "blocks": [
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
            },
            {
                "id": "blok_halaman_p1",
                "code": "HALAMAN",
                "title": "NOMOR HALAMAN 1",
                "bbox_pt": [280.00, 824.00, 316.00, 840.00],
                "sub_blocks": ["sub_hal_p1"]
            }
        ],
        "sub_blocks": [
            {
                "id": "sub_kesediaan",
                "block_id": "blok_kesediaan",
                "code": "KESEDIAAN",
                "name": "Kesediaan Responden Diwawancarai (3 Opsi Respon)",
                "type": "respondent_consent",
                "bbox_pt": [14.16, 185.00, 425.00, 251.66]
            },
            {
                "id": "sub_waktu_mulai",
                "block_id": "blok_kesediaan",
                "code": "WAKTU_MULAI",
                "name": "Waktu Mulai Wawancara (Jam : Menit)",
                "type": "interview_time",
                "bbox_pt": [425.00, 185.00, 582.00, 251.66]
            },
            {
                "id": "sub_101_111",
                "block_id": "blok_1",
                "code": "R101-R111",
                "name": "Pengenalan Wilayah Administrasi & Sampel",
                "type": "admin_sample",
                "bbox_pt": [14.16, 251.66, 582.00, 542.47]
            },
            {
                "id": "sub_201_203",
                "block_id": "blok_2",
                "code": "R201-R203",
                "name": "Keterangan Petugas PCL/PML & Hasil Pendataan",
                "type": "survey_officer",
                "bbox_pt": [14.16, 560.23, 582.00, 809.50]
            },
            {
                "id": "sub_hal_p1",
                "block_id": "blok_halaman_p1",
                "code": "HAL_1",
                "name": "Nomor Halaman 1",
                "type": "page_number",
                "bbox_pt": [280.00, 824.00, 316.00, 840.00]
            }
        ]
    },
    2: {
        "blocks": [
            {
                "id": "blok_3",
                "code": "BLOK III",
                "title": "KETERANGAN RUMAH TANGGA",
                "bbox_pt": [14.16, 14.16, 580.80, 335.93],
                "sub_blocks": ["sub_301_307"]
            },
            {
                "id": "blok_4_1",
                "code": "BLOK IV.1",
                "title": "KONSUMSI BAHAN MAKANAN (R405)",
                "bbox_pt": [14.16, 345.17, 580.80, 406.73],
                "sub_blocks": ["sub_405"]
            },
            {
                "id": "blok_4_2_a_p1",
                "code": "BLOK IV.2.A",
                "title": "PENGELUARAN BUKAN MAKANAN (BAGIAN 1)",
                "bbox_pt": [14.16, 415.01, 582.60, 826.06],
                "sub_blocks": ["sub_406_414"]
            },
            {
                "id": "blok_halaman_p2",
                "code": "HALAMAN",
                "title": "NOMOR HALAMAN 2",
                "bbox_pt": [280.00, 826.50, 316.00, 840.00],
                "sub_blocks": ["sub_hal_p2"]
            }
        ],
        "sub_blocks": [
            {
                "id": "sub_301_307",
                "block_id": "blok_3",
                "code": "R301-R307",
                "name": "Roster 10 Anggota Rumah Tangga (ART)",
                "type": "roster_art_demography",
                "bbox_pt": [14.16, 14.16, 580.80, 335.93]
            },
            {
                "id": "sub_405",
                "block_id": "blok_4_1",
                "code": "R405",
                "name": "Checkbox Konsumsi Bahan Makanan (215 Kotak Centang)",
                "type": "checkbox_matrix",
                "bbox_pt": [14.16, 345.17, 580.80, 406.73]
            },
            {
                "id": "sub_406_414",
                "block_id": "blok_4_2_a_p1",
                "code": "R406-R414",
                "name": "Pengeluaran Bukan Makanan Sebulan & Tiga Bulan",
                "type": "expense_nonfood",
                "bbox_pt": [14.16, 415.01, 582.60, 826.06]
            },
            {
                "id": "sub_hal_p2",
                "block_id": "blok_halaman_p2",
                "code": "HAL_2",
                "name": "Nomor Halaman 2",
                "type": "page_number",
                "bbox_pt": [280.00, 826.50, 316.00, 840.00]
            }
        ]
    },
    3: {
        "blocks": [
            {
                "id": "blok_4_2_a_p2",
                "code": "BLOK IV.2.A (Lanjutan)",
                "title": "PENGELUARAN BUKAN MAKANAN (SEBULAN & TIGA BULAN)",
                "bbox_pt": [14.16, 14.16, 582.60, 800.50],
                "sub_blocks": ["sub_415_451"]
            },
            {
                "id": "blok_halaman_p3",
                "code": "HALAMAN",
                "title": "NOMOR HALAMAN 3",
                "bbox_pt": [280.00, 804.00, 316.00, 840.00],
                "sub_blocks": ["sub_hal_p3"]
            }
        ],
        "sub_blocks": [
            {
                "id": "sub_415_451",
                "block_id": "blok_4_2_a_p2",
                "code": "R415-R451",
                "name": "Pengeluaran Bukan Makanan Lanjutan & Total R451",
                "type": "expense_nonfood_detail",
                "bbox_pt": [14.16, 14.16, 582.60, 800.50]
            },
            {
                "id": "sub_hal_p3",
                "block_id": "blok_halaman_p3",
                "code": "HAL_3",
                "name": "Nomor Halaman 3",
                "type": "page_number",
                "bbox_pt": [280.00, 804.00, 316.00, 840.00]
            }
        ]
    },
    4: {
        "blocks": [
            {
                "id": "blok_4_2_b",
                "code": "BLOK IV.2.B",
                "title": "PENGELUARAN KESEHATAN DAN PENDIDIKAN",
                "bbox_pt": [14.16, 22.56, 582.60, 318.89],
                "sub_blocks": ["sub_452_463"]
            },
            {
                "id": "blok_4_3",
                "code": "BLOK IV.3",
                "title": "PERSEPSI KONDISI EKONOMI RUMAH TANGGA",
                "bbox_pt": [14.16, 328.01, 582.00, 721.18],
                "sub_blocks": ["sub_464_469", "sub_470"]
            },
            {
                "id": "blok_catatan_p4",
                "code": "CATATAN",
                "title": "KODE KETERANGAN BLOK III",
                "bbox_pt": [14.16, 723.58, 582.00, 821.38],
                "sub_blocks": ["sub_ref_p4"]
            },
            {
                "id": "blok_halaman_p4",
                "code": "HALAMAN",
                "title": "NOMOR HALAMAN 4",
                "bbox_pt": [280.00, 824.00, 316.00, 840.00],
                "sub_blocks": ["sub_hal_p4"]
            }
        ],
        "sub_blocks": [
            {
                "id": "sub_452_463",
                "block_id": "blok_4_2_b",
                "code": "R452-R463",
                "name": "Pengeluaran Kesehatan & Pendidikan (Total & OOP)",
                "type": "expense_health_edu",
                "bbox_pt": [14.16, 22.56, 582.60, 318.89]
            },
            {
                "id": "sub_464_469",
                "block_id": "blok_4_3",
                "code": "R464-R469",
                "name": "Tabel Kecenderungan Persepsi Ekonomi Triwulanan",
                "type": "perception_matrix",
                "bbox_pt": [14.16, 328.01, 582.00, 594.43]
            },
            {
                "id": "sub_470",
                "block_id": "blok_4_3",
                "code": "R470",
                "name": "Kecukupan Pendapatan & Sumber Dana (470a, 470bi-iii)",
                "type": "perception_funding",
                "bbox_pt": [14.16, 594.43, 582.00, 721.18]
            },
            {
                "id": "sub_ref_p4",
                "block_id": "blok_catatan_p4",
                "code": "REF-P4",
                "name": "Daftar Kode Kolom (3) & (7) Blok III",
                "type": "reference_code",
                "bbox_pt": [14.16, 723.58, 582.00, 821.38]
            },
            {
                "id": "sub_hal_p4",
                "block_id": "blok_halaman_p4",
                "code": "HAL_4",
                "name": "Nomor Halaman 4",
                "type": "page_number",
                "bbox_pt": [280.00, 824.00, 316.00, 840.00]
            }
        ]
    },
    5: {
        "blocks": [
            {
                "id": "blok_5_p1",
                "code": "BLOK V",
                "title": "PENDAPATAN, PENERIMAAN, DAN PENGELUARAN SELAIN KONSUMSI",
                "bbox_pt": [13.44, 14.16, 822.94, 580.44],
                "sub_blocks": ["sub_501", "sub_502", "sub_503"]
            },
            {
                "id": "blok_halaman_p5",
                "code": "HALAMAN",
                "title": "NOMOR HALAMAN 5",
                "bbox_pt": [403.00, 580.80, 439.00, 592.50],
                "sub_blocks": ["sub_hal_p5"]
            }
        ],
        "sub_blocks": [
            {
                "id": "sub_501",
                "block_id": "blok_5_p1",
                "code": "R501",
                "name": "Pendapatan Upah/Gaji (Pekerjaan Utama & Tambahan)",
                "type": "roster_art_wage",
                "bbox_pt": [13.44, 14.16, 822.94, 221.35],
                "description": "Roster ART: No ART, Uraian, Kategori Lap Usaha, Jenis Pekerjaan, Upah Uang, Barang, Total"
            },
            {
                "id": "sub_502",
                "block_id": "blok_5_p1",
                "code": "R502",
                "name": "Pendapatan Usaha Rumah Tangga",
                "type": "roster_business",
                "bbox_pt": [13.44, 221.35, 822.94, 428.92],
                "description": "Roster Usaha: No ART, Kegiatan, Kategori, Pekerjaan, Omset, Biaya, Persen Gaji, Stok Akhir"
            },
            {
                "id": "sub_503",
                "block_id": "blok_5_p1",
                "code": "R503",
                "name": "Pendapatan Kepemilikan (Diterima & Dibayar)",
                "type": "matrix_property_income",
                "bbox_pt": [13.44, 428.92, 822.94, 580.44],
                "description": "Matriks 4 Baris: Bunga Simpanan, Dividen, Sewa Tanah, Royalti & Subtotal"
            },
            {
                "id": "sub_hal_p5",
                "block_id": "blok_halaman_p5",
                "code": "HAL_5",
                "name": "Nomor Halaman 5",
                "type": "page_number",
                "bbox_pt": [403.00, 580.80, 439.00, 592.50]
            }
        ]
    },
    6: {
        "blocks": [
            {
                "id": "blok_5_p2",
                "code": "BLOK V (Lanjutan)",
                "title": "PENDAPATAN, PENERIMAAN, DAN PENGELUARAN (Lanjutan)",
                "bbox_pt": [13.44, 14.16, 822.94, 470.86],
                "sub_blocks": ["sub_504", "sub_505"]
            },
            {
                "id": "blok_catatan_p6",
                "code": "CATATAN",
                "title": "KLASIFIKASI LAPANGAN USAHA & PEKERJAAN",
                "bbox_pt": [14.16, 472.66, 823.18, 579.36],
                "sub_blocks": ["sub_ref_p6"]
            },
            {
                "id": "blok_halaman_p6",
                "code": "HALAMAN",
                "title": "NOMOR HALAMAN 6",
                "bbox_pt": [403.00, 580.80, 439.00, 592.50],
                "sub_blocks": ["sub_hal_p6"]
            }
        ],
        "sub_blocks": [
            {
                "id": "sub_504",
                "block_id": "blok_5_p2",
                "code": "R504",
                "name": "Transfer Berjalan Tiga Bulan Terakhir",
                "type": "matrix_transfers_4quadrant",
                "bbox_pt": [13.44, 14.16, 822.94, 213.53],
                "description": "Transfer: Pensiun, Bansos/Zakat, RT Lain, Lainnya x Diterima (Uang/Barang) vs Diberikan"
            },
            {
                "id": "sub_505",
                "block_id": "blok_5_p2",
                "code": "R505",
                "name": "Aset Rumah Tangga / Usaha Rumah Tangga",
                "type": "matrix_asset_balance",
                "bbox_pt": [13.44, 213.53, 822.94, 470.86],
                "description": "Neraca Transaksi Aset: Beli, Beri, Perbaiki, Jual, Nilai Sekarang x 4 Kelas Aset"
            },
            {
                "id": "sub_ref_p6",
                "block_id": "blok_catatan_p6",
                "code": "REF-P6",
                "name": "Panduan Kode 21 Kategori Lapangan Usaha & 10 Jenis Pekerjaan",
                "type": "reference_code",
                "bbox_pt": [14.16, 472.66, 823.18, 579.36]
            },
            {
                "id": "sub_hal_p6",
                "block_id": "blok_halaman_p6",
                "code": "HAL_6",
                "name": "Nomor Halaman 6",
                "type": "page_number",
                "bbox_pt": [403.00, 580.80, 439.00, 592.50]
            }
        ]
    },
    7: {
        "blocks": [
            {
                "id": "blok_6",
                "code": "BLOK VI",
                "title": "POSISI ASET DAN KEWAJIBAN KEUANGAN",
                "bbox_pt": [14.16, 30.96, 822.94, 410.11],
                "sub_blocks": ["sub_601_607", "sub_608_612"]
            },
            {
                "id": "blok_7",
                "code": "BLOK VII",
                "title": "CATATAN / KUNJUNGAN PENCACAHAN",
                "bbox_pt": [14.16, 424.39, 822.94, 568.32],
                "sub_blocks": ["sub_kunjungan"]
            },
            {
                "id": "blok_waktu_selesai",
                "code": "WAKTU_SELESAI",
                "title": "WAKTU SELESAI WAWANCARA",
                "bbox_pt": [14.16, 568.32, 200.00, 588.00],
                "sub_blocks": ["sub_waktu_selesai"]
            },
            {
                "id": "blok_halaman_p7",
                "code": "HALAMAN",
                "title": "NOMOR HALAMAN 7",
                "bbox_pt": [403.00, 577.00, 439.00, 592.50],
                "sub_blocks": ["sub_hal_p7"]
            }
        ],
        "sub_blocks": [
            {
                "id": "sub_601_607",
                "block_id": "blok_6",
                "code": "R601-R607",
                "name": "Posisi Aset Keuangan Rumah Tangga",
                "type": "financial_assets",
                "bbox_pt": [14.16, 30.96, 426.29, 410.11]
            },
            {
                "id": "sub_608_612",
                "block_id": "blok_6",
                "code": "R608-R612",
                "name": "Posisi Kewajiban Keuangan (Utang/Pinjaman)",
                "type": "financial_liabilities",
                "bbox_pt": [426.29, 30.96, 822.94, 410.11]
            },
            {
                "id": "sub_kunjungan",
                "block_id": "blok_7",
                "code": "R701-R703",
                "name": "Jadwal Kunjungan Pencacahan, Waktu Wawancara, & Catatan",
                "type": "survey_visits",
                "bbox_pt": [14.16, 424.39, 822.94, 568.32]
            },
            {
                "id": "sub_waktu_selesai",
                "block_id": "blok_waktu_selesai",
                "code": "WAKTU_SELESAI",
                "name": "Waktu Selesai Wawancara (Jam : Menit)",
                "type": "interview_time",
                "bbox_pt": [14.16, 568.32, 200.00, 588.00]
            },
            {
                "id": "sub_hal_p7",
                "block_id": "blok_halaman_p7",
                "code": "HAL_7",
                "name": "Nomor Halaman 7",
                "type": "page_number",
                "bbox_pt": [403.00, 577.00, 439.00, 592.50]
            }
        ]
    }
}

def inject():
    print("=== Menyuntikkan Metadata Presisi Sub-Blok ke Seluruh Template JSON ===")
    for pno in range(1, 8):
        fn = TEMPLATES_DIR / f"template_p{pno}.json"
        if not fn.exists():
            print(f"File {fn} tidak ditemukan!")
            continue
            
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        hier = PAGE_HIERARCHY.get(pno)
        if not hier:
            continue
            
        cells = data.get("cells", [])
        fields = data.get("fields", [])
        
        # 0. Pastikan sel dan field nomor halaman terdaftar
        c_id = f"p{pno}_c_hal"
        f_id = f"p{pno}_F_hal"
        hal_bboxes = {
            1: [280.00, 824.00, 316.00, 840.00],
            2: [280.00, 826.50, 316.00, 840.00],
            3: [280.00, 804.00, 316.00, 840.00],
            4: [280.00, 824.00, 316.00, 840.00],
            5: [403.00, 580.80, 439.00, 592.50],
            6: [403.00, 580.80, 439.00, 592.50],
            7: [403.00, 577.00, 439.00, 592.50],
        }
        hal_bbox = hal_bboxes.get(pno, [280.00, 824.00, 316.00, 840.00])
        page_w, page_h = data.get("size_pt", [595.32, 841.92])
        
        existing_cell = next((c for c in cells if c["id"] == c_id), None)
        if existing_cell is None:
            cells.append({
                "id": c_id,
                "kind": "digit",
                "bbox_pt": hal_bbox,
                "bbox_n": [round(hal_bbox[0]/page_w, 5), round(hal_bbox[1]/page_h, 5), round(hal_bbox[2]/page_w, 5), round(hal_bbox[3]/page_h, 5)],
                "fill": "white",
                "fill_rgb": [255, 255, 255],
                "text": str(pno)
            })
        else:
            existing_cell["bbox_pt"] = hal_bbox
            existing_cell["bbox_n"] = [round(hal_bbox[0]/page_w, 5), round(hal_bbox[1]/page_h, 5), round(hal_bbox[2]/page_w, 5), round(hal_bbox[3]/page_h, 5)]
            existing_cell["text"] = str(pno)
            
        existing_field = next((f for f in fields if f["id"] == f_id), None)
        if existing_field is None:
            fields.append({
                "id": f_id,
                "row_label": f"HAL_{pno}",
                "n_boxes": 1,
                "boxes": [c_id],
                "parent": None,
                "parent_text": f"Nomor Halaman {pno}",
                "printed": str(pno),
                "bbox_pt": hal_bbox
            })
        else:
            existing_field["bbox_pt"] = hal_bbox
            existing_field["printed"] = str(pno)
            existing_field["row_label"] = f"HAL_{pno}"
            
        if "counts" in data:
            data["counts"]["digit"] = len([c for c in cells if c["kind"] == "digit" and min(c["fill_rgb"]) >= 245])
            data["counts"]["fields"] = len(fields)
            
        # Valid digit ids (white cells)
        valid_digit_ids = set(c["id"] for c in cells if c["kind"] == "digit" and min(c["fill_rgb"]) >= 245)
        
        # Spatial linking per sub-block
        sub_blocks = []
        for sb in hier["sub_blocks"]:
            sb_copy = dict(sb)
            bx = sb_copy["bbox_pt"]
            x_min, y_min, x_max, y_max = bx
            
            # Find all cells whose center falls inside the sub-block bbox
            matched_cells = []
            for c in cells:
                cx = (c["bbox_pt"][0] + c["bbox_pt"][2]) / 2.0
                cy = (c["bbox_pt"][1] + c["bbox_pt"][3]) / 2.0
                if x_min - 1.5 <= cx <= x_max + 1.5 and y_min - 1.5 <= cy <= y_max + 1.5:
                    matched_cells.append(c["id"])
                    
            matched_cell_set = set(matched_cells)
            matched_fields = [f["id"] for f in fields if any(b in matched_cell_set for b in f["boxes"])]
            matched_digits = [cid for cid in matched_cells if cid in valid_digit_ids]
            
            sb_copy["cells_count"] = len(matched_cells)
            sb_copy["fields_count"] = len(matched_fields)
            sb_copy["digits_count"] = len(matched_digits)
            sb_copy["fields"] = matched_fields
            sub_blocks.append(sb_copy)
            
        data["blocks"] = hier["blocks"]
        data["sub_blocks"] = sub_blocks
        
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Halaman {pno}: Injeksi {len(hier['blocks'])} Blocks, {len(sub_blocks)} Sub-Blocks:")
        for sb in sub_blocks:
            print(f"   -> {sb['code']:<14}: {sb['name'][:38]:<38} | Bbox: {[round(v, 2) for v in sb['bbox_pt']]} | Sel: {sb['cells_count']:<3} | Field: {sb['fields_count']:<2} | Digit: {sb['digits_count']}")

if __name__ == "__main__":
    inject()
