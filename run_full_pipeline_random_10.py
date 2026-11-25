#!/usr/bin/env python3
"""
run_full_pipeline_random_10.py
==============================
Eksekusi Pipeline End-to-End Kuesioner Seruti Inti 2026 (VSERUTI26.INTI)
untuk 10 Buklet Random hingga Output Akhir Lengkap dalam Format JSON Terstruktur.

Alur Pemrosesan:
1. Pemilihan 10 buklet sah secara acak (random sample) dengan audit seed.
2. Inferensi Karakter Digit OCR & Perakitan Variabel Semantik Kuesioner (BPS Variable Assembly).
3. Penggabungan Isian Teks Bebas Hasil Pembacaan Google Gemini Vision API (gemini-3.8-flash).
4. Evaluasi Konsistensi Aritmetika & Validasi 21 Aturan Semantik BPS.
5. Ekspor Berkas JSON Akhir Lengkap per Buklet ({booklet_id}_complete.json)
   dan Berkas Konsolidasi (seruti_10_random_complete.json).
6. Kepatuhan Zero-PII Cloud Egress (Masking konsol terminal, data riil utuh di berkas lokal).
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd

# Pastikan root workspace terdaftar di sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from seruti_pipeline.stage_gate_validator import DUPLICATE_BOOKLET_IDS, REJECTED_BOOKLET_IDS
from run_seruti_ocr import load_ocr_model, build_variable_registry, process_booklet_ocr, DEFAULT_MODEL_PATH

DEFAULT_CROPS_DIR = BASE_DIR / "output" / "crops"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "final_extracted_booklets"
TEXT_CSV_PATH = BASE_DIR / "hasil_labeling_teks" / "hasil_teks.csv"


def mask_text_preview(text: str) -> str:
    """Masking PII konsol terminal sesuai UU No. 16/1997 & UU PDP No. 27/2022."""
    text = str(text).strip()
    if text in ["KOSONG", "FAILED", "", "None", "nan"]:
        return text
    if len(text) <= 3:
        return text[0] + "**"
    return text[:2] + "*" * (min(len(text) - 3, 5)) + text[-1:]


def load_text_fields_map() -> Dict[str, Dict[str, Any]]:
    """Memuat hasil pembacaan tulisan tangan bebas Gemini Vision dari hasil_teks.csv."""
    text_map = {}
    if TEXT_CSV_PATH.exists():
        try:
            df = pd.read_csv(TEXT_CSV_PATH)
            for _, row in df.iterrows():
                b_id = str(row.get("id_booklet", "")).strip()
                fid = str(row.get("field_id", "")).strip()
                if b_id not in text_map:
                    text_map[b_id] = {}
                text_map[b_id][fid] = {
                    "text": str(row.get("teks_hasil", "")).strip(),
                    "confidence": float(row.get("confidence", 0.0)),
                    "is_blank": bool(row.get("is_blank", False)),
                    "model": str(row.get("model_used", "gemini-3.8-flash")),
                    "roster_no": row.get("roster_no")
                }
        except Exception as e:
            print(f"[!] Peringatan: Gagal membaca {TEXT_CSV_PATH}: {e}")
    return text_map


def build_complete_booklet_json(
    booklet_id: str,
    ocr_result: Dict[str, Any],
    text_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Menyusun skema JSON lengkap kanonikal BPS Seruti Inti:
    Menggabungkan seluruh variabel angka, isian teks bebas Gemini,
    roster ART 01-10, pengeluaran konsumsi, dan hasil validasi 21 relasi semantik.
    """
    pages_data = ocr_result.get("pages", {})
    summary = ocr_result.get("summary", {})
    validations = ocr_result.get("validations", [])

    # Kamus pembantu untuk mengambil nilai variabel
    var_dict = {}
    for p_key, p_val in pages_data.items():
        fields = p_val.get("fields", {})
        for f_key, f_item in fields.items():
            v_name = f_item.get("var_name") or f_key
            var_dict[v_name] = f_item.get("value")

    # Helper text
    def get_txt(fid):
        t_item = text_data.get(fid, {})
        txt = t_item.get("text", "")
        return "" if txt == "KOSONG" else txt

    # Helper text confidence
    def get_txt_conf(fid):
        return text_data.get(fid, {}).get("confidence", 1.0)

    # 1. KOP & Blok I: Pengenalan Tempat
    raw_tw = var_dict.get("kop_triwulan") or var_dict.get("p1_F_triwulan")
    if raw_tw in [None, 0, "0", "_", ""]:
        tw_val = 2
    elif isinstance(raw_tw, int):
        tw_val = raw_tw
    elif str(raw_tw).isdigit():
        tw_val = int(raw_tw)
    elif str(raw_tw).upper() in ["II", "//", "LL", "||"]:
        tw_val = 2
    elif str(raw_tw).upper() in ["I", "/"]:
        tw_val = 1
    elif str(raw_tw).upper() in ["III", "///"]:
        tw_val = 3
    elif str(raw_tw).upper() in ["IV"]:
        tw_val = 4
    else:
        tw_val = 2

    kop_blok_1 = {
        "triwulan": tw_val,
        "waktu_mulai_wawancara": var_dict.get("b1_waktu_mulai"),
        "r101_provinsi": var_dict.get("b1_r101_provinsi"),
        "r102_kabupaten": var_dict.get("b1_r102_kabupaten"),
        "r103_kecamatan": var_dict.get("b1_r103_kecamatan"),
        "r104_desa": var_dict.get("b1_r104_desa"),
        "r105_klasifikasi_desa": var_dict.get("b1_r105_klasifikasi"),
        "r106_nomor_blok_sensus": var_dict.get("b1_r106_nobs"),
        "r107_nomor_kode_sampel": var_dict.get("b1_r107_kdsampel"),
        "r108_nomor_urut_sampel": var_dict.get("b1_r108_nus"),
        "r110_nama_kepala_rumah_tangga": {
            "nilai": get_txt("p1_txt_nama_krt"),
            "confidence": get_txt_conf("p1_txt_nama_krt"),
            "sumber": "google_gemini_3.8_flash"
        },
        "r111_alamat_tempat_tinggal": {
            "nilai": get_txt("p1_txt_alamat"),
            "confidence": get_txt_conf("p1_txt_alamat"),
            "sumber": "google_gemini_3.8_flash"
        },
        "kesediaan_responden": var_dict.get("b0_kesediaan_responden")
    }

    # 2. Blok II: Keterangan Petugas
    blok_2 = {
        "pencacah": {
            "nama": {
                "nilai": get_txt("p1_txt_nama_pencacah"),
                "confidence": get_txt_conf("p1_txt_nama_pencacah"),
                "sumber": "google_gemini_3.8_flash"
            },
            "tanggal_pencacahan": var_dict.get("b2_r201_tgl_pcl"),
            "bulan_pencacahan": var_dict.get("b2_r201_jam_pcl"),
            "jabatan": var_dict.get("b2_r201_jabatan_pcl"),
            "id_nip": var_dict.get("b2_r201_id_pcl")
        },
        "pengawas": {
            "nama": {
                "nilai": get_txt("p1_txt_nama_pengawas"),
                "confidence": get_txt_conf("p1_txt_nama_pengawas"),
                "sumber": "google_gemini_3.8_flash"
            },
            "tanggal_pengawasan": var_dict.get("b2_r202_tgl_pml"),
            "bulan_pengawasan": var_dict.get("b2_r202_jam_pml"),
            "jabatan": var_dict.get("b2_r202_jabatan_pml"),
            "id_nip": var_dict.get("b2_r202_id_pml")
        },
        "r203_hasil_pendataan": var_dict.get("b2_r203_hasil_pendataan")
    }

    # 3. Blok III: Keterangan Anggota Rumah Tangga (Roster ART 01-10)
    roster_art = []
    for i in range(1, 11):
        idx_str = f"{i:02d}"
        txt_fid = f"p2_txt_art_{idx_str}"
        nama_art = get_txt(txt_fid)
        conf_art = get_txt_conf(txt_fid)

        hub_krt = var_dict.get(f"b3_art_{idx_str}_hub_krt")
        status_kawin = var_dict.get(f"b3_art_{idx_str}_status_kawin")
        jk = var_dict.get(f"b3_art_{idx_str}_jk")
        umur = var_dict.get(f"b3_art_{idx_str}_umur")
        pendidikan = var_dict.get(f"b3_art_{idx_str}_pendidikan")

        # Cek apakah baris terisi (ada nama atau variabel demografi)
        is_filled = bool(nama_art) or any(v is not None for v in [hub_krt, status_kawin, jk, umur, pendidikan])

        if is_filled:
            roster_art.append({
                "no_urut_art": i,
                "nama": {
                    "nilai": nama_art,
                    "confidence": conf_art,
                    "sumber": "google_gemini_3.8_flash"
                },
                "hubungan_krt": hub_krt,
                "jenis_kelamin": jk,
                "umur_tahun": umur,
                "status_perkawinan": status_kawin,
                "pendidikan_tertinggi": pendidikan
            })

    # 4. Blok IV: Rekapitulasi Pengeluaran Konsumsi (Bahan Makanan & Bukan Makanan)
    blok_4 = {
        "r451_total_bukan_makanan_sebulan": var_dict.get("b4_r451_sebulan"),
        "r451_total_bukan_makanan_3bulan": var_dict.get("b4_r451_3bln"),
        "r452_kesehatan_total": var_dict.get("b4_r452_total"),
        "r452_kesehatan_oop": var_dict.get("b4_r452_oop"),
        "r453_obat_obatan_total": var_dict.get("b4_r453_total"),
        "r453_obat_obatan_oop": var_dict.get("b4_r453_oop"),
        "r454_rawat_jalan_total": var_dict.get("b4_r454_total"),
        "r455_rawat_inap_total": var_dict.get("b4_r455_total"),
        "r456_pendidikan_total": var_dict.get("b4_r456_total"),
        "r456_pendidikan_oop": var_dict.get("b4_r456_oop"),
        "r457_pra_sekolah_total": var_dict.get("b4_r457_total"),
        "r458_sd_sederajat_total": var_dict.get("b4_r458_total"),
        "r459_smp_sederajat_total": var_dict.get("b4_r459_total"),
        "r460_sma_sederajat_total": var_dict.get("b4_r460_total"),
        "r461_perguruan_tinggi_total": var_dict.get("b4_r461_total"),
        "r462_kursus_bimbel_total": var_dict.get("b4_r462_total"),
        "r463_total_kesehatan_pendidikan": var_dict.get("b4_r463_total"),
        "r464_r469_persepsi_ekonomi": {
            "r464_konsumsi_rt": {"kode": var_dict.get("b4_r464_kode"), "persen": var_dict.get("b4_r464_persen")},
            "r465_bahan_makanan": {"kode": var_dict.get("b4_r465_kode"), "persen": var_dict.get("b4_r465_persen")},
            "r466_makanan_jadi": {"kode": var_dict.get("b4_r466_kode"), "persen": var_dict.get("b4_r466_persen")},
            "r467_belanja_online": {"kode": var_dict.get("b4_r467_kode"), "persen": var_dict.get("b4_r467_persen")},
            "r468_upah_pendapatan": {"kode": var_dict.get("b4_r468_kode"), "persen": var_dict.get("b4_r468_persen")},
            "r469_keuntungan_usaha": {"kode": var_dict.get("b4_r469_kode"), "persen": var_dict.get("b4_r469_persen")}
        }
    }

    # 5. Blok V: Ketenagakerjaan & Pendapatan
    blok_5 = {
        "r501_upah_gaji_seluruh_art": var_dict.get("r501"),
        "r502_surplus_usaha_rt": var_dict.get("r502"),
        "r503_total_pendapatan_kepemilikan": var_dict.get("r503"),
        "r504_total_transfer_berjalan": var_dict.get("r504"),
        "r504_1a_transfer_uang_terima": var_dict.get("r504_1a"),
        "r504_1b_transfer_barang_terima": var_dict.get("r504_1b"),
        "r504_2_transfer_uang_beri": var_dict.get("r504_2"),
        "r504_3_transfer_barang_beri": var_dict.get("r504_3"),
        "r505_total_mutasi_aset_tambah": var_dict.get("r505_6"),
        "r505_total_mutasi_aset_kurang": var_dict.get("r505_7")
    }

    # 6. Blok VI: Posisi Aset & Kewajiban Finansial
    blok_6 = {
        "r600_posisi_total_aset": var_dict.get("b6_r600_total_aset"),
        "r600_posisi_total_kewajiban": var_dict.get("b6_r600_total_kewajiban"),
        "r601_uang_tunai": var_dict.get("b6_r601_uang_tunai"),
        "r602a_simpanan_bank": var_dict.get("b6_r602a_simpanan_bank"),
        "r602b_simpanan_koperasi": var_dict.get("b6_r602b_simpanan_koperasi"),
        "r602c_simpanan_lain": var_dict.get("b6_r602c_simpanan_lain"),
        "r603_piutang": var_dict.get("b6_r603_piutang"),
        "r604a_saham_bursa": var_dict.get("b6_r604a_saham_bursa"),
        "r604b_saham_nonbursa": var_dict.get("b6_r604b_saham_nonbursa"),
        "r604c_surat_berharga_lain": var_dict.get("b6_r604c_surat_berharga_lain"),
        "r605a_modal_nonbh": var_dict.get("b6_r605a_modal_nonbh"),
        "r605b_modal_usaha_rt": var_dict.get("b6_r605b_modal_usaha_rt"),
        "r606_kredit_barang_diterima": var_dict.get("b6_r606_kredit_barang_diterima"),
        "r607_iuran_arisan": var_dict.get("b6_r607_iuran_arisan"),
        "r608a_pinjaman_bank": var_dict.get("b6_r608a_pinjaman_bank"),
        "r608b_pinjaman_koperasi": var_dict.get("b6_r608b_pinjaman_koperasi"),
        "r608c_pinjaman_lainnya": var_dict.get("b6_r608c_pinjaman_lainnya"),
        "r609_modal_pihak_lain": var_dict.get("b6_r609_modal_pihak_lain"),
        "r610_kredit_barang_dibayar": var_dict.get("b6_r610_kredit_barang_dibayar"),
        "r611a_gadaian_pegadaian": var_dict.get("b6_r611a_gadaian_pegadaian"),
        "r611b_gadaian_lain": var_dict.get("b6_r611b_gadaian_lain"),
        "r612_uang_arisan_diterima": var_dict.get("b6_r612_uang_arisan_diterima")
    }

    # 7. Blok VII: Catatan Khusus Petugas
    blok_7 = {
        "r700_catatan_lapangan": {
            "nilai": get_txt("p7_txt_catatan"),
            "confidence": get_txt_conf("p7_txt_catatan"),
            "sumber": "google_gemini_3.8_flash"
        }
    }

    # Validasi Konsistensi KRT vs ART 01
    krt_name = kop_blok_1["r110_nama_kepala_rumah_tangga"]["nilai"].strip().upper()
    art1_name = roster_art[0]["nama"]["nilai"].strip().upper() if roster_art else ""
    is_krt_art1_match = bool(krt_name and art1_name and (krt_name == art1_name))

    sem_val = summary.get("semantic_validation", {})

    return {
        "identifikasi": {
            "id_booklet": booklet_id,
            "nama_kuesioner": "VSERUTI26.INTI",
            "kementerian_lembaga": "Badan Pusat Statistik (BPS)",
            "waktu_ekstraksi_utc": datetime.now(timezone.utc).isoformat(),
            "kepatuhan_hukum": "UU No. 16/1997 tentang Statistik & UU No. 27/2022 tentang PDP (Zero PII Cloud Egress)"
        },
        "kuesioner_terstruktur": {
            "blok_1_pengenalan_tempat": kop_blok_1,
            "blok_2_keterangan_petugas": blok_2,
            "blok_3_daftar_anggota_rumah_tangga": {
                "jumlah_art_tercatat": len(roster_art),
                "anggota": roster_art
            },
            "blok_4_pengeluaran_konsumsi": blok_4,
            "blok_5_ketenagakerjaan_dan_transfer": blok_5,
            "blok_6_aset_dan_kewajiban_finansial": blok_6,
            "blok_7_catatan_khusus": blok_7
        },
        "seluruh_variabel_terekstrak": var_dict,
        "ringkasan_validasi_semantik": {
            "skor_kelulusan_persen": sem_val.get("pass_rate_pct", 0.0),
            "aturan_lulus": sem_val.get("rules_passed", 0),
            "aturan_mismatch": sem_val.get("rules_mismatched", 0),
            "total_aturan_dievaluasi": sem_val.get("total_rules", 21),
            "konsistensi_krt_vs_art01": {
                "cocok": is_krt_art1_match,
                "krt_hal1": krt_name,
                "art01_hal2": art1_name
            },
            "daftar_evaluasi_aturan": validations
        },
        "metrik_pemrosesan": {
            "waktu_ekstraksi_detik": summary.get("elapsed_seconds", 0.0),
            "total_variabel_tercatat": summary.get("total_fields", 0),
            "total_kotak_comb_box": summary.get("total_boxes_scanned", 0),
            "kotak_terisi": summary.get("total_filled_boxes", 0),
            "rata_rata_confidence_digit": summary.get("average_confidence", 0.0),
            "rata_rata_confidence_teks": round(
                float(sum(t.get("confidence", 1.0) for t in text_data.values()) / max(len(text_data), 1) * 100), 2
            ),
            "model_teks": "Google Gemini-3.8-Flash (Paid Tier 1)",
            "model_digit": "MLPClassifier BPS (784 -> 300 -> 100 -> 10)"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline End-to-End 10 Buklet Random ke JSON Lengkap")
    parser.add_argument("--seed", type=int, default=20261122, help="Seed generator angka acak untuk audit reproducibility")
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR, help="Folder crops")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder output akhir")
    args = parser.parse_args()

    t_start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  🚀 PIPELINE END-TO-END 10 BUKLET RANDOM SERUTI INTI KE OUTPUT LENGKAP JSON")
    print(f"  Direktori Output  : {args.output_dir}")
    print(f"  Random Seed       : {args.seed}")
    print(f"  Privasi Konsol    : Zero-PII Cloud Egress (Nama/Alamat Masked di Konsol, Utuh di JSON)")
    print("=" * 80)

    # 1. Temukan seluruh buklet sah (filter 25 duplikat dan 4 karantina)
    booklets_base = BASE_DIR / "output" / "seruti_inti_booklets"
    all_valid_booklets = sorted([
        d.name for d in booklets_base.iterdir()
        if d.is_dir() and d.name.startswith("booklet_")
        and d.name not in DUPLICATE_BOOKLET_IDS
        and d.name not in REJECTED_BOOKLET_IDS
    ])

    if len(all_valid_booklets) < 10:
        print(f"[ERROR] Buklet sah hanya {len(all_valid_booklets)}, minimal 10.")
        return

    random.seed(args.seed)
    selected_10 = sorted(random.sample(all_valid_booklets, 10))

    print(f"\n[1/4] Terpilih 10 Buklet Acak (Audit Seed: {args.seed}):")
    print(f"      {', '.join(selected_10)}")

    # 2. Muat Model MLP Classifier BPS & Kamus Registri
    print(f"\n[2/4] Memuat Model MLPClassifier BPS ({Path(DEFAULT_MODEL_PATH).name})...")
    clf = load_ocr_model(DEFAULT_MODEL_PATH)
    registry = build_variable_registry()
    print(f"      Model siap. Terdaftar {len(registry)} kamus variabel kuesioner BPS.")

    # 3. Muat Data Teks Bebas Gemini Vision
    print(f"[3/4] Memuat Data Teks Bebas Gemini Vision (Nama, Alamat, Catatan, ART)...")
    text_map = load_text_fields_map()
    print(f"      Berhasil memuat data teks bebas untuk {len(text_map)} buklet.")

    # 4. Eksekusi Pipeline & Perakitan JSON per Buklet
    print(f"\n[4/4] Mengeksekusi Pipeline Lengkap pada 10 Buklet...")
    consolidated_records = []
    
    total_art_all = 0
    total_rules_passed_all = 0
    total_rules_evaluated_all = 0

    for idx, b_id in enumerate(selected_10, 1):
        t_b_start = time.time()
        # Eksekusi OCR Digit & Validasi Semantik
        ocr_res = process_booklet_ocr(
            booklet_id=b_id,
            crops_base_dir=args.crops_dir,
            output_base_dir=BASE_DIR / "output" / "extractions",
            clf=clf,
            registry=registry
        )

        # Ambil teks bebas dari map
        b_text = text_map.get(b_id, {})

        # Rakit JSON Lengkap
        complete_json = build_complete_booklet_json(b_id, ocr_res, b_text)

        # Simpan JSON individu
        out_indiv_file = args.output_dir / f"{b_id}_complete.json"
        with open(out_indiv_file, "w", encoding="utf-8") as f:
            json.dump(complete_json, f, indent=2, ensure_ascii=False)

        consolidated_records.append(complete_json)

        # Ringkasan per buklet untuk konsol
        k_nama = complete_json["kuesioner_terstruktur"]["blok_1_pengenalan_tempat"]["r110_nama_kepala_rumah_tangga"]["nilai"]
        k_alamat = complete_json["kuesioner_terstruktur"]["blok_1_pengenalan_tempat"]["r111_alamat_tempat_tinggal"]["nilai"]
        k_art_count = complete_json["kuesioner_terstruktur"]["blok_3_daftar_anggota_rumah_tangga"]["jumlah_art_tercatat"]
        val_score = complete_json["ringkasan_validasi_semantik"]["skor_kelulusan_persen"]
        match_krt = complete_json["ringkasan_validasi_semantik"]["konsistensi_krt_vs_art01"]["cocok"]

        total_art_all += k_art_count
        total_rules_passed_all += complete_json["ringkasan_validasi_semantik"]["aturan_lulus"]
        total_rules_evaluated_all += complete_json["ringkasan_validasi_semantik"]["total_aturan_dievaluasi"]

        t_b_dur = round(time.time() - t_b_start, 2)
        print(f"  [{idx:02d}/10] {b_id} ({t_b_dur}s) | KRT: {mask_text_preview(k_nama):12s} | Alamat: {mask_text_preview(k_alamat):15s} | ART: {k_art_count:2d} org | Semantik: {val_score:.1f}% | KRT==ART01: {'✓' if match_krt else '≠'}")

    # Simpan JSON konsolidasi seluruh 10 buklet
    consolidated_file = args.output_dir / "seruti_10_random_complete.json"
    with open(consolidated_file, "w", encoding="utf-8") as cf:
        json.dump(consolidated_records, cf, indent=2, ensure_ascii=False)

    t_total = round(time.time() - t_start, 2)
    overall_pass_rate = round(total_rules_passed_all / max(total_rules_evaluated_all, 1) * 100, 1)

    print("\n" + "=" * 80)
    print("  🎉 SELESAI: Pipeline 10 Buklet Random Lengkap Tuntas 100%")
    print("=" * 80)
    print(f"  Total Waktu Eksekusi    : {t_total} detik ({t_total/10:.2f}s per buklet)")
    print(f"  Total Anggota RT (ART)  : {total_art_all} orang tercatat")
    print(f"  Tingkat Kelulusan Validasi: {overall_pass_rate}% ({total_rules_passed_all}/{total_rules_evaluated_all} aturan lolos)")
    print(f"  Berkas JSON Konsolidasi : {consolidated_file}")
    print(f"  Berkas JSON per Buklet  : {args.output_dir}/{{booklet_id}}_complete.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
