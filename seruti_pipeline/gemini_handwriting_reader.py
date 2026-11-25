"""
Mesin Pengenal Tulisan Tangan Bebas (Freeform Handwriting OCR) Kuesioner Seruti BPS
Menggunakan Google Gemini Vision Flagship (gemini-3.8-flash) dengan Pydantic Structured Output.
"""

import os
import time
from typing import Dict, Any, List, Optional
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Muat kredensial dari .env (Windows UTF-8 BOM safe)
for _env_path in [".env", os.path.expanduser("~/.env")]:
    if os.path.exists(_env_path):
        try:
            with open(_env_path, "r", encoding="utf-8-sig") as _f:
                for _line in _f.read().splitlines():
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        _k = _k.strip().lstrip("\ufeff")
                        _v = _v.strip().strip("'\"")
                        if _k and _v:
                            os.environ[_k] = _v
        except Exception:
            pass

API_KEY = os.getenv("GEMINI_API_KEY", "")

CANDIDATE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

MODEL_COOLDOWN = {m: 0.0 for m in CANDIDATE_MODELS}
COOLDOWN_DURATION = 60.0


class HasilTeksField(BaseModel):
    text: str = Field(
        description="Teks tulisan tangan yang terbaca bersih. Jika kosong atau hanya garis titik-titik, isi 'KOSONG'."
    )
    is_blank: bool = Field(
        description="True jika bidang kosong tanpa goresan tulisan tangan, False jika ada tulisan"
    )
    confidence: float = Field(
        description="Tingkat keyakinan pembacaan karakter 0.0 sampai 1.0"
    )
    catatan: str = Field(
        description="Catatan jika ada keraguan, coretan revisi, atau tulisan terpotong"
    )


PROMPTS_BY_TYPE = {
    "single_line_name": (
        "Anda adalah pakar verifikasi OCR kuesioner sensus/survei BPS.\n"
        "Tugas: Baca tulisan tangan NAMA ORANG pada potongan formulir ini.\n"
        "Aturan Khusus:\n"
        "1. Abaikan garis cetak titik-titik panduan formulir di bawah teks.\n"
        "2. Tuliskan nama dalam HURUF KAPITAL (UPPERCASE) bersih tanpa tanda baca berulang.\n"
        "3. Pertahankan singkatan nama atau gelar jika ada (misal: 'M. YUSUF', 'HJ. FATIMAH').\n"
        "4. Jika kotak kosong atau hanya memuat garis titik-titik formulir tanpa coretan pena, kembalikan text 'KOSONG' dan is_blank True.\n"
        "5. Berikan estimasi keyakinan pembacaan (confidence) dari 0.0 sampai 1.0."
    ),
    "multi_line_address": (
        "Anda adalah pakar verifikasi OCR kuesioner sensus/survei BPS.\n"
        "Tugas: Baca tulisan tangan ALAMAT (Nama Jalan/Gang, RT/RW, Dusun) pada potongan formulir ini.\n"
        "Aturan Khusus:\n"
        "1. Abaikan garis cetak titik-titik formulir.\n"
        "2. Gabungkan baris alamat menjadi satu teks alamat yang utuh dan terbaca wajar.\n"
        "3. Gunakan format huruf kapital standar BPS (misal: 'JL. MERDEKA NO. 14 RT. 02').\n"
        "4. Jika kosong tanpa tulisan, kembalikan text 'KOSONG' dan is_blank True.\n"
        "5. Berikan skor confidence 0.0 sampai 1.0."
    ),
    "freeform_paragraph": (
        "Anda adalah pakar verifikasi OCR kuesioner sensus/survei BPS.\n"
        "Tugas: Baca teks narasi/paragraf tulisan tangan pada BLOK CATATAN KHUSUS ini.\n"
        "Aturan Khusus:\n"
        "1. Baca seluruh catatan secara utuh dan terstruktur.\n"
        "2. Perbaiki singkatan lapangan jika maknanya jelas (misal: 'art' -> Anggota Rumah Tangga, 'krt' -> Kepala Rumah Tangga, 'blm' -> belum).\n"
        "3. Jika kotak bersih tanpa coretan tulisan tangan petugas, kembalikan text 'KOSONG' dan is_blank True.\n"
        "4. Berikan skor confidence 0.0 sampai 1.0."
    ),
    "single_line_occupation": (
        "Anda adalah pakar verifikasi OCR kuesioner sensus/survei BPS.\n"
        "Tugas: Baca tulisan tangan URAIAN PEKERJAAN atau KEGIATAN USAHA (nama profesi, jabatan, bidang kerja) pada potongan formulir ini.\n"
        "Aturan Khusus:\n"
        "1. Abaikan garis cetak titik-titik panduan formulir di bawah teks.\n"
        "2. Tuliskan uraian pekerjaan dalam HURUF KAPITAL (UPPERCASE) bersih tanpa tanda baca berulang (misal: 'BURUH TUKANG BANGUNAN', 'SATPAM / SECURITY KANTOR', 'PETANI SAWIT').\n"
        "3. Pertahankan keterangan instansi, tempat kerja, atau lokasi jika tertulis.\n"
        "4. Jika kotak kosong atau hanya memuat garis titik-titik formulir tanpa coretan pena, kembalikan text 'KOSONG' dan is_blank True.\n"
        "5. Berikan estimasi keyakinan pembacaan (confidence) dari 0.0 sampai 1.0."
    )
}


class GeminiHandwritingReader:
    def __init__(self):
        self.client = genai.Client(
            api_key=API_KEY,
            http_options={"api_version": "v1beta"}
        )

    def recognize_single_field(
        self,
        crop_image_path: str,
        field_type: str = "single_line_name"
    ) -> Dict[str, Any]:
        """
        Mengenali tulisan tangan pada satu potongan citra teks menggunakan Gemini Vision.
        """
        if not os.path.exists(crop_image_path):
            return {
                "text": "KOSONG",
                "is_blank": True,
                "confidence": 0.0,
                "catatan": "Berkas citra tidak ditemukan",
                "model_used": "none"
            }

        prompt = PROMPTS_BY_TYPE.get(field_type, PROMPTS_BY_TYPE["single_line_name"])
        img_pil = Image.open(crop_image_path).convert("RGB")

        global MODEL_COOLDOWN
        now = time.time()

        for model_name in CANDIDATE_MODELS:
            if now < MODEL_COOLDOWN.get(model_name, 0.0):
                continue

            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[img_pil, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=HasilTeksField,
                        temperature=0.0
                    )
                )
                if response and response.parsed:
                    res_p: HasilTeksField = response.parsed
                    text_clean = res_p.text.strip()
                    if text_clean.upper() in ["KOSONG", "BLANK", "EMPTY", "NONE", "-", ""]:
                        text_clean = "KOSONG"
                        is_b = True
                    else:
                        is_b = res_p.is_blank

                    return {
                        "text": text_clean,
                        "is_blank": is_b,
                        "confidence": round(float(res_p.confidence), 4),
                        "catatan": res_p.catatan,
                        "model_used": model_name
                    }
            except Exception as e:
                err_str = str(e)
                if any(tok in err_str for tok in ["503", "504", "DEADLINE_EXCEEDED", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                    MODEL_COOLDOWN[model_name] = time.time() + COOLDOWN_DURATION
                else:
                    time.sleep(0.5)

        return {
            "text": "FAILED",
            "is_blank": False,
            "confidence": 0.0,
            "catatan": "Seluruh model API gagal / quota exhausted",
            "model_used": "FAILED"
        }
