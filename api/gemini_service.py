"""
api/gemini_service.py — Modul Layanan Google Gemini API Vision untuk Inferensi Digit Si-Periksa v2.0
Mendukung:
- Model Berbayar / Cloud: gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro
- Dual-Execution: Modern google.genai Client SDK + Fallback Direct REST HTTP
- Zero-PII Compliance (UU No. 16/1997 & UU No. 27/2022): Hanya potongan crop comb-box anonim yang diproses
- SHA-256 Local Result Caching: Mencegah biaya API berulang untuk citra crop yang sama
"""

import os
import io
import time
import json
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from seruti_pipeline.advanced_filter import is_blank_cell, remove_comb_box_border
from services.label_normalizer import normalize_ai_label, is_checkmark_or_tick, is_roman_numeral

# Coba import google.genai
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import dotenv
    dotenv.load_dotenv()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_FILE = BASE_DIR / "db" / "gemini_inference_cache.json"

AVAILABLE_MODELS = [
    {
        "id": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash (Paling Mutakhir)",
        "description": "Model multimodal generasi 3.8 super cepat dengan akurasi vision tertinggi untuk pengenalan tulisan tangan.",
        "tier": "paid",
        "is_default": True
    },
    {
        "id": "gemini-3.8-flash-high",
        "name": "Gemini 3.8 Flash (High Reasoning)",
        "description": "Model 3.8 Flash dengan budget penalaran tinggi untuk digit yang sangat rusak atau tercoreng.",
        "tier": "paid",
        "is_default": False
    },
    {
        "id": "gemini-3.8-flash-medium",
        "name": "Gemini 3.8 Flash (Medium Reasoning)",
        "description": "Model 3.8 Flash dengan penalaran seimbang dan latensi optimal.",
        "tier": "paid",
        "is_default": False
    },
    {
        "id": "gemini-3.7-flash",
        "name": "Gemini 3.7 Flash",
        "description": "Model multimodal generasi 3.7 dengan latensi rendah.",
        "tier": "paid",
        "is_default": False
    },
    {
        "id": "gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "description": "Model multimodal generasi 3.5 stabil dengan performa vision tajam.",
        "tier": "paid",
        "is_default": False
    },
    {
        "id": "gemini-3.1-pro-preview",
        "name": "Gemini 3.1 Pro Preview",
        "description": "Model penalaran mendalam (deep reasoning) generasi 3.1 untuk visualisasi kompleks.",
        "tier": "paid",
        "is_default": False
    },
    {
        "id": "gemini-3-flash-preview",
        "name": "Gemini 3.0 Flash Preview",
        "description": "Model generasi 3.0 Flash berkecepatan tinggi.",
        "tier": "paid",
        "is_default": False
    },
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "description": "Multimodal ultra-cepat dengan akurasi vision tinggi untuk tulisan tangan & hemat biaya.",
        "tier": "paid",
        "is_default": False
    },
    {
        "id": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "description": "Model penalaran vision paling mutakhir generasi 2.5 untuk digit tulisan tangan yang sangat rusak/ambigu.",
        "tier": "paid",
        "is_default": False
    },
    {
        "id": "gemini-1.5-flash",
        "name": "Gemini 1.5 Flash",
        "description": "Model generasi 1.5 Flash berkecepatan tinggi dan berbiaya minimal.",
        "tier": "paid",
        "is_default": False
    },
    {
        "id": "gemini-1.5-pro",
        "name": "Gemini 1.5 Pro",
        "description": "Model vision kompleks generasi 1.5.",
        "tier": "paid",
        "is_default": False
    }
]

# In-memory result cache
_CACHE: Dict[str, Any] = {}


def _load_cache():
    global _CACHE
    if CACHE_FILE.exists():
        try:
            _CACHE = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = {}


def _save_cache():
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


_load_cache()


def get_gemini_api_key() -> Optional[str]:
    """Mengambil API key dari environment variables."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


def set_gemini_api_key(api_key: str) -> bool:
    """Menyimpan API key baru ke environment dan berkas .env."""
    clean_key = api_key.strip()
    os.environ["GEMINI_API_KEY"] = clean_key

    env_path = BASE_DIR / ".env"
    try:
        lines = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()

        found = False
        new_lines = []
        for line in lines:
            if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                new_lines.append(f"GEMINI_API_KEY={clean_key}")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"GEMINI_API_KEY={clean_key}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception as e:
        print(f"[GeminiService] Gagal memperbarui .env: {e}")
        return True  # Tetap aktif di memory os.environ


def mask_api_key(api_key: Optional[str]) -> Optional[str]:
    """Menyamarkan API key untuk tampilan UI yang aman."""
    if not api_key:
        return None
    s = api_key.strip()
    if len(s) <= 8:
        return "••••••••"
    return f"{s[:6]}••••••••{s[-4:]}"


def get_gemini_status() -> Dict[str, Any]:
    """Mengambil status konfigurasi model Gemini API."""
    key = get_gemini_api_key()
    return {
        "configured": bool(key),
        "masked_key": mask_api_key(key),
        "default_model": "gemini-3.8-flash",
        "has_sdk": HAS_GENAI_SDK,
        "available_models": AVAILABLE_MODELS,
        "cached_predictions_count": len(_CACHE)
    }


def get_genai_client(api_key: str):
    """
    Membuat klien Google GenAI dengan konfigurasi endpoint v1beta.
    Endpoint v1beta diperlukan untuk mendukung fitur response_mime_type='application/json' (JSON mode).
    """
    if not HAS_GENAI_SDK:
        return None
    try:
        return genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
    except Exception:
        return genai.Client(api_key=api_key)


def test_gemini_connection(api_key: Optional[str] = None, model: str = "gemini-3.8-flash") -> Dict[str, Any]:
    """Melakukan pengujian koneksi ke Google Gemini API."""
    key = api_key or get_gemini_api_key()
    if not key:
        return {
            "success": False,
            "error": "API Key belum dikonfigurasi. Silakan masukkan Google Gemini API Key Anda.",
            "latency_ms": 0
        }

    t0 = time.time()
    try:
        if HAS_GENAI_SDK:
            client = get_genai_client(key)
            resp = client.models.generate_content(
                model=model,
                contents="Hitung hasil 7 + 5. Berikan jawaban dalam angka saja.",
            )
            text_out = resp.text.strip() if hasattr(resp, "text") else ""
            latency = int((time.time() - t0) * 1000)
            return {
                "success": True,
                "model": model,
                "latency_ms": latency,
                "response_sample": text_out,
                "message": f"Koneksi Gemini API berhasil! Respons diterima dalam {latency} ms."
            }
        else:
            # Fallback HTTP REST
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            req_data = {
                "contents": [{"parts": [{"text": "Hitung hasil 7 + 5. Angka saja."}]}]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                latency = int((time.time() - t0) * 1000)
                return {
                    "success": True,
                    "model": model,
                    "latency_ms": latency,
                    "response_sample": str(res_body)[:100],
                    "message": f"Koneksi REST Gemini API berhasil ({latency} ms)."
                }
    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        return {
            "success": False,
            "error": f"Gagal terhubung ke Gemini API: {str(e)}",
            "latency_ms": latency
        }


def _calc_image_hash(image_path: Path) -> str:
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def classify_digit_crop(
    image_path: str,
    model: str = "gemini-2.5-flash",
    use_cache: bool = True,
    force_eval: bool = False
) -> Dict[str, Any]:
    """
    Melakukan inferensi vision pada satu citra potongan kotak digit (*crop*)
    menggunakan Google Gemini Vision API berbayar.
    
    Zero-PII Safe: Hanya citra crop kotak digit biner/grayscale anonim tanpa identitas responden.
    OpenCV Blank Filter: Kotak kosong yang terdeteksi OpenCV langsung dilewati tanpa memanggil API.
    """
    p = Path(image_path)
    if not p.is_absolute():
        p = BASE_DIR / p.as_posix().lstrip("/")
    if not p.exists():
        return {
            "error": f"Citra crop tidak ditemukan pada jalur: {p}",
            "digit": None,
            "confidence": 0.0
        }

    # ATURAN MUTLAK: Inferensi hanya dilakukan untuk selain kotak kosong yang sudah difilter OpenCV
    if not force_eval:
        try:
            cv_img = cv2.imread(str(p))
            if cv_img is not None and is_blank_cell(cv_img):
                return {
                    "model": "opencv_blank_filter",
                    "pred_digit": "_",
                    "char": "_",
                    "char_type": "blank",
                    "is_checked": False,
                    "numeric_value": None,
                    "display_label": "KOSONG (_)",
                    "conf": 1.0,
                    "conf_pct": 100.0,
                    "is_blank": True,
                    "alt_digit": "-",
                    "alt_char": "-",
                    "alt_conf": 0.0,
                    "probabilities": [0.0] * 10,
                    "top3": [
                        {"digit": "KOSONG (_)", "char": "_", "char_type": "blank", "conf": 100.0},
                        {"digit": "Tinta Pensil", "char": "-", "char_type": "blank", "conf": 0.0},
                        {"digit": "Noise", "char": "-", "char_type": "blank", "conf": 0.0}
                    ],
                    "reasoning": "Kotak kosong terdeteksi oleh OpenCV Blank Detector (Zero Tinta/Grafis). Inferensi AI dilewati 100% hemat biaya.",
                    "latency_ms": 0,
                    "from_cache": False,
                    "bypassed_by_opencv": True,
                    "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as cv_err:
            print(f"[GeminiService] Warning saat memeriksa is_blank_cell: {cv_err}")

    img_hash = _calc_image_hash(p)
    cache_key = f"{model}:{img_hash}"
    if use_cache and cache_key in _CACHE:
        cached = dict(_CACHE[cache_key])
        cached["from_cache"] = True
        return cached

    key = get_gemini_api_key()
    if not key:
        return {
            "error": "GEMINI_API_KEY belum dikonfigurasi. Silakan atur API Key terlebih dahulu.",
            "digit": None,
            "confidence": 0.0
        }

    prompt = (
        "Anda adalah sistem OCR vision cerdas pembaca formulir dan kuesioner resmi BPS (Badan Pusat Statistik).\n"
        "Periksa gambar potongan (crop) kotak isian comb-box ini secara teliti.\n"
        "Tugas:\n"
        "1. KOTAK KOSONG & HANYA BERISI KOTAK DIGIT:\n"
        "   - Jika crop hanya berisi kotak kosong tanpa coretan tangan, kembalikan char: null, digit: null, is_blank: true, char_type: 'blank', confidence: 1.0.\n"
        "   - ATURAN KHUSUS KOTAK DIGIT: Jika crop HANYA BERISI KOTAK DIGIT (yaitu hanya memuat garis pembatas/bingkai cetakan kotak comb-box, garis kisi tabel, atau sudut kotak formulir tanpa adanya goresan tulisan tangan pensil/pulpen responden di dalamnya), MAKA WAJIB DIBERI LABEL KOSONG (char: null, digit: null, is_blank: true, char_type: 'blank', confidence: 1.0).\n"
        "   - PERINGATAN MUTLAK: JANGAN SEKALI-KALI mengira garis tepi/garis batas vertikal kotak cetakan sebagai angka '1', dan JANGAN mengira bingkai kotak persegi sebagai angka '0'. Bila tidak ada goresan tinta tulisan tangan responden di bagian dalam kotak, mutlak kembalikan is_blank: true!\n"
        "2. ANGKA ARAB (0 sampai 9): Jika berisi angka standar 0–9 tulisan tangan responden di dalam kotak, kembalikan char: '0'..'9', char_type: 'digit', numeric_value: int.\n"
        "3. TANDA CENTANG / CEKLIS & PERSETUJUAN/KESEDIAAN RESPONDEN:\n"
        "   - ATURAN KHUSUS CITRA CENTANG (Persetujuan Responden / Checkbox):\n"
        "     * Tanda centang memiliki pola khas: goresan pendek menurun ke arah kanan-bawah, lalu berbalik tajam (sudut lancip) naik memanjang ke arah kanan-atas (bentuk '✓' atau 'v').\n"
        "     * Variasi Goresan Lapangan: Akui segala bentuk centang, baik pensil 2B tipis, pulpen hitam/biru tebal, ceklis cepat (swoosh/hook), centang bersudut membulat, maupun tanda centang yang ekornya menembus bingkai kotak (overshooting stroke).\n"
        "     * Pembedaan Kritis Centang (✓) vs Angka 1: JANGAN mengira tanda centang sebagai angka '1'. Angka '1' berupa garis lurus vertikal tegak. Jika terdapat belokan lancip di bagian bawah dan lengan naik ke kanan atas, itu MUTLAK adalah tanda CENTANG ('✓').\n"
        "     * Persetujuan Responden (Kuesioner BPS): Pada kotak persetujuan/kesediaan responden untuk diwawancarai, tanda centang (✓), tanda silang (X/x), huruf v/V, atau coretan penandaan pilihan menandakan persetujuan afirmatif (1: Bersedia). Kembalikan char: '✓', digit: '✓', char_type: 'checkmark', is_checked: true, numeric_value: 1, confidence: 0.95–1.0.\n"
        "     * Tanda Silang: Jika berisi tanda silang tegas (x/X), kembalikan char: 'X', digit: 'X', char_type: 'cross', is_checked: true, numeric_value: 1.\n"
        "     * Jika kotak persetujuan kosong tanpa goresan tinta, kembalikan char: null, digit: null, is_blank: true, is_checked: false, char_type: 'blank', numeric_value: 0.\n"
        "4. ANGKA ROMAWI: Jika berisi angka Romawi (I, II, III, IV, V, VI, VII, VIII, IX, X), kembalikan char: angka romawi kapital tersebut (misal 'I', 'IV', 'VII'), char_type: 'roman', numeric_value: nilai bilangan bulat Arab ekuivalen (contoh IV -> 4, VII -> 7).\n"
        "5. ATURAN KHUSUS KOTAK TRIWULAN (label/var 'triwulan' atau 'p1_F_triwulan'): Kotak ini berisi periode triwulan survei (hanya bernilai 1 s.d. 4). Pencacah di lapangan sering menuliskan dua garis vertikal sejajar ('//') atau angka Romawi 'II' untuk menandai Triwulan 2. JANGAN PERNAH memberi label 'ragu_ragu' pada angka Romawi atau coretan turus triwulan! Jika melihat dua garis vertikal sejajar ('//') atau Romawi 'II', kembalikan char: '2', digit: '2', numeric_value: 2, confidence: 0.98. Jika satu garis ('/' atau 'I'), kembalikan char: '1', numeric_value: 1. Jika tiga garis ('///' atau 'III'), kembalikan char: '3', numeric_value: 3. Jika 'IV', kembalikan char: '4', numeric_value: 4.\n"
        "6. Berikan skor keyakinan (confidence: 0.00 hingga 1.00) dan karakter alternatif (alt_char).\n"
        "Format respon WAJIB JSON murni tanpa markdown pembungkus:\n"
        "{\n"
        '  "char": "0".."9" | "✓" | "X" | "I".."X" | null,\n'
        '  "digit": "0".."9" | "✓" | "X" | "I".."X" | null,\n'
        '  "char_type": "digit" | "checkmark" | "cross" | "roman" | "blank",\n'
        '  "confidence": 0.00..1.00,\n'
        '  "is_blank": true/false,\n'
        '  "is_checked": true/false,\n'
        '  "numeric_value": integer atau null,\n'
        '  "alt_char": string atau null,\n'
        '  "alt_digit": string atau null,\n'
        '  "alt_confidence": 0.00..1.00,\n'
        '  "probabilities": [0.0, 0.0, ..., 0.0],\n'
        '  "reasoning": "penjelasan singkat karakteristik visual goresan tinta"\n'
        "}"
    )

    t0 = time.time()
    result = None

    # Helper untuk membersihkan dan mengurai teks JSON secara tangguh
    def _parse_json_response(raw_text: str) -> dict:
        txt = raw_text.strip()
        if txt.startswith("```json"):
            txt = txt[7:]
        elif txt.startswith("```"):
            txt = txt[3:]
        if txt.endswith("```"):
            txt = txt[:-3]
        txt = txt.strip()
        try:
            return json.loads(txt)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                return json.loads(m.group(0))
            raise

    try:
        # Jalur SDK modern dengan endpoint v1beta
        if HAS_GENAI_SDK and HAS_PIL:
            client = get_genai_client(key)
            with Image.open(p) as pil_img:
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=[pil_img, prompt],
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        ) if hasattr(genai_types, "GenerateContentConfig") else None
                    )
                except Exception as sdk_err:
                    # Fallback adaptif: jika endpoint v1 menolak response_mime_type, panggil tanpa JSON mode
                    err_str = str(sdk_err)
                    if "JSON mode" in err_str or "v1beta" in err_str or "INVALID_ARGUMENT" in err_str:
                        resp = client.models.generate_content(
                            model=model,
                            contents=[pil_img, prompt]
                        )
                    else:
                        raise sdk_err

                text_out = resp.text.strip() if hasattr(resp, "text") else ""
                result = _parse_json_response(text_out)
        else:
            # Jalur REST HTTP langsung (selalu v1beta)
            import urllib.request
            with open(p, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            req_data = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/png", "data": img_b64}}
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.1
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                text_out = res_body["candidates"][0]["content"]["parts"][0]["text"].strip()
                result = _parse_json_response(text_out)

    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        return {
            "error": f"Inferensi Gemini API gagal: {str(e)}",
            "digit": None,
            "confidence": 0.0,
            "latency_ms": latency
        }

    latency = int((time.time() - t0) * 1000)
    if not isinstance(result, dict):
        result = {"digit": None, "char": None, "confidence": 0.0, "is_blank": True}

    raw_char = result.get("char") if "char" in result else result.get("digit")
    norm = normalize_ai_label(raw_char)
    char = norm["char"]
    char_type = result.get("char_type") or norm["char_type"]
    conf = float(result.get("confidence", 0.98))
    is_blank = bool(result.get("is_blank", norm["is_blank"]))

    # Guard: Jika AI mendeteksi hanya garis kotak/bingkai kotak digit cetakan tanpa coretan tangan
    reasoning_str = str(result.get("reasoning", "")).lower()
    if any(k in reasoning_str for k in ["hanya kotak", "hanya bingkai", "hanya garis batas", "garis comb-box", "empty box", "kotak digit saja", "garis cetak"]):
        is_blank = True
        char = ""
        char_type = "blank"
        conf = 1.0

    is_checked = bool(result.get("is_checked", norm["is_checked"]))
    numeric_value = result.get("numeric_value") if result.get("numeric_value") is not None else norm["numeric_value"]

    probs = result.get("probabilities")
    if not probs or len(probs) != 10:
        probs = [0.001] * 10
        if char and str(char).isdigit():
            c_int = int(char)
            probs[c_int] = round(conf, 4)
            rem = max(0.0, 1.0 - conf)
            for i in range(10):
                if i != c_int:
                    probs[i] = round(rem / 9.0, 4)
        elif char_type in ["checkmark", "roman"]:
            probs[1] = round(conf, 4)
        else:
            probs[0] = 0.02

    alt_char = result.get("alt_char") or result.get("alt_digit") or ("1" if char != "1" else "7")
    alt_conf = float(result.get("alt_confidence", max(0.01, round(1.0 - conf, 4))))

    top3 = [
        {"digit": str(char) if char != "" else "_", "char": char, "char_type": char_type, "conf": round(conf * 100, 1)},
        {"digit": str(alt_char) if alt_char is not None else "0", "char": str(alt_char), "conf": round(alt_conf * 100, 1)},
        {"digit": "3" if str(char) not in ["3", str(alt_char)] else "8", "conf": 0.5}
    ]

    out_data = {
        "model": model,
        "pred_digit": str(char) if char != "" else "_",
        "char": char,
        "char_type": char_type,
        "is_checked": is_checked,
        "numeric_value": numeric_value,
        "display_label": (norm["display_label"] + " (Prediksi Terkuat)") if is_doubtful and not is_blank else norm["display_label"],
                "is_doubtful": is_doubtful,
        "conf": round(conf, 4),
        "conf_pct": round(conf * 100, 1),
        "is_blank": is_blank,
        "alt_digit": str(alt_char),
        "alt_char": str(alt_char),
        "alt_conf": round(alt_conf, 4),
        "probabilities": probs,
        "top3": top3,
        "reasoning": result.get("reasoning", "Analisis fitur goresan vision model Gemini"),
        "latency_ms": latency,
        "from_cache": False,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    _CACHE[cache_key] = out_data
    _save_cache()

    return out_data


def build_crop_grid_mosaic(
    crop_items: List[Dict[str, Any]],
    cols: int = 10,
    cell_w: int = 95,
    cell_h: int = 75,
    header_h: int = 18
) -> tuple:
    """
    Menyatukan daftar potongan citra kotak digit comb-box menjadi satu gambar kisi komposit (*mosaic grid*)
    dengan kapasitas standar 50 crop per grid (10 kolom x 5 baris) untuk efisiensi token ekstrem.
    Setiap kotak diberi tag penanda numerik terstandar (contoh: D01, D02, ..., D50).
    
    Zero-PII Safe: Hanya potongan comb-box anonim yang ditempel pada kisi.
    """
    if not HAS_PIL:
        raise RuntimeError("Pillow (PIL) tidak terpasang untuk membuat kisi komposit.")

    from PIL import ImageDraw, ImageFont

    total_items = len(crop_items)
    if total_items == 0:
        raise ValueError("Daftar crop_items kosong.")

    if total_items <= 15:
        cols = min(cols, 5)
    elif total_items <= 25:
        cols = min(cols, 5)
    else:
        cols = min(cols, 10)

    rows = (total_items + cols - 1) // cols
    grid_w = cols * cell_w
    grid_h = rows * cell_h

    mosaic = Image.new("RGB", (grid_w, grid_h), color=(15, 23, 42))
    draw = ImageDraw.Draw(mosaic)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    tag_map = {}

    for idx, item in enumerate(crop_items):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = r * cell_h
        tag = f"D{idx+1:02d}"
        pred_label = item.get("char") if item.get("char") is not None else item.get("pred_digit")
        if pred_label is not None and str(pred_label).strip() != "" and str(pred_label) != "_":
            header_text = f"{tag}: [{pred_label}]"
        else:
            header_text = tag

        # Header bar tag
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + header_h - 1], fill=(30, 41, 59))
        draw.text((x0 + 4, y0 + 3), header_text, fill=(56, 189, 248), font=font)

        # Box area
        draw.rectangle([x0 + 2, y0 + header_h + 1, x0 + cell_w - 3, y0 + cell_h - 3], fill=(0, 0, 0), outline=(51, 65, 85))

        # Muat dan tempel crop
        raw_p = item.get("crop_path") or item.get("raw_crop_url") or ""
        p = Path(raw_p)
        if not p.is_absolute():
            p = BASE_DIR / raw_p.lstrip("/")

        if p.exists():
            try:
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    bw = cell_w - 6
                    bh = cell_h - header_h - 6
                    img.thumbnail((bw, bh), Image.Resampling.LANCZOS)
                    px = x0 + 3 + (bw - img.width) // 2
                    py = y0 + header_h + 3 + (bh - img.height) // 2
                    mosaic.paste(img, (px, py))
            except Exception as e:
                print(f"[GeminiService] Gagal memuat citra {p}: {e}")

        tag_map[tag] = {
            "index": idx,
            "tag": tag,
            "id": item.get("id"),
            "label": item.get("label"),
            "crop_path": str(p),
            "original_item": item
        }

    # Garis pembatas kisi
    for c in range(cols + 1):
        draw.line([(c * cell_w, 0), (c * cell_w, grid_h)], fill=(71, 85, 105), width=1)
    for r in range(rows + 1):
        draw.line([(0, r * cell_h), (grid_w, r * cell_h)], fill=(71, 85, 105), width=1)

    return mosaic, tag_map


def classify_crop_grid_batch(
    crop_items: List[Dict[str, Any]],
    model: str = "gemini-3.8-flash",
    batch_size: int = 50,
    use_cache: bool = True,
    force_all: bool = False
) -> Dict[str, Any]:
    """
    Melakukan inferensi vision secara batch dengan menyatukan 50 potongan citra (crop)
    menjadi satu kisi komposit (*mosaic grid*), menghemat hingga ~96% token dan panggilan API.
    
    ATURAN MUTLAK:
    Inferensi hanya dilakukan untuk selain kotak kosong yang sudah difilter OpenCV.
    Kotak kosong langsung dilabeli '_' (100% conf) tanpa memanggil Gemini API.
    """
    key = get_gemini_api_key()
    if not key:
        return {
            "success": False,
            "error": "GEMINI_API_KEY belum dikonfigurasi. Silakan atur API Key terlebih dahulu.",
            "total_digits": 0,
            "results": {}
        }

    total_digits = len(crop_items)
    if total_digits == 0:
        return {
            "success": True,
            "total_digits": 0,
            "total_batches": 0,
            "results": {},
            "summary": {"auto_accepted": 0, "needs_review": 0, "avg_confidence": 0.0}
        }

    all_results = {}
    t0_all = time.time()
    total_api_calls = 0

    # 1. Pisahkan kotak kosong (OpenCV Filter) vs kotak terisi
    blank_items = []
    filled_items = []

    for item in crop_items:
        is_item_blank = item.get("is_blank", False)
        item_id = item.get("id") or item.get("label")

        if not is_item_blank and not force_all:
            raw_p = item.get("crop_path") or item.get("raw_crop_url") or ""
            p = Path(raw_p)
            if not p.is_absolute():
                p = BASE_DIR / raw_p.lstrip("/")
            if p.exists():
                try:
                    cv_img = cv2.imread(str(p))
                    if cv_img is not None and is_blank_cell(cv_img):
                        is_item_blank = True
                except Exception:
                    pass

        if is_item_blank and not force_all:
            blank_items.append(item)
            all_results[item_id] = {
                "model": "opencv_blank_filter",
                "pred_digit": "_",
                "char": "_",
                "char_type": "blank",
                "is_checked": False,
                "numeric_value": None,
                "display_label": "KOSONG (_)",
                "conf": 1.0,
                "conf_pct": 100.0,
                "is_blank": True,
                "alt_digit": "-",
                "alt_char": "-",
                "alt_conf": 0.0,
                "probabilities": [0.0] * 10,
                "top3": [
                    {"digit": "KOSONG (_)", "char": "_", "char_type": "blank", "conf": 100.0},
                    {"digit": "Tinta Pensil", "char": "-", "char_type": "blank", "conf": 0.0},
                    {"digit": "Noise", "char": "-", "char_type": "blank", "conf": 0.0}
                ],
                "reasoning": "Kotak kosong terdeteksi oleh OpenCV Blank Detector. Inferensi AI dilewati 100% hemat biaya.",
                "latency_ms": 0,
                "from_cache": False,
                "bypassed_by_opencv": True,
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            filled_items.append(item)

    # Jika seluruh kotak adalah kotak kosong terfilter OpenCV, selesaikan instan tanpa API
    if not filled_items:
        t_elapsed = int((time.time() - t0_all) * 1000)
        return {
            "success": True,
            "total_digits": total_digits,
            "total_evaluated": len(all_results),
            "blank_bypassed_count": len(blank_items),
            "filled_evaluated_count": 0,
            "total_batches": 0,
            "total_api_calls": 0,
            "tokens_saved_pct": 100.0,
            "total_latency_ms": t_elapsed,
            "results": all_results,
            "summary": {
                "auto_accepted": len(blank_items),
                "needs_review": 0,
                "avg_confidence": 100.0
            }
        }

    batch_size = max(5, min(batch_size, 60))
    chunks = [filled_items[i:i + batch_size] for i in range(0, len(filled_items), batch_size)]

    client = get_genai_client(key)

    def _parse_json_response_local(raw_text: str) -> dict:
        txt = raw_text.strip()
        if txt.startswith("```json"):
            txt = txt[7:]
        elif txt.startswith("```"):
            txt = txt[3:]
        if txt.endswith("```"):
            txt = txt[:-3]
        txt = txt.strip()
        try:
            return json.loads(txt)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                return json.loads(m.group(0))
            raise

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_chunk(b_idx: int, chunk: list):
        chunk_results = {}
        calls = 0
        uncached_chunk = []
        for item in chunk:
            raw_p = item.get("crop_path") or item.get("raw_crop_url") or ""
            p = Path(raw_p)
            if not p.is_absolute():
                p = BASE_DIR / raw_p.lstrip("/")
            if p.exists() and use_cache:
                c_key = f"{model}:{_calc_image_hash(p)}"
                if c_key in _CACHE:
                    cached_res = dict(_CACHE[c_key])
                    cached_res["from_cache"] = True
                    chunk_results[item.get("id")] = cached_res
                    continue
            uncached_chunk.append(item)

        if not uncached_chunk:
            # Bangun citra kisi komposit dari chunk agar pratinjau kisi 50 crop selalu tersedia di UI
            try:
                mosaic_img, tag_map = build_crop_grid_mosaic(chunk, cols=10, cell_w=95, cell_h=75)
                m_dir = BASE_DIR / "output" / "crops" / "grid_mosaics"
                m_dir.mkdir(parents=True, exist_ok=True)
                m_filename = f"grid_50_{int(time.time())}_{b_idx+1}.png"
                mosaic_img.save(str(m_dir / m_filename), format="PNG")
                mosaic_rel_url = f"/output/crops/grid_mosaics/{m_filename}"
                m_meta = {
                    "batch_index": b_idx + 1,
                    "crop_count": len(tag_map),
                    "mosaic_url": mosaic_rel_url,
                    "tags": [
                        {
                            "tag": t,
                            "id": tag_map[t].get("id"),
                            "label": tag_map[t].get("label"),
                            "pred_digit": chunk_results.get(tag_map[t].get("id"), {}).get("pred_digit"),
                            "char": chunk_results.get(tag_map[t].get("id"), {}).get("char"),
                            "conf_pct": chunk_results.get(tag_map[t].get("id"), {}).get("conf_pct", 98.0),
                            "char_type": chunk_results.get(tag_map[t].get("id"), {}).get("char_type", "digit"),
                            "reasoning": chunk_results.get(tag_map[t].get("id"), {}).get("reasoning", "Dari cache hasil Gemini Vision")
                        } for t in tag_map
                    ]
                }
                return chunk_results, 0, m_meta
            except Exception:
                return chunk_results, 0, None

        # Buat kisi komposit untuk uncached_chunk (kapasitas 50 crop per grid)
        mosaic_img, tag_map = build_crop_grid_mosaic(uncached_chunk, cols=10, cell_w=95, cell_h=75)
        tags_str = ", ".join(tag_map.keys())

        # Simpan pratinjau kisi komposit untuk inspeksi visual pengguna
        mosaic_rel_url = None
        try:
            m_dir = BASE_DIR / "output" / "crops" / "grid_mosaics"
            m_dir.mkdir(parents=True, exist_ok=True)
            m_filename = f"grid_50_{int(time.time())}_{b_idx+1}.png"
            mosaic_img.save(str(m_dir / m_filename), format="PNG")
            mosaic_rel_url = f"/output/crops/grid_mosaics/{m_filename}"
        except Exception:
            pass

        prompt = (
            "Anda adalah sistem OCR vision cerdas yang membaca formulir sensus resmi BPS (Badan Pusat Statistik).\n"
            f"Gambar ini adalah kisi komposit (mosaic grid) yang menggabungkan {len(tag_map)} potongan kotak comb-box bertanda:\n"
            f"{tags_str}\n\n"
            "Tugas Anda untuk setiap kotak bertanda:\n"
            "1. KOTAK KOSONG & HANYA BERISI KOTAK DIGIT:\n"
            "   - Jika kotak kosong atau HANYA BERISI KOTAK DIGIT (hanya memuat garis pembatas/bingkai cetakan kotak comb-box, garis kisi tabel, atau sudut kotak formulir tanpa adanya goresan tulisan tangan pensil/pulpen responden di bagian dalam), MAKA WAJIB DIBERI LABEL KOSONG (char: null, digit: null, is_blank: true, char_type: 'blank', confidence: 1.0).\n"
            "   - PERINGATAN MUTLAK: JANGAN SEKALI-KALI mengira garis tepi/garis batas vertikal kotak cetakan sebagai angka '1', dan JANGAN mengira bingkai kotak persegi sebagai angka '0'. Bila tidak ada tinta tulisan tangan responden di dalam kotak, mutlak kembalikan is_blank: true!\n"
            "2. ANGKA ARAB (0 sampai 9): Jika berisi angka tulisan tangan 0-9 di dalam kotak, kembalikan char: '0'..'9', char_type: 'digit', numeric_value: int.\n"
            "3. TANDA CENTANG (✓) & PERSETUJUAN/KESEDIAAN RESPONDEN:\n"
            "   - Akui segala bentuk centang (✓), huruf 'v'/'V', ceklis cepat (swoosh), goresan pensil/pulpen, maupun centang yang ekornya menembus tepi kotak (overshooting stroke).\n"
            "   - JANGAN tertukar dengan angka '1': Tanda centang memiliki 2 segmen (turun pendek -> belokan tajam lancip -> naik panjang ke kanan-atas). Bila ada belokan lancip dan lengan naik, itu MUTLAK adalah tanda CENTANG ('✓').\n"
            "   - Pada kotak persetujuan responden / checkbox: Centang (✓), silang (X), atau coretan afirmatif menandakan persetujuan (1: Bersedia). Kembalikan char: '✓', digit: '✓', char_type: 'checkmark', is_checked: true, numeric_value: 1.\n"
            "   - Jika silang tegas (X/x), kembalikan char: 'X', digit: 'X', char_type: 'cross', is_checked: true, numeric_value: 1. Jika kotak kosong, kembalikan char: null, is_blank: true, is_checked: false, numeric_value: 0.\n"
            "4. ANGKA ROMAWI (I..X): Jika berisi angka Romawi (I, II, III, IV, V, VI, VII, VIII, IX, X), kembalikan char: huruf romawi kapital (contoh 'IV', 'VII'), char_type: 'roman', numeric_value: ekuivalen angka Arab (contoh IV -> 4).\n"
            "5. ATURAN KHUSUS KOTAK TRIWULAN (label 'p1_F_triwulan' atau 'TRIWULAN'): Kotak ini berisi periode triwulan survei (hanya bernilai 1 s.d. 4). Pencacah di lapangan sering menuliskan dua garis vertikal sejajar ('//') atau angka Romawi 'II' untuk menandai Triwulan 2. JANGAN PERNAH memberi label 'ragu_ragu' pada angka Romawi atau coretan turus triwulan! Jika melihat dua garis sejajar ('//') atau Romawi 'II', kembalikan char: '2', digit: '2', numeric_value: 2, confidence: 0.98. Jika satu garis ('/' atau 'I'), kembalikan char: '1', numeric_value: 1. Jika tiga garis ('///' atau 'III'), kembalikan char: '3', numeric_value: 3. Jika 'IV', kembalikan char: '4', numeric_value: 4.\n"
            "6. Berikan skor keyakinan (confidence: 0.00 hingga 1.00), alt_char, dan reasoning visual singkat.\n\n"
            "Kembalikan respon WAJIB JSON murni dengan format persis:\n"
            "{\n"
            '  "results": [\n'
            '    {\n'
            '      "tag": "D01",\n'
            '      "char": "0".."9" | "✓" | "X" | "I".."X" | null,\n'
            '      "digit": "0".."9" | "✓" | "X" | "I".."X" | null,\n'
            '      "char_type": "digit" | "checkmark" | "cross" | "roman" | "blank",\n'
            '      "confidence": float,\n'
            '      "is_blank": bool,\n'
            '      "is_checked": bool,\n'
            '      "numeric_value": int atau null,\n'
            '      "alt_char": string atau null,\n'
            '      "reasoning": "..."\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        t_batch = time.time()
        parsed_batch = None
        calls += 1

        try:
            if HAS_GENAI_SDK:
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=[mosaic_img, prompt],
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        ) if hasattr(genai_types, "GenerateContentConfig") else None
                    )
                except Exception as sdk_err:
                    err_str = str(sdk_err)
                    if "JSON mode" in err_str or "v1beta" in err_str or "INVALID_ARGUMENT" in err_str:
                        resp = client.models.generate_content(
                            model=model,
                            contents=[mosaic_img, prompt]
                        )
                    else:
                        raise sdk_err

                text_out = resp.text.strip() if hasattr(resp, "text") else ""
                parsed_batch = _parse_json_response_local(text_out)
            else:
                # Fallback REST
                import urllib.request
                buf = io.BytesIO()
                mosaic_img.save(buf, format="PNG")
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                req_data = {
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/png", "data": b64_data}}
                        ]
                    }],
                    "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1}
                }
                req = urllib.request.Request(url, data=json.dumps(req_data).encode("utf-8"), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=25) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    text_out = res_body["candidates"][0]["content"]["parts"][0]["text"].strip()
                    parsed_batch = _parse_json_response_local(text_out)

        except Exception as e:
            print(f"[GeminiService] Gagal memproses batch {b_idx + 1}: {e}")
            return chunk_results, calls, None

        latency_batch = int((time.time() - t_batch) * 1000)
        per_digit_latency = max(1, latency_batch // max(len(uncached_chunk), 1))

        if not parsed_batch:
            return chunk_results, calls, None

        raw_list = []
        if isinstance(parsed_batch, dict):
            raw_list = parsed_batch.get("results") or parsed_batch.get("digits") or []
        elif isinstance(parsed_batch, list):
            raw_list = parsed_batch

        items_recognized = {r.get("tag"): r for r in raw_list if isinstance(r, dict)}

        for tag, tag_info in tag_map.items():
            r_data = items_recognized.get(tag, {})
            raw_char = r_data.get("char") if "char" in r_data else r_data.get("digit")
            f_ident = tag_info.get("label") or tag_info.get("id") or tag_info.get("crop_path") or ""
            norm = normalize_ai_label(raw_char, field_id=f_ident)
            char = norm["char"]
            char_type = r_data.get("char_type") or norm["char_type"]
            conf = float(r_data.get("confidence", 0.98))
            is_blank = bool(r_data.get("is_blank", norm["is_blank"]))

            # Guard: Jika AI mendeteksi hanya garis kotak/bingkai kotak digit cetakan tanpa coretan tangan
            reasoning_str = str(r_data.get("reasoning", "")).lower()
            if any(k in reasoning_str for k in ["hanya kotak", "hanya bingkai", "hanya garis batas", "garis comb-box", "empty box", "kotak digit saja", "garis cetak"]):
                is_blank = True
                char = ""
                char_type = "blank"
                conf = 1.0

            is_checked = bool(r_data.get("is_checked", norm["is_checked"]))
            numeric_value = r_data.get("numeric_value") if r_data.get("numeric_value") is not None else norm["numeric_value"]

            # ATURAN MUTLAK USER: Untuk label ragu-ragu, tetap ditulis label prediksi terkuatnya
            alt_char = r_data.get("alt_char") or r_data.get("alt_digit")
            is_doubtful = False

            if not is_blank and (char in ["ragu_ragu", "ragu", "?", "", None] or not str(char).strip()):
                is_doubtful = True
                if alt_char and str(alt_char).strip() not in ["ragu_ragu", "ragu", "?", "null", "none"]:
                    char = str(alt_char).strip()
                else:
                    char = "0"
                if str(char).isdigit():
                    char_type = "digit"
                    numeric_value = int(char)
                conf = min(conf, 0.65)
            elif not is_blank and conf < 0.85:
                is_doubtful = True

            if not alt_char or str(alt_char) == str(char):
                alt_char = "1" if str(char) != "1" else "7"

            probs = [0.001] * 10
            if char and str(char).isdigit():
                c_int = int(char)
                probs[c_int] = round(conf, 4)
                rem = max(0.0, 1.0 - conf)
                for i in range(10):
                    if i != c_int:
                        probs[i] = round(rem / 9.0, 4)
            elif char_type in ["checkmark", "roman"]:
                probs[1] = round(conf, 4)
            else:
                probs[0] = 0.02

            alt_conf = max(0.01, round(1.0 - conf, 4))

            top3 = [
                {"digit": str(char) if char != "" else "_", "char": char, "char_type": char_type, "conf": round(conf * 100, 1)},
                {"digit": str(alt_char), "char": str(alt_char), "conf": round(alt_conf * 100, 1)},
                {"digit": "3" if str(char) not in ["3", str(alt_char)] else "8", "conf": 0.5}
            ]

            digit_out = {
                "model": model,
                "pred_digit": str(char) if char != "" else "_",
                "char": char,
                "char_type": char_type,
                "is_checked": is_checked,
                "numeric_value": numeric_value,
                "display_label": (norm["display_label"] + " (Prediksi Terkuat)") if is_doubtful and not is_blank else norm["display_label"],
                "is_doubtful": is_doubtful,
                "conf": round(conf, 4),
                "conf_pct": round(conf * 100, 1),
                "is_blank": is_blank,
                "alt_digit": str(alt_char),
                "alt_char": str(alt_char),
                "alt_conf": round(alt_conf, 4),
                "probabilities": probs,
                "top3": top3,
                "reasoning": r_data.get("reasoning", "Analisis fitur goresan vision model Gemini 3.8 Flash"),
                "latency_ms": per_digit_latency,
                "from_cache": False,
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            if mosaic_rel_url:
                digit_out["mosaic_url"] = mosaic_rel_url
                digit_out["grid_tag"] = tag
                digit_out["grid_batch"] = b_idx + 1

            p_crop = Path(tag_info["crop_path"])
            if p_crop.exists():
                c_key = f"{model}:{_calc_image_hash(p_crop)}"
                _CACHE[c_key] = digit_out

            chunk_results[tag_info["id"]] = digit_out

        mosaic_meta = {
            "batch_index": b_idx + 1,
            "crop_count": len(tag_map),
            "mosaic_url": mosaic_rel_url,
            "tags": [
                {
                    "tag": t,
                    "id": tag_map[t].get("id"),
                    "label": tag_map[t].get("label"),
                    "pred_digit": chunk_results.get(tag_map[t].get("id"), {}).get("pred_digit"),
                    "char": chunk_results.get(tag_map[t].get("id"), {}).get("char"),
                    "conf_pct": chunk_results.get(tag_map[t].get("id"), {}).get("conf_pct", 98.0),
                    "char_type": chunk_results.get(tag_map[t].get("id"), {}).get("char_type", "digit"),
                    "reasoning": chunk_results.get(tag_map[t].get("id"), {}).get("reasoning", "")
                } for t in tag_map
            ]
        }
        return chunk_results, calls, mosaic_meta

    mosaic_batches = []

    # Eksekusi paralel hingga 3 batch bersamaan
    max_workers = min(3, len(chunks)) if len(chunks) > 1 else 1
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_process_chunk, b_idx, chunk): b_idx for b_idx, chunk in enumerate(chunks)}
            for future in as_completed(future_to_idx):
                res_dict, calls_cnt, m_meta = future.result()
                total_api_calls += calls_cnt
                all_results.update(res_dict)
                if m_meta and m_meta.get("mosaic_url"):
                    mosaic_batches.append(m_meta)
    else:
        for b_idx, chunk in enumerate(chunks):
            res_dict, calls_cnt, m_meta = _process_chunk(b_idx, chunk)
            total_api_calls += calls_cnt
            all_results.update(res_dict)
            if m_meta and m_meta.get("mosaic_url"):
                mosaic_batches.append(m_meta)

    mosaic_batches.sort(key=lambda x: x.get("batch_index", 0))
    _save_cache()

    total_time_ms = int((time.time() - t0_all) * 1000)
    saved_tokens_pct = round(max(0, (1 - (total_api_calls / max(total_digits, 1)))) * 100, 1)

    # Ringkasan
    total_eval = len(all_results)
    auto_cnt = sum(1 for v in all_results.values() if v.get("conf", 0) >= 0.95 and not v.get("is_blank"))
    rev_cnt = total_eval - auto_cnt
    avg_c = round(sum(v.get("conf", 0) for v in all_results.values()) / max(total_eval, 1) * 100, 2)

    return {
        "success": True,
        "total_digits": total_digits,
        "total_evaluated": total_eval,
        "blank_bypassed_count": len(blank_items),
        "filled_evaluated_count": len(filled_items),
        "total_batches": len(chunks),
        "batch_size": batch_size,
        "total_api_calls": total_api_calls,
        "tokens_saved_pct": saved_tokens_pct,
        "total_latency_ms": total_time_ms,
        "mosaic_batches": mosaic_batches,
        "results": all_results,
        "summary": {
            "auto_accepted": auto_cnt,
            "needs_review": rev_cnt,
            "avg_confidence": avg_c
        }
    }
