"""
services/label_normalizer.py
Modul Normalisasi Semantik Karakter Kuesioner BPS
Mendukung:
- Deteksi & Normalisasi Tanda Centang / Ceklis (✓, v, V, checkmark)
- Deteksi, Konversi & Normalisasi Angka Romawi (I, II, III, IV, V, VI, VII, VIII, IX, X, dst.)
- Penanganan Angka Arab (0–9)
- Penanganan Kotak Kosong (Blank / ∅)
"""

import re
from typing import Optional, Dict, Any, Union

# Kamus konversi Romawi <-> Arab
ROMAN_TO_ARABIC_MAP = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
    "XVIII": 18,
    "XIX": 19,
    "XX": 20
}

ARABIC_TO_ROMAN_MAP = {v: k for k, v in ROMAN_TO_ARABIC_MAP.items()}


def is_checkmark_or_tick(val: Any) -> bool:
    """Mengecek apakah nilai merepresentasikan tanda centang/ceklis atau persetujuan responden."""
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {
        "✓", "✔", "check", "checked", "tick", "centang", "ceklis",
        "[v]", "(v)", "yes", "true", "bersedia", "setuju", "persetujuan", "ok"
    } or (s == "v" and len(str(val).strip()) == 1)


def is_cross(val: Any) -> bool:
    """Mengecek apakah nilai merepresentasikan tanda silang."""
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"x", "✕", "✖", "cross", "silang", "[x]", "(x)"}


def is_roman_numeral(val: Any) -> bool:
    """Mengecek apakah teks adalah angka Romawi yang valid (I s.d. XX)."""
    if val is None:
        return False
    s = str(val).strip().upper()
    return s in ROMAN_TO_ARABIC_MAP


def roman_to_arabic(text: str) -> Optional[int]:
    """Mengonversi angka Romawi (I..XX) ke bilangan bulat Arab (1..20)."""
    if not text:
        return None
    s = text.strip().upper()
    return ROMAN_TO_ARABIC_MAP.get(s)


def arabic_to_roman(n: int) -> Optional[str]:
    """Mengonversi bilangan bulat (1..20) ke string angka Romawi."""
    return ARABIC_TO_ROMAN_MAP.get(n)


def normalize_ai_label(
    raw_char: Any,
    field_id: Optional[str] = None,
    allow_roman: bool = True,
    allow_checkmark: bool = True,
    prefer_roman: bool = False
) -> Dict[str, Any]:
    """
    Menormalkan output mentah inferensi AI menjadi struktur standar BPS.
    """
    if raw_char is None or raw_char == "":
        return {
            "char": "",
            "char_type": "blank",
            "is_blank": True,
            "is_checked": False,
            "numeric_value": None,
            "display_label": "∅ Kosong"
        }

    s = str(raw_char).strip()

    # 1. Blank variations (termasuk crop yang hanya berisi garis/bingkai kotak digit)
    if s.lower() in {"", " ", "_", "null", "none", "blank", "kosong", "∅", "kotak", "kotak_digit", "bingkai", "border", "empty_box", "box"}:
        return {
            "char": "",
            "char_type": "blank",
            "is_blank": True,
            "is_checked": False,
            "numeric_value": None,
            "display_label": "∅ Kosong"
        }

    s_upper = s.upper()

    # 1.1 ATURAN KHUSUS KOTAK TRIWULAN (KOP PENDATAAN BPS)
    # Di lapangan pencacah menuliskan angka Romawi (I, II, III, IV), angka Arab (1-4), atau coretan turus (//)
    if field_id and ("triwulan" in str(field_id).lower()):
        if s in {"//", "II", "ii", "2", "ll", "||", "2.0"} or s_upper == "II":
            return {
                "char": "2",
                "char_type": "digit",
                "is_blank": False,
                "is_checked": False,
                "numeric_value": 2,
                "display_label": "Triwulan 2 (II)"
            }
        elif s in {"/", "I", "i", "1", "|", "1.0"} or s_upper == "I":
            return {
                "char": "1",
                "char_type": "digit",
                "is_blank": False,
                "is_checked": False,
                "numeric_value": 1,
                "display_label": "Triwulan 1 (I)"
            }
        elif s in {"///", "III", "iii", "3", "|||", "3.0"} or s_upper == "III":
            return {
                "char": "3",
                "char_type": "digit",
                "is_blank": False,
                "is_checked": False,
                "numeric_value": 3,
                "display_label": "Triwulan 3 (III)"
            }
        elif s in {"IV", "iv", "4", "4.0"} or s_upper == "IV":
            return {
                "char": "4",
                "char_type": "digit",
                "is_blank": False,
                "is_checked": False,
                "numeric_value": 4,
                "display_label": "Triwulan 4 (IV)"
            }
        elif s.lower() in {"ragu_ragu", "ragu", "?"}:
            # Di kuesioner Seruti 2026, coretan dua garis paralel '//' atau Romawi 'II'
            # seringkali ditandai 'ragu_ragu' oleh model digit standar 0-9.
            # Normalizer merekonsiliasi coretan tinta pada kotak triwulan menjadi Triwulan 2.
            return {
                "char": "2",
                "char_type": "digit",
                "is_blank": False,
                "is_checked": False,
                "numeric_value": 2,
                "display_label": "Triwulan 2 (II) [Auto-Resolved]"
            }

    # Prioritas Romawi jika prefer_roman aktif (misal pada field blok Romawi)
    if prefer_roman and allow_roman and is_roman_numeral(s_upper):
        arabic_num = roman_to_arabic(s_upper)
        return {
            "char": s_upper,
            "char_type": "roman",
            "is_blank": False,
            "is_checked": False,
            "numeric_value": arabic_num,
            "display_label": f"Romawi {s_upper} ({arabic_num})"
        }

    # 2. Checkmark / Centang (✓)
    if allow_checkmark and is_checkmark_or_tick(s):
        return {
            "char": "✓",
            "char_type": "checkmark",
            "is_blank": False,
            "is_checked": True,
            "numeric_value": 1,
            "display_label": "✓ Tercentang"
        }

    # 3. Cross / Tanda Silang ("x" atau "X" atau simbol silang)
    if is_cross(s):
        return {
            "char": "X",
            "char_type": "cross",
            "is_blank": False,
            "is_checked": False,
            "numeric_value": 0,
            "display_label": "X Silang"
        }

    # 4. Angka Romawi (misal: "IV", "III", "I", "II", "V", "VI", "VII", "X")
    if allow_roman and is_roman_numeral(s_upper):
        arabic_num = roman_to_arabic(s_upper)
        return {
            "char": s_upper,
            "char_type": "roman",
            "is_blank": False,
            "is_checked": False,
            "numeric_value": arabic_num,
            "display_label": f"Romawi {s_upper} ({arabic_num})"
        }

    # 5. Angka Arab (0 sampai 9)
    if s.isdigit():
        num_val = int(s)
        return {
            "char": s,
            "char_type": "digit",
            "is_blank": False,
            "is_checked": False,
            "numeric_value": num_val,
            "display_label": f"Angka {s}"
        }

    # 6. Fallback karakter tunggal umum
    return {
        "char": s[0].upper() if len(s) == 1 else s,
        "char_type": "digit",
        "is_blank": False,
        "is_checked": False,
        "numeric_value": None,
        "display_label": f"Karakter {s}"
    }
