"""
Inspeksi detail 5 buklet bermasalah: booklet_21, booklet_38, booklet_49, booklet_55, booklet_67.
"""
import json
from pathlib import Path
import cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
output_dir = Path("output/seruti_inti_booklets")
raw_dir = Path("inti")

target_booklets = ["booklet_21", "booklet_38", "booklet_49", "booklet_55", "booklet_67"]

for b_id in target_booklets:
    mf_file = output_dir / b_id / "manifest.json"
    if not mf_file.exists():
        continue
    mf = json.loads(mf_file.read_text(encoding="utf-8"))
    spreads = mf["spread_sources"]
    feed = mf.get("feed_direction", "unknown")
    print(f"\n==================================================")
    print(f"{b_id} (Manifest feed_direction: {feed})")
    print(f"==================================================")

    for s_idx, s_name in enumerate(spreads):
        p_list = list(raw_dir.rglob(s_name))
        if not p_list:
            # Cari di root jika tidak ada di inti
            p_list = list(Path(".").glob(f"*/{s_name}"))
        if not p_list:
            print(f"  Spread {s_idx} ({s_name}): FILE NOT FOUND")
            continue
        p = p_list[0]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Spread {s_idx} ({s_name}): UNREADABLE")
            continue
        h, w = img.shape[:2]
        left = img[:, :w//2]
        right = img[:, w//2:]
        std_l, std_r = float(left.std()), float(right.std())

        # Cek OCR top & bottom dari left dan right
        def get_text(half):
            hh = half.shape[0]
            top = half[:int(0.25*hh), :]
            bot = half[int(0.75*hh):, :]
            top_s = cv2.resize(top, (800, int(top.shape[0]*800/top.shape[1])))
            bot_s = cv2.resize(bot, (800, int(bot.shape[0]*800/bot.shape[1])))
            res_t, _ = ocr(top_s)
            res_b, _ = ocr(bot_s)
            t_str = " ".join(r[1] for r in res_t).upper() if res_t else ""
            b_str = " ".join(r[1] for r in res_b).upper() if res_b else ""
            return t_str, b_str

        txt_l_t, txt_l_b = get_text(left)
        txt_r_t, txt_r_b = get_text(right)

        def label_page(txt_t, txt_b, std_val):
            if std_val < 12.0:
                return "BLANK_HAL_8"
            # Cek top
            if any(k in txt_t for k in ["VSERUTI", "SURVEI EKONOMI", "REPUBLIK", "BLOK I", "BLOK II"]):
                return "HAL_1(COVER)"
            if any(k in txt_b for k in ["VSERUTI", "SURVEI EKONOMI", "REPUBLIK", "BLOK I", "BLOK II"]):
                return "HAL_1(UPSIDE_DOWN_180)"
            if any(k in txt_t for k in ["BLOK III", "KETERANGAN RUMAH TANGGA", "ART"]):
                return "HAL_2(ART)"
            if any(k in txt_t for k in ["IV.2.A", "PENGELUARAN", "PEMBERSIH"]):
                return "HAL_3"
            if any(k in txt_t for k in ["IV.2.B", "IV.3", "KESEHATAN"]):
                return "HAL_4"
            if any(k in txt_t for k in ["BLOK V", "PENDAPATAN", "UPAH"]):
                return "HAL_5"
            if any(k in txt_b for k in ["BLOK V", "PENDAPATAN", "UPAH"]):
                return "HAL_5(UPSIDE_DOWN_270)"
            if any(k in txt_t for k in ["BLOK V.2", "ASET", "USAHA"]):
                return "HAL_6"
            if any(k in txt_t for k in ["BLOK VI", "POSISI ASET"]):
                return "HAL_7"
            return f"UNKNOWN({txt_t[:20] if txt_t else txt_b[:20]})"

        label_l = label_page(txt_l_t, txt_l_b, std_l)
        label_r = label_page(txt_r_t, txt_r_b, std_r)
        print(f"  Spread {s_idx} ({s_name}): Left={label_l} (std={std_l:.1f}) | Right={label_r} (std={std_r:.1f})", flush=True)
