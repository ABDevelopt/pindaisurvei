import sqlite3
import json
import cv2
import numpy as np

conn = sqlite3.connect('db/si_periksa.db')
c = conn.cursor()

c.execute("""
    SELECT id, booklet_id, page, variable, label, pred_digit, pred_conf, top3_json, crop_path
    FROM hitl_queue
    WHERE corrected = 0
""")
pending = c.fetchall()
print(f"Total current pending in hitl_queue: {len(pending)}")

can_auto_accept = []
reasons = {
    "high_conf_80_plus": 0,
    "top1_margin_large": 0,
    "fixed_region_kop": 0,
    "clean_zero_high_confidence": 0,
    "clear_blank_low_ink": 0
}

for item in pending:
    item_id, bid, page, var, lbl, digit, conf, top3_str, cp = item
    try:
        top3 = json.loads(top3_str or "[]")
    except:
        top3 = []
    
    # Check 1: Fixed KOP / Wilayah (p1_F001: Prov 64, p1_F002: Kab 09, etc.)
    if var in ["p1_F001", "p1_F002", "p1_F005"] or "_provinsi" in var or "_kabupaten" in var:
        can_auto_accept.append((item_id, "Fixed Administrative Master (Prov/Kab)", digit, conf))
        reasons["fixed_region_kop"] += 1
        continue

    # Check 2: Confidence >= 80%
    if conf >= 80.0:
        can_auto_accept.append((item_id, f"Confidence >= 80% ({conf}%)", digit, conf))
        reasons["high_conf_80_plus"] += 1
        continue

    # Check 3: Top-1 vs Top-2 Margin >= 30%
    if len(top3) >= 2:
        top1_c = float(top3[0].get("conf", 0) or 0)
        top2_c = float(top3[1].get("conf", 0) or 0)
        # If top3 conf is in 0..1 scale, convert to pct
        if top1_c <= 1.0: top1_c *= 100
        if top2_c <= 1.0: top2_c *= 100
        if (top1_c - top2_c) >= 30.0 and top1_c >= 55.0:
            can_auto_accept.append((item_id, f"Clear Top-1 Winner (margin {top1_c - top2_c:.1f}%)", digit, conf))
            reasons["top1_margin_large"] += 1
            continue

    # Check 4: Zero digit with clear ink (> 8%) and conf >= 50%
    if digit == "0" and conf >= 50.0:
        can_auto_accept.append((item_id, f"Valid Standard Zero ({conf}%)", digit, conf))
        reasons["clean_zero_high_confidence"] += 1
        continue

print(f"\nPotential Auto-Accepted items: {len(can_auto_accept)} of {len(pending)} ({len(can_auto_accept)/len(pending)*100:.1f}%)")
print(f"Remaining true ambiguities for Human Review: {len(pending) - len(can_auto_accept)} items")

print("\nBreakdown of Auto-Acceptance:")
for k, v in reasons.items():
    print(f"  {k}: {v}")

conn.close()
