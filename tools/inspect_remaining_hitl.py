import sqlite3
import json

conn = sqlite3.connect('db/si_periksa.db')
c = conn.cursor()

c.execute("""
    SELECT id, booklet_id, page, variable, label, pred_digit, pred_conf, top3_json, priority
    FROM hitl_queue
    WHERE corrected = 0
""")
pending = c.fetchall()

remaining = []
for item in pending:
    item_id, bid, page, var, lbl, digit, conf, top3_str, prio = item
    # Rule 1: High conf >= 80%
    if conf >= 80.0:
        continue
    # Rule 2: Valid zero with conf >= 50%
    if digit == "0" and conf >= 50.0:
        continue
    remaining.append(item)

print(f"Remaining items: {len(remaining)}")

# Breakdown of remaining items
rem_digits = {}
rem_conf_buckets = {"< 40%": 0, "40-50%": 0, "50-60%": 0, "60-70%": 0, "70-80%": 0}
rem_variables = {}

for item in remaining:
    item_id, bid, page, var, lbl, digit, conf, top3_str, prio = item
    rem_digits[digit] = rem_digits.get(digit, 0) + 1
    rem_variables[var] = rem_variables.get(var, 0) + 1
    if conf < 40: rem_conf_buckets["< 40%"] += 1
    elif conf < 50: rem_conf_buckets["40-50%"] += 1
    elif conf < 60: rem_conf_buckets["50-60%"] += 1
    elif conf < 70: rem_conf_buckets["60-70%"] += 1
    else: rem_conf_buckets["70-80%"] += 1

print("\nRemaining digits breakdown:")
for d, cnt in sorted(rem_digits.items(), key=lambda x: x[1], reverse=True):
    print(f"  '{d}': {cnt}")

print("\nRemaining conf breakdown:")
for b, cnt in rem_conf_buckets.items():
    print(f"  {b}: {cnt}")

print("\nTop remaining variables:")
for v, cnt in sorted(rem_variables.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {v}: {cnt}")

# Let's see some samples of pred_digit='0' with conf < 50%
c.execute("""
    SELECT id, booklet_id, variable, pred_digit, pred_conf, top3_json
    FROM hitl_queue
    WHERE corrected = 0 AND pred_digit = '0' AND pred_conf < 50.0
    LIMIT 10
""")
print("\nSample 0 with conf < 50%:")
for r in c.fetchall():
    print(' ', r[:5])

conn.close()
