import sqlite3
import json

conn = sqlite3.connect('db/si_periksa.db')
c = conn.cursor()

c.execute("""
    SELECT pred_digit, pred_conf, priority, variable, label, crop_path
    FROM hitl_queue
    WHERE corrected = 0
    ORDER BY priority DESC, pred_conf ASC
""")
rows = c.fetchall()
print(f"Total pending items: {len(rows)}")

# Group by reason / characteristics
digit_counts = {}
conf_buckets = {"< 50%": 0, "50-70%": 0, "70-80%": 0, "80-85%": 0, "85-90%": 0, ">= 90%": 0}
zeros_leading = 0
zeros_normal = 0

for r in rows:
    digit, conf, prio, var, lbl, cp = r
    digit_counts[digit] = digit_counts.get(digit, 0) + 1
    if conf < 50: conf_buckets["< 50%"] += 1
    elif conf < 70: conf_buckets["50-70%"] += 1
    elif conf < 80: conf_buckets["70-80%"] += 1
    elif conf < 85: conf_buckets["80-85%"] += 1
    elif conf < 90: conf_buckets["85-90%"] += 1
    else: conf_buckets[">= 90%"] += 1

print("\nDigit breakdown:")
for d, cnt in sorted(digit_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  '{d}': {cnt} ({cnt/len(rows)*100:.1f}%)")

print("\nConfidence breakdown:")
for b, cnt in conf_buckets.items():
    print(f"  {b}: {cnt} ({cnt/len(rows)*100:.1f}%)")

conn.close()
