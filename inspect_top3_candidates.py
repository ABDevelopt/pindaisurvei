import sqlite3
import json

conn = sqlite3.connect("db/si_periksa.db")
c = conn.cursor()
c.execute("SELECT id, pred_digit, top3_json FROM hitl_queue WHERE pred_digit = 'ragu_ragu'")
rows = c.fetchall()

candidates = {}
for r in rows:
    top3_raw = r[2]
    try:
        top3 = json.loads(top3_raw) if top3_raw else []
        non_ragu = [it.get("d") or it.get("digit") for it in top3 if (it.get("d") or it.get("digit")) not in ["ragu_ragu", "?", "∅", " ", ""]]
        best = non_ragu[0] if non_ragu else "None"
        candidates[best] = candidates.get(best, 0) + 1
    except Exception as e:
        candidates["err"] = candidates.get("err", 0) + 1

print(f"Total ragu_ragu rows: {len(rows)}")
print("Strongest candidate distribution for ragu_ragu rows:")
for k, v in sorted(candidates.items(), key=lambda x: -x[1]):
    print(f"  '{k}': {v}")
