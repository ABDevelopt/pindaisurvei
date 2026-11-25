import sqlite3

conn = sqlite3.connect('db/si_periksa.db')
c = conn.cursor()

c.execute("SELECT count(*) FROM booklet_inferences")
inf_cnt = c.fetchone()[0]

c.execute("SELECT count(*) FROM booklet_field_values")
fld_cnt = c.fetchone()[0]

c.execute("SELECT count(*) FROM booklets WHERE ocr_status = 'COMPLETED'")
bk_cnt = c.fetchone()[0]

print("Database Verification:")
print("  booklet_inferences rows  :", inf_cnt)
print("  booklet_field_values rows:", fld_cnt)
print("  booklets COMPLETED       :", bk_cnt)

c.execute("SELECT booklet_id, model_name, total_digits, auto_accepted, needs_review, avg_confidence FROM booklet_inferences LIMIT 5")
print("\nSample inferences:")
for r in c.fetchall():
    print(" ", r)

c.execute("SELECT booklet_id, page, var_name, var_label, value, confidence FROM booklet_field_values LIMIT 5")
print("\nSample field values:")
for r in c.fetchall():
    print(" ", r)

conn.close()
