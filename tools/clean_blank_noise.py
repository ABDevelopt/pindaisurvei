import sqlite3, cv2, os, numpy as np

conn = sqlite3.connect('db/si_periksa.db')
c = conn.cursor()
c.execute('SELECT id, crop_path FROM hitl_queue WHERE corrected = 0')
rows = c.fetchall()

to_blank = []
for item_id, cp in rows:
    if not cp or not os.path.exists(cp): continue
    img = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    h, w = img.shape
    pad_y, pad_x = max(1, int(h * 0.12)), max(1, int(w * 0.12))
    inner = img[pad_y:h-pad_y, pad_x:w-pad_x]
    ink_density = np.sum(inner < 170) / (inner.shape[0] * inner.shape[1])
    if ink_density < 0.035:
        to_blank.append(item_id)

print(f"Auto-accepting {len(to_blank)} pure blank noise boxes")
for bid in to_blank:
    c.execute("UPDATE hitl_queue SET corrected = 1, final_digit = ' ', corrected_at = datetime('now') WHERE id = ?", (bid,))

conn.commit()

c.execute('SELECT count(*) FROM hitl_queue WHERE corrected = 0')
print('Remaining true ambiguous pending:', c.fetchone()[0])
conn.close()
