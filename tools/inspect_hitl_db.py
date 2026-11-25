import sqlite3
import json

conn = sqlite3.connect('db/si_periksa.db')
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables:', [r[0] for r in c.fetchall()])

try:
    c.execute('SELECT count(*), sum(corrected) FROM hitl_queue')
    total, corr = c.fetchone()
    corr = corr or 0
    print(f'hitl_queue: total={total}, corrected={corr}, pending={total - corr}')

    c.execute('SELECT min(pred_conf), max(pred_conf), avg(pred_conf) FROM hitl_queue WHERE corrected=0')
    print('Pending conf (min, max, avg):', c.fetchone())

    # Group by pred_conf ranges
    print("\nDistribution of pending by pred_conf:")
    ranges = [(0, 50), (50, 70), (70, 80), (80, 85), (85, 90), (90, 95), (95, 100)]
    for low, high in ranges:
        c.execute('SELECT count(*) FROM hitl_queue WHERE corrected=0 AND pred_conf >= ? AND pred_conf < ?', (low, high))
        cnt = c.fetchone()[0]
        print(f"  {low}% - {high}%: {cnt} items")

    # Check pred_digit values
    print("\nPending by pred_digit:")
    c.execute('SELECT pred_digit, count(*) FROM hitl_queue WHERE corrected=0 GROUP BY pred_digit ORDER BY count(*) DESC LIMIT 15')
    for r in c.fetchall():
        print(f"  digit '{r[0]}': {r[1]} items")

    # Check booklet counts
    print("\nPending by booklet:")
    c.execute('SELECT booklet_id, count(*) FROM hitl_queue WHERE corrected=0 GROUP BY booklet_id ORDER BY count(*) DESC LIMIT 10')
    for r in c.fetchall():
        print(f"  {r[0]}: {r[1]} items")

except Exception as e:
    print('Error:', e)

conn.close()
