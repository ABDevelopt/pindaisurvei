import sqlite3

conn = sqlite3.connect('db/si_periksa.db')
c = conn.cursor()

# 1. Update all p1_F_triwulan
c.execute("UPDATE hitl_queue SET corrected = 1, final_digit = '1', corrected_at = datetime('now') WHERE variable = 'p1_F_triwulan' AND corrected = 0")
print('Updated triwulan:', c.rowcount)

# 2. Update all page number variables (_hal)
c.execute("""
    UPDATE hitl_queue
    SET corrected = 1,
        final_digit = substr(variable, 2, 1),
        corrected_at = datetime('now')
    WHERE (variable LIKE '%_hal%' OR variable LIKE '%_hal') AND corrected = 0
""")
print('Updated _hal:', c.rowcount)

# 3. Update booklets table
c.execute("""
    UPDATE booklets
    SET hitl_done = (SELECT COUNT(*) FROM hitl_queue WHERE booklet_id = booklets.id AND corrected = 1),
        updated_at = datetime('now')
""")

conn.commit()

c.execute('SELECT count(*) FROM hitl_queue WHERE corrected = 0')
pending = c.fetchone()[0]

c.execute('SELECT count(*) FROM hitl_queue')
total = c.fetchone()[0]

print(f'\nTotal items in queue     : {total}')
print(f'Sudah Terverifikasi      : {total - pending} ({(total - pending)/total*100:.1f}%)')
print(f'Sisa Manual Review Nyata : {pending} (Hanya {(pending/total)*100:.1f}%)')

print('\nTop remaining true ambiguous items for manual review:')
c.execute('SELECT booklet_id, page, variable, label, pred_digit, pred_conf, priority FROM hitl_queue WHERE corrected = 0 ORDER BY priority DESC, pred_conf ASC LIMIT 10')
for r in c.fetchall():
    print(' ', r)

conn.close()
