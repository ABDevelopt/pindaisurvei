import sqlite3
import json
from datetime import datetime

def run_smart_auto_accept():
    conn = sqlite3.connect('db/si_periksa.db')
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Total awal
    c.execute("SELECT count(*), sum(corrected) FROM hitl_queue")
    tot_before, corr_before = c.fetchone()
    pending_before = tot_before - (corr_before or 0)
    print(f"[*] Kondisi Awal: Total={tot_before}, Sudah Koreksi={corr_before}, Pending={pending_before}")

    # -------------------------------------------------------------------------
    # Kategori 1: Fixed Pre-Printed Template Metadata (Nomor Halaman)
    # Halaman 1..7 adalah angka statis cetakan template footer.
    # -------------------------------------------------------------------------
    c.execute("""
        UPDATE hitl_queue
        SET corrected = 1,
            final_digit = CASE 
                WHEN variable LIKE 'p1_F_hal%' OR variable = 'p1_F_hal' THEN '1'
                WHEN variable LIKE 'p2_F_hal%' OR variable = 'p2_F_hal' THEN '2'
                WHEN variable LIKE 'p3_F_hal%' OR variable = 'p3_F_hal' THEN '3'
                WHEN variable LIKE 'p4_F_hal%' OR variable = 'p4_F_hal' THEN '4'
                WHEN variable LIKE 'p5_F_hal%' OR variable = 'p5_F_hal' THEN '5'
                WHEN variable LIKE 'p6_F_hal%' OR variable = 'p6_F_hal' THEN '6'
                WHEN variable LIKE 'p7_F_hal%' OR variable = 'p7_F_hal' THEN '7'
                ELSE final_digit
            END,
            corrected_at = ?
        WHERE corrected = 0 AND variable LIKE '%_hal%'
    """, (now,))
    count_hal = c.rowcount
    print(f"  [1] Auto-Accept Nomor Halaman: {count_hal} items")

    # -------------------------------------------------------------------------
    # Kategori 2: Angka Berkonfidensi Tinggi (pred_conf >= 80.0%)
    # Angka dengan konfidensi >= 80% tidak memerlukan tinjauan manual operator.
    # -------------------------------------------------------------------------
    c.execute("""
        UPDATE hitl_queue
        SET corrected = 1,
            final_digit = pred_digit,
            corrected_at = ?
        WHERE corrected = 0 AND pred_conf >= 80.0
    """, (now,))
    count_high_conf = c.rowcount
    print(f"  [2] Auto-Accept Konfidensi Tinggi (>= 80%): {count_high_conf} items")

    # -------------------------------------------------------------------------
    # Kategori 3: Angka Nol Standar Valid (pred_digit = '0' dengan conf >= 50.0%)
    # Mengeliminasi ratusan leading/trailing zeros pada kotak angka kuesioner.
    # -------------------------------------------------------------------------
    c.execute("""
        UPDATE hitl_queue
        SET corrected = 1,
            final_digit = '0',
            corrected_at = ?
        WHERE corrected = 0 AND pred_digit = '0' AND pred_conf >= 50.0
    """, (now,))
    count_zeros = c.rowcount
    print(f"  [3] Auto-Accept Nol Standar Valid (conf >= 50%): {count_zeros} items")

    # -------------------------------------------------------------------------
    # Kategori 4: Wilayah Administratif Statis Master BPS PPU (Prov 64, Kab 09)
    # -------------------------------------------------------------------------
    c.execute("""
        UPDATE hitl_queue
        SET corrected = 1,
            final_digit = CASE
                WHEN variable = 'p1_F001' THEN '64'
                WHEN variable = 'p1_F002' THEN '09'
                ELSE pred_digit
            END,
            corrected_at = ?
        WHERE corrected = 0 AND variable IN ('p1_F001', 'p1_F002')
    """, (now,))
    count_adm = c.rowcount
    print(f"  [4] Auto-Accept Wilayah Master BPS (64 & 09): {count_adm} items")

    # Catat ke correction_log sebagai batch auto-accept
    c.execute("""
        INSERT INTO correction_log (hitl_id, booklet_id, variable, old_digit, new_digit, operator, corrected_at)
        SELECT id, booklet_id, variable, pred_digit, final_digit, 'smart_auto_accept_engine', corrected_at
        FROM hitl_queue
        WHERE corrected_at = ?
    """, (now,))

    # Update hitl_done di tabel booklets
    c.execute("""
        UPDATE booklets
        SET hitl_done = (SELECT COUNT(*) FROM hitl_queue WHERE booklet_id = booklets.id AND corrected = 1),
            updated_at = ?
    """, (now,))

    conn.commit()

    # Hitung kondisi akhir
    c.execute("SELECT count(*), sum(corrected) FROM hitl_queue")
    tot_after, corr_after = c.fetchone()
    pending_after = tot_after - (corr_after or 0)
    print(f"\n[+] HASIL AKHIR:")
    print(f"    Total Item              : {tot_after}")
    print(f"    Sudah Terverifikasi     : {corr_after} ({corr_after/tot_after*100:.1f}%)")
    print(f"    Pending Manual Review   : {pending_after} (Terkurang dari {pending_before})")
    print(f"    Efisiensi Penghematan   : {(pending_before - pending_after)/pending_before*100:.1f}% beban verifikasi operator terpangkas!")

    conn.close()

if __name__ == '__main__':
    run_smart_auto_accept()
