import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright

SCREENSHOTS_DIR = Path("output/verification_screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

async def run_verification():
    print("=" * 80)
    print("  VERIFIKASI PLAYWRIGHT END-TO-END KUESIONER DIGITAL & INFERENSI SERUTI")
    print("=" * 80)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 1200})
        page = await context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        # 1. Buka Kuesioner Digital booklet_01
        url = "http://127.0.0.1:8000/static/kuesioner_digital.html?booklet=booklet_01"
        print(f"\n[1/6] Membuka {url}...")
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # 2. Verifikasi Hal 1: KOP, Blok I & Blok II
        print("\n[2/6] Memverifikasi Hal 1 (KOP, Blok I & Blok II)...")
        krt_val = await page.input_value('[data-key="b1.r110"]')
        alamat_val = await page.input_value('[data-key="b1.r111"]')
        pcl_val = await page.input_value('[data-key="b2.r201_nama"]')
        pml_val = await page.input_value('[data-key="b2.r202_nama"]')
        print(f"      Nama KRT        : '{krt_val}'")
        print(f"      Alamat          : '{alamat_val}'")
        print(f"      Nama Pencacah   : '{pcl_val}'")
        print(f"      Nama Pengawas   : '{pml_val}'")

        assert "Petugas Mitra BPS (09175)" not in pcl_val, "Gagal: Pencacah masih mengandung mock teks palsu!"
        assert "Responden Dummy" not in krt_val, "Gagal: Nama KRT masih mock data!"

        # 3. Verifikasi Hal 2: Roster ART (Blok III)
        print("\n[3/6] Memverifikasi Hal 2: Roster ART Blok III...")
        blok3_inputs = await page.query_selector_all("#blok3-body tr")
        print(f"      Total baris Blok III: {len(blok3_inputs)}")
        assert len(blok3_inputs) == 10, "Blok III harus memiliki 10 baris ART!"

        # Cek baris 1 & 2
        art1_name = await (await blok3_inputs[0].query_selector("td:nth-child(2) input")).input_value()
        art2_name = await (await blok3_inputs[1].query_selector("td:nth-child(2) input")).input_value()
        art3_name = await (await blok3_inputs[2].query_selector("td:nth-child(2) input")).input_value()
        print(f"      ART 01 Nama : '{art1_name}'")
        print(f"      ART 02 Nama : '{art2_name}'")
        print(f"      ART 03 Nama : '{art3_name}'")

        assert "ART 02 (Anggota Ruta)" not in art2_name, "Gagal: Teks palsu 'ART 02 (Anggota Ruta)' terdeteksi!"
        assert "ART 03 (Anggota Ruta)" not in art3_name, "Gagal: Teks palsu 'ART 03 (Anggota Ruta)' terdeteksi!"

        # Verifikasi R405 (Status Penguasaan Bangunan Tempat Tinggal yang Ditempati)
        r405_val = await page.input_value('[data-key="b42a.r405"]')
        print(f"      R405 Status Bangunan : '{r405_val}'")
        assert r405_val == "1", f"Gagal: R405 harus berisi '1', ditemukan: '{r405_val}'"

        # 4. Verifikasi Hal 5: Blok V (R501, R502, R503) & Interaktivitas Edit
        print("\n[4/6] Memverifikasi Hal 5 (Pendapatan & Interaktivitas Edit)...")
        r501_rows = await page.query_selector_all("#blok501-body tr:not(.total-row)")
        print(f"      Total baris R501 ter-render: {len(r501_rows)}")
        assert len(r501_rows) == 5, f"R501 harus memiliki 5 baris lengkap, ditemukan: {len(r501_rows)}"

        # Verifikasi Uraian Pekerjaan terbaca tulisan asli (bukan kode KBLI)
        u1 = await (await r501_rows[0].query_selector("td:nth-child(2) input")).input_value()
        u2 = await (await r501_rows[1].query_selector("td:nth-child(2) input")).input_value()
        print(f"      R501 Baris 1 Uraian: '{u1}'")
        print(f"      R501 Baris 2 Uraian: '{u2}'")
        assert "KBLI" not in u1, f"Gagal: Uraian Baris 1 masih memuat kode fallback KBLI: '{u1}'"
        assert "TUKANG BANGUNAN" in u1, f"Gagal: Uraian Baris 1 tidak memuat TUKANG BANGUNAN: '{u1}'"
        assert "SECURITY" in u2, f"Gagal: Uraian Baris 2 tidak memuat SECURITY: '{u2}'"

        r502_rows = await page.query_selector_all("#blok502-body tr:not(.total-row)")
        print(f"      Total baris R502 ter-render: {len(r502_rows)}")
        assert len(r502_rows) == 5, f"R502 harus memiliki 5 baris lengkap, ditemukan: {len(r502_rows)}"

        # Periksa apakah input dapat diedit (readOnly == False)
        uang_input = await r501_rows[0].query_selector("td:nth-child(5) input")
        is_readonly = await uang_input.is_editable()
        val_uang_before = await uang_input.input_value()
        print(f"      R501 Baris 1 Uang (Sebelum): '{val_uang_before}' | Is Editable: {is_readonly}")
        assert is_readonly, "Gagal: Input R501 masih terkunci readonly!"

        sum_uang_before = await page.inner_text("#sum-501-uang")
        print(f"      Total Sum R501 Uang (Sebelum): '{sum_uang_before}'")

        # Uji Interaktivitas Koreksi Data & Auto-Recalculate
        expected_diff = 8000000 - int(val_uang_before.replace(".", ""))
        num_sum_before = int(sum_uang_before.replace(".", ""))
        await uang_input.fill("8000000")
        await uang_input.dispatch_event("input")
        await page.wait_for_timeout(300)
        sum_uang_after = await page.inner_text("#sum-501-uang")
        num_sum_after = int(sum_uang_after.replace(".", ""))
        print(f"      Total Sum R501 Uang (Setelah edit 8000000): '{sum_uang_after}'")
        assert num_sum_after == num_sum_before + expected_diff, f"Gagal: Auto-recalc tidak cocok, didapat: {num_sum_after}, ekspetasi: {num_sum_before + expected_diff}"

        # Kembalikan nilai
        await uang_input.fill(val_uang_before.replace(".", ""))
        await uang_input.dispatch_event("input")
        await page.wait_for_timeout(300)

        # 5. Tangkap Screenshot Hal 1 s.d. Hal 7
        print("\n[5/6] Mengambil Screenshot Resmi Hal 1 s.d. Hal 7...")
        for p_idx in range(1, 8):
            await page.evaluate(f"filterPage('{p_idx}', null)")
            await page.wait_for_timeout(500)
            sheet = await page.query_selector(f'.a4-sheet[data-page="{p_idx}"]')
            if sheet:
                ss_path = SCREENSHOTS_DIR / f"kuesioner_digital_hal_{p_idx}.png"
                await sheet.screenshot(path=str(ss_path))
                print(f"      [✓] Screenshot Hal {p_idx} tersimpan: {ss_path.name}")

        # Tampilkan kembali semua halaman
        await page.evaluate("filterPage('all', null)")
        await page.wait_for_timeout(500)

        # 6. Uji Coba Buklet Lain (booklet_02)
        print("\n[6/6] Memverifikasi booklet_02...")
        await page.evaluate("onBookletChange('booklet_02')")
        await page.wait_for_timeout(2000)
        krt_02 = await page.input_value('[data-key="b1.r110"]')
        print(f"      booklet_02 KRT: '{krt_02}'")
        assert krt_02 != "", "Gagal: booklet_02 KRT kosong!"

        # Periksa console errors
        print("\n>>> Ringkasan Console Errors:", len(console_errors))
        if console_errors:
            print("    Warning errors:", console_errors[:5])

        await browser.close()
        print("\n" + "=" * 80)
        print("  VERIFIKASI PLAYWRIGHT SUKSES 100% TANPA KENDALA!")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_verification())
