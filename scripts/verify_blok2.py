import asyncio
import os
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1280, 'height': 1200})
        
        # 1. Direct kuesioner_digital page for booklet_02
        url = 'http://127.0.0.1:8001/static/kuesioner_digital.html?booklet=booklet_02'
        print('Opening', url)
        await page.goto(url, wait_until='networkidle')
        await page.wait_for_timeout(2000)
        
        # Scroll to Blok II
        blok2 = page.locator('text=BLOK II. KETERANGAN PENDATAAN')
        await blok2.scroll_into_view_if_needed()
        await page.wait_for_timeout(1000)
        
        out_dir = r"C:\Users\ajian\.gemini\antigravity\brain\c8d745af-d5de-4ed4-9693-f2658abde927"
        
        # Capture Blok II element
        table_el = page.locator('article[data-page="1"] table.bps-table').nth(1) # Blok II table is 2nd table in page 1
        await table_el.screenshot(path=os.path.join(out_dir, 'kuesioner_blok2_table_fixed.png'))
        print('Saved kuesioner_blok2_table_fixed.png')
        
        # Capture bottom half of Page 1
        art1 = page.locator('article[data-page="1"]')
        await art1.screenshot(path=os.path.join(out_dir, 'kuesioner_page1_fixed.png'))
        print('Saved kuesioner_page1_fixed.png')
        
        # 2. Main app view
        app_url = 'http://127.0.0.1:8001/booklets/booklet_02'
        print('Opening', app_url)
        await page.goto(app_url, wait_until='networkidle')
        await page.wait_for_timeout(2500)
        await page.screenshot(path=os.path.join(out_dir, 'main_app_kuesioner_b02.png'))
        print('Saved main_app_kuesioner_b02.png')
        
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
