import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

ARTIFACTS_DIR = Path(r"C:\Users\ajian\.gemini\antigravity\brain\c8d745af-d5de-4ed4-9693-f2658abde927")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 1050})
        page = await context.new_page()

        print("Navigating to http://127.0.0.1:8001/ ...")
        await page.goto("http://127.0.0.1:8001/", wait_until="networkidle")
        await page.wait_for_timeout(1000)

        # Full page screenshot of Stage 01 (Integrated Input & Queue)
        stage1_path = ARTIFACTS_DIR / "pipeline_integrated_stage1_full.png"
        await page.screenshot(path=str(stage1_path), full_page=True)
        print(f"Captured Stage 01 Full Page screenshot: {stage1_path}")

        # Test navigating to /input
        print("Navigating to /input ...")
        await page.goto("http://127.0.0.1:8001/input", wait_until="networkidle")
        await page.wait_for_timeout(800)
        input_routed_path = ARTIFACTS_DIR / "pipeline_input_routed.png"
        await page.screenshot(path=str(input_routed_path))
        print(f"Captured /input routed screenshot: {input_routed_path}")

        await browser.close()
        print("All verification screenshots captured successfully.")

if __name__ == "__main__":
    asyncio.run(main())
