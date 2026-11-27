import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

ARTIFACTS_DIR = Path(r"C:\Users\ajian\.gemini\antigravity\brain\c8d745af-d5de-4ed4-9693-f2658abde927")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        print("Navigating to http://127.0.0.1:8001/pipeline ...")
        await page.goto("http://127.0.0.1:8001/pipeline", wait_until="networkidle")
        await page.wait_for_timeout(1000)

        # Stage 01: Jobs
        stage1_path = ARTIFACTS_DIR / "pipeline_clean_stage1_jobs.png"
        await page.screenshot(path=str(stage1_path))
        print(f"Captured Stage 01 screenshot: {stage1_path}")

        # Switch to Stage 04: Inference
        print("Switching to Stage 04 (Inferensi)...")
        await page.evaluate("setPipelineSubMode('inference')")
        await page.wait_for_timeout(1500)
        stage4_path = ARTIFACTS_DIR / "pipeline_clean_stage4_inference.png"
        await page.screenshot(path=str(stage4_path))
        print(f"Captured Stage 04 screenshot: {stage4_path}")

        # Switch to Stage 02: Overlay
        print("Switching to Stage 02 (Overlay)...")
        await page.evaluate("setPipelineSubMode('overlay')")
        await page.wait_for_timeout(1000)
        stage2_path = ARTIFACTS_DIR / "pipeline_clean_stage2_overlay.png"
        await page.screenshot(path=str(stage2_path))
        print(f"Captured Stage 02 screenshot: {stage2_path}")

        # Switch to Stage 03: Crop
        print("Switching to Stage 03 (Crop)...")
        await page.evaluate("setPipelineSubMode('crop')")
        await page.wait_for_timeout(1000)
        stage3_path = ARTIFACTS_DIR / "pipeline_clean_stage3_crop.png"
        await page.screenshot(path=str(stage3_path))
        print(f"Captured Stage 03 screenshot: {stage3_path}")

        await browser.close()
        print("All screenshots captured successfully.")

if __name__ == "__main__":
    asyncio.run(main())
