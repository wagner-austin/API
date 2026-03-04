"""Fetch McKinnon's summer_extremes source code and paper via Playwright stealth."""

from pathlib import Path

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

# Source code from GitHub
SOURCE_URLS = {
    "utils.py": "https://raw.githubusercontent.com/karenamckinnon/summer_extremes/master/summer_extremes/utils.py",
    "rank_trends.py": "https://raw.githubusercontent.com/karenamckinnon/summer_extremes/master/scripts/rank_trends_summer_extremes.py",
}

# PNAS paper PDF (open access, CC BY-NC-ND)
PAPER_PDF_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11494304/pdf/pnas.2406143121.pdf"
PAPER_PDF_NAME = "mckinnon_2024_pace_of_change.pdf"

OUTPUT_DIR = Path(__file__).parent / "mckinnon_sources"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Fetch source code (plain text)
        for name, url in SOURCE_URLS.items():
            print(f"Fetching {name} from {url}...")
            page.goto(url, wait_until="networkidle", timeout=30000)
            text = page.evaluate("() => document.body.innerText")
            out_path = OUTPUT_DIR / name
            out_path.write_text(text, encoding="utf-8")
            print(f"  Saved {len(text)} chars to {out_path}")

        # Fetch paper PDF (binary download via PMC)
        print(f"Fetching {PAPER_PDF_NAME} from PMC...")
        pdf_path = OUTPUT_DIR / PAPER_PDF_NAME
        with page.expect_download() as download_info:
            page.goto(PAPER_PDF_URL, timeout=60000)
        download = download_info.value
        download.save_as(str(pdf_path))
        pdf_size = pdf_path.stat().st_size
        print(f"  Saved {pdf_size:,} bytes to {pdf_path}")

        browser.close()

    print("Done.")


if __name__ == "__main__":
    main()
