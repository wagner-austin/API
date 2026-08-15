"""Fetch the McKinnon 2024 paper PDF via Playwright stealth.

One-shot vendoring script for the reference material in mckinnon_sources/.
It ran once; the PDF it produced is committed. It is kept so the provenance
of that file is recorded as code rather than as prose.

Only the paper is fetched. The two source files this also used to download --
summer_extremes' utils.py and rank_trends_summer_extremes.py -- are gone: the
second URL 404s, so what landed on disk was the string "404: Not Found", and
the first was 47KB of research code that nothing here imported and that no
gate could check, since cartopy and xarray are not dependencies of this
service. The paper is the reference; the code was never the reference.
"""

from pathlib import Path

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

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
