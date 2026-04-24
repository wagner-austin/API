"""PDF text extraction pipeline.

Extracts page content from PDFs using two methods:
1. pdfplumber for pages with embedded text (fast, exact)
2. docTR OCR for every page (GPU-accelerated image recognition)

For each page, both methods run and the longer result wins.
This ensures tables rendered as images, charts with data labels,
and scanned pages all get full coverage without arbitrary thresholds.
"""

from __future__ import annotations

import io

from . import _test_hooks
from .types import ExtractedPage, decode_extracted_page, format_table_rows

_TABLE_SETTINGS: dict[str, int] = {"text_x_tolerance": 1}


def extract_page_text(page: _test_hooks.PdfPlumberPage) -> ExtractedPage:
    """Extract text from a single pdfplumber page.

    Tries both table extraction and plain text extraction,
    keeps whichever yields more content.

    Args:
        page: A pdfplumber page.

    Returns:
        An ExtractedPage with the best content and extraction method.
    """
    text = page.extract_text(x_tolerance=1)
    text_content = text if text is not None else ""

    tables = page.extract_tables(_TABLE_SETTINGS)
    table_content = format_table_rows(tables) if len(tables) > 0 else ""

    if len(table_content.strip()) > len(text_content.strip()):
        return decode_extracted_page(
            page_number=page.page_number,
            content=table_content,
            method="pdfplumber-table",
        )

    return decode_extracted_page(
        page_number=page.page_number,
        content=text_content,
        method="pdfplumber-text",
    )


def extract_pdf_pages(pdf_bytes: bytes) -> list[ExtractedPage]:
    """Extract all pages from a PDF.

    Runs pdfplumber on every page for embedded text, then docTR OCR
    on every page for image content (if the OCR hook is configured).
    For each page, keeps whichever method produced more content.

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        List of ExtractedPage objects, one per page, ordered by page number.
    """
    pdf = _test_hooks.pdfplumber_open(io.BytesIO(pdf_bytes))
    pages: list[ExtractedPage] = []

    for page in pdf.pages:
        extracted = extract_page_text(page)
        pages.append(extracted)

    pdf.close()

    page_count = len(pages)

    ocr_hook = _test_hooks.ocr_pdf
    if ocr_hook is None:
        return pages

    all_indices = list(range(page_count))
    ocr_results = ocr_hook(pdf_bytes, all_indices)

    for i in range(page_count):
        ocr_content = ocr_results.get(i, "")
        pdfplumber_content = pages[i]["content"]
        if len(ocr_content.strip()) > len(pdfplumber_content.strip()):
            pages[i] = decode_extracted_page(
                page_number=pages[i]["page_number"],
                content=ocr_content,
                method="doctr-ocr",
            )

    return pages


__all__ = [
    "extract_page_text",
    "extract_pdf_pages",
]
