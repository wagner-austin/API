"""Tests for doc_extract_api.extract."""

from __future__ import annotations

from doc_extract_api import _test_hooks
from doc_extract_api.extract import extract_page_text, extract_pdf_pages

from ._test_hooks import FakePdfPlumberPage, make_fake_ocr, make_fake_pdfplumber_open


class TestExtractPageText:
    def test_text_only(self) -> None:
        page = FakePdfPlumberPage(1, text="Hello world")
        result = extract_page_text(page)
        assert result["page_number"] == 1
        assert result["content"] == "Hello world"
        assert result["method"] == "pdfplumber-text"

    def test_table_wins_when_longer(self) -> None:
        page = FakePdfPlumberPage(
            1,
            text="short",
            tables=[[["a very long table cell", "another cell"]]],
        )
        result = extract_page_text(page)
        assert result["method"] == "pdfplumber-table"
        assert "a very long table cell" in result["content"]

    def test_text_wins_when_longer(self) -> None:
        page = FakePdfPlumberPage(
            1,
            text="this is a longer text than the table",
            tables=[[["a", "b"]]],
        )
        result = extract_page_text(page)
        assert result["method"] == "pdfplumber-text"

    def test_none_text(self) -> None:
        page = FakePdfPlumberPage(1, text=None)
        result = extract_page_text(page)
        assert result["content"] == ""
        assert result["method"] == "pdfplumber-text"

    def test_empty_tables(self) -> None:
        page = FakePdfPlumberPage(1, text="some text", tables=[])
        result = extract_page_text(page)
        assert result["method"] == "pdfplumber-text"


class TestExtractPdfPages:
    def test_pdfplumber_only_no_ocr(self) -> None:
        pages = [
            FakePdfPlumberPage(1, text="page one"),
            FakePdfPlumberPage(2, text="page two"),
        ]
        _test_hooks.pdfplumber_open = make_fake_pdfplumber_open(pages)
        _test_hooks.ocr_pdf = None

        result = extract_pdf_pages(b"fake pdf bytes")
        assert len(result) == 2
        assert result[0]["content"] == "page one"
        assert result[1]["content"] == "page two"
        assert result[0]["method"] == "pdfplumber-text"

    def test_ocr_wins_when_longer(self) -> None:
        pages = [
            FakePdfPlumberPage(1, text="short"),
            FakePdfPlumberPage(2, text="also short"),
        ]
        _test_hooks.pdfplumber_open = make_fake_pdfplumber_open(pages)
        _test_hooks.ocr_pdf = make_fake_ocr(
            {
                0: "much longer OCR result that beats pdfplumber",
                1: "tiny",
            }
        )

        result = extract_pdf_pages(b"fake pdf bytes")
        assert result[0]["method"] == "doctr-ocr"
        assert result[0]["content"] == "much longer OCR result that beats pdfplumber"
        assert result[1]["method"] == "pdfplumber-text"

    def test_pdfplumber_wins_when_longer(self) -> None:
        pages = [
            FakePdfPlumberPage(1, text="pdfplumber has much more content here"),
        ]
        _test_hooks.pdfplumber_open = make_fake_pdfplumber_open(pages)
        _test_hooks.ocr_pdf = make_fake_ocr({0: "short"})

        result = extract_pdf_pages(b"fake pdf bytes")
        assert result[0]["method"] == "pdfplumber-text"

    def test_ocr_missing_page(self) -> None:
        pages = [
            FakePdfPlumberPage(1, text="has text"),
        ]
        _test_hooks.pdfplumber_open = make_fake_pdfplumber_open(pages)
        _test_hooks.ocr_pdf = make_fake_ocr({})

        result = extract_pdf_pages(b"fake pdf bytes")
        assert result[0]["method"] == "pdfplumber-text"
        assert result[0]["content"] == "has text"

    def test_empty_pdf(self) -> None:
        _test_hooks.pdfplumber_open = make_fake_pdfplumber_open([])
        _test_hooks.ocr_pdf = None

        result = extract_pdf_pages(b"fake pdf bytes")
        assert len(result) == 0
