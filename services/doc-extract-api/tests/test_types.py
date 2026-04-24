"""Tests for doc_extract_api.types."""

from __future__ import annotations

import pytest

from doc_extract_api.types import (
    VALID_CATEGORIES,
    _require_int,
    _require_job_status,
    _require_str,
    decode_extracted_page,
    decode_extraction_job,
    decode_extraction_request,
    encode_extracted_page,
    encode_extraction_job,
    encode_extraction_job_response,
    format_table_rows,
    validate_category,
)


class TestExtractedPage:
    def test_decode_extracted_page(self) -> None:
        page = decode_extracted_page(1, "hello", "pdfplumber-text")
        assert page["page_number"] == 1
        assert page["content"] == "hello"
        assert page["method"] == "pdfplumber-text"

    def test_decode_extracted_page_table_method(self) -> None:
        page = decode_extracted_page(2, "col1\tcol2", "pdfplumber-table")
        assert page["method"] == "pdfplumber-table"

    def test_decode_extracted_page_ocr_method(self) -> None:
        page = decode_extracted_page(3, "ocr text", "doctr-ocr")
        assert page["method"] == "doctr-ocr"

    def test_encode_extracted_page(self) -> None:
        page = decode_extracted_page(1, "hello world", "pdfplumber-text")
        result = encode_extracted_page(page)
        assert "Page 1" in result
        assert "pdfplumber-text" in result
        assert "11 chars" in result


class TestExtractionJob:
    def test_decode_extraction_job(self) -> None:
        raw = {
            "status": "queued",
            "title": "Test Doc",
            "source": "https://example.com",
            "category": "general",
            "file_path": "/tmp/test.pdf",
            "pages_total": "10",
            "pages_done": "0",
            "document_id": "",
            "error": "",
        }
        job = decode_extraction_job(raw, "job-123")
        assert job["job_id"] == "job-123"
        assert job["status"] == "queued"
        assert job["title"] == "Test Doc"
        assert job["pages_total"] == 10
        assert job["pages_done"] == 0

    def test_decode_extraction_job_missing_title(self) -> None:
        raw = {
            "status": "queued",
            "category": "general",
            "file_path": "/tmp/test.pdf",
            "pages_total": "10",
            "pages_done": "0",
        }
        with pytest.raises(KeyError, match="title"):
            decode_extraction_job(raw, "job-123")

    def test_encode_extraction_job(self) -> None:
        from doc_extract_api.types import ExtractionJob

        job = ExtractionJob(
            job_id="j1",
            status="processing",
            title="Doc",
            source="src",
            category="budget",
            file_path="/f.pdf",
            pages_total=5,
            pages_done=2,
            document_id="",
            error="",
        )
        encoded = encode_extraction_job(job)
        assert encoded["status"] == "processing"
        assert encoded["pages_total"] == "5"
        assert encoded["pages_done"] == "2"

    def test_encode_extraction_job_response(self) -> None:
        from doc_extract_api.types import ExtractionJob

        job = ExtractionJob(
            job_id="j1",
            status="completed",
            title="Doc",
            source="",
            category="audit",
            file_path="/f.pdf",
            pages_total=10,
            pages_done=10,
            document_id="doc-uuid",
            error="",
        )
        resp = encode_extraction_job_response(job)
        assert resp["job_id"] == "j1"
        assert resp["pages_total"] == 10
        assert resp["document_id"] == "doc-uuid"


class TestExtractionRequest:
    def test_decode_extraction_request(self) -> None:
        req = decode_extraction_request("Title", "/path.pdf", "general", "src")
        assert req["title"] == "Title"
        assert req["file_path"] == "/path.pdf"

    def test_decode_extraction_request_empty_title(self) -> None:
        with pytest.raises(ValueError, match="title"):
            decode_extraction_request("", "/path.pdf", "general", "")

    def test_decode_extraction_request_whitespace_title(self) -> None:
        with pytest.raises(ValueError, match="title"):
            decode_extraction_request("   ", "/path.pdf", "general", "")

    def test_decode_extraction_request_empty_file_path(self) -> None:
        with pytest.raises(ValueError, match="file_path"):
            decode_extraction_request("Title", "", "general", "")


class TestFormatTableRows:
    def test_empty_tables(self) -> None:
        assert format_table_rows([]) == ""

    def test_single_table(self) -> None:
        tables: list[list[list[str | None]]] = [[["a", "b"], ["c", None]]]
        result = format_table_rows(tables)
        assert result == "a\tb\nc\t"

    def test_newlines_in_cells(self) -> None:
        tables: list[list[list[str | None]]] = [[["line1\nline2", "ok"]]]
        result = format_table_rows(tables)
        assert result == "line1 line2\tok"

    def test_multiple_tables(self) -> None:
        tables: list[list[list[str | None]]] = [[["a"]], [["b"]]]
        result = format_table_rows(tables)
        assert result == "a\nb"


class TestValidation:
    def test_require_job_status_valid(self) -> None:
        assert _require_job_status("queued") == "queued"
        assert _require_job_status("processing") == "processing"
        assert _require_job_status("completed") == "completed"
        assert _require_job_status("failed") == "failed"

    def test_require_job_status_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid job status"):
            _require_job_status("unknown")

    def test_require_str_present(self) -> None:
        assert _require_str({"key": "val"}, "key") == "val"

    def test_require_str_missing(self) -> None:
        with pytest.raises(KeyError, match="key"):
            _require_str({}, "key")

    def test_require_int_valid(self) -> None:
        assert _require_int({"n": "42"}, "n") == 42

    def test_require_int_missing(self) -> None:
        with pytest.raises(KeyError, match="n"):
            _require_int({}, "n")

    def test_require_int_invalid(self) -> None:
        with pytest.raises(ValueError):
            _require_int({"n": "abc"}, "n")

    def test_validate_category_valid(self) -> None:
        for cat in VALID_CATEGORIES:
            assert validate_category(cat) == cat

    def test_validate_category_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid category"):
            validate_category("nonexistent")
