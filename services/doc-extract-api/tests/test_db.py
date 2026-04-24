"""Tests for doc_extract_api.db."""

from __future__ import annotations

from doc_extract_api.db import document_exists, insert_document, insert_pages_batch
from doc_extract_api.types import decode_extracted_page

from ._test_hooks import FakeDbConnection


class TestDocumentExists:
    def test_exists_true(self) -> None:
        conn = FakeDbConnection()
        conn.add_response([(1,)])
        result = document_exists(conn, "My Doc", "budget")
        assert result is True
        assert len(conn.executed) == 1
        assert "SELECT COUNT" in conn.executed[0][0]

    def test_exists_false(self) -> None:
        conn = FakeDbConnection()
        conn.add_response([(0,)])
        result = document_exists(conn, "Missing", "general")
        assert result is False

    def test_exists_none_row(self) -> None:
        conn = FakeDbConnection()
        result = document_exists(conn, "Missing", "general")
        assert result is False


class TestInsertDocument:
    def test_inserts_and_returns_id(self) -> None:
        conn = FakeDbConnection()
        conn.add_response([("doc-uuid-123",)])
        doc_id = insert_document(conn, "Test Doc", "https://example.com", 5, "budget")
        assert doc_id == "doc-uuid-123"
        assert conn.committed >= 1
        assert "INSERT INTO documents" in conn.executed[0][0]


class TestInsertPagesBatch:
    def test_inserts_pages(self) -> None:
        conn = FakeDbConnection()
        pages = [
            decode_extracted_page(1, "page one", "pdfplumber-text"),
            decode_extracted_page(2, "page two", "doctr-ocr"),
        ]
        count = insert_pages_batch(conn, "doc-123", pages)
        assert count == 2
        assert conn.committed >= 1
        assert len(conn.executed) == 2
        assert "INSERT INTO document_pages" in conn.executed[0][0]

    def test_empty_pages(self) -> None:
        conn = FakeDbConnection()
        count = insert_pages_batch(conn, "doc-123", [])
        assert count == 0
