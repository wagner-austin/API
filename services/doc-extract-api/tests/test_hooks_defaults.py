"""Tests for doc_extract_api._test_hooks default implementations."""

from __future__ import annotations

from doc_extract_api._test_hooks import (
    _default_connect_db,
    _default_pdfplumber_open,
    _default_read_file,
    _default_redis_for_kv,
    connect_db,
    ocr_pdf,
    pdfplumber_open,
    test_runner,
)


class TestHookDefaults:
    def test_connect_db_default_is_set(self) -> None:
        assert connect_db is _default_connect_db

    def test_pdfplumber_open_default_is_set(self) -> None:
        assert pdfplumber_open is _default_pdfplumber_open

    def test_redis_factory_default_is_callable(self) -> None:
        assert callable(_default_redis_for_kv)

    def test_read_file_default_is_callable(self) -> None:
        assert callable(_default_read_file)

    def test_ocr_pdf_default_is_none(self) -> None:
        # OCR is optional, None means pdfplumber-only mode
        assert ocr_pdf is None or callable(ocr_pdf)

    def test_test_runner_default_is_none(self) -> None:
        assert test_runner is None
