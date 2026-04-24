"""Tests for doc_extract_api.asgi."""

from __future__ import annotations

from doc_extract_api.asgi import app


class TestAsgi:
    def test_app_exists(self) -> None:
        assert app.title == "doc-extract-api"
