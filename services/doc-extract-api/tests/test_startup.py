"""Tests for doc_extract_api.startup."""

from __future__ import annotations

from doc_extract_api.startup import make_app


class TestMakeApp:
    def test_creates_app(self) -> None:
        app = make_app()
        assert app.title == "doc-extract-api"
        assert app.version == "0.1.0"

    def test_has_routes(self) -> None:
        app = make_app()
        # 4 explicit routes (readyz, healthz, jobs POST, jobs/{id} GET)
        # plus OpenAPI auto-generated routes
        assert len(app.routes) >= 4
