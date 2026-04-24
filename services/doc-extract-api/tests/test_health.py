"""Tests for doc_extract_api.health."""

from __future__ import annotations

from doc_extract_api.health import _healthz, _readyz


class TestHealth:
    def test_readyz(self) -> None:
        result = _readyz()
        assert result == {"status": "ok"}

    def test_healthz(self) -> None:
        result = _healthz()
        assert result == {"status": "ok"}
