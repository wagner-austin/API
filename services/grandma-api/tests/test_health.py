"""Tests for grandma_api.health module."""

from __future__ import annotations

from grandma_api.health import healthz_endpoint


def test_healthz_endpoint_returns_ok() -> None:
    """Test healthz_endpoint returns status ok."""
    result = healthz_endpoint()
    assert result == {"status": "ok"}
