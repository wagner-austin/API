from __future__ import annotations

from platform_core.http_utils import add_correlation_header


def test_add_correlation_header_adds_and_preserves() -> None:
    base = {"Accept": "application/json"}
    out = add_correlation_header(base, "req-1")
    assert out["X-Request-ID"] == "req-1"
    assert out["Accept"] == "application/json"
    # Original not mutated
    assert "X-Request-ID" not in base
