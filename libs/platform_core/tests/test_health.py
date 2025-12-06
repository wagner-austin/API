"""Tests for platform_core.health module."""

from __future__ import annotations

from platform_core.health import (
    HealthResponse,
    ReadyResponse,
    healthz,
)


def test_healthz_returns_ok() -> None:
    """Test healthz always returns status ok."""
    result: HealthResponse = healthz()
    assert result == {"status": "ok"}


def test_health_response_type_structure() -> None:
    """Test HealthResponse TypedDict structure."""
    response: HealthResponse = {"status": "ok"}
    assert response["status"] == "ok"


def test_ready_response_type_structure_ready() -> None:
    """Test ReadyResponse TypedDict structure for ready state."""
    response: ReadyResponse = {"status": "ready", "reason": None}
    assert response["status"] == "ready"
    assert response["reason"] is None


def test_ready_response_type_structure_degraded() -> None:
    """Test ReadyResponse TypedDict structure for degraded state."""
    response: ReadyResponse = {"status": "degraded", "reason": "test reason"}
    assert response["status"] == "degraded"
    assert response["reason"] == "test reason"
