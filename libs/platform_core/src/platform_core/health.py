"""Standardized health check utilities for all services.

This module provides:
- TypedDict definitions for health check responses (HealthResponse, ReadyResponse)
- Liveness probe function (healthz)

For readiness probes (/readyz), use platform_workers.health which checks Redis.
All services in this platform require Redis for job queue infrastructure.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class HealthResponse(TypedDict):
    """Response for liveness probe (/healthz)."""

    status: Literal["ok"]


class ReadyResponse(TypedDict):
    """Response for readiness probe (/readyz).

    When ready: {"status": "ready", "reason": None}
    When degraded: {"status": "degraded", "reason": "description of issue"}
    """

    status: Literal["ready", "degraded"]
    reason: str | None


def healthz() -> HealthResponse:
    """Standard liveness probe - always returns ok.

    Liveness probes check if the process is running and responsive.
    They should NOT check external dependencies.
    """
    return {"status": "ok"}


__all__ = [
    "HealthResponse",
    "ReadyResponse",
    "healthz",
]
