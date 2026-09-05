"""Standardized health check utilities for all services.

This module provides:
- TypedDict definitions for health check responses (HealthResponse, ReadyResponse)
- Liveness probe function (healthz)

For readiness probes (/readyz), use platform_workers.health which checks Redis.
All services in this platform require Redis for job queue infrastructure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter
from starlette.responses import Response
from typing_extensions import TypedDict

SERVICE_UNAVAILABLE = 503
"""What a degraded readiness probe answers.

Named here because five services each wrote it into their own router, two of
them as the bare integer and three via ``status.HTTP_503_SERVICE_UNAVAILABLE``.
A readiness probe that returns 200 while reporting "degraded" is a service an
orchestrator keeps sending traffic to.
"""


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


def build_health_router(
    *,
    healthz_route: Callable[[], HealthResponse],
    readyz_route: Callable[[], ReadyResponse],
) -> APIRouter:
    """Build the two-route health router every service in this monorepo has.

    Five services had their own copy of this, in two variants: Art-Trainer and
    Model-Trainer taking a ServiceContainer and logging, and music-wrapped-api,
    qr-api and transcript-api driving a generator dependency by hand. What all
    five repeated was the ROUTER SHAPE and the degraded-to-503 mapping, and
    that mapping is the part worth having once -- a readiness probe that
    answers 200 while reporting "degraded" is a service the orchestrator keeps
    routing traffic to.

    The differences stay with the callers, as arguments: how a service obtains
    its Redis client, whether it logs, and what it considers ready are its own
    business and are not the same across the five.

    Args:
        healthz_route: Liveness probe. Called with no arguments; a service that
            needs a container or a client closes over it.
        readyz_route: Readiness probe. Called with no arguments and returning
            the readiness result; this function maps a degraded result onto the
            response status, so the caller does not set it and cannot forget.

    Returns:
        A router serving GET /healthz and GET /readyz.
    """
    router = APIRouter()

    def _healthz() -> HealthResponse:
        return healthz_route()

    def _readyz(response: Response) -> ReadyResponse:
        result = readyz_route()
        if result["status"] == "degraded":
            response.status_code = SERVICE_UNAVAILABLE
        return result

    router.add_api_route("/healthz", _healthz, methods=["GET"])
    router.add_api_route("/readyz", _readyz, methods=["GET"])
    return router


__all__ = [
    "SERVICE_UNAVAILABLE",
    "HealthResponse",
    "ReadyResponse",
    "build_health_router",
    "healthz",
]
