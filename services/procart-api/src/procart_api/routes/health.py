from __future__ import annotations

from fastapi import APIRouter
from platform_core.health import HealthResponse, ReadyResponse, healthz


def build_router() -> APIRouter:
    """Build health router.

    Returns:
        APIRouter: Router exposing /healthz and /readyz endpoints.
    """
    r = APIRouter()

    def _healthz() -> HealthResponse:
        """Liveness probe.

        Returns:
            Health status.
        """
        return healthz()

    def _readyz() -> ReadyResponse:
        """Readiness probe.

        Rendering runs in-process against registries that are populated at
        import time, so a process that answers at all can serve a render
        request. There is no queue, database or upstream service whose
        reachability could make this answer differ from liveness.

        Returns:
            Ready status.
        """
        return {"status": "ready", "reason": None}

    # Use add_api_route to avoid decorator-induced Any types under mypy --strict.
    r.add_api_route("/healthz", _healthz, methods=["GET"])
    r.add_api_route("/readyz", _readyz, methods=["GET"])
    return r


__all__ = ["build_router"]
