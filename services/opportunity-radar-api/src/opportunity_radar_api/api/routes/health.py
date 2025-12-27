"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter
from platform_core.health import HealthResponse, healthz


def build_router() -> APIRouter:
    """Build health check router.

    Returns:
        Configured APIRouter with health endpoints.
    """
    router = APIRouter(tags=["health"])

    def _healthz() -> HealthResponse:
        """Liveness probe endpoint.

        Returns:
            Health status dict.
        """
        return healthz()

    router.add_api_route("/healthz", _healthz, methods=["GET"])
    return router
