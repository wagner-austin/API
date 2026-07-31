"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter
from platform_core.health import HealthResponse, ReadyResponse, healthz


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

    def _readyz() -> ReadyResponse:
        """Readiness probe endpoint.

        The service holds no queue and no database; discovery calls go out to
        Devpost, Kaggle and GitHub per request. Their credentials are optional
        (`KAGGLE_API_TOKEN`, `GITHUB_TOKEN`), so their absence degrades
        individual routes rather than the service, and an upstream outage must
        not take this probe down with it.

        Returns:
            Ready status dict.
        """
        return {"status": "ready", "reason": None}

    router.add_api_route("/healthz", _healthz, methods=["GET"])
    router.add_api_route("/readyz", _readyz, methods=["GET"])
    return router
