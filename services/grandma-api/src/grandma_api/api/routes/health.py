"""Health check route for grandma-api."""

from __future__ import annotations

from fastapi import APIRouter
from platform_core.health import HealthResponse

from grandma_api.health import healthz_endpoint


def build_router() -> APIRouter:
    """Build health router with /healthz endpoint.

    Returns:
        APIRouter with health endpoints configured.
    """
    router = APIRouter()

    def _healthz() -> HealthResponse:
        return healthz_endpoint()

    router.add_api_route("/healthz", _healthz, methods=["GET"])
    return router


__all__ = ["build_router"]
