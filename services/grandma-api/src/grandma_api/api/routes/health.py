"""Health check routes for grandma-api."""

from __future__ import annotations

from fastapi import APIRouter
from platform_core.health import HealthResponse, ReadyResponse

from grandma_api.health import healthz_endpoint, readyz_endpoint


def build_router() -> APIRouter:
    """Build health router with /healthz and /readyz endpoints.

    Returns:
        APIRouter with health endpoints configured.
    """
    router = APIRouter()

    def _healthz() -> HealthResponse:
        return healthz_endpoint()

    def _readyz() -> ReadyResponse:
        return readyz_endpoint()

    router.add_api_route("/healthz", _healthz, methods=["GET"])
    router.add_api_route("/readyz", _readyz, methods=["GET"])
    return router


__all__ = ["build_router"]
