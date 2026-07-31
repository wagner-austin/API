"""Health check routes for transcript-api."""

from __future__ import annotations

from fastapi import APIRouter
from platform_core.health import HealthResponse, ReadyResponse
from starlette.responses import Response

from ...dependencies import get_redis
from ...health import healthz_endpoint, readyz_endpoint


def build_router() -> APIRouter:
    """Build health router with /healthz and /readyz endpoints.

    Returns:
        APIRouter with health endpoints configured.
    """
    router = APIRouter()

    def _healthz() -> HealthResponse:
        return healthz_endpoint()

    def _readyz(resp: Response) -> ReadyResponse:
        # get_redis is a generator dependency; drive it by hand so close() runs
        # deterministically whether or not the readiness check raises.
        gen = get_redis()
        client = next(gen)
        try:
            result = readyz_endpoint(client)
        finally:
            _ = next(gen, None)
        if result["status"] == "degraded":
            resp.status_code = 503
        return result

    router.add_api_route("/healthz", _healthz, methods=["GET"])
    router.add_api_route("/readyz", _readyz, methods=["GET"])
    return router


__all__ = ["build_router"]
