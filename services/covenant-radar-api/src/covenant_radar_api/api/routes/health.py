"""Health check routes for covenant-radar-api."""

from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, status
from platform_core.health import HealthResponse, ReadyResponse
from platform_workers.redis import RedisStrProto
from starlette.responses import Response

from ...health import healthz_endpoint, readyz_endpoint


class HealthContainerProtocol(Protocol):
    """Protocol for health check container."""

    redis: RedisStrProto


def build_router(get_container: HealthContainerProtocol) -> APIRouter:
    """Build health router with /healthz and /readyz endpoints.

    Args:
        get_container: Container with redis attribute for health checks.
    """
    router = APIRouter()

    def _healthz() -> HealthResponse:
        return healthz_endpoint()

    def _readyz(resp: Response) -> ReadyResponse:
        result = readyz_endpoint(redis=get_container.redis)
        if result["status"] == "degraded":
            resp.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    router.add_api_route("/healthz", _healthz, methods=["GET"], response_model=None)
    router.add_api_route("/readyz", _readyz, methods=["GET"], response_model=None)
    return router


__all__ = ["HealthContainerProtocol", "build_router"]
