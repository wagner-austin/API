"""Health check routes for covenant-radar-api."""

from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, status
from platform_core.health import HealthResponse, ReadyResponse
from platform_core.json_utils import JSONValue
from platform_workers.redis import RedisStrProto
from starlette.responses import Response

from ...health import healthz_endpoint, readyz_endpoint

# OpenAPI response schemas (no type annotation for FastAPI compatibility)
_HEALTHZ_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Service is alive",
        "content": {
            "application/json": {
                "example": {"status": "ok"},
            },
        },
    },
}

_READYZ_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Service is ready",
        "content": {
            "application/json": {
                "example": {"status": "ready", "reason": None},
            },
        },
    },
    503: {
        "description": "Service is degraded",
        "content": {
            "application/json": {
                "example": {"status": "degraded", "reason": "redis-unavailable"},
            },
        },
    },
}


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
        """Liveness probe for container orchestration.

        Returns {"status": "ok"} if the service is running.
        Use this endpoint for Kubernetes liveness probes.
        """
        return healthz_endpoint()

    def _readyz(resp: Response) -> ReadyResponse:
        """Readiness probe checking Redis connectivity and worker availability.

        Returns 200 with {"status": "ready"} if all dependencies are healthy.
        Returns 503 with {"status": "degraded", "reason": "..."} if:
        - Redis is unavailable ("redis-unavailable")
        - No RQ workers are registered ("no-worker")
        """
        result = readyz_endpoint(redis=get_container.redis)
        if result["status"] == "degraded":
            resp.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    router.add_api_route(
        "/healthz",
        _healthz,
        methods=["GET"],
        response_model=None,
        summary="Liveness probe",
        description="Liveness probe for container orchestration. Returns 200 if running.",
        response_description="Health status",
        responses=_HEALTHZ_RESPONSES,
        tags=["health"],
    )
    router.add_api_route(
        "/readyz",
        _readyz,
        methods=["GET"],
        response_model=None,
        summary="Readiness probe",
        description=(
            "Readiness probe checking Redis and RQ worker availability. "
            "Returns 503 if dependencies are unhealthy."
        ),
        response_description="Readiness status with optional reason",
        responses=_READYZ_RESPONSES,
        tags=["health"],
    )
    return router


__all__ = ["HealthContainerProtocol", "build_router"]
