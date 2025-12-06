from __future__ import annotations

from fastapi import APIRouter, Response, status
from platform_core.health import HealthResponse, ReadyResponse, healthz
from platform_core.logging import get_logger
from platform_workers.health import readyz_redis_with_workers
from platform_workers.redis import RedisStrProto

from ...core.services.container import ServiceContainer

_logger = get_logger(__name__)


def build_router(container: ServiceContainer) -> APIRouter:
    router = APIRouter()

    def healthz_route() -> HealthResponse:
        _logger.info("healthz", extra={"category": "api", "service": "health", "event": "healthz"})
        return healthz()

    def readyz_route(response: Response) -> ReadyResponse:
        client: RedisStrProto = container.redis
        # Check Redis connectivity and worker presence using shared helper
        result = readyz_redis_with_workers(client)
        if result["status"] == "degraded":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            _logger.info(
                "readyz degraded",
                extra={
                    "category": "api",
                    "service": "health",
                    "event": "readyz",
                    "reason": result["reason"],
                },
            )
            return result
        _logger.info(
            "readyz",
            extra={"category": "api", "service": "health", "event": "readyz", "status": "ready"},
        )
        return {"status": "ready", "reason": None}

    router.add_api_route("/healthz", healthz_route, methods=["GET"])
    router.add_api_route("/readyz", readyz_route, methods=["GET"])
    return router
