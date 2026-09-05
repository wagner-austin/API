"""Health check routes for Art-Trainer API.

This module provides health and readiness endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter
from platform_core.health import (
    HealthResponse,
    ReadyResponse,
    build_health_router,
    healthz,
)
from platform_core.logging import get_logger
from platform_workers.health import readyz_redis_with_workers
from platform_workers.redis import RedisStrProto

from art_trainer.core.services.container import ServiceContainer

_logger = get_logger(__name__)


def build_router(container: ServiceContainer) -> APIRouter:
    """Build health check router.

    The router shape and the degraded-to-503 mapping live in
    ``platform_core.health``; what stays here is what is this service's own --
    where its Redis client comes from, and its logging.

    Args:
        container: Service container.

    Returns:
        Configured API router.
    """

    def healthz_route() -> HealthResponse:
        _logger.info(
            "healthz",
            extra={"category": "api", "service": "health", "event": "healthz"},
        )
        return healthz()

    def readyz_route() -> ReadyResponse:
        client: RedisStrProto = container.redis
        result = readyz_redis_with_workers(client)
        if result["status"] == "degraded":
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
            extra={
                "category": "api",
                "service": "health",
                "event": "readyz",
                "status": "ready",
            },
        )
        return {"status": "ready", "reason": None}

    return build_health_router(healthz_route=healthz_route, readyz_route=readyz_route)


__all__ = [
    "build_router",
]
