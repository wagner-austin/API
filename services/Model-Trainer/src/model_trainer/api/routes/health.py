from __future__ import annotations

from fastapi import APIRouter
from platform_core.health import (
    HealthResponse,
    ReadyResponse,
    build_health_router,
    healthz,
)
from platform_core.logging import get_logger
from platform_workers.health import logged_readyz_with_workers

from ...core.services.container import ServiceContainer

_logger = get_logger(__name__)


def build_router(container: ServiceContainer) -> APIRouter:
    """Build this service's health router.

    The router shape and the degraded-to-503 mapping live in
    ``platform_core.health``; what stays here is what is actually this
    service's own -- where its Redis client comes from, and its logging.

    Args:
        container: The service container holding the Redis client.

    Returns:
        A router serving GET /healthz and GET /readyz.
    """

    def healthz_route() -> HealthResponse:
        _logger.info("healthz", extra={"category": "api", "service": "health", "event": "healthz"})
        return healthz()

    def readyz_route() -> ReadyResponse:
        return logged_readyz_with_workers(container.redis, _logger)

    return build_health_router(healthz_route=healthz_route, readyz_route=readyz_route)
