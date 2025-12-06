"""Standardized health check utilities for qr-api.

Uses platform_core.health for liveness and platform_workers.health for Redis
readiness with worker presence.
"""

from __future__ import annotations

from platform_core.health import HealthResponse, ReadyResponse, healthz
from platform_workers.health import readyz_redis_with_workers
from platform_workers.redis import RedisStrProto


def healthz_endpoint() -> HealthResponse:
    """Liveness probe - always returns ok."""
    return healthz()


def readyz_endpoint(redis: RedisStrProto) -> ReadyResponse:
    """Readiness probe - checks Redis connectivity and worker presence.

    Args:
        redis: Redis client for connectivity and worker checks

    Returns:
        Ready response with status and optional reason
    """
    return readyz_redis_with_workers(redis)
