"""Standardized health check utilities for covenant-radar-api.

Uses platform_core.health for liveness and platform_workers.health for Redis checks.
"""

from __future__ import annotations

from platform_core.health import HealthResponse, ReadyResponse, healthz
from platform_workers.health import readyz_redis_with_workers
from platform_workers.redis import RedisStrProto


def healthz_endpoint() -> HealthResponse:
    """Liveness probe - always returns ok."""
    return healthz()


def readyz_endpoint(redis: RedisStrProto) -> ReadyResponse:
    """Readiness probe - checks Redis and workers.

    Args:
        redis: Redis client for connectivity check

    Returns:
        Ready response with status and optional reason
    """
    redis_result = readyz_redis_with_workers(redis)
    if redis_result["status"] == "degraded":
        return redis_result

    return {"status": "ready", "reason": None}


__all__ = [
    "healthz_endpoint",
    "readyz_endpoint",
]
