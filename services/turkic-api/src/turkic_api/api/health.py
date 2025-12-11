"""Standardized health check utilities for turkic-api.

Uses platform_core.health for liveness and platform_workers.health for Redis,
with additional volume check specific to this service.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.health import HealthResponse, ReadyResponse, healthz
from platform_workers.health import readyz_redis_with_workers
from platform_workers.redis import RedisStrProto

from turkic_api import _test_hooks


def healthz_endpoint() -> HealthResponse:
    """Liveness probe - always returns ok."""
    return healthz()


def readyz_endpoint(redis: RedisStrProto, data_dir: str) -> ReadyResponse:
    """Readiness probe - checks Redis and data volume.

    Args:
        redis: Redis client for connectivity check
        data_dir: Path to data directory that must exist

    Returns:
        Ready response with status and optional reason
    """
    # Check Redis connectivity and worker presence first
    redis_result = readyz_redis_with_workers(redis)
    if redis_result["status"] == "degraded":
        return redis_result

    # Check volume exists
    if not _test_hooks.path_exists(Path(data_dir)):
        return {"status": "degraded", "reason": "data volume not found"}

    return {"status": "ready", "reason": None}
