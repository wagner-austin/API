"""Health check utilities for transcript-api.

Uses platform_core.health for the standardized liveness probe and
platform_workers.health for Redis readiness with worker presence.
"""

from __future__ import annotations

from platform_core.health import HealthResponse, ReadyResponse, healthz
from platform_workers.health import readyz_redis_with_workers
from platform_workers.redis import RedisStrProto


def healthz_endpoint() -> HealthResponse:
    """Liveness probe - always returns ok.

    Returns:
        HealthResponse with status "ok".
    """
    return healthz()


def readyz_endpoint(redis: RedisStrProto) -> ReadyResponse:
    """Readiness probe - checks Redis connectivity and worker presence.

    STT jobs are enqueued to Redis and executed by transcript-rq-worker, so a
    reachable Redis with no registered worker is not ready: requests would be
    accepted and then never run.

    Args:
        redis: Redis client for connectivity and worker checks.

    Returns:
        Ready response with status and optional reason.
    """
    return readyz_redis_with_workers(redis)


__all__ = ["healthz_endpoint", "readyz_endpoint"]
