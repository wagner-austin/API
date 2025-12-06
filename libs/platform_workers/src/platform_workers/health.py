"""Redis-dependent health check utilities.

This module provides readiness probes that check Redis connectivity.
For stateless health checks, use platform_core.health instead.
"""

from __future__ import annotations

from platform_core.health import ReadyResponse
from platform_core.logging import get_logger

from platform_workers.redis import RedisStrProto, is_redis_error

_logger = get_logger(__name__)


def readyz_redis(redis: RedisStrProto) -> ReadyResponse:
    """Readiness probe that checks Redis connectivity.

    Args:
        redis: Redis client satisfying RedisStrProto

    Returns:
        Ready response with status and optional reason
    """
    pong: bool
    try:
        pong = redis.ping()
    except Exception as exc:
        if not is_redis_error(exc):
            _logger.error("readyz_redis non-redis error", exc_info=True)
            raise
        _logger.warning("readyz_redis redis error: %s", exc)
        return {"status": "degraded", "reason": "redis error"}

    if not pong:
        _logger.warning("readyz_redis ping returned false")
        return {"status": "degraded", "reason": "redis no-pong"}

    return {"status": "ready", "reason": None}


def readyz_redis_with_workers(
    redis: RedisStrProto,
    *,
    workers_key: str = "rq:workers",
) -> ReadyResponse:
    """Readiness probe that checks Redis connectivity AND worker presence.

    Args:
        redis: Redis client satisfying RedisStrProto
        workers_key: Redis set key containing worker registrations

    Returns:
        Ready response with status and optional reason
    """
    # First check basic connectivity
    pong: bool
    try:
        pong = redis.ping()
    except Exception as exc:
        if not is_redis_error(exc):
            _logger.error("readyz_redis_with_workers ping non-redis error", exc_info=True)
            raise
        _logger.warning("readyz_redis_with_workers ping redis error: %s", exc)
        return {"status": "degraded", "reason": "redis error"}

    if not pong:
        _logger.warning("readyz_redis_with_workers ping returned false")
        return {"status": "degraded", "reason": "redis no-pong"}

    # Check worker presence
    worker_count: int
    try:
        worker_count = redis.scard(workers_key)
    except Exception as exc:
        if not is_redis_error(exc):
            _logger.error("readyz_redis_with_workers scard non-redis error", exc_info=True)
            raise
        _logger.warning("readyz_redis_with_workers scard redis error: %s", exc)
        return {"status": "degraded", "reason": "redis error"}

    if worker_count <= 0:
        _logger.warning("readyz_redis_with_workers no workers found")
        return {"status": "degraded", "reason": "no-worker"}

    return {"status": "ready", "reason": None}


__all__ = [
    "readyz_redis",
    "readyz_redis_with_workers",
]
