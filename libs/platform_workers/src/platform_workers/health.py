"""Redis-dependent health check utilities.

This module provides readiness probes that check Redis connectivity.
For stateless health checks, use platform_core.health instead.
"""

from __future__ import annotations

from platform_core.health import ReadyResponse
from platform_core.logging import get_logger

from platform_workers._fakes_logging import LoggerProtocol
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


def logged_readyz_with_workers(
    redis: RedisStrProto,
    logger: LoggerProtocol,
    *,
    workers_key: str = "rq:workers",
) -> ReadyResponse:
    """Run the worker readiness probe and log what it decided.

    Art-Trainer and Model-Trainer had this identical, down to the log messages
    and the `extra` keys -- the two services whose readiness means the same
    thing, because both are trainers backed by an RQ worker pool. Everything
    below the probe was already shared; the LOGGING was the part each had
    written out again.

    Args:
        redis: Redis client satisfying RedisStrProto.
        logger: Where to record the outcome. Typed as the package's own
            LoggerProtocol rather than logging.Logger, because a direct
            logging import is refused here and the minimal interface is
            what this needs anyway.
        workers_key: Redis set key holding worker registrations.

    Returns:
        The probe's result, unchanged. Logging is a side effect here and
        never alters the verdict -- a readiness answer that depended on
        whether anyone was listening would be the wrong shape entirely.
    """
    result = readyz_redis_with_workers(redis, workers_key=workers_key)
    if result["status"] == "degraded":
        logger.info(
            "readyz degraded",
            extra={
                "category": "api",
                "service": "health",
                "event": "readyz",
                "reason": result["reason"],
            },
        )
        return result
    logger.info(
        "readyz",
        extra={"category": "api", "service": "health", "event": "readyz", "status": "ready"},
    )
    return result


__all__ = [
    "logged_readyz_with_workers",
    "readyz_redis",
    "readyz_redis_with_workers",
]
