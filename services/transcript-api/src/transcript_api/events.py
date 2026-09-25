from __future__ import annotations

from platform_core.config import _require_env_str
from platform_core.job_events import (
    ErrorKind,
    JobCompletedV1,
    JobDomain,
    JobFailedV1,
    default_events_channel,
    encode_job_event,
    make_completed_event,
    make_failed_event,
)
from platform_workers.redis import RedisStrProto

from . import _test_hooks

_DEFAULT_CHANNEL = default_events_channel(JobDomain.TRANSCRIPT)


def _load_redis() -> RedisStrProto:
    """Load a Redis client for publishing events."""
    redis_url = _require_env_str("REDIS_URL")
    return _test_hooks.redis_factory(redis_url)


def publish_completed(*, request_id: str, user_id: int, url: str, text: str) -> None:
    """Publish a generic completed job event for synchronous transcript requests."""
    event: JobCompletedV1 = make_completed_event(
        domain=JobDomain.TRANSCRIPT,
        job_id=request_id,
        user_id=int(user_id),
        result_id=url,
        result_bytes=len(text.encode("utf-8")),
    )
    redis = _load_redis()
    try:
        redis.publish(_DEFAULT_CHANNEL, encode_job_event(event))
    finally:
        redis.close()


def publish_failed(*, request_id: str, user_id: int, error_kind: ErrorKind, message: str) -> None:
    """Publish a generic failed job event for synchronous transcript requests."""
    event: JobFailedV1 = make_failed_event(
        domain=JobDomain.TRANSCRIPT,
        job_id=request_id,
        user_id=int(user_id),
        error_kind=error_kind,
        message=message,
    )
    redis = _load_redis()
    try:
        redis.publish(_DEFAULT_CHANNEL, encode_job_event(event))
    finally:
        redis.close()


__all__ = [
    "publish_completed",
    "publish_failed",
]
