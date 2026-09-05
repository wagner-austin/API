from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Annotated

from fastapi import Depends
from platform_core.config import _require_env_str
from platform_core.logging import get_logger
from platform_core.queues import TRANSCRIPT_QUEUE
from platform_workers.redis import RedisStrProto
from platform_workers.rq_harness import QueueProtocol, connecting_queue

from transcript_api.types import (
    LoggerProtocol,
)

from . import _test_hooks

# Provider types for dependency injection
RedisProviderType = Callable[[], RedisStrProto]
QueueProviderType = Callable[[], QueueProtocol]
LoggerProviderType = Callable[[], LoggerProtocol]


class _ProviderContext:
    """Global context for injectable providers (used in testing)."""

    def __init__(self) -> None:
        self.redis_provider: RedisProviderType | None = None
        self.queue_provider: QueueProviderType | None = None
        self.logger_provider: LoggerProviderType | None = None


provider_context = _ProviderContext()


def _get_redis_url() -> str:
    """Get Redis URL from environment, raising if not set."""
    return _require_env_str("REDIS_URL")


def get_redis() -> Generator[RedisStrProto, None, None]:
    """Dependency: typed Redis (strings) using URL from env; closes on teardown."""
    if provider_context.redis_provider is not None:
        yield provider_context.redis_provider()
        return
    redis_url = _get_redis_url()
    client = _test_hooks.redis_factory(redis_url)
    try:
        yield client
    finally:
        client.close()


def get_request_logger() -> LoggerProtocol:
    """Dependency: request-scoped logger (delegates to global logger)."""
    if provider_context.logger_provider is not None:
        return provider_context.logger_provider()
    return get_logger(__name__)


def get_queue() -> QueueProtocol:
    """Dependency: RQ queue bound to a dedicated binary Redis connection.

    Uses shared platform helpers and a fixed queue name from platform_core.
    Imports RQ at runtime to allow strict tests to inject fakes. Return type is
    a minimal JobLike object to avoid leaking untyped values.
    """
    if provider_context.queue_provider is not None:
        return provider_context.queue_provider()
    return connecting_queue(TRANSCRIPT_QUEUE, _get_redis_url())


RedisDep = Annotated[RedisStrProto, Depends(get_redis)]
LoggerDep = Annotated[LoggerProtocol, Depends(get_request_logger)]
QueueDep = Annotated[QueueProtocol, Depends(get_queue)]


__all__ = [
    "LoggerDep",
    "LoggerProviderType",
    "QueueDep",
    "QueueProviderType",
    "RedisDep",
    "RedisProviderType",
    "get_queue",
    "get_redis",
    "get_request_logger",
    "provider_context",
]
