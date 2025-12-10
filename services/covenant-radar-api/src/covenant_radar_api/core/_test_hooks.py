"""Test hooks for container - allows injecting factory functions for testing."""

from __future__ import annotations

from typing import Protocol

from covenant_persistence import ConnectionProtocol
from platform_workers.redis import RedisStrProto
from platform_workers.rq_harness import _RedisBytesClient


class RedisFactoryProtocol(Protocol):
    """Protocol for redis_for_kv factory."""

    def __call__(self, url: str) -> RedisStrProto:
        """Create Redis client from URL."""
        ...


class ConnectionFactoryProtocol(Protocol):
    """Protocol for psycopg connect factory."""

    def __call__(self, dsn: str) -> ConnectionProtocol:
        """Create database connection from DSN."""
        ...


class RedisRqFactoryProtocol(Protocol):
    """Protocol for redis_raw_for_rq factory."""

    def __call__(self, url: str) -> _RedisBytesClient:
        """Create Redis RQ client from URL."""
        ...


# Injectable factories for testing.
# When set, from_settings() uses these instead of real implementations.
redis_factory: RedisFactoryProtocol | None = None
connection_factory: ConnectionFactoryProtocol | None = None
redis_rq_factory: RedisRqFactoryProtocol | None = None
