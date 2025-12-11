"""Hooks for container factories - production defaults, tests override.

Production code initializes these to real implementations at module level.
Tests replace them with fakes before exercising the code under test.
No conditionals needed - just call the hook directly.
"""

from __future__ import annotations

from typing import Protocol

from covenant_persistence import ConnectCallable, ConnectionProtocol
from platform_workers.redis import RedisStrProto, redis_for_kv
from platform_workers.rq_harness import (
    RQClientQueue,
    _RedisBytesClient,
    redis_raw_for_rq,
    rq_queue,
)


class KvClientFactoryProtocol(Protocol):
    """Protocol for key-value client factory."""

    def __call__(self, url: str) -> RedisStrProto:
        """Create KV client from URL."""
        ...


class ConnectionFactoryProtocol(Protocol):
    """Protocol for psycopg connect factory."""

    def __call__(self, dsn: str) -> ConnectionProtocol:
        """Create database connection from DSN."""
        ...


class RqClientFactoryProtocol(Protocol):
    """Protocol for RQ client factory."""

    def __call__(self, url: str) -> _RedisBytesClient:
        """Create RQ client from URL."""
        ...


class QueueFactoryProtocol(Protocol):
    """Protocol for rq_queue factory."""

    def __call__(self, name: str, connection: _RedisBytesClient) -> RQClientQueue:
        """Create RQ queue from name and connection."""
        ...


def _default_psycopg_connect(dsn: str) -> ConnectionProtocol:
    """Production psycopg connect - used as default hook."""
    psycopg = __import__("psycopg")
    connect_fn: ConnectCallable = psycopg.connect
    return connect_fn(dsn)


def _default_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
    """Production rq_queue - used as default hook."""
    return rq_queue(name, connection)


# Factory hooks - initialized to production implementations.
# Tests replace these with fakes before calling container code.
kv_factory: KvClientFactoryProtocol = redis_for_kv
connection_factory: ConnectionFactoryProtocol = _default_psycopg_connect
rq_client_factory: RqClientFactoryProtocol = redis_raw_for_rq
queue_factory: QueueFactoryProtocol = _default_rq_queue
