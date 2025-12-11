"""Hooks for container factories - production defaults, tests override.

Production code initializes these to real implementations at module level.
Tests replace them with fakes before exercising the code under test.
No conditionals needed - just call the hook directly.
"""

from __future__ import annotations

from typing import Protocol

from covenant_persistence import ConnectionProtocol
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


class PsycopgModuleProtocol(Protocol):
    """Protocol for psycopg module with connect method."""

    def connect(self, dsn: str, autocommit: bool = False) -> ConnectionProtocol:
        """Connect to database."""
        ...


class LoadPsycopgModuleHook(Protocol):
    """Protocol for psycopg module loader hook."""

    def __call__(self) -> PsycopgModuleProtocol:
        """Load psycopg module."""
        ...


# Hook for loading psycopg module - tests override to provide fake
load_psycopg_module_hook: LoadPsycopgModuleHook | None = None


def _load_psycopg_module() -> PsycopgModuleProtocol:
    """Load psycopg module dynamically.

    If load_psycopg_module_hook is set (by tests), uses that.
    Otherwise loads the real psycopg module.
    """
    if load_psycopg_module_hook is not None:
        return load_psycopg_module_hook()
    module: PsycopgModuleProtocol = __import__("psycopg")
    return module


def _psycopg_connect_autocommit(dsn: str) -> ConnectionProtocol:
    """Connect to postgres with autocommit enabled.

    Uses autocommit=True to prevent failed transactions from blocking
    subsequent queries. Each statement commits immediately.
    """
    module = _load_psycopg_module()
    conn: ConnectionProtocol = module.connect(dsn, autocommit=True)
    return conn


# Factory hooks - initialized to production implementations.
# Tests replace these with fakes before calling container code.
# Production defaults call real external services (redis, postgres).
kv_factory: KvClientFactoryProtocol = redis_for_kv
connection_factory: ConnectionFactoryProtocol = _psycopg_connect_autocommit
rq_client_factory: RqClientFactoryProtocol = redis_raw_for_rq
queue_factory: QueueFactoryProtocol = rq_queue
