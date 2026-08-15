"""Testing utilities for platform_workers.

This module provides typed stubs for testing services that use platform_workers
infrastructure. These stubs implement the public protocols with in-memory storage.

It also provides a HooksContainer for dependency injection in tests. Production
code sets hooks to real implementations at startup; tests set them to fakes.
"""

from __future__ import annotations

import importlib as _importlib
from typing import Protocol

# Re-export all fakes from _fakes module
from ._fakes import (
    EnqueuedJob,
    FakeAsyncRedis,
    FakeFetchedJob,
    FakeJob,
    FakeLogger,
    FakePubSub,
    FakeQueue,
    FakeRedis,
    FakeRedisAsyncioModule,
    FakeRedisBytesClient,
    FakeRedisBytesModule,
    FakeRedisClient,
    FakeRedisConditionalHsetError,
    FakeRedisConditionalHsetRedisError,
    FakeRedisError,
    FakeRedisHsetError,
    FakeRedisHsetRedisError,
    FakeRedisNonRedisError,
    FakeRedisNonRedisScardError,
    FakeRedisNoPong,
    FakeRedisPublishError,
    FakeRedisScardError,
    FakeRedisStrModule,
    FakeRetry,
    FakeRQModule,
    LoggerProtocol,
    LogRecord,
    MethodCall,
    Published,
    _FakeCurrentJob,
    _FakeRQQueueInternal,
    _FakeRQWorkerInternal,
)
from .redis import (
    RedisBytesProto,
    RedisStrProto,
    _RedisAsyncioModule,
    _RedisBytesClient,
    _RedisBytesModule,
    _RedisStrModule,
)
from .rq_harness import (
    FetchedJobProto,
    RQRetryLike,
    _RQJobClassProto,
    _RQModuleProtocol,
)

# =============================================================================
# Hook Protocol Definitions
# =============================================================================


class LoadStrModuleHook(Protocol):
    """Protocol for loading the string-mode kv module.

    Typed against the real module contract, not the fake, so the production
    implementation satisfies it and the hook needs no nullable fallback.
    """

    def __call__(self) -> _RedisStrModule: ...


class LoadBytesModuleHook(Protocol):
    """Protocol for loading the bytes-mode kv module."""

    def __call__(self) -> _RedisBytesModule: ...


class LoadAsyncModuleHook(Protocol):
    """Protocol for loading the async kv module."""

    def __call__(self) -> _RedisAsyncioModule: ...


class LoadRQModuleHook(Protocol):
    """Protocol for loading the RQ module."""

    def __call__(self) -> _RQModuleProtocol: ...


class FetchJobHook(Protocol):
    """Protocol for fetching RQ jobs by ID."""

    def __call__(self, job_id: str, connection: _RedisBytesClient) -> FetchedJobProto: ...


# =============================================================================
# Hooks Container for Dependency Injection
# =============================================================================


def _real_load_redis_str_module() -> _RedisStrModule:
    """Production implementation importing the redis module.

    Returns:
        The imported redis module.
    """
    return __import__("redis")


def _real_load_redis_bytes_module() -> _RedisBytesModule:
    """Production implementation importing the redis module for bytes clients.

    Returns:
        The imported redis module.
    """
    return __import__("redis")


def _real_load_redis_asyncio_module() -> _RedisAsyncioModule:
    """Production implementation importing the redis.asyncio module.

    Returns:
        The imported redis.asyncio module.
    """
    module: _RedisAsyncioModule = _importlib.import_module("redis.asyncio")
    return module


def _real_load_rq_module() -> _RQModuleProtocol:
    """Production implementation importing the rq module.

    Returns:
        The imported rq module.
    """
    return __import__("rq")


def _real_fetch_job(job_id: str, connection: _RedisBytesClient) -> FetchedJobProto:
    """Production implementation fetching an RQ job by id.

    Args:
        job_id: Identifier of the job to fetch.
        connection: Redis connection the job lives on.

    Returns:
        The fetched job.

    Raises:
        NoSuchJobError: If the job does not exist. Use load_no_such_job_error()
            to get the exception type for catching.
    """
    rq_job_mod = __import__("rq.job", fromlist=["Job"])
    job_cls: _RQJobClassProto = rq_job_mod.Job
    result: FetchedJobProto = job_cls.fetch(job_id, connection=connection)
    return result


class HooksContainer:
    """Container for dependency injection hooks in platform_workers.

    Every hook is bound to its real implementation here, so callers invoke it
    directly and there is no conditional dispatch. Tests rebind a hook to a
    fake and call reset() to restore the real implementations.

    Attributes:
        load_redis_str_module: Hook to load redis module for str clients.
        load_redis_bytes_module: Hook to load redis module for bytes clients.
        load_redis_asyncio_module: Hook to load redis.asyncio module.
        load_rq_module: Hook to load rq module.
        fetch_job: Hook for fetching RQ jobs by ID.
    """

    # Redis module loaders
    load_redis_str_module: LoadStrModuleHook = _real_load_redis_str_module
    load_redis_bytes_module: LoadBytesModuleHook = _real_load_redis_bytes_module
    load_redis_asyncio_module: LoadAsyncModuleHook = _real_load_redis_asyncio_module

    # RQ module loader
    load_rq_module: LoadRQModuleHook = _real_load_rq_module

    # RQ job fetch hook
    fetch_job: FetchJobHook = _real_fetch_job

    @classmethod
    def reset(cls) -> None:
        """Restore every hook to its real implementation."""
        cls.load_redis_str_module = _real_load_redis_str_module
        cls.load_redis_bytes_module = _real_load_redis_bytes_module
        cls.load_redis_asyncio_module = _real_load_redis_asyncio_module
        cls.load_rq_module = _real_load_rq_module
        cls.fetch_job = _real_fetch_job


# Global hooks instance
hooks = HooksContainer


# =============================================================================
# Factory Functions for Test Hooks Injection
# =============================================================================


def fake_kv_store_factory(url: str) -> RedisStrProto:
    """Factory that returns a FakeRedis for kv_store_factory hook."""
    return FakeRedis()


def fake_rq_connection_factory(url: str) -> RedisBytesProto:
    """Factory that returns a FakeRedisBytesClient for rq_connection_factory hook."""
    return FakeRedisBytesClient()


def fake_rq_queue_factory(name: str, connection: _RedisBytesClient) -> FakeQueue:
    """Factory that returns a FakeQueue for rq_queue_factory hook."""
    return FakeQueue()


def fake_rq_retry_factory(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
    """Factory that returns a FakeRetry for rq_retry_factory hook."""
    return FakeRetry(max=max_retries, interval=intervals)


# =============================================================================
# Fake Factory Helpers for Hook Setup
# =============================================================================


def make_fake_load_redis_str_module(
    client: FakeRedisClient,
) -> tuple[LoadStrModuleHook, FakeRedisStrModule]:
    """Create a hook function and module for str client testing."""
    module = FakeRedisStrModule(client)

    def _hook() -> FakeRedisStrModule:
        return module

    return _hook, module


def make_fake_load_redis_bytes_module() -> tuple[LoadBytesModuleHook, FakeRedisBytesModule]:
    """Create a hook function and module for bytes client testing."""
    module = FakeRedisBytesModule()

    def _hook() -> FakeRedisBytesModule:
        return module

    return _hook, module


def make_fake_load_redis_asyncio_module() -> tuple[LoadAsyncModuleHook, FakeRedisAsyncioModule]:
    """Create a hook function and module for async client testing."""
    module = FakeRedisAsyncioModule()

    def _hook() -> FakeRedisAsyncioModule:
        return module

    return _hook, module


def make_fake_load_rq_module(
    *, current_job: _FakeCurrentJob | None = None
) -> tuple[LoadRQModuleHook, FakeRQModule]:
    """Create a hook function and module for RQ testing."""
    module = FakeRQModule(current_job=current_job)

    def _hook() -> _RQModuleProtocol:
        return module

    return _hook, module


def make_fake_fetch_job_found(job: FakeFetchedJob) -> FetchJobHook:
    """Create a fetch_job hook that returns the given fake job.

    Args:
        job: The FakeFetchedJob to return.

    Returns:
        A hook function suitable for hooks.fetch_job.
    """

    def _hook(job_id: str, connection: _RedisBytesClient) -> FetchedJobProto:
        return job

    return _hook


def make_fake_fetch_job_not_found() -> FetchJobHook:
    """Create a fetch_job hook that raises NoSuchJobError.

    Returns:
        A hook function that raises NoSuchJobError when called.
    """
    from .rq_harness import load_no_such_job_error

    exc_cls = load_no_such_job_error()

    def _hook(job_id: str, connection: _RedisBytesClient) -> FetchedJobProto:
        raise exc_cls(f"Job {job_id} not found")

    return _hook


__all__ = [
    # Re-exported from _fakes
    "EnqueuedJob",
    "FakeAsyncRedis",
    "FakeFetchedJob",
    "FakeJob",
    "FakeLogger",
    "FakePubSub",
    "FakeQueue",
    "FakeRQModule",
    "FakeRedis",
    "FakeRedisAsyncioModule",
    "FakeRedisBytesClient",
    "FakeRedisBytesModule",
    "FakeRedisClient",
    "FakeRedisConditionalHsetError",
    "FakeRedisConditionalHsetRedisError",
    "FakeRedisError",
    "FakeRedisHsetError",
    "FakeRedisHsetRedisError",
    "FakeRedisNoPong",
    "FakeRedisNonRedisError",
    "FakeRedisNonRedisScardError",
    "FakeRedisPublishError",
    "FakeRedisScardError",
    "FakeRedisStrModule",
    "FakeRetry",
    # Hook Protocols
    "FetchJobHook",
    # Hooks
    "HooksContainer",
    "LoadAsyncModuleHook",
    "LoadBytesModuleHook",
    "LoadRQModuleHook",
    "LoadStrModuleHook",
    "LogRecord",
    "LoggerProtocol",
    "MethodCall",
    "Published",
    "_FakeCurrentJob",
    "_FakeRQQueueInternal",
    "_FakeRQWorkerInternal",
    # Factory functions
    "fake_kv_store_factory",
    "fake_rq_connection_factory",
    "fake_rq_queue_factory",
    "fake_rq_retry_factory",
    "hooks",
    "make_fake_fetch_job_found",
    "make_fake_fetch_job_not_found",
    "make_fake_load_redis_asyncio_module",
    "make_fake_load_redis_bytes_module",
    "make_fake_load_redis_str_module",
    "make_fake_load_rq_module",
]
