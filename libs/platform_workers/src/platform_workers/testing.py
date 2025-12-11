"""Testing utilities for platform_workers.

This module provides typed stubs for testing services that use platform_workers
infrastructure. These stubs implement the public protocols with in-memory storage.

It also provides a HooksContainer for dependency injection in tests. Production
code sets hooks to real implementations at startup; tests set them to fakes.
"""

from __future__ import annotations

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
    _RedisBytesClient,
)
from .rq_harness import (
    FetchedJobProto,
    RQRetryLike,
    _RQModuleProtocol,
)

# =============================================================================
# Hook Protocol Definitions
# =============================================================================


class LoadStrModuleHook(Protocol):
    """Protocol for loading the string-mode kv module."""

    def __call__(self) -> FakeRedisStrModule: ...


class LoadBytesModuleHook(Protocol):
    """Protocol for loading the bytes-mode kv module."""

    def __call__(self) -> FakeRedisBytesModule: ...


class LoadAsyncModuleHook(Protocol):
    """Protocol for loading the async kv module."""

    def __call__(self) -> FakeRedisAsyncioModule: ...


class LoadRQModuleHook(Protocol):
    """Protocol for loading the RQ module."""

    def __call__(self) -> _RQModuleProtocol: ...


class PathIsDirHook(Protocol):
    """Protocol for checking if a path is a directory."""

    def __call__(self, path: str) -> bool: ...


class FetchJobHook(Protocol):
    """Protocol for fetching RQ jobs by ID."""

    def __call__(self, job_id: str, connection: _RedisBytesClient) -> FetchedJobProto: ...


# =============================================================================
# Hooks Container for Dependency Injection
# =============================================================================


class HooksContainer:
    """Container for dependency injection hooks in platform_workers.

    Production code sets these hooks to real implementations at startup.
    Tests set them to fakes to avoid external dependencies.

    Attributes:
        load_redis_str_module: Hook to load redis module for str clients.
        load_redis_bytes_module: Hook to load redis module for bytes clients.
        load_redis_asyncio_module: Hook to load redis.asyncio module.
        load_rq_module: Hook to load rq module.
        fetch_job: Hook for fetching RQ jobs by ID.
        path_is_dir: Hook for Path.is_dir() calls.
    """

    # Redis module loaders
    load_redis_str_module: LoadStrModuleHook | None = None
    load_redis_bytes_module: LoadBytesModuleHook | None = None
    load_redis_asyncio_module: LoadAsyncModuleHook | None = None

    # RQ module loader
    load_rq_module: LoadRQModuleHook | None = None

    # RQ job fetch hook
    fetch_job: FetchJobHook | None = None

    # Path checking for guard scripts
    path_is_dir: PathIsDirHook | None = None

    @classmethod
    def reset(cls) -> None:
        """Reset all hooks to None (production defaults)."""
        cls.load_redis_str_module = None
        cls.load_redis_bytes_module = None
        cls.load_redis_asyncio_module = None
        cls.load_rq_module = None
        cls.fetch_job = None
        cls.path_is_dir = None


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


def fake_path_is_dir_false(path: str) -> bool:
    """Hook that always returns False for path_is_dir."""
    return False


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
    "PathIsDirHook",
    "Published",
    "_FakeCurrentJob",
    "_FakeRQQueueInternal",
    "_FakeRQWorkerInternal",
    # Factory functions
    "fake_kv_store_factory",
    "fake_path_is_dir_false",
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
