"""Tests for transcript_api.dependencies module."""

from __future__ import annotations

import pytest
from platform_core.config import _test_hooks as platform_hooks
from platform_core.testing import make_fake_env
from platform_workers.redis import RedisStrProto
from platform_workers.rq_harness import RQJobLike, RQRetryLike
from platform_workers.testing import FakeRedis

from transcript_api import _test_hooks
from transcript_api.dependencies import (
    get_queue,
    get_redis,
    get_request_logger,
    provider_context,
)
from transcript_api.types import JsonValue, QueueProtocol, _EnqCallable


class _JobStub:
    """Stub job implementing RQJobLike."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def get_id(self) -> str:
        return self._job_id


class _QueueStub:
    """Stub queue for testing implementing QueueProtocol."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, tuple[JsonValue, ...]]] = []

    def enqueue(
        self,
        func: str | _EnqCallable,
        *args: JsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike:
        # For test purposes, we only support string func_refs
        func_ref = func if isinstance(func, str) else "callable"
        self.enqueued.append((func_ref, args))
        return _JobStub("job-id")


def test_get_redis_yields_and_closes() -> None:
    """Test get_redis yields client and closes on exit."""
    redis = FakeRedis()

    def _redis_for_kv(url: str) -> RedisStrProto:
        assert url == "redis://example/0"
        return redis

    platform_hooks.get_env = make_fake_env({"REDIS_URL": "redis://example/0"})
    _test_hooks.redis_factory = _redis_for_kv

    gen = get_redis()
    client = next(gen)
    assert client is redis
    with pytest.raises(StopIteration):
        next(gen)
    assert redis.closed
    redis.assert_only_called({"close"})


def test_get_queue_via_provider_context() -> None:
    """Test get_queue uses provider_context when set."""
    queue = _QueueStub()

    def _queue_provider() -> QueueProtocol:
        return queue

    platform_hooks.get_env = make_fake_env({"REDIS_URL": "redis://example/1"})
    provider_context.queue_provider = _queue_provider

    q = get_queue()
    job = q.enqueue("transcript_api.jobs.process_stt", {"url": "u"})
    assert job.get_id() == "job-id"
    assert queue.enqueued


def test_get_request_logger_returns_named_logger() -> None:
    """Test get_request_logger returns a working logger."""
    logger = get_request_logger()
    logger.info("ok", extra={})


def test_get_queue_production_adapter() -> None:
    """Test get_queue creates _QueueAdapter when no provider_context is set."""
    from platform_workers.testing import (
        hooks,
        make_fake_load_redis_bytes_module,
        make_fake_load_rq_module,
    )

    # Clear provider_context to use production path
    provider_context.queue_provider = None

    # Set environment
    platform_hooks.get_env = make_fake_env({"REDIS_URL": "redis://test-queue"})

    # Save and setup hooks for platform_workers
    orig_bytes_hook = hooks.load_redis_bytes_module
    orig_rq_hook = hooks.load_rq_module

    bytes_hook, _bytes_mod = make_fake_load_redis_bytes_module()
    rq_hook, _rq_mod = make_fake_load_rq_module()
    hooks.load_redis_bytes_module = bytes_hook
    hooks.load_rq_module = rq_hook

    # Get queue - this should use the production _QueueAdapter path
    q = get_queue()

    # Enqueue a job to verify the adapter works
    job = q.enqueue("my.func", "arg1", job_timeout=60, description="test")
    assert job.get_id() == "job-my.func"

    # Restore hooks
    hooks.load_redis_bytes_module = orig_bytes_hook
    hooks.load_rq_module = orig_rq_hook
