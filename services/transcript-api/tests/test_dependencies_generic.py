from __future__ import annotations

import pytest
from platform_workers.testing import FakeRedis

from transcript_api.dependencies import get_queue, get_redis, get_request_logger
from transcript_api.types import JsonValue


class _JobStub:
    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def get_id(self) -> str:
        return self._job_id


class _QueueStub:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, tuple[JsonValue, ...]]] = []

    def enqueue(self, func_ref: str, *args: JsonValue, **kwargs: JsonValue) -> _JobStub:
        self.enqueued.append((func_ref, args))
        return _JobStub("job-id")


def test_get_redis_yields_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis()

    def _redis_for_kv(url: str) -> FakeRedis:
        assert url == "redis://example/0"
        return redis

    monkeypatch.setenv("REDIS_URL", "redis://example/0")
    monkeypatch.setattr("transcript_api.dependencies.redis_for_kv", _redis_for_kv)

    gen = get_redis()
    client = next(gen)
    assert client is redis
    with pytest.raises(StopIteration):
        next(gen)
    assert redis.closed
    redis.assert_only_called({"close"})


def test_get_queue_builds_stub_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis()
    queue = _QueueStub()

    def _raw(url: str) -> FakeRedis:
        assert url == "redis://example/1"
        return redis

    def _rq_queue(name: str, *, connection: FakeRedis) -> _QueueStub:
        assert name == "transcript"
        assert connection is redis
        return queue

    monkeypatch.setenv("REDIS_URL", "redis://example/1")
    monkeypatch.setattr("transcript_api.dependencies.redis_raw_for_rq", _raw)
    monkeypatch.setattr("transcript_api.dependencies.rq_queue", _rq_queue)

    q = get_queue()
    job = q.enqueue("transcript_api.jobs.process_stt", {"url": "u"})
    assert job.get_id() == "job-id"
    assert queue.enqueued
    redis.assert_only_called(set())


def test_get_request_logger_returns_named_logger() -> None:
    logger = get_request_logger()
    logger.info("ok", extra={})
