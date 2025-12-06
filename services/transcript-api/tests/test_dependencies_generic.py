from __future__ import annotations

import pytest

from transcript_api.dependencies import get_queue, get_redis, get_request_logger
from transcript_api.types import JsonValue


class _RedisStub:
    def __init__(self) -> None:
        self.closed = False

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self, key: str, value: str) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        return 1

    def hget(self, key: str, field: str) -> str | None:
        return None

    def hgetall(self, key: str) -> dict[str, str]:
        return {}

    def publish(self, channel: str, message: str) -> int:
        return 1

    def scard(self, key: str) -> int:
        return 0

    def sadd(self, key: str, member: str) -> int:
        return 1

    def sismember(self, key: str, member: str) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


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
    redis = _RedisStub()

    def _redis_for_kv(url: str) -> _RedisStub:
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


def test_get_queue_builds_stub_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _RedisStub()
    queue = _QueueStub()

    def _raw(url: str) -> _RedisStub:
        assert url == "redis://example/1"
        return redis

    def _rq_queue(name: str, *, connection: _RedisStub) -> _QueueStub:
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


def test_get_request_logger_returns_named_logger() -> None:
    logger = get_request_logger()
    logger.info("ok", extra={})
