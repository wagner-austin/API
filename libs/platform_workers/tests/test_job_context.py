from __future__ import annotations

from typing import NamedTuple, TypeGuard

from platform_core.job_events import (
    JobCompletedV1,
    JobEventV1,
    JobFailedV1,
    JobProgressV1,
    decode_job_event,
)

from platform_workers.job_context import JobContext, make_job_context
from platform_workers.redis import RedisStrProto


class _Published(NamedTuple):
    channel: str
    payload: str


class _FakeRedis(RedisStrProto):
    def __init__(self) -> None:
        self.messages: list[_Published] = []
        self._hashes: dict[str, dict[str, str]] = {}
        self._strings: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self, key: str, value: str) -> bool:
        self._strings[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._strings.get(key)

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._hashes[key] = dict(mapping)
        return 1

    def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(field)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def publish(self, channel: str, message: str) -> int:
        self.messages.append(_Published(channel, message))
        return 1

    def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    def sadd(self, key: str, member: str) -> int:
        bucket = self._sets.setdefault(key, set())
        before = len(bucket)
        bucket.add(member)
        return 1 if len(bucket) > before else 0

    def sismember(self, key: str, member: str) -> bool:
        return member in self._sets.get(key, set())

    def delete(self, key: str) -> int:
        removed = 0
        if key in self._strings:
            del self._strings[key]
            removed += 1
        if key in self._hashes:
            del self._hashes[key]
            removed += 1
        if key in self._sets:
            del self._sets[key]
            removed += 1
        return removed

    def expire(self, key: str, time: int) -> bool:
        return key in self._strings or key in self._hashes or key in self._sets

    def close(self) -> None:
        self._hashes.clear()
        self._strings.clear()
        self._sets.clear()


def _last(redis: _FakeRedis) -> JobEventV1:
    assert redis.messages
    _, payload = redis.messages[-1]
    return decode_job_event(payload)


def _is_progress(event: JobEventV1) -> TypeGuard[JobProgressV1]:
    return "progress" in event


def _is_completed(event: JobEventV1) -> TypeGuard[JobCompletedV1]:
    return "result_id" in event


def _is_failed(event: JobEventV1) -> TypeGuard[JobFailedV1]:
    return "error_kind" in event


def test_job_context_publishes_started_and_progress() -> None:
    redis = _FakeRedis()
    ctx: JobContext = make_job_context(
        redis=redis,
        domain="turkic",
        events_channel="turkic:events",
        job_id="job-1",
        user_id=7,
        queue_name="queue-x",
    )

    ctx.publish_started()
    started = _last(redis)
    assert started == {
        "type": "turkic.job.started.v1",
        "domain": "turkic",
        "job_id": "job-1",
        "user_id": 7,
        "queue": "queue-x",
    }

    ctx.publish_progress(50, "halfway")
    progress = _last(redis)
    assert _is_progress(progress)
    assert progress["progress"] == 50
    assert progress["message"] == "halfway"

    ctx.publish_progress(75, None)
    progress_no_msg = _last(redis)
    assert _is_progress(progress_no_msg)
    assert "message" not in progress_no_msg


def test_job_context_publishes_completion_and_failure() -> None:
    redis = _FakeRedis()
    ctx: JobContext = make_job_context(
        redis=redis,
        domain="transcript",
        events_channel="transcript:events",
        job_id="job-2",
        user_id=11,
        queue_name="queue-y",
    )

    ctx.publish_completed("result-1", 4096)
    completed = _last(redis)
    assert _is_completed(completed)
    assert completed["result_id"] == "result-1"
    assert completed["result_bytes"] == 4096

    ctx.publish_failed("system", "boom")
    failed = _last(redis)
    assert _is_failed(failed)
    assert failed["error_kind"] == "system"
    assert failed["message"] == "boom"

    ctx.publish_failed("user", "bad input")
    failed_user = _last(redis)
    assert _is_failed(failed_user)
    assert failed_user["error_kind"] == "user"
