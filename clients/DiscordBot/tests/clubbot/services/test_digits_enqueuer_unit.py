from __future__ import annotations

import logging
from typing import TypedDict

import pytest
from platform_workers.rq_harness import (
    RQClientQueue,
    RQRetryLike,
    _RedisBytesClient,
)
from platform_workers.rq_harness import (
    _JsonValue as _RQJsonValue,
)

from clubbot.services.jobs.digits_enqueuer import RQDigitsEnqueuer


class _Retry:
    def __init__(self, *, max: int, interval: list[int]) -> None:
        self.max = max
        self.interval = interval


class _FakeRedisConnection:
    pass


class _CallState(TypedDict, total=False):
    queue_name: str
    func_name: str
    payload: dict[str, int | str | float | bool | None]
    opts: dict[str, int | str | _Retry]
    connection: _FakeRedisConnection | None


CALLS: _CallState = {}


class _Job:
    def __init__(self, jid: str) -> None:
        self._id = jid

    def get_id(self) -> str:
        return self._id


class _Queue(RQClientQueue):
    def __init__(
        self, name: str, connection: _RedisBytesClient | _FakeRedisConnection | None = None
    ) -> None:
        CALLS["queue_name"] = name
        CALLS["connection"] = connection if isinstance(connection, _FakeRedisConnection) else None

    def enqueue(
        self,
        func_ref: str,
        *args: _RQJsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> _Job:
        CALLS["func_name"] = func_ref
        pl: dict[str, int | str | float | bool | None] = {}
        if len(args) > 0 and isinstance(args[0], dict):
            for k, v in args[0].items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    pl[str(k)] = v
        CALLS["payload"] = pl
        CALLS["opts"] = {
            "job_timeout": int(job_timeout or 0),
            "result_ttl": int(result_ttl or 0),
            "failure_ttl": int(failure_ttl or 0),
            "retry": (retry if isinstance(retry, _Retry) else _Retry(max=0, interval=[])),
            "description": str(description or ""),
        }
        return _Job("jid1")


def _fake_from_url(_: str) -> _FakeRedisConnection:
    return _FakeRedisConnection()


def _fake_rq_queue(
    name: str, *, connection: _RedisBytesClient | _FakeRedisConnection
) -> RQClientQueue:
    return _Queue(name, connection)


def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
    return _Retry(max=max_retries, interval=intervals)


def test_digits_enqueuer_builds_job_with_expected_args(monkeypatch: pytest.MonkeyPatch) -> None:
    global CALLS
    CALLS = {}

    # Patch module-level helpers to avoid importing rq directly
    monkeypatch.setattr(
        "clubbot.services.jobs.digits_enqueuer._redis_from_url", _fake_from_url, raising=True
    )
    monkeypatch.setattr(
        "clubbot.services.jobs.digits_enqueuer._rq_queue",
        _fake_rq_queue,
        raising=True,
    )
    monkeypatch.setattr(
        "clubbot.services.jobs.digits_enqueuer._rq_retry",
        _fake_rq_retry,
        raising=True,
    )

    enq = RQDigitsEnqueuer(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        job_timeout_s=25200,
        result_ttl_s=86400,
        failure_ttl_s=604800,
        retry_max=2,
        retry_intervals_s=(60, 300),
    )
    job_id = enq.enqueue_train(
        request_id="r1",
        user_id=9,
        model_id="m",
        epochs=5,
        batch_size=32,
        lr=0.001,
        seed=42,
        augment=True,
        notes="hello",
    )
    assert job_id == "jid1"
    assert "queue_name" in CALLS
    assert CALLS["queue_name"] == "digits"
    assert "connection" in CALLS
    assert type(CALLS["connection"]) is _FakeRedisConnection
    assert "func_name" in CALLS
    assert CALLS["func_name"] == "handwriting_ai.jobs.digits.process_train_job"
    assert "payload" in CALLS
    payload = CALLS["payload"]
    assert type(payload) is dict
    assert payload["type"] == "digits.train.v1"
    assert payload["request_id"] == "r1"
    assert payload["user_id"] == 9
    assert payload["model_id"] == "m"
    assert payload["epochs"] == 5
    assert payload["batch_size"] == 32
    assert payload["lr"] == 0.001
    assert payload["seed"] == 42
    assert payload["augment"] is True
    assert "opts" in CALLS
    opts = CALLS["opts"]
    retry_obj = opts["retry"]
    assert type(retry_obj) is _Retry
    assert retry_obj.max == 2
    assert retry_obj.interval == [60, 300]
    assert opts["job_timeout"] == 25200
    assert opts["result_ttl"] == 86400
    assert opts["failure_ttl"] == 604800
    assert opts["description"] == "digits:r1"
    # No global rq module patching required


logger = logging.getLogger(__name__)
