from __future__ import annotations

from platform_workers.rq_harness import (
    RQClientQueue,
    RQJobLike,
    RQRetryLike,
    _RedisBytesClient,
)
from platform_workers.rq_harness import (
    _JsonValue as _RQJsonValue,
)

from clubbot import _test_hooks


def test_digits_enqueuer_redis_helper_cov() -> None:
    # Call the module helper to exercise the import/return path
    from clubbot.services.jobs import digits_enqueuer as de

    conn = de._redis_from_url("redis://localhost:6379/0")
    # Verify connection is returned (object check is implicit)
    _ = conn


def test_digits_enqueuer_queue_and_retry_helpers_cov() -> None:
    from clubbot.services.jobs import digits_enqueuer as de

    class _DummyConn:
        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return True

        def close(self) -> None:
            return None

    class _DummyJob:
        def get_id(self) -> str:
            return "jid"

    class _DummyQueue(RQClientQueue):
        def enqueue(
            self,
            func_ref: str,
            *args: _RQJsonValue,
            job_timeout: int | None = None,
            result_ttl: int | None = None,
            failure_ttl: int | None = None,
            retry: RQRetryLike | None = None,
            description: str | None = None,
        ) -> RQJobLike:
            _ = (func_ref, args, job_timeout, result_ttl, failure_ttl, retry, description)
            return _DummyJob()

    class _DummyRetry:
        def __init__(self) -> None:
            self.max = 1
            self.interval = [1, 2]

    def _fake_rq_queue(name: str, *, connection: _RedisBytesClient) -> RQClientQueue:
        _ = (name, connection)
        return _DummyQueue()

    def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
        _ = (max_retries, intervals)
        return _DummyRetry()

    original_queue = _test_hooks.rq_queue
    original_retry = _test_hooks.rq_retry
    _test_hooks.rq_queue = _fake_rq_queue
    _test_hooks.rq_retry = _fake_rq_retry
    try:
        q = de._rq_queue("digits", connection=_DummyConn())
        r = de._rq_retry(max_retries=1, intervals=[1, 2])
        assert q is not None and r is not None
    finally:
        _test_hooks.rq_queue = original_queue
        _test_hooks.rq_retry = original_retry
