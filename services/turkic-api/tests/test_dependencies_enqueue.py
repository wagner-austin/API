from __future__ import annotations

import pytest
from platform_workers.rq_harness import RQClientQueue, RQRetryLike, _JsonValue
from platform_workers.testing import FakeQueue, FakeRedis
from typing_extensions import TypedDict

from turkic_api.api.dependencies import get_queue


class _EnqueueCall(TypedDict):
    """Captured enqueue call with explicit RQ parameters."""

    func_ref: str
    args: tuple[_JsonValue, ...]
    job_timeout: int | None
    result_ttl: int | None
    failure_ttl: int | None
    retry: RQRetryLike | None
    description: str | None


def test_queue_enqueue_calls_underlying(monkeypatch: pytest.MonkeyPatch) -> None:
    import turkic_api.api.dependencies as deps

    created_queue: list[FakeQueue] = []
    created_redis: list[FakeRedis] = []

    def _fake_redis_raw(url: str) -> FakeRedis:
        r = FakeRedis()
        created_redis.append(r)
        return r

    def _fake_rq_queue(name: str, connection: FakeRedis) -> RQClientQueue:
        queue = FakeQueue(job_id="job-id")
        created_queue.append(queue)
        return queue

    import platform_workers.redis as redis_mod

    monkeypatch.setattr(redis_mod, "redis_raw_for_rq", _fake_redis_raw)
    monkeypatch.setattr(deps, "redis_raw_for_rq", _fake_redis_raw)
    monkeypatch.setattr(deps, "rq_queue", _fake_rq_queue)

    # Build queue using settings dependency
    q = get_queue(deps.get_settings())

    # Enqueue using a string reference with valid RQ parameters
    res = q.enqueue("pkg.add", 1, 2, job_timeout=30, description="test job")
    if res is None:
        pytest.fail("expected enqueue result")
    queue = created_queue[0]
    assert queue.jobs
    job = queue.jobs[0]
    assert job.func == "pkg.add"
    assert job.args[:2] == (1, 2)
    assert job.job_timeout == 30
    assert job.description == "test job"
    assert job.result_ttl is None
    assert job.failure_ttl is None
    for r in created_redis:
        r.assert_only_called(set())
