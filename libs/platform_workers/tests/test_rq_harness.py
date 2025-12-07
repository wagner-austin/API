from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Protocol

import pytest
from _pytest.monkeypatch import MonkeyPatch

from platform_workers.rq_harness import (
    WorkerConfig,
    get_current_job,
    rq_queue,
    rq_retry,
    run_rq_worker,
)


def _make_rq_mock(fake_module: ModuleType) -> Callable[[str], ModuleType]:
    """Create a mock for __import__ that returns fake_module for 'rq'."""
    real_import: Callable[[str], ModuleType] = __import__

    def _mock(name: str) -> ModuleType:
        if name == "rq":
            return fake_module
        return real_import(name)

    return _mock


class _RedisBytesProto(Protocol):
    def ping(self, **kwargs: str | int | float | bool | None) -> bool: ...
    def close(self) -> None: ...


def test_run_rq_worker_invokes_worker(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeConn:
        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return True

    class _Q:
        pass

    class _W:
        def work(self, *, with_scheduler: bool) -> None:
            calls.append(f"work:{with_scheduler}")

    def _fake_queue(_name: str, connection: _FakeConn) -> _Q:
        assert type(connection) is _FakeConn
        return _Q()

    def _fake_worker(_queues: list[_Q], connection: _FakeConn) -> _W:
        assert type(connection) is _FakeConn
        assert len(_queues) == 1
        assert type(_queues[0]) is _Q
        return _W()

    monkeypatch.setattr("platform_workers.rq_harness._rq_queue_raw", _fake_queue, raising=True)
    monkeypatch.setattr("platform_workers.rq_harness._rq_simple_worker", _fake_worker, raising=True)

    def _fake_conn_factory(_url: str) -> _FakeConn:
        return _FakeConn()

    monkeypatch.setattr(
        "platform_workers.rq_harness.redis_raw_for_rq",
        _fake_conn_factory,
        raising=True,
    )

    cfg: WorkerConfig = {
        "redis_url": "redis://x",
        "queue_name": "turkic",
        "events_channel": "turkic:events",
    }
    run_rq_worker(cfg)
    assert calls == ["work:True"]


def test_rq_queue_enqueue_wrapper(monkeypatch: MonkeyPatch) -> None:
    from platform_workers import rq_harness as rh

    class _FakeJob:
        def __init__(self, job_id: str) -> None:
            self._id = job_id

        def get_id(self) -> str:
            return self._id

    class _FakeQueue:
        def __init__(self, name: str, *, connection: _RedisBytesProto) -> None:
            self.name = name
            self.connection = connection

        def enqueue(
            self,
            func_ref: str,
            *args: rh._JsonValue,
            **kwargs: rh._JsonValue,
        ) -> _FakeJob:
            return _FakeJob(f"job-{func_ref}")

    class _FakeRQModule(ModuleType):
        Queue = _FakeQueue

    mock_import = _make_rq_mock(_FakeRQModule("rq"))
    monkeypatch.setattr("builtins.__import__", mock_import)

    class _FakeConn:
        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return True

        def close(self) -> None:
            pass

    conn = _FakeConn()
    q_adapter = rq_queue("test", connection=conn)
    job = q_adapter.enqueue("my_func", "arg1", job_timeout=60, description="test job")
    assert job.get_id() == "job-my_func"


def test_rq_worker_work_wrapper(monkeypatch: MonkeyPatch) -> None:
    from platform_workers import rq_harness as rh

    work_called = False

    class _FakeJob:
        def get_id(self) -> str:
            return "fake-job-id"

    class _FakeQueue:
        def enqueue(
            self,
            func_ref: str,
            *args: rh._JsonValue,
            job_timeout: int | None = None,
            result_ttl: int | None = None,
            failure_ttl: int | None = None,
            retry: rh.RQRetryLike | None = None,
            description: str | None = None,
        ) -> _FakeJob:
            return _FakeJob()

    class _FakeWorker:
        def __init__(
            self, queues: list[rh._RQQueueInternal], *, connection: _RedisBytesProto
        ) -> None:
            self.queues = queues
            self.connection = connection

        def work(self, *, with_scheduler: bool) -> None:
            nonlocal work_called
            work_called = True
            assert with_scheduler

    class _FakeRQModule(ModuleType):
        SimpleWorker = _FakeWorker

    mock_import = _make_rq_mock(_FakeRQModule("rq"))
    monkeypatch.setattr("builtins.__import__", mock_import)

    class _FakeConn:
        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return True

        def close(self) -> None:
            pass

    conn = _FakeConn()
    q: rh._RQQueueInternal = _FakeQueue()
    worker = rh._rq_simple_worker([q], connection=conn)
    worker.work(with_scheduler=True)
    assert work_called


def test_get_current_job_returns_none_outside_worker(monkeypatch: MonkeyPatch) -> None:
    class _FakeRQModule(ModuleType):
        @staticmethod
        def get_current_job() -> None:
            return None

    mock_import = _make_rq_mock(_FakeRQModule("rq"))
    monkeypatch.setattr("builtins.__import__", mock_import)
    result = get_current_job()
    assert result is None


def test_get_current_job_returns_job_inside_worker(monkeypatch: MonkeyPatch) -> None:
    class _FakeJob:
        origin: str | None = "test-queue"

        def get_id(self) -> str:
            return "job-123"

    class _FakeRQModule(ModuleType):
        @staticmethod
        def get_current_job() -> _FakeJob:
            return _FakeJob()

    mock_import = _make_rq_mock(_FakeRQModule("rq"))
    monkeypatch.setattr("builtins.__import__", mock_import)
    result = get_current_job()
    if result is None:
        pytest.fail("expected current job")
    assert result.get_id() == "job-123"
    assert result.origin == "test-queue"


def test_rq_retry_creates_retry_object(monkeypatch: MonkeyPatch) -> None:
    class _FakeRetry:
        def __init__(self, *, max: int, interval: list[int]) -> None:
            self.max = max
            self.interval = interval

    class _FakeRQModule(ModuleType):
        Retry = _FakeRetry

    mock_import = _make_rq_mock(_FakeRQModule("rq"))
    monkeypatch.setattr("builtins.__import__", mock_import)

    retry = rq_retry(max_retries=3, intervals=[10, 30, 60])
    assert type(retry) is _FakeRetry
    assert retry.max == 3
    assert retry.interval == [10, 30, 60]
