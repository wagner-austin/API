from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Protocol

from _pytest.monkeypatch import MonkeyPatch

import platform_workers.rq_harness as rh


class _RedisBytesProto(Protocol):
    def ping(self, **kwargs: str | int | float | bool | None) -> bool: ...
    def close(self) -> None: ...


def _make_rq_mock(fake_module: ModuleType) -> Callable[[str], ModuleType]:
    """Create a mock for __import__ that returns fake_module for 'rq'."""
    real_import: Callable[[str], ModuleType] = __import__

    def _mock(name: str) -> ModuleType:
        if name == "rq":
            return fake_module
        return real_import(name)

    return _mock


def test_rq_runtime_imports_queue_and_worker(monkeypatch: MonkeyPatch) -> None:
    class _FakeJob:
        def get_id(self) -> str:
            return "fake-job-id"

    class _Q:
        def __init__(self, name: str, *, connection: _RedisBytesProto) -> None:
            self.name = name
            self.connection = connection

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

    class _W:
        def __init__(
            self, queues: list[rh._RQQueueInternal], *, connection: _RedisBytesProto
        ) -> None:
            self.queues = queues
            self.connection = connection

        def work(self, *, with_scheduler: bool) -> None:
            pass

    class _FakeRQModule(ModuleType):
        Queue = _Q
        SimpleWorker = _W

    mock_import = _make_rq_mock(_FakeRQModule("rq"))
    monkeypatch.setattr("builtins.__import__", mock_import)

    class _FakeConn:
        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return True

        def close(self) -> None:
            pass

    conn = _FakeConn()
    raw_queue = rh._rq_queue_raw("turkic", connection=conn)
    worker = rh._rq_simple_worker([raw_queue], connection=conn)
    assert type(raw_queue) is _Q
    assert raw_queue.name == "turkic"
    worker.work(with_scheduler=True)


def test_public_rq_queue_factory(monkeypatch: MonkeyPatch) -> None:
    class _FakeJob:
        def get_id(self) -> str:
            return "fake-job-id"

    class _Q:
        def __init__(self, name: str, *, connection: _RedisBytesProto) -> None:
            self.name = name
            self.connection = connection

        def enqueue(
            self,
            func_ref: str,
            *args: rh._JsonValue,
            **kwargs: rh._JsonValue,
        ) -> _FakeJob:
            return _FakeJob()

    class _FakeRQModule(ModuleType):
        Queue = _Q

    mock_import = _make_rq_mock(_FakeRQModule("rq"))
    monkeypatch.setattr("builtins.__import__", mock_import)

    class _FakeConn:
        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return True

        def close(self) -> None:
            pass

    conn = _FakeConn()
    q_adapter = rh.rq_queue("turkic", connection=conn)
    assert type(q_adapter) is rh._RQQueueAdapter
    assert type(q_adapter._inner).__name__ == "_Q"
