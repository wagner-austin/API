from __future__ import annotations

import sys
from types import ModuleType
from typing import Protocol

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedisBytesClient
from pytest import MonkeyPatch

from music_wrapped_api.app import create_app


class _JobStub(Protocol):
    def get_status(self) -> str: ...

    @property
    def is_finished(self) -> bool: ...

    @property
    def meta(self) -> dict[str, JSONValue]: ...

    @property
    def result(self) -> str | None: ...


class _QueuedJob:
    def get_status(self) -> str:
        return "queued"

    @property
    def is_finished(self) -> bool:
        return False

    @property
    def meta(self) -> dict[str, JSONValue]:
        return {}

    @property
    def result(self) -> str | None:
        return None


class _FinishedJob:
    def __init__(self, rid: str) -> None:
        self._rid = rid

    def get_status(self) -> str:
        return "finished"

    @property
    def is_finished(self) -> bool:
        return True

    @property
    def meta(self) -> dict[str, JSONValue]:
        return {"progress": 100}

    @property
    def result(self) -> str | None:
        return self._rid


def test_status_routes(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import music_wrapped_api.routes.wrapped as routes

    def _conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    seq: list[_JobStub] = [_QueuedJob(), _FinishedJob("wrapped:9:2024")]

    def _get(job_id: str, *, connection: FakeRedisBytesClient) -> _JobStub:
        return seq.pop(0)

    monkeypatch.setattr(routes, "_rq_conn", _conn)
    monkeypatch.setattr(routes, "_get_job", _get)

    client = TestClient(create_app(), raise_server_exceptions=False)

    r1 = client.get("/v1/wrapped/status/abc")
    assert r1.status_code == 200
    body1 = load_json_str(r1.text)
    if not isinstance(body1, dict):
        raise AssertionError("response must be an object")
    assert body1.get("status") == "queued"
    assert body1.get("progress") == 0
    assert body1.get("result_id") is None

    r2 = client.get("/v1/wrapped/status/abc")
    assert r2.status_code == 200
    body2 = load_json_str(r2.text)
    if not isinstance(body2, dict):
        raise AssertionError("response must be an object")
    assert body2.get("status") == "finished"
    assert body2.get("progress") == 100
    assert body2.get("result_id") == "wrapped:9:2024"


def test_status_failure_branch(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import music_wrapped_api.routes.wrapped as routes

    def _conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    def _raise(job_id: str, *, connection: FakeRedisBytesClient) -> _JobStub:
        raise RuntimeError("boom")

    monkeypatch.setattr(routes, "_rq_conn", _conn)
    monkeypatch.setattr(routes, "_get_job", _raise)

    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.get("/v1/wrapped/status/abc")
    # Unhandled runtime error propagates as 500
    assert resp.status_code == 500


def test_get_job_dynamic_import(monkeypatch: MonkeyPatch) -> None:
    import music_wrapped_api.routes.wrapped as routes

    # Create a fake rq.job module with a Job class exposing fetch()
    fake_module = ModuleType("rq.job")

    class _Job:
        @staticmethod
        def fetch(job_id: str, *, connection: FakeRedisBytesClient) -> _JobStub:
            return _QueuedJob()

    object.__setattr__(fake_module, "Job", _Job)
    sys.modules["rq.job"] = fake_module

    out = routes._get_job("abc", connection=FakeRedisBytesClient())
    if not isinstance(out, _QueuedJob):
        raise AssertionError("expected _QueuedJob instance")
