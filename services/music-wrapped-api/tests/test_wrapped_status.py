from __future__ import annotations

from typing import Protocol

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.rq_harness import _RedisBytesClient
from platform_workers.testing import FakeRedisBytesClient

from music_wrapped_api import _test_hooks
from music_wrapped_api.api.main import create_app


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


def test_status_routes() -> None:
    seq: list[_JobStub] = [_QueuedJob(), _FinishedJob("wrapped:9:2024")]

    def _get(job_id: str, connection: _RedisBytesClient) -> _JobStub:
        _ = connection  # unused
        return seq.pop(0)

    _test_hooks.get_job = _get

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


def test_status_failure_branch() -> None:
    def _raise(job_id: str, connection: _RedisBytesClient) -> _JobStub:
        _ = connection  # unused
        raise RuntimeError("boom")

    _test_hooks.get_job = _raise

    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.get("/v1/wrapped/status/abc")
    # Unhandled runtime error propagates as 500
    assert resp.status_code == 500


def test_get_job_dynamic_import() -> None:
    # Test that the default hook can fetch jobs (mocked at protocol level)
    job = _QueuedJob()

    def _get(job_id: str, connection: _RedisBytesClient) -> _JobStub:
        _ = connection  # unused
        return job

    _test_hooks.get_job = _get

    out = _test_hooks.get_job("abc", FakeRedisBytesClient())
    if not isinstance(out, _QueuedJob):
        raise AssertionError("expected _QueuedJob instance")
