from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch

from music_wrapped_api.app import create_app


def _client(monkeypatch: MonkeyPatch, *, workers: int) -> TestClient:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    from music_wrapped_api import app as app_mod

    def _fake(url: str) -> FakeRedis:
        r = FakeRedis()
        for i in range(workers):
            r.sadd("rq:workers", f"worker-{i}")
        return r

    monkeypatch.setattr(app_mod, "redis_for_kv", _fake)
    return TestClient(create_app())


def test_readyz_degraded_without_worker(monkeypatch: MonkeyPatch) -> None:
    client = _client(monkeypatch, workers=0)
    r = client.get("/readyz")
    assert r.status_code == 503
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        pytest.fail("expected dict response body")
    body: dict[str, JSONValue] = body_raw
    assert body.get("status") == "degraded"
    assert body.get("reason") == "no-worker"


def test_readyz_ready_with_worker(monkeypatch: MonkeyPatch) -> None:
    client = _client(monkeypatch, workers=1)
    r = client.get("/readyz")
    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        pytest.fail("expected dict response body")
    body: dict[str, JSONValue] = body_raw
    assert body.get("status") == "ready"
