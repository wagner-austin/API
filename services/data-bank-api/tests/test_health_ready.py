from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch

from data_bank_api.app import create_app
from data_bank_api.config import Settings


def _with_tmp_root(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    *,
    workers: int = 1,
) -> TestClient:
    root = tmp_path / "files"
    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    # Patch redis_for_kv to return a fake client
    from data_bank_api.api.routes import health as health_route_mod

    def _fake(url: str) -> FakeRedis:
        r = FakeRedis()
        for i in range(workers):
            r.sadd("rq:workers", f"worker-{i}")
        return r

    monkeypatch.setattr(health_route_mod, "redis_for_kv", _fake)
    return TestClient(create_app(s))


def test_healthz_ok(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _with_tmp_root(tmp_path, monkeypatch)
    r = client.get("/healthz")
    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        raise AssertionError("expected dict")
    body = body_raw
    assert body["status"] == "ok"


def test_readyz_ready_when_writable(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _with_tmp_root(tmp_path, monkeypatch, workers=1)
    r = client.get("/readyz")
    assert r.status_code == 200
    body2_raw = load_json_str(r.text)
    if type(body2_raw) is not dict:
        raise AssertionError("expected dict")
    body2 = body2_raw
    assert body2["status"] == "ready"


def test_readyz_degraded_no_worker(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _with_tmp_root(tmp_path, monkeypatch, workers=0)
    r = client.get("/readyz")
    assert r.status_code == 503
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        raise AssertionError("expected dict")
    body = body_raw
    assert body["status"] == "degraded"
    assert body["reason"] == "no-worker"


def test_readyz_missing_storage_creates_and_is_ready(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # With new semantics, missing storage is created and treated as writable
    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(tmp_path / "missing"),
        "min_free_gb": 0,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    from data_bank_api.api.routes import health as health_route_mod

    def _fake2(url: str) -> FakeRedis:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")
        return r

    monkeypatch.setattr(health_route_mod, "redis_for_kv", _fake2)
    client = TestClient(create_app(s))
    r = client.get("/readyz")
    assert r.status_code == 200


def test_readyz_degraded_low_disk(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Build settings with a positive min_free_gb to trigger low disk
    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(tmp_path / "files"),
        "min_free_gb": 10,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    from data_bank_api.api.routes import health as health_route_mod

    def _fake_redis(url: str) -> FakeRedis:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")
        return r

    monkeypatch.setattr(health_route_mod, "redis_for_kv", _fake_redis)
    client = TestClient(create_app(s))
    # Patch _free_gb to simulate low disk
    from data_bank_api import health as health_mod

    def _fake_free_gb(p: Path) -> float:
        return 0.0

    monkeypatch.setattr(health_mod, "_free_gb", _fake_free_gb)
    r = client.get("/readyz")
    assert r.status_code == 503
    obj_raw = load_json_str(r.text)
    if type(obj_raw) is not dict:
        raise AssertionError("expected dict")
    obj: dict[str, JSONValue] = obj_raw
    assert obj.get("reason") == "low disk"
