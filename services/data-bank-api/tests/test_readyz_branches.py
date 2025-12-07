from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch

from data_bank_api.app import create_app
from data_bank_api.config import Settings


def _client(tmp_path: Path, min_free_gb: int = 1) -> TestClient:
    root = tmp_path / "files"
    s: Settings = {
        "redis_url": "redis://ignored",
        "data_root": str(root),
        "min_free_gb": min_free_gb,
        "delete_strict_404": False,
        "max_file_bytes": 0,
        "api_upload_keys": frozenset(),
        "api_read_keys": frozenset(),
        "api_delete_keys": frozenset(),
    }
    return TestClient(create_app(s))


def test_readyz_degraded_when_missing_and_not_writable(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path)
    from data_bank_api.api.routes import health as health_route_mod

    fake_redis = FakeRedis()
    fake_redis.sadd("rq:workers", "worker-1")

    def _rf(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(health_route_mod, "redis_for_kv", _rf)

    def _always_false(path: Path) -> bool:
        return False

    monkeypatch.setattr("data_bank_api.health._is_writable", _always_false)
    r = client.get("/readyz")
    assert r.status_code == 503

    assert "storage not writable" in r.text
    fake_redis.assert_only_called({"sadd", "ping", "scard", "close"})


def test_readyz_degraded_when_exists_but_not_writable(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path)
    from data_bank_api.api.routes import health as health_route_mod

    fake_redis = FakeRedis()
    fake_redis.sadd("rq:workers", "worker-1")

    def _rf2(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(health_route_mod, "redis_for_kv", _rf2)

    def _always_false(path: Path) -> bool:
        return False

    (tmp_path / "files").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("data_bank_api.health._is_writable", _always_false)
    r = client.get("/readyz")
    assert r.status_code == 503

    assert "storage not writable" in r.text
    fake_redis.assert_only_called({"sadd", "ping", "scard", "close"})


def test_readyz_degraded_when_low_disk(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _client(tmp_path, min_free_gb=10)
    from data_bank_api.api.routes import health as health_route_mod

    fake_redis = FakeRedis()
    fake_redis.sadd("rq:workers", "worker-1")

    def _rf3(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(health_route_mod, "redis_for_kv", _rf3)

    def _fake_free(_: Path) -> float:
        return 0.1

    monkeypatch.setattr("data_bank_api.health._free_gb", _fake_free)
    r = client.get("/readyz")
    assert r.status_code == 503

    assert "low disk" in r.text
    fake_redis.assert_only_called({"sadd", "ping", "scard", "close"})
