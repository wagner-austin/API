"""Tests for health check endpoints."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from covenant_persistence import ConnectionProtocol, CursorProtocol
from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str
from platform_workers.rq_harness import _RedisBytesClient
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

from covenant_radar_api.core.config import Settings
from covenant_radar_api.core.container import ServiceContainer
from covenant_radar_api.main import create_app


class _FakeCursor:
    """Fake cursor for testing."""

    @property
    def rowcount(self) -> int:
        """Return row count."""
        return 0

    def execute(self, query: str, params: tuple[str | int | bool | None, ...] = ()) -> None:
        """Execute query (no-op for tests)."""
        pass

    def fetchone(self) -> tuple[str | int | bool | None, ...] | None:
        """Return None."""
        return None

    def fetchall(self) -> Sequence[tuple[str | int | bool | None, ...]]:
        """Return empty list."""
        return []


class _FakeConnection:
    """Fake connection that satisfies ConnectionProtocol for testing."""

    def __init__(self) -> None:
        """Initialize fake connection."""
        self._closed = False
        self._cursor = _FakeCursor()

    def cursor(self) -> CursorProtocol:
        """Return fake cursor."""
        return self._cursor

    def commit(self) -> None:
        """No-op commit."""
        pass

    def rollback(self) -> None:
        """No-op rollback."""
        pass

    def close(self) -> None:
        """Mark as closed."""
        self._closed = True


class _TestableContainer(ServiceContainer):
    """Container subclass that accepts fake dependencies for testing."""

    pass


def _create_test_container(*, workers: int = 1) -> _TestableContainer:
    """Create test container with fake Redis."""
    settings: Settings = {
        "redis_url": "redis://test:6379/0",
        "database_url": "postgresql://test:test@localhost/test",
    }
    fake_redis = FakeRedis()
    for i in range(workers):
        fake_redis.sadd("rq:workers", f"worker-{i}")

    fake_conn: ConnectionProtocol = _FakeConnection()
    fake_rq: _RedisBytesClient = FakeRedisBytesClient()

    return _TestableContainer(
        settings=settings,
        redis=fake_redis,
        db_conn=fake_conn,
        redis_rq=fake_rq,
        model_path="./test_model.ubj",
        model_output_dir=Path("./models"),
        sector_encoder={"Tech": 0},
        region_encoder={"NA": 0},
    )


def test_healthz_ok() -> None:
    """Test liveness probe returns ok."""
    container = _create_test_container()
    client: TestClient = TestClient(create_app(container))

    r = client.get("/healthz")

    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        raise AssertionError("expected dict")
    body = body_raw
    assert body["status"] == "ok"


def test_readyz_ready() -> None:
    """Test readiness probe returns ready when workers present."""
    container = _create_test_container(workers=1)
    client: TestClient = TestClient(create_app(container))

    r = client.get("/readyz")

    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        raise AssertionError("expected dict")
    body = body_raw
    assert body["status"] == "ready"


def test_readyz_degraded_no_worker() -> None:
    """Test readiness probe returns degraded when no workers."""
    container = _create_test_container(workers=0)
    client: TestClient = TestClient(create_app(container))

    r = client.get("/readyz")

    assert r.status_code == 503
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        raise AssertionError("expected dict")
    body = body_raw
    assert body["status"] == "degraded"
    assert body["reason"] == "no-worker"
