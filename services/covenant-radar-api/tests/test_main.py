"""Tests for application factory."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from covenant_persistence import ConnectionProtocol, CursorProtocol
from fastapi.testclient import TestClient
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
    """Create test container with fake dependencies."""
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


def test_app_factory_creates_fastapi_app() -> None:
    """Test create_app returns a FastAPI application."""
    container = _create_test_container()
    app = create_app(container)

    assert app.title == "covenant-radar-api"
    assert app.version == "0.1.0"


def test_app_factory_health_endpoints() -> None:
    """Test app factory creates working application with health endpoints."""
    container = _create_test_container()
    client: TestClient = TestClient(create_app(container))

    r1 = client.get("/healthz")
    assert r1.status_code == 200
    assert '"status"' in r1.text
    assert '"ok"' in r1.text

    r2 = client.get("/readyz")
    assert r2.status_code == 200
    assert '"status"' in r2.text


def test_app_factory_includes_crud_routes() -> None:
    """Test app factory includes CRUD routes."""
    container = _create_test_container()
    client: TestClient = TestClient(create_app(container))

    # CRUD routes are registered (will return empty list since no data)
    r_deals = client.get("/deals")
    assert r_deals.status_code == 200
    assert r_deals.text == "[]"

    r_covenants = client.get("/covenants/by-deal/test-deal")
    assert r_covenants.status_code == 200
    assert r_covenants.text == "[]"

    r_measurements = client.get("/measurements/by-deal/test-deal")
    assert r_measurements.status_code == 200
    assert r_measurements.text == "[]"
