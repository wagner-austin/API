"""Tests for service container dependency injection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from covenant_ml.trainer import save_model, train_model
from covenant_ml.types import TrainConfig
from covenant_persistence import ConnectionProtocol, CursorProtocol
from numpy.typing import NDArray
from platform_workers.rq_harness import _RedisBytesClient
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

from covenant_radar_api.core import ServiceContainer, Settings, _test_hooks


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

    @property
    def closed(self) -> bool:
        """Check if closed."""
        return self._closed


class _TestableContainer(ServiceContainer):
    """Container subclass that accepts fake dependencies for testing."""

    pass


def test_container_init_stores_dependencies() -> None:
    """Test container constructor stores settings and redis."""
    settings: Settings = {
        "redis_url": "redis://test:6379/0",
        "database_url": "postgresql://test:test@localhost/test",
    }
    fake_redis = FakeRedis()
    fake_conn: ConnectionProtocol = _FakeConnection()
    fake_rq: _RedisBytesClient = FakeRedisBytesClient()

    container = _TestableContainer(
        settings=settings,
        redis=fake_redis,
        db_conn=fake_conn,
        redis_rq=fake_rq,
        model_path="./test_model.ubj",
        model_output_dir=Path("./models"),
        sector_encoder={"Tech": 0},
        region_encoder={"NA": 0},
    )

    assert container.settings == settings
    assert container.redis is fake_redis
    assert container.db_conn is fake_conn


def test_container_close_closes_all_resources() -> None:
    """Test close() calls close on redis and db_conn."""
    settings: Settings = {
        "redis_url": "redis://test:6379/0",
        "database_url": "postgresql://test:test@localhost/test",
    }
    fake_redis = FakeRedis()
    fake_conn = _FakeConnection()
    fake_rq = FakeRedisBytesClient()

    container = _TestableContainer(
        settings=settings,
        redis=fake_redis,
        db_conn=fake_conn,
        redis_rq=fake_rq,
        model_path="./test_model.ubj",
        model_output_dir=Path("./models"),
        sector_encoder={"Tech": 0},
        region_encoder={"NA": 0},
    )
    container.close()

    fake_redis.assert_only_called({"close"})
    assert fake_conn.closed is True
    assert fake_rq.closed is True


def test_container_exported_from_core() -> None:
    """Test ServiceContainer is exported from core package."""
    from covenant_radar_api.core import ServiceContainer as ExportedContainer

    assert ExportedContainer is ServiceContainer


def _make_test_container() -> _TestableContainer:
    """Create test container with all dependencies."""
    settings: Settings = {
        "redis_url": "redis://test:6379/0",
        "database_url": "postgresql://test:test@localhost/test",
    }
    fake_redis = FakeRedis()
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


def test_container_deal_repo_returns_repository() -> None:
    """Test deal_repo returns a DealRepository."""
    container = _make_test_container()
    repo = container.deal_repo()

    # Type is DealRepository - verify we got a valid repository
    assert repo is not None


def test_container_covenant_repo_returns_repository() -> None:
    """Test covenant_repo returns a CovenantRepository."""
    container = _make_test_container()
    repo = container.covenant_repo()

    # Type is CovenantRepository - verify we got a valid repository
    assert repo is not None


def test_container_measurement_repo_returns_repository() -> None:
    """Test measurement_repo returns a MeasurementRepository."""
    container = _make_test_container()
    repo = container.measurement_repo()

    # Type is MeasurementRepository - verify we got a valid repository
    assert repo is not None


def test_container_covenant_result_repo_returns_repository() -> None:
    """Test covenant_result_repo returns a CovenantResultRepository."""
    container = _make_test_container()
    repo = container.covenant_result_repo()

    # Type is CovenantResultRepository - verify we got a valid repository
    assert repo is not None


def test_container_rq_queue_returns_queue() -> None:
    """Test rq_queue returns an RQClientQueue."""
    container = _make_test_container()
    queue = container.rq_queue()

    # Verify we got a valid queue
    assert queue is not None


def test_container_get_model_info_returns_info() -> None:
    """Test get_model_info returns model info."""
    container = _make_test_container()
    info = container.get_model_info()

    assert info["model_id"] == "default"
    assert info["model_path"] == "./test_model.ubj"
    assert info["is_loaded"] is False


def test_container_get_sector_encoder_returns_dict() -> None:
    """Test get_sector_encoder returns encoder dict."""
    container = _make_test_container()
    encoder = container.get_sector_encoder()

    assert encoder == {"Tech": 0}


def test_container_get_region_encoder_returns_dict() -> None:
    """Test get_region_encoder returns encoder dict."""
    container = _make_test_container()
    encoder = container.get_region_encoder()

    assert encoder == {"NA": 0}


def test_container_get_model_output_dir_returns_path() -> None:
    """Test get_model_output_dir returns path."""
    container = _make_test_container()
    output_dir = container.get_model_output_dir()

    assert output_dir == Path("./models")


def test_container_get_model_loads_and_caches_model(tmp_path: Path) -> None:
    """Test get_model loads model from file and caches it."""
    # Create a real model file
    x_train: NDArray[np.float64] = np.zeros((4, 8), dtype=np.float64)
    x_train[0, 0] = 2.0
    x_train[1, 0] = 3.0
    x_train[2, 0] = 5.0
    x_train[3, 0] = 6.0

    y_train: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    y_train[2] = 1
    y_train[3] = 1

    config = TrainConfig(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=10,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
    )
    model = train_model(x_train, y_train, config)
    model_path = tmp_path / "test_model.ubj"
    save_model(model, str(model_path))

    settings: Settings = {
        "redis_url": "redis://test:6379/0",
        "database_url": "postgresql://test:test@localhost/test",
    }
    fake_redis = FakeRedis()
    fake_conn: ConnectionProtocol = _FakeConnection()
    fake_rq: _RedisBytesClient = FakeRedisBytesClient()

    container = _TestableContainer(
        settings=settings,
        redis=fake_redis,
        db_conn=fake_conn,
        redis_rq=fake_rq,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Tech": 0},
        region_encoder={"NA": 0},
    )

    # Initially not loaded
    assert container.get_model_info()["is_loaded"] is False

    # Load model
    loaded_model = container.get_model()
    assert loaded_model is not None

    # Now should be marked as loaded
    assert container.get_model_info()["is_loaded"] is True

    # Second call should return cached model
    second_call = container.get_model()
    assert second_call is loaded_model


def test_from_settings_uses_injected_factories() -> None:
    """Test from_settings uses test hook factories when set."""
    settings: Settings = {
        "redis_url": "redis://from-settings-test:6379/0",
        "database_url": "postgresql://from-settings-test:test@localhost/test",
    }
    fake_redis = FakeRedis()
    fake_conn = _FakeConnection()
    fake_rq = FakeRedisBytesClient()

    # Track which URLs were passed to factories
    redis_urls: list[str] = []
    connection_dsns: list[str] = []
    redis_rq_urls: list[str] = []

    def _redis_factory(url: str) -> FakeRedis:
        redis_urls.append(url)
        return fake_redis

    def _connection_factory(dsn: str) -> _FakeConnection:
        connection_dsns.append(dsn)
        return fake_conn

    def _redis_rq_factory(url: str) -> FakeRedisBytesClient:
        redis_rq_urls.append(url)
        return fake_rq

    # Set test hooks
    orig_redis = _test_hooks.redis_factory
    orig_conn = _test_hooks.connection_factory
    orig_rq = _test_hooks.redis_rq_factory

    _test_hooks.redis_factory = _redis_factory
    _test_hooks.connection_factory = _connection_factory
    _test_hooks.redis_rq_factory = _redis_rq_factory

    # Create container via from_settings
    container = ServiceContainer.from_settings(
        settings,
        model_path="./test_model.ubj",
        model_output_dir=Path("./models"),
        sector_encoder={"Tech": 0},
        region_encoder={"NA": 0},
    )

    # Restore hooks
    _test_hooks.redis_factory = orig_redis
    _test_hooks.connection_factory = orig_conn
    _test_hooks.redis_rq_factory = orig_rq

    # Verify factories were called with correct URLs
    assert redis_urls == ["redis://from-settings-test:6379/0"]
    assert connection_dsns == ["postgresql://from-settings-test:test@localhost/test"]
    assert redis_rq_urls == ["redis://from-settings-test:6379/0"]

    # Verify container has the injected dependencies
    assert container.redis is fake_redis
    assert container.db_conn is fake_conn


def test_from_settings_with_default_encoders() -> None:
    """Test from_settings uses empty dicts when encoders not provided."""
    settings: Settings = {
        "redis_url": "redis://test:6379/0",
        "database_url": "postgresql://test:test@localhost/test",
    }
    fake_redis = FakeRedis()
    fake_conn = _FakeConnection()
    fake_rq = FakeRedisBytesClient()

    def _redis_factory(url: str) -> FakeRedis:
        return fake_redis

    def _connection_factory(dsn: str) -> _FakeConnection:
        return fake_conn

    def _redis_rq_factory(url: str) -> FakeRedisBytesClient:
        return fake_rq

    # Set test hooks
    orig_redis = _test_hooks.redis_factory
    orig_conn = _test_hooks.connection_factory
    orig_rq = _test_hooks.redis_rq_factory

    _test_hooks.redis_factory = _redis_factory
    _test_hooks.connection_factory = _connection_factory
    _test_hooks.redis_rq_factory = _redis_rq_factory

    # Create container without providing encoders
    container = ServiceContainer.from_settings(settings)

    # Restore hooks
    _test_hooks.redis_factory = orig_redis
    _test_hooks.connection_factory = orig_conn
    _test_hooks.redis_rq_factory = orig_rq

    # Verify defaults
    assert container.get_sector_encoder() == {}
    assert container.get_region_encoder() == {}
    assert container.get_model_output_dir() == Path("./models")


def test_get_psycopg_connect_without_hook_attempts_real_connection() -> None:
    """Test _get_psycopg_connect calls psycopg.connect when hook is None."""
    import pytest

    from covenant_radar_api.core.container import _get_psycopg_connect

    # Get the actual psycopg error type with proper typing
    psycopg = __import__("psycopg")
    operational_error: type[Exception] = psycopg.OperationalError

    # Ensure hook is None
    orig = _test_hooks.connection_factory
    _test_hooks.connection_factory = None

    # Calling with invalid DSN should raise psycopg OperationalError
    # Use "host= dbname=x" which fails fast without DNS resolution
    try:
        with pytest.raises(operational_error):
            _get_psycopg_connect("host= dbname=x")
    finally:
        _test_hooks.connection_factory = orig


def test_get_redis_for_kv_without_hook_attempts_real_connection() -> None:
    """Test _get_redis_for_kv calls redis_for_kv when hook is None."""
    import pytest
    from platform_workers.redis import _load_redis_error_class

    from covenant_radar_api.core.container import _get_redis_for_kv

    redis_error: type[BaseException] = _load_redis_error_class()

    # Ensure hook is None
    orig = _test_hooks.redis_factory
    _test_hooks.redis_factory = None

    # Calling with invalid URL should return a client that fails on ping
    try:
        client = _get_redis_for_kv("redis://nonexistent-host-12345:6379/0")
        # Try to ping to trigger actual connection - will raise redis error or OSError
        with pytest.raises((redis_error, OSError)):
            client.ping()
    finally:
        _test_hooks.redis_factory = orig


def test_get_redis_raw_for_rq_without_hook_attempts_real_connection() -> None:
    """Test _get_redis_raw_for_rq calls redis_raw_for_rq when hook is None."""
    import pytest
    from platform_workers.redis import _load_redis_error_class

    from covenant_radar_api.core.container import _get_redis_raw_for_rq

    redis_error: type[BaseException] = _load_redis_error_class()

    # Ensure hook is None
    orig = _test_hooks.redis_rq_factory
    _test_hooks.redis_rq_factory = None

    # Calling with invalid URL should return a client that fails on ping
    try:
        client = _get_redis_raw_for_rq("redis://nonexistent-host-12345:6379/0")
        # Try to ping to trigger actual connection - will raise redis error or OSError
        with pytest.raises((redis_error, OSError)):
            client.ping()
    finally:
        _test_hooks.redis_rq_factory = orig
