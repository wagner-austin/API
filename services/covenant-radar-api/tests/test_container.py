"""Tests for service container dependency injection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_domain import Covenant, CovenantId, Deal, DealId, Measurement
from covenant_ml.testing import make_train_config
from covenant_ml.trainer import save_model, train_model
from covenant_persistence.testing import InMemoryConnection, InMemoryStore
from lightgbm.basic import LightGBMError
from numpy.typing import NDArray
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

from covenant_radar_api.core import ServiceContainer
from covenant_radar_api.core.config import Settings

from .conftest import ContainerAndStore


def _add_test_deal(store: InMemoryStore, deal_id: str) -> None:
    """Add a test deal to the store."""
    store.deals[deal_id] = Deal(
        id=DealId(value=deal_id),
        name="Test Deal",
        borrower="Acme Corp",
        sector="Technology",
        region="North America",
        commitment_amount_cents=100000000,
        currency="USD",
        maturity_date_iso="2025-12-31",
    )
    store._deal_order.append(deal_id)


def _add_test_covenant(
    store: InMemoryStore,
    cov_id: str,
    deal_id: str,
) -> None:
    """Add a test covenant to the store."""
    store.covenants[cov_id] = Covenant(
        id=CovenantId(value=cov_id),
        deal_id=DealId(value=deal_id),
        name="Test Covenant",
        formula="debt / ebitda",
        threshold_value_scaled=4_000_000,
        threshold_direction="<=",
        frequency="QUARTERLY",
    )
    store._covenant_order.append(cov_id)


def _add_test_measurement(store: InMemoryStore, deal_id: str) -> None:
    """Add a test measurement to the store."""
    store.measurements.append(
        Measurement(
            deal_id=DealId(value=deal_id),
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            metric_name="debt",
            metric_value_scaled=10_000_000,
        )
    )


def test_container_exported_from_core() -> None:
    """Test ServiceContainer is exported from core package."""
    from covenant_radar_api.core import ServiceContainer as ExportedContainer

    assert ExportedContainer is ServiceContainer


def test_container_deal_repo_returns_repository(
    container_with_store: ContainerAndStore,
) -> None:
    """Test deal_repo returns a DealRepository that can create/get deals."""
    repo = container_with_store.container.deal_repo()
    store = container_with_store.store

    # Verify we can create and retrieve
    _add_test_deal(store, "test-deal-1")

    retrieved = repo.get(DealId(value="test-deal-1"))
    assert retrieved["id"]["value"] == "test-deal-1"
    assert retrieved["name"] == "Test Deal"
    assert retrieved["borrower"] == "Acme Corp"

    # Verify list_all works
    all_deals = repo.list_all()
    assert len(all_deals) == 1


def test_container_covenant_repo_returns_repository(
    container_with_store: ContainerAndStore,
) -> None:
    """Test covenant_repo returns a CovenantRepository that can create/get covenants."""
    store = container_with_store.store
    covenant_repo = container_with_store.container.covenant_repo()

    # Add test data
    _add_test_deal(store, "deal-for-cov")
    _add_test_covenant(store, "test-cov-1", "deal-for-cov")

    # Verify we can retrieve
    retrieved = covenant_repo.get(CovenantId(value="test-cov-1"))
    assert retrieved["id"]["value"] == "test-cov-1"
    assert retrieved["name"] == "Test Covenant"
    assert retrieved["formula"] == "debt / ebitda"


def test_container_measurement_repo_returns_repository(
    container_with_store: ContainerAndStore,
) -> None:
    """Test measurement_repo returns a MeasurementRepository that can add/list measurements."""
    store = container_with_store.store
    measurement_repo = container_with_store.container.measurement_repo()

    # Add test data
    _add_test_deal(store, "deal-for-meas")
    _add_test_measurement(store, "deal-for-meas")

    # Verify we can list
    retrieved = measurement_repo.list_for_deal(DealId(value="deal-for-meas"))
    assert len(retrieved) == 1
    assert retrieved[0]["metric_name"] == "debt"


def test_container_covenant_result_repo_returns_repository(
    container_with_store: ContainerAndStore,
) -> None:
    """Test covenant_result_repo returns a CovenantResultRepository."""
    result_repo = container_with_store.container.covenant_result_repo()

    # Verify the repo has expected methods by calling list_for_deal
    results = result_repo.list_for_deal(DealId(value="nonexistent"))
    assert len(results) == 0


def test_container_rq_queue_returns_queue(
    container_with_store: ContainerAndStore,
) -> None:
    """Test rq_queue returns an RQClientQueue that can enqueue jobs."""
    queue = container_with_store.container.rq_queue()

    # Test that we can enqueue a job
    job = queue.enqueue("test_func", "arg1", "arg2")
    # FakeQueue returns a job-like object with get_id method
    assert job.get_id() == "test-job-id"


def test_container_get_model_info_returns_info(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_model_info returns model info with correct fields."""
    info = container_with_store.container.get_model_info()
    assert info["model_id"] == "default"
    assert info["model_path"].endswith("test_model.ubj")
    assert info["is_loaded"] is False


def test_container_get_sector_encoder_returns_dict(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_sector_encoder returns encoder dict."""
    encoder = container_with_store.container.get_sector_encoder()
    assert encoder == {"Technology": 0, "Finance": 1, "Healthcare": 2}


def test_container_get_region_encoder_returns_dict(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_region_encoder returns encoder dict."""
    encoder = container_with_store.container.get_region_encoder()
    assert encoder == {"North America": 0, "Europe": 1, "Asia": 2}


def test_container_get_model_output_dir_returns_path(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_model_output_dir returns a Path to an existing directory."""
    output_dir = container_with_store.container.get_model_output_dir()
    # Verify it's a Path that exists (tmp_path is created by pytest)
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_container_get_model_loads_and_caches_model(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_model loads model from file and caches it."""
    # Create a real model file at the expected path
    model_path = Path(container_with_store.container.get_model_info()["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    x_train: NDArray[np.float64] = np.zeros((4, 8), dtype=np.float64)
    x_train[0, 0] = 2.0
    x_train[1, 0] = 3.0
    x_train[2, 0] = 5.0
    x_train[3, 0] = 6.0

    y_train: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    y_train[2] = 1
    y_train[3] = 1

    config = make_train_config(
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )
    model = train_model(x_train, y_train, config)
    save_model(model, str(model_path))

    # Initially not loaded
    assert container_with_store.container.get_model_info()["is_loaded"] is False

    # Load model and verify it can make predictions
    loaded_model = container_with_store.container.get_model()

    # Verify the model works by making a prediction
    x_test: NDArray[np.float64] = np.zeros((1, 8), dtype=np.float64)
    x_test[0, 0] = 4.0
    prediction = loaded_model.predict_proba(x_test)
    assert prediction.shape == (1, 2)  # Binary classification

    # Now should be marked as loaded
    assert container_with_store.container.get_model_info()["is_loaded"] is True

    # Second call should return cached model (same object)
    second_call = container_with_store.container.get_model()
    assert second_call is loaded_model


def test_container_close_closes_resources(
    container_with_store: ContainerAndStore,
) -> None:
    """Test close() closes all resources."""
    # Just verify close doesn't raise
    container_with_store.container.close()


# =============================================================================
# Tests for production hook defaults
# =============================================================================


def test_the_bound_hook_loads_the_real_module() -> None:
    """The hook the module binds by default imports the real psycopg."""
    from covenant_radar_api.core import _test_hooks

    # Ensure hook is None (default production behavior)
    orig_hook = _test_hooks.load_psycopg_module
    _test_hooks.load_psycopg_module = _test_hooks._real_load_psycopg_module

    try:
        module = _test_hooks.load_psycopg_module()
        # Verify module has connect method (required by protocol)
        # We call it with invalid DSN expecting OperationalError
        psycopg = __import__("psycopg")
        operational_error: type[Exception] = psycopg.OperationalError
        import pytest

        with pytest.raises(operational_error):
            module.connect("host= dbname=x", autocommit=True)
    finally:
        _test_hooks.load_psycopg_module = orig_hook


def test_load_psycopg_module_uses_a_rebound_hook() -> None:
    """A rebound hook is what the loader returns."""
    from covenant_persistence.testing import InMemoryConnection, InMemoryStore

    from covenant_radar_api.core import _test_hooks
    from covenant_radar_api.core._test_hooks import PsycopgModuleProtocol

    store = InMemoryStore()
    hook_called = [False]

    class FakePsycopgModule:
        """Fake psycopg module for testing."""

        def connect(self, dsn: str, autocommit: bool = False) -> InMemoryConnection:
            hook_called[0] = True
            return InMemoryConnection(store)

    def fake_hook() -> PsycopgModuleProtocol:
        fake: PsycopgModuleProtocol = FakePsycopgModule()
        return fake

    orig_hook = _test_hooks.load_psycopg_module
    _test_hooks.load_psycopg_module = fake_hook

    try:
        module = _test_hooks.load_psycopg_module()
        # Call connect to verify it's the fake
        module.connect("test-dsn", autocommit=True)
        assert hook_called[0] is True
    finally:
        _test_hooks.load_psycopg_module = orig_hook


def test_psycopg_connect_autocommit_with_hook() -> None:
    """Test _psycopg_connect_autocommit returns connection via hook."""
    from covenant_persistence.testing import InMemoryConnection, InMemoryStore

    from covenant_radar_api.core import _test_hooks
    from covenant_radar_api.core._test_hooks import PsycopgModuleProtocol

    store = InMemoryStore()

    class FakePsycopgModule:
        """Fake psycopg module for testing."""

        def connect(self, dsn: str, autocommit: bool = False) -> InMemoryConnection:
            return InMemoryConnection(store)

    def fake_hook() -> PsycopgModuleProtocol:
        fake: PsycopgModuleProtocol = FakePsycopgModule()
        return fake

    orig_hook = _test_hooks.load_psycopg_module
    _test_hooks.load_psycopg_module = fake_hook

    try:
        # This will now hit the return conn line
        conn = _test_hooks._psycopg_connect_autocommit("test-dsn")
        # Verify we got a connection that works
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
    finally:
        _test_hooks.load_psycopg_module = orig_hook


def test_psycopg_connect_autocommit_calls_real_psycopg() -> None:
    """Test _psycopg_connect_autocommit calls real psycopg.connect."""

    from covenant_radar_api.core import _test_hooks

    # Ensure hook is None to use real psycopg
    orig_hook = _test_hooks.load_psycopg_module
    _test_hooks.load_psycopg_module = _test_hooks._real_load_psycopg_module

    try:
        psycopg = __import__("psycopg")
        operational_error: type[Exception] = psycopg.OperationalError

        with pytest.raises(operational_error):
            _test_hooks._psycopg_connect_autocommit("host= dbname=x")
    finally:
        _test_hooks.load_psycopg_module = orig_hook


def test_default_kv_factory_calls_real_redis() -> None:
    """Test redis_for_kv (production default) attempts real connection."""
    from platform_workers.redis import _load_redis_error_class, redis_for_kv

    redis_error: type[BaseException] = _load_redis_error_class()

    client = redis_for_kv("redis://nonexistent-host:6379/0")
    with pytest.raises((redis_error, OSError)):
        client.ping()


def test_default_rq_client_factory_calls_real_redis() -> None:
    """Test redis_raw_for_rq (production default) attempts real connection."""
    from platform_workers.redis import _load_redis_error_class
    from platform_workers.rq_harness import redis_raw_for_rq

    redis_error: type[BaseException] = _load_redis_error_class()

    client = redis_raw_for_rq("redis://nonexistent-host:6379/0")
    with pytest.raises((redis_error, OSError)):
        client.ping()


def test_rq_queue_factory_creates_real_queue() -> None:
    """Test rq_queue (production default) creates real RQ queue from redis connection."""
    from platform_workers.redis import _load_redis_error_class
    from platform_workers.rq_harness import redis_raw_for_rq, rq_queue

    redis_error: type[BaseException] = _load_redis_error_class()

    # Get a real connection (will fail but proves the factory path works)
    client = redis_raw_for_rq("redis://nonexistent-host:6379/0")
    queue = rq_queue("test-queue", client)

    # Verify queue was created (attempting to enqueue will fail with connection error)
    with pytest.raises((redis_error, OSError)):
        queue.enqueue("some_func")


# =============================================================================
# Tests for get_job_status
# =============================================================================


def test_container_get_job_status_not_found(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status returns not_found when job doesn't exist."""
    from platform_workers.testing import hooks as workers_hooks
    from platform_workers.testing import make_fake_fetch_job_not_found

    workers_hooks.fetch_job = make_fake_fetch_job_not_found()

    status = container_with_store.container.get_job_status("nonexistent-job-id")
    assert status["job_id"] == "nonexistent-job-id"
    assert status["status"] == "not_found"
    assert status["result"] is None


def test_container_get_job_status_queued(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status returns queued status."""
    from platform_workers.testing import FakeFetchedJob, make_fake_fetch_job_found
    from platform_workers.testing import hooks as workers_hooks

    fake_job = FakeFetchedJob(job_id="job-queued", status="queued", result=None)
    workers_hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-queued")
    assert status["job_id"] == "job-queued"
    assert status["status"] == "queued"
    assert status["result"] is None


def test_container_get_job_status_started(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status returns started status."""
    from platform_workers.testing import FakeFetchedJob, make_fake_fetch_job_found
    from platform_workers.testing import hooks as workers_hooks

    fake_job = FakeFetchedJob(job_id="job-started", status="started", result=None)
    workers_hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-started")
    assert status["job_id"] == "job-started"
    assert status["status"] == "started"
    assert status["result"] is None


def test_container_get_job_status_finished_with_result(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status returns finished status with result."""
    from platform_workers.testing import FakeFetchedJob, make_fake_fetch_job_found
    from platform_workers.testing import hooks as workers_hooks

    fake_job = FakeFetchedJob(
        job_id="job-finished",
        status="finished",
        result={"model_path": "/path/to/model.ubj"},
    )
    workers_hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-finished")
    assert status["job_id"] == "job-finished"
    assert status["status"] == "finished"
    assert status["result"] == {"model_path": "/path/to/model.ubj"}


def test_container_get_job_status_failed(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status returns failed status."""
    from platform_workers.testing import FakeFetchedJob, make_fake_fetch_job_found
    from platform_workers.testing import hooks as workers_hooks

    fake_job = FakeFetchedJob(job_id="job-failed", status="failed", result=None)
    workers_hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-failed")
    assert status["job_id"] == "job-failed"
    assert status["status"] == "failed"
    assert status["result"] is None


def test_container_get_job_status_unknown_status(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status maps unknown RQ status to not_found."""
    from platform_workers.testing import FakeFetchedJob, make_fake_fetch_job_found
    from platform_workers.testing import hooks as workers_hooks

    fake_job = FakeFetchedJob(job_id="job-unknown", status="deferred", result=None)
    workers_hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-unknown")
    assert status["job_id"] == "job-unknown"
    assert status["status"] == "not_found"
    assert status["result"] is None


def test_container_get_job_status_finished_with_non_dict_result(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status ignores non-dict result."""
    from platform_workers.testing import FakeFetchedJob, make_fake_fetch_job_found
    from platform_workers.testing import hooks as workers_hooks

    # Result is a string, not a dict - should be ignored
    fake_job = FakeFetchedJob(job_id="job-string-result", status="finished", result="ok")
    workers_hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-string-result")
    assert status["job_id"] == "job-string-result"
    assert status["status"] == "finished"
    assert status["result"] is None  # Non-dict result is ignored


# =============================================================================
# Tests for load_model_now
# =============================================================================


def test_container_load_model_now_returns_false_when_file_missing(
    container_with_store: ContainerAndStore,
) -> None:
    """Test load_model_now returns False when model file doesn't exist."""
    # Don't create model file - it should not exist
    result = container_with_store.container.load_model_now()
    assert result is False
    # Model should still not be loaded
    assert container_with_store.container.get_model_info()["is_loaded"] is False


def test_container_load_model_now_returns_true_when_file_exists(
    container_with_store: ContainerAndStore,
) -> None:
    """Test load_model_now returns True and loads model when file exists."""
    # Create a real model file at the expected path
    model_path = Path(container_with_store.container.get_model_info()["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    x_train: NDArray[np.float64] = np.zeros((4, 8), dtype=np.float64)
    x_train[0, 0] = 2.0
    x_train[1, 0] = 3.0
    x_train[2, 0] = 5.0
    x_train[3, 0] = 6.0

    y_train: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    y_train[2] = 1
    y_train[3] = 1

    config = make_train_config(
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )
    model = train_model(x_train, y_train, config)
    save_model(model, str(model_path))

    # Now load model
    result = container_with_store.container.load_model_now()
    assert result is True
    assert container_with_store.container.get_model_info()["is_loaded"] is True


# =============================================================================
# Tests for MLP backend (raises RuntimeError)
# =============================================================================


def test_container_load_mlp_model_raises_file_not_found(
    container_with_store: ContainerAndStore,
    tmp_path: Path,
) -> None:
    """Test _load_mlp_model raises FileNotFoundError when metadata missing."""
    container = container_with_store.container
    # Create model file but not metadata
    model_path = tmp_path / "test_mlp.pt"
    model_path.write_bytes(b"fake model")

    with pytest.raises(FileNotFoundError) as exc_info:
        container._load_mlp_model(str(model_path))

    assert "active_mlp_meta.json" in str(exc_info.value)


def test_container_load_model_now_mlp_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now raises FileNotFoundError when MLP metadata missing."""
    # Create a model file but no metadata so load_model_now fails
    model_path = tmp_path / "test_mlp.pt"
    model_path.write_bytes(b"fake mlp model")

    # Create container with MLP backend
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="mlp",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        container.load_model_now()

    assert "active_mlp_meta.json" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_get_model_mlp_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model raises FileNotFoundError when MLP metadata missing."""
    # Create model file but no metadata
    model_path = tmp_path / "test_mlp.pt"
    model_path.write_bytes(b"fake mlp model")

    # Create container with MLP backend
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="mlp",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        container.get_model()

    assert "active_mlp_meta.json" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_load_lstm_model_raises_file_not_found(
    container_with_store: ContainerAndStore,
    tmp_path: Path,
) -> None:
    """Test _load_lstm_model raises FileNotFoundError when metadata missing."""
    container = container_with_store.container
    # Create model file but not metadata
    model_path = tmp_path / "test_lstm.pt"
    model_path.write_bytes(b"fake model")

    with pytest.raises(FileNotFoundError) as exc_info:
        container._load_lstm_model(str(model_path))

    assert "active_lstm_meta.json" in str(exc_info.value)


def test_container_load_lightgbm_model_raises_lightgbm_error(
    container_with_store: ContainerAndStore,
    tmp_path: Path,
) -> None:
    """Test _load_lightgbm_model raises LightGBMError when model missing."""
    container = container_with_store.container
    model_path = str(tmp_path / "test_lightgbm.txt")

    with pytest.raises(LightGBMError) as exc_info:
        container._load_lightgbm_model(model_path)

    assert "test_lightgbm.txt" in str(exc_info.value)


def test_container_load_model_now_lstm_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now raises FileNotFoundError when LSTM metadata missing."""
    # Create a model file but no metadata
    model_path = tmp_path / "test_lstm.pt"
    model_path.write_bytes(b"fake lstm model")

    # Create container with LSTM backend
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="lstm",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        container.load_model_now()

    assert "active_lstm_meta.json" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_load_model_now_lightgbm_backend_returns_false_when_missing(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now returns False when LightGBM model missing."""
    # Create container with LightGBM backend but no model file
    model_path = tmp_path / "test_lightgbm.txt"
    # Don't create the file

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="lightgbm",
    )

    result = container.load_model_now()

    assert result is False
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_load_model_now_lightgbm_backend_returns_true(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now returns True when LightGBM model loads successfully.

    Uses test hook injection to provide a fake model loader, covering
    the lightgbm branch in load_model_now (container.py line 326).
    """
    from covenant_radar_api.worker import _test_hooks as worker_hooks

    # Create model file
    model_path = tmp_path / "test_lightgbm.txt"
    model_path.write_text("fake lightgbm model")

    # Create fake predictor
    class FakeLightGBMPredictor:
        """Fake predictor for LightGBM model."""

        def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            """Return fake predictions."""
            n_samples = int(x.shape[0])
            return np.column_stack(
                [
                    np.full(n_samples, 0.3, dtype=np.float64),
                    np.full(n_samples, 0.7, dtype=np.float64),
                ]
            )

    fake_predictor = FakeLightGBMPredictor()
    orig_loader = worker_hooks.lightgbm_loader
    worker_hooks.lightgbm_loader = lambda model_path: fake_predictor

    try:
        container = ServiceContainer(
            settings=test_settings,
            redis=fake_kv_client,
            db_conn=InMemoryConnection(in_memory_store),
            redis_rq=fake_rq_client,
            model_path=str(model_path),
            model_output_dir=tmp_path,
            sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
            region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
            ml_backend="lightgbm",
        )

        result = container.load_model_now()

        assert result is True
        assert container.get_model_info()["is_loaded"] is True
        container.close()
    finally:
        worker_hooks.lightgbm_loader = orig_loader


def test_container_get_model_lstm_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model raises FileNotFoundError when LSTM metadata missing."""
    # Create model file but no metadata
    model_path = tmp_path / "test_lstm.pt"
    model_path.write_bytes(b"fake lstm model")

    # Create container with LSTM backend
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="lstm",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        container.get_model()

    assert "active_lstm_meta.json" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_get_model_lightgbm_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model raises FileNotFoundError when LightGBM model missing.

    When the model file doesn't exist and data-bank is not configured,
    get_model raises FileNotFoundError with a descriptive message.
    """
    # Create container with LightGBM backend but no model file
    model_path = tmp_path / "test_lightgbm.txt"
    # Don't create the file

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="lightgbm",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        container.get_model()

    # Check that the error message includes the path and helpful info
    assert "test_lightgbm.txt" in str(exc_info.value)
    assert "Train a model first" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


# =============================================================================
# Data Bank Integration Tests
# =============================================================================


def test_container_get_model_file_id_xgboost(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _get_model_file_id returns correct file ID for XGBoost backend."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_model_file_id="active_xgb.ubj",
    )
    assert container._get_model_file_id() == "active_xgb.ubj"
    container.close()


def test_container_get_model_file_id_mlp(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _get_model_file_id returns correct file ID for MLP backend."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.pt"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="mlp",
        data_bank_model_file_id="active_mlp.pt",
    )
    assert container._get_model_file_id() == "active_mlp.pt"
    container.close()


def test_container_get_model_file_id_lstm(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _get_model_file_id returns correct file ID for LSTM backend."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.pt"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="lstm",
        data_bank_model_file_id="active_lstm.pt",
    )
    assert container._get_model_file_id() == "active_lstm.pt"
    container.close()


def test_container_get_model_file_id_lightgbm(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _get_model_file_id returns correct file ID for LightGBM backend."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.txt"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="lightgbm",
        data_bank_model_file_id="active_lgbm.txt",
    )
    assert container._get_model_file_id() == "active_lgbm.txt"
    container.close()


def test_container_download_model_from_data_bank_not_configured(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _download_model_from_data_bank returns False when not configured."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="",  # Not configured
        data_bank_key="",
    )
    result = container._download_model_from_data_bank(tmp_path / "model.ubj")
    assert result is False
    container.close()


def test_container_download_model_from_data_bank_success(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _download_model_from_data_bank successfully downloads model."""
    from platform_core.data_bank_client import HeadInfo

    from covenant_radar_api.core import _test_hooks

    # Track download calls
    download_calls: list[tuple[str, Path]] = []

    class FakeDataBankDownloader:
        """Fake downloader that writes a file."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            download_calls.append((file_id, dest))
            dest.write_bytes(b"fake model content")
            return HeadInfo(size=18, etag="abc123", content_type="application/octet-stream")

    def fake_factory(base_url: str, api_key: str) -> FakeDataBankDownloader:
        return FakeDataBankDownloader()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    dest_path = tmp_path / "models" / "active_xgb.ubj"
    result = container._download_model_from_data_bank(dest_path)

    assert result is True
    assert len(download_calls) == 1
    assert download_calls[0][0] == "active_xgb.ubj"
    assert download_calls[0][1] == dest_path
    assert dest_path.exists()
    assert dest_path.read_bytes() == b"fake model content"

    container.close()
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_download_model_from_data_bank_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _download_model_from_data_bank returns False when file not found."""
    from platform_core.data_bank_client import HeadInfo, NotFoundError

    from covenant_radar_api.core import _test_hooks

    class FakeDataBankDownloaderNotFound:
        """Fake downloader that raises NotFoundError."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            raise NotFoundError(f"File {file_id} not found")

    def fake_factory(base_url: str, api_key: str) -> FakeDataBankDownloaderNotFound:
        return FakeDataBankDownloaderNotFound()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    dest_path = tmp_path / "models" / "active_xgb.ubj"
    result = container._download_model_from_data_bank(dest_path)

    assert result is False
    assert not dest_path.exists()

    container.close()
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_download_model_from_data_bank_client_error(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _download_model_from_data_bank returns False on client errors."""
    from platform_core.data_bank_client import DataBankClientError, HeadInfo

    from covenant_radar_api.core import _test_hooks

    class FakeDataBankDownloaderError:
        """Fake downloader that raises DataBankClientError."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            raise DataBankClientError("transport error: Connection refused")

    def fake_factory(base_url: str, api_key: str) -> FakeDataBankDownloaderError:
        return FakeDataBankDownloaderError()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    dest_path = tmp_path / "models" / "active_xgb.ubj"
    result = container._download_model_from_data_bank(dest_path)

    assert result is False
    assert not dest_path.exists()

    container.close()
    _test_hooks.data_bank_client_factory = orig_factory


# =============================================================================
# Tests for _default_data_bank_client_factory
# =============================================================================


def test_default_data_bank_client_factory_creates_client() -> None:
    """Test _default_data_bank_client_factory creates a DataBankClient."""
    from covenant_radar_api.core._test_hooks import _default_data_bank_client_factory

    client = _default_data_bank_client_factory(
        "https://data-bank.example.com",
        "test-api-key",
    )
    assert client.__class__.__name__ == "DataBankClient"


# =============================================================================
# Tests for data-bank download in load_model_now and get_model
# =============================================================================


def test_container_load_model_now_tries_data_bank_when_configured(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now downloads from data-bank when local file missing."""
    from platform_core.data_bank_client import HeadInfo

    from covenant_radar_api.core import _test_hooks

    x_train: NDArray[np.float64] = np.zeros((4, 8), dtype=np.float64)
    x_train[0, 0] = 2.0
    x_train[1, 0] = 3.0
    x_train[2, 0] = 5.0
    x_train[3, 0] = 6.0
    y_train: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    y_train[2] = 1
    y_train[3] = 1
    config = make_train_config(
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )
    xgb_model = train_model(x_train, y_train, config)
    staging_path = tmp_path / "staging" / "model.ubj"
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(xgb_model, str(staging_path))
    model_bytes = staging_path.read_bytes()

    class FakeDownloader:
        """Fake downloader that writes model bytes to dest path."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(model_bytes)
            return HeadInfo(
                size=len(model_bytes),
                etag="fake-etag",
                content_type="application/octet-stream",
            )

    def fake_factory(base_url: str, api_key: str) -> FakeDownloader:
        return FakeDownloader()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0},
        region_encoder={"North America": 0},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    result = container.load_model_now()

    assert result is True
    assert container.get_model_info()["is_loaded"] is True

    container.close()
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_load_model_now_returns_false_when_data_bank_fails(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now returns False when data-bank download fails."""
    from platform_core.data_bank_client import DataBankClientError, HeadInfo

    from covenant_radar_api.core import _test_hooks

    class FakeDownloaderFail:
        """Fake downloader that raises error."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            raise DataBankClientError("connection refused")

    def fake_factory(base_url: str, api_key: str) -> FakeDownloaderFail:
        return FakeDownloaderFail()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    result = container.load_model_now()

    assert result is False
    assert container.get_model_info()["is_loaded"] is False

    container.close()
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_load_model_now_skips_when_no_file_id(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now skips data-bank download when file_id is empty."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="",
    )

    result = container.load_model_now()

    assert result is False
    assert container.get_model_info()["is_loaded"] is False

    container.close()


def test_container_get_model_tries_data_bank_when_configured(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model downloads from data-bank when local file missing."""
    from platform_core.data_bank_client import HeadInfo

    from covenant_radar_api.core import _test_hooks

    x_train: NDArray[np.float64] = np.zeros((4, 8), dtype=np.float64)
    x_train[0, 0] = 2.0
    x_train[1, 0] = 3.0
    x_train[2, 0] = 5.0
    x_train[3, 0] = 6.0
    y_train: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    y_train[2] = 1
    y_train[3] = 1
    config = make_train_config(
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )
    xgb_model = train_model(x_train, y_train, config)
    staging_path = tmp_path / "staging" / "model.ubj"
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(xgb_model, str(staging_path))
    model_bytes = staging_path.read_bytes()

    class FakeDownloader:
        """Fake downloader that writes model bytes to dest path."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(model_bytes)
            return HeadInfo(
                size=len(model_bytes),
                etag="fake-etag",
                content_type="application/octet-stream",
            )

    def fake_factory(base_url: str, api_key: str) -> FakeDownloader:
        return FakeDownloader()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0},
        region_encoder={"North America": 0},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    loaded_model = container.get_model()

    x_test: NDArray[np.float64] = np.zeros((1, 8), dtype=np.float64)
    prediction = loaded_model.predict_proba(x_test)
    assert prediction.shape == (1, 2)
    assert container.get_model_info()["is_loaded"] is True

    container.close()
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_get_model_raises_when_data_bank_fails(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model raises FileNotFoundError when data-bank download fails."""
    from platform_core.data_bank_client import DataBankClientError, HeadInfo

    from covenant_radar_api.core import _test_hooks

    class FakeDownloaderFail:
        """Fake downloader that raises error."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            raise DataBankClientError("connection refused")

    def fake_factory(base_url: str, api_key: str) -> FakeDownloaderFail:
        return FakeDownloaderFail()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    with pytest.raises(FileNotFoundError, match="Train a model first"):
        container.get_model()

    container.close()
    _test_hooks.data_bank_client_factory = orig_factory


# =============================================================================
# Tests for get_model with LightGBM backend (covers line 483)
# =============================================================================


def test_container_get_model_lightgbm_backend_loads_successfully(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model loads LightGBM model via hook when file exists."""
    from covenant_radar_api.worker import _test_hooks as worker_hooks

    model_path = tmp_path / "test_lightgbm.txt"
    model_path.write_text("fake lightgbm model")

    class FakeLightGBMPredictor:
        """Fake predictor for LightGBM model."""

        def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            """Return fake predictions."""
            n_samples = int(x.shape[0])
            return np.column_stack(
                [
                    np.full(n_samples, 0.3, dtype=np.float64),
                    np.full(n_samples, 0.7, dtype=np.float64),
                ]
            )

    fake_predictor = FakeLightGBMPredictor()
    orig_loader = worker_hooks.lightgbm_loader
    worker_hooks.lightgbm_loader = lambda model_path: fake_predictor

    try:
        container = ServiceContainer(
            settings=test_settings,
            redis=fake_kv_client,
            db_conn=InMemoryConnection(in_memory_store),
            redis_rq=fake_rq_client,
            model_path=str(model_path),
            model_output_dir=tmp_path,
            sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
            region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
            ml_backend="lightgbm",
        )

        loaded_model = container.get_model()

        x_test: NDArray[np.float64] = np.zeros((1, 8), dtype=np.float64)
        prediction = loaded_model.predict_proba(x_test)
        assert prediction.shape == (1, 2)
        assert container.get_model_info()["is_loaded"] is True

        container.close()
    finally:
        worker_hooks.lightgbm_loader = orig_loader
