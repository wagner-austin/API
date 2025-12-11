"""Tests for service container dependency injection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from covenant_domain import Covenant, CovenantId, Deal, DealId, Measurement
from covenant_ml.testing import make_train_config
from covenant_ml.trainer import save_model, train_model
from covenant_persistence.testing import InMemoryStore
from numpy.typing import NDArray

from covenant_radar_api.core import ServiceContainer

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


def test_load_psycopg_module_returns_real_module() -> None:
    """Test _load_psycopg_module returns real psycopg when hook is None."""
    from covenant_radar_api.core import _test_hooks

    # Ensure hook is None (default production behavior)
    orig_hook = _test_hooks.load_psycopg_module_hook
    _test_hooks.load_psycopg_module_hook = None

    try:
        module = _test_hooks._load_psycopg_module()
        # Verify module has connect method (required by protocol)
        # We call it with invalid DSN expecting OperationalError
        psycopg = __import__("psycopg")
        operational_error: type[Exception] = psycopg.OperationalError
        import pytest

        with pytest.raises(operational_error):
            module.connect("host= dbname=x", autocommit=True)
    finally:
        _test_hooks.load_psycopg_module_hook = orig_hook


def test_load_psycopg_module_uses_hook_when_set() -> None:
    """Test _load_psycopg_module uses hook when set."""
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

    orig_hook = _test_hooks.load_psycopg_module_hook
    _test_hooks.load_psycopg_module_hook = fake_hook

    try:
        module = _test_hooks._load_psycopg_module()
        # Call connect to verify it's the fake
        module.connect("test-dsn", autocommit=True)
        assert hook_called[0] is True
    finally:
        _test_hooks.load_psycopg_module_hook = orig_hook


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

    orig_hook = _test_hooks.load_psycopg_module_hook
    _test_hooks.load_psycopg_module_hook = fake_hook

    try:
        # This will now hit the return conn line
        conn = _test_hooks._psycopg_connect_autocommit("test-dsn")
        # Verify we got a connection that works
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
    finally:
        _test_hooks.load_psycopg_module_hook = orig_hook


def test_psycopg_connect_autocommit_calls_real_psycopg() -> None:
    """Test _psycopg_connect_autocommit calls real psycopg.connect."""
    import pytest

    from covenant_radar_api.core import _test_hooks

    # Ensure hook is None to use real psycopg
    orig_hook = _test_hooks.load_psycopg_module_hook
    _test_hooks.load_psycopg_module_hook = None

    try:
        psycopg = __import__("psycopg")
        operational_error: type[Exception] = psycopg.OperationalError

        with pytest.raises(operational_error):
            _test_hooks._psycopg_connect_autocommit("host= dbname=x")
    finally:
        _test_hooks.load_psycopg_module_hook = orig_hook


def test_default_kv_factory_calls_real_redis() -> None:
    """Test redis_for_kv (production default) attempts real connection."""
    import pytest
    from platform_workers.redis import _load_redis_error_class, redis_for_kv

    redis_error: type[BaseException] = _load_redis_error_class()

    client = redis_for_kv("redis://nonexistent-host:6379/0")
    with pytest.raises((redis_error, OSError)):
        client.ping()


def test_default_rq_client_factory_calls_real_redis() -> None:
    """Test redis_raw_for_rq (production default) attempts real connection."""
    import pytest
    from platform_workers.redis import _load_redis_error_class
    from platform_workers.rq_harness import redis_raw_for_rq

    redis_error: type[BaseException] = _load_redis_error_class()

    client = redis_raw_for_rq("redis://nonexistent-host:6379/0")
    with pytest.raises((redis_error, OSError)):
        client.ping()


def test_rq_queue_factory_creates_real_queue() -> None:
    """Test rq_queue (production default) creates real RQ queue from redis connection."""
    import pytest
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
    from platform_workers.testing import hooks, make_fake_fetch_job_not_found

    hooks.fetch_job = make_fake_fetch_job_not_found()

    status = container_with_store.container.get_job_status("nonexistent-job-id")
    assert status["job_id"] == "nonexistent-job-id"
    assert status["status"] == "not_found"
    assert status["result"] is None


def test_container_get_job_status_queued(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status returns queued status."""
    from platform_workers.testing import FakeFetchedJob, hooks, make_fake_fetch_job_found

    fake_job = FakeFetchedJob(job_id="job-queued", status="queued", result=None)
    hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-queued")
    assert status["job_id"] == "job-queued"
    assert status["status"] == "queued"
    assert status["result"] is None


def test_container_get_job_status_started(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status returns started status."""
    from platform_workers.testing import FakeFetchedJob, hooks, make_fake_fetch_job_found

    fake_job = FakeFetchedJob(job_id="job-started", status="started", result=None)
    hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-started")
    assert status["job_id"] == "job-started"
    assert status["status"] == "started"
    assert status["result"] is None


def test_container_get_job_status_finished_with_result(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status returns finished status with result."""
    from platform_workers.testing import FakeFetchedJob, hooks, make_fake_fetch_job_found

    fake_job = FakeFetchedJob(
        job_id="job-finished",
        status="finished",
        result={"model_path": "/path/to/model.ubj"},
    )
    hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-finished")
    assert status["job_id"] == "job-finished"
    assert status["status"] == "finished"
    assert status["result"] == {"model_path": "/path/to/model.ubj"}


def test_container_get_job_status_failed(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status returns failed status."""
    from platform_workers.testing import FakeFetchedJob, hooks, make_fake_fetch_job_found

    fake_job = FakeFetchedJob(job_id="job-failed", status="failed", result=None)
    hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-failed")
    assert status["job_id"] == "job-failed"
    assert status["status"] == "failed"
    assert status["result"] is None


def test_container_get_job_status_unknown_status(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status maps unknown RQ status to not_found."""
    from platform_workers.testing import FakeFetchedJob, hooks, make_fake_fetch_job_found

    fake_job = FakeFetchedJob(job_id="job-unknown", status="deferred", result=None)
    hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    status = container_with_store.container.get_job_status("job-unknown")
    assert status["job_id"] == "job-unknown"
    assert status["status"] == "not_found"
    assert status["result"] is None


def test_container_get_job_status_finished_with_non_dict_result(
    container_with_store: ContainerAndStore,
) -> None:
    """Test get_job_status ignores non-dict result."""
    from platform_workers.testing import FakeFetchedJob, hooks, make_fake_fetch_job_found

    # Result is a string, not a dict - should be ignored
    fake_job = FakeFetchedJob(job_id="job-string-result", status="finished", result="ok")
    hooks.fetch_job = make_fake_fetch_job_found(fake_job)

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
