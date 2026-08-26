"""Shared test fixtures for covenant-radar-api."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from covenant_ml.testing import reset_cuda_hook, set_cuda_hook
from covenant_persistence import ConnectionProtocol
from covenant_persistence.testing import InMemoryConnection, InMemoryStore
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from platform_core.config import _test_hooks as config_test_hooks
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_workers.redis import RedisBytesProto, RedisStrProto
from platform_workers.rq_harness import RQClientQueue, _RedisBytesClient
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient

from covenant_radar_api import _test_hooks as app_test_hooks
from covenant_radar_api.api.error_handlers import install_covenant_error_handlers
from covenant_radar_api.core import _test_hooks
from covenant_radar_api.core.config import Settings
from covenant_radar_api.core.container import ServiceContainer
from covenant_radar_api.integrations.datadog import _test_hooks as datadog_test_hooks
from covenant_radar_api.integrations.datadog.tracing import reset_tracing_state
from covenant_radar_api.seeding import _test_hooks as seeding_test_hooks
from covenant_radar_api.worker import _test_hooks as worker_job_hooks

# =============================================================================
# Container And Store for Testing
# =============================================================================


class ContainerAndStore:
    """Container and store pair for testing."""

    def __init__(
        self,
        container: ServiceContainer,
        store: InMemoryStore,
        queue: FakeQueue,
    ) -> None:
        """Initialize with container and store."""
        self.container = container
        self.store = store
        self.queue = queue


# =============================================================================
# Fixture Implementations
# =============================================================================


def _make_in_memory_store() -> InMemoryStore:
    return InMemoryStore()


def _make_fake_kv_client() -> FakeRedis:
    kv = FakeRedis()
    kv.sadd("rq:workers", "worker-1")
    return kv


def _make_fake_rq_client() -> FakeRedisBytesClient:
    return FakeRedisBytesClient()


def _make_fake_queue() -> FakeQueue:
    return FakeQueue()


def _make_test_settings(tmp_path: Path) -> Settings:
    # Roots are real per-test directories, not the fictional "/data" tree. The
    # API confines caller-supplied model paths to models_root, so that root has
    # to be somewhere a test can actually write a model.
    data_root = tmp_path / "data"
    models_root = tmp_path / "models"
    logs_root = tmp_path / "logs"
    return {
        "app_env": "dev",
        "logging": {"level": "INFO"},
        "redis": {"enabled": True, "url": "redis://test:6379/0"},
        "rq": {
            "queue_name": "covenant",
            "job_timeout_sec": 3600,
            "result_ttl_sec": 86400,
            "failure_ttl_sec": 604800,
        },
        "app": {
            "data_root": str(data_root),
            "models_root": str(models_root),
            "logs_root": str(logs_root),
            "ml_backend": "xgboost",
            "active_model_path_xgb": str(models_root / "active_xgb.ubj"),
            "active_model_path_mlp": str(models_root / "active_mlp.pt"),
            "data_bank_api_url": "",
            "data_bank_api_key": "",
            "data_bank_model_file_id": "",
        },
        "datadog": {
            "enabled": False,
            "service": "covenant-radar-api",
            "env": "dev",
            "version": "0.0.0",
            "agent_host": "localhost",
            "dogstatsd_port": 8125,
            "trace_enabled": False,
        },
        "database_url": "postgresql://test:test@localhost/test",
    }


def _make_container_with_store(
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    fake_queue: FakeQueue,
    test_settings: Settings,
    tmp_path: Path,
) -> Generator[ContainerAndStore, None, None]:
    """Provide ServiceContainer with injected fakes via _test_hooks."""

    def kv_factory(url: str) -> RedisStrProto:
        return fake_kv_client

    def connection_factory(dsn: str) -> ConnectionProtocol:
        return InMemoryConnection(in_memory_store)

    def rq_client_factory(url: str) -> RedisBytesProto:
        return fake_rq_client

    def queue_factory(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        return fake_queue

    orig_kv = _test_hooks.kv_factory
    orig_conn = _test_hooks.connection_factory
    orig_rq = _test_hooks.rq_client_factory
    orig_queue = _test_hooks.queue_factory

    _test_hooks.kv_factory = kv_factory
    _test_hooks.connection_factory = connection_factory
    _test_hooks.rq_client_factory = rq_client_factory
    _test_hooks.queue_factory = queue_factory

    # Pass ml_backend explicitly since ServiceContainer.__init__ now requires it
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "test_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="xgboost",
    )

    yield ContainerAndStore(container, in_memory_store, fake_queue)

    _test_hooks.kv_factory = orig_kv
    _test_hooks.connection_factory = orig_conn
    _test_hooks.rq_client_factory = orig_rq
    _test_hooks.queue_factory = orig_queue
    container.close()


def _reset_test_hooks_impl() -> Generator[None, None, None]:
    """Reset test hooks after each test to production defaults."""
    # Save original hooks (production defaults)
    orig_kv = _test_hooks.kv_factory
    orig_conn = _test_hooks.connection_factory
    orig_rq = _test_hooks.rq_client_factory
    orig_queue = _test_hooks.queue_factory
    orig_data_bank = _test_hooks.data_bank_client_factory
    yield
    # Restore to production defaults
    _test_hooks.kv_factory = orig_kv
    _test_hooks.connection_factory = orig_conn
    _test_hooks.rq_client_factory = orig_rq
    _test_hooks.queue_factory = orig_queue
    _test_hooks.data_bank_client_factory = orig_data_bank


def _reset_config_hooks_impl() -> Generator[None, None, None]:
    """Reset platform_core config hooks after each test."""
    orig_get_env = config_test_hooks.get_env
    orig_tomllib_loads = config_test_hooks.tomllib_loads
    yield
    config_test_hooks.get_env = orig_get_env
    config_test_hooks.tomllib_loads = orig_tomllib_loads


def _reset_worker_hooks_impl() -> Generator[None, None, None]:
    """Reset worker entry test hooks after each test."""
    orig_runner = app_test_hooks.worker_runner
    orig_data_bank_uploader = worker_job_hooks.data_bank_uploader
    yield
    app_test_hooks.worker_runner = orig_runner
    worker_job_hooks.data_bank_uploader = orig_data_bank_uploader


def _reset_seeding_hooks_impl() -> Generator[None, None, None]:
    """Reset seeding module hooks after each test."""
    orig_conn = seeding_test_hooks.connection_factory
    orig_uuid = seeding_test_hooks.uuid_generator
    yield
    seeding_test_hooks.connection_factory = orig_conn
    seeding_test_hooks.uuid_generator = orig_uuid


def _disable_cuda_impl() -> Generator[None, None, None]:
    """Disable CUDA in tests to avoid XGBoost GPU warnings."""
    set_cuda_hook(lambda: False)
    yield
    reset_cuda_hook()


def _reset_datadog_hooks_impl() -> Generator[None, None, None]:
    """Reset Datadog integration hooks after each test."""
    orig_metrics_factory = datadog_test_hooks.metrics_sink_factory
    orig_tracing_setup = datadog_test_hooks.tracing_setup
    yield
    datadog_test_hooks.metrics_sink_factory = orig_metrics_factory
    datadog_test_hooks.tracing_setup = orig_tracing_setup
    reset_tracing_state()


# =============================================================================
# Route Test Client
# =============================================================================


def make_route_test_client(router: APIRouter) -> TestClient:
    """Build a TestClient wired the same way the production app is.

    Installs the exception handlers that `create_app` installs, so route tests
    observe the production error contract (an AppError surfaces as its declared
    http_status) instead of a bare 500 from an unhandled exception. Route test
    modules previously each built a plain FastAPI() with no handlers, which
    made every AppError look like a 500.

    Args:
        router: Router under test, already bound to a container.

    Returns:
        TestClient that returns server exceptions as responses.
    """
    app = FastAPI()
    install_exception_handlers_fastapi(app)
    install_covenant_error_handlers(app)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Pytest Fixtures
# =============================================================================

in_memory_store = pytest.fixture(_make_in_memory_store)
fake_kv_client = pytest.fixture(_make_fake_kv_client)
fake_rq_client = pytest.fixture(_make_fake_rq_client)
fake_queue = pytest.fixture(_make_fake_queue)
test_settings = pytest.fixture(_make_test_settings)
container_with_store = pytest.fixture(_make_container_with_store)
_reset_test_hooks = pytest.fixture(autouse=True)(_reset_test_hooks_impl)
_reset_config_hooks = pytest.fixture(autouse=True)(_reset_config_hooks_impl)
_reset_worker_hooks = pytest.fixture(autouse=True)(_reset_worker_hooks_impl)
_reset_seeding_hooks = pytest.fixture(autouse=True)(_reset_seeding_hooks_impl)
_disable_cuda = pytest.fixture(autouse=True)(_disable_cuda_impl)
_reset_datadog_hooks = pytest.fixture(autouse=True)(_reset_datadog_hooks_impl)
