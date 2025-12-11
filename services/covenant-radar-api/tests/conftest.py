"""Shared test fixtures for covenant-radar-api."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from covenant_ml.testing import set_cuda_hook
from covenant_persistence import ConnectionProtocol
from covenant_persistence.testing import InMemoryConnection, InMemoryStore
from platform_core.config import _test_hooks as config_test_hooks
from platform_workers.redis import RedisBytesProto, RedisStrProto
from platform_workers.rq_harness import RQClientQueue, _RedisBytesClient
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient

from covenant_radar_api import _test_hooks as worker_test_hooks
from covenant_radar_api.core import _test_hooks
from covenant_radar_api.core.config import Settings
from covenant_radar_api.core.container import ServiceContainer
from covenant_radar_api.seeding import _test_hooks as seeding_test_hooks
from scripts import guard as guard_mod

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


def _make_test_settings() -> Settings:
    return Settings(
        app_env="dev",
        logging={"level": "INFO"},
        redis={"enabled": True, "url": "redis://test:6379/0"},
        rq={
            "queue_name": "covenant",
            "job_timeout_sec": 3600,
            "result_ttl_sec": 86400,
            "failure_ttl_sec": 604800,
        },
        app={
            "data_root": "/data",
            "models_root": "/data/models",
            "logs_root": "/data/logs",
            "active_model_path": "/data/models/active.ubj",
        },
        database_url="postgresql://test:test@localhost/test",
    )


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

    container = ServiceContainer.from_settings(
        test_settings,
        model_path=str(tmp_path / "test_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
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
    yield
    # Restore to production defaults
    _test_hooks.kv_factory = orig_kv
    _test_hooks.connection_factory = orig_conn
    _test_hooks.rq_client_factory = orig_rq
    _test_hooks.queue_factory = orig_queue


def _reset_config_hooks_impl() -> Generator[None, None, None]:
    """Reset platform_core config hooks after each test."""
    orig_get_env = config_test_hooks.get_env
    orig_tomllib_loads = config_test_hooks.tomllib_loads
    yield
    config_test_hooks.get_env = orig_get_env
    config_test_hooks.tomllib_loads = orig_tomllib_loads


def _reset_worker_hooks_impl() -> Generator[None, None, None]:
    """Reset worker entry test hooks after each test."""
    orig_runner = worker_test_hooks.test_runner
    yield
    worker_test_hooks.test_runner = orig_runner


def _reset_guard_hooks_impl() -> Generator[None, None, None]:
    """Reset guard script hooks after each test."""
    orig_is_dir = guard_mod._is_dir
    yield
    guard_mod._is_dir = orig_is_dir


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
    set_cuda_hook(None)


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
_reset_guard_hooks = pytest.fixture(autouse=True)(_reset_guard_hooks_impl)
_reset_seeding_hooks = pytest.fixture(autouse=True)(_reset_seeding_hooks_impl)
_disable_cuda = pytest.fixture(autouse=True)(_disable_cuda_impl)
