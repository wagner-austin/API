"""Tests for application factory."""

from __future__ import annotations

from covenant_persistence import ConnectionProtocol
from covenant_persistence.testing import InMemoryConnection
from fastapi.testclient import TestClient

from covenant_radar_api.api.main import create_app
from covenant_radar_api.core import _test_hooks as core_hooks
from covenant_radar_api.core.config import Settings
from covenant_radar_api.integrations.datadog import _test_hooks as datadog_test_hooks

from .conftest import ContainerAndStore


def test_app_factory_creates_fastapi_app(
    container_with_store: ContainerAndStore,
) -> None:
    """Test create_app returns a FastAPI application."""
    # Hooks are set by container_with_store fixture
    app = create_app(container_with_store.container.settings)

    assert app.title == "covenant-radar-api"
    assert app.version == "0.1.0"


def test_app_lifespan_closes_container_connection(
    container_with_store: ContainerAndStore,
) -> None:
    """App shutdown releases the database connection the container holds.

    Regression guard: create_app built a ServiceContainer and never registered
    a shutdown hook, so every API process leaked its psycopg connection and
    both Redis clients.

    Args:
        container_with_store: Fixture installing in-memory fakes via hooks.
    """
    created: list[InMemoryConnection] = []

    def recording_connection_factory(dsn: str) -> ConnectionProtocol:
        """Create an in-memory connection and record it for inspection.

        Args:
            dsn: Connection string; ignored by the in-memory implementation.

        Returns:
            A fresh InMemoryConnection backed by the fixture's store.
        """
        del dsn
        conn = InMemoryConnection(container_with_store.store)
        created.append(conn)
        return conn

    original_factory = core_hooks.connection_factory
    core_hooks.connection_factory = recording_connection_factory

    app = create_app(container_with_store.container.settings)
    with TestClient(app):
        pass

    core_hooks.connection_factory = original_factory

    assert len(created) == 1
    assert created[0].closed is True


def test_app_factory_health_endpoints(
    container_with_store: ContainerAndStore,
) -> None:
    """Test app factory creates working application with health endpoints."""
    client: TestClient = TestClient(create_app(container_with_store.container.settings))

    r1 = client.get("/healthz")
    assert r1.status_code == 200
    assert '"status"' in r1.text
    assert '"ok"' in r1.text

    r2 = client.get("/readyz")
    assert r2.status_code == 200
    assert '"status"' in r2.text


def test_app_factory_includes_crud_routes(
    container_with_store: ContainerAndStore,
) -> None:
    """Test app factory includes CRUD routes."""
    client: TestClient = TestClient(create_app(container_with_store.container.settings))

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


class TestCreateAppDatadogIntegration:
    """Tests for Datadog integration in create_app."""

    def test_create_app_with_datadog_enabled_calls_tracing_setup(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that create_app calls setup_datadog_tracing when enabled."""
        call_args: list[tuple[str, str, str]] = []

        def fake_tracing_setup(service: str, env: str, version: str) -> bool:
            call_args.append((service, env, version))
            return True

        datadog_test_hooks.tracing_setup = fake_tracing_setup

        # Create settings with Datadog enabled
        base_settings = container_with_store.container.settings
        settings: Settings = {
            **base_settings,
            "datadog": {
                "enabled": True,
                "service": "test-service",
                "env": "staging",
                "version": "1.2.3",
                "agent_host": "localhost",
                "dogstatsd_port": 8125,
                "trace_enabled": True,
            },
        }

        _app = create_app(settings)

        assert len(call_args) == 1
        assert call_args[0] == ("test-service", "staging", "1.2.3")

    def test_create_app_with_datadog_disabled_skips_tracing_setup(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that create_app skips tracing when datadog is disabled."""
        call_args: list[tuple[str, str, str]] = []

        def fake_tracing_setup(service: str, env: str, version: str) -> bool:
            call_args.append((service, env, version))
            return True

        datadog_test_hooks.tracing_setup = fake_tracing_setup

        # Create settings with Datadog disabled
        base_settings = container_with_store.container.settings
        settings: Settings = {
            **base_settings,
            "datadog": {
                "enabled": False,
                "service": "test-service",
                "env": "dev",
                "version": "1.0.0",
                "agent_host": "localhost",
                "dogstatsd_port": 8125,
                "trace_enabled": True,
            },
        }

        _app = create_app(settings)

        assert len(call_args) == 0

    def test_create_app_with_trace_disabled_skips_tracing_setup(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that create_app skips tracing when trace_enabled is false."""
        call_args: list[tuple[str, str, str]] = []

        def fake_tracing_setup(service: str, env: str, version: str) -> bool:
            call_args.append((service, env, version))
            return True

        datadog_test_hooks.tracing_setup = fake_tracing_setup

        # Create settings with trace_enabled=False
        base_settings = container_with_store.container.settings
        settings: Settings = {
            **base_settings,
            "datadog": {
                "enabled": True,
                "service": "test-service",
                "env": "production",
                "version": "2.0.0",
                "agent_host": "localhost",
                "dogstatsd_port": 8125,
                "trace_enabled": False,
            },
        }

        _app = create_app(settings)

        assert len(call_args) == 0
