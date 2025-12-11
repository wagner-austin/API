"""Tests for application factory."""

from __future__ import annotations

from fastapi.testclient import TestClient

from covenant_radar_api.api.main import create_app

from .conftest import ContainerAndStore


def test_app_factory_creates_fastapi_app(
    container_with_store: ContainerAndStore,
) -> None:
    """Test create_app returns a FastAPI application."""
    # Hooks are set by container_with_store fixture
    app = create_app(container_with_store.container.settings)

    assert app.title == "covenant-radar-api"
    assert app.version == "0.1.0"


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
