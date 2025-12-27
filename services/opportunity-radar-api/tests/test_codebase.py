"""Tests for codebase routes."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_list

from opportunity_radar_api.api.container import ServiceContainer
from opportunity_radar_api.api.main import create_app


def test_get_profile(fake_container: ServiceContainer) -> None:
    """Test getting codebase profile."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/codebase/profile")

    assert response.status_code == 200
    data = narrow_json_to_dict(load_json_str(response.text))
    assert data["technologies"] == ["python"]
    assert data["frameworks"] == ["fastapi"]
    assert data["ml_backends"] == ["xgboost"]


def test_list_libs(fake_container: ServiceContainer) -> None:
    """Test listing libraries."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/codebase/libs")

    assert response.status_code == 200
    data = narrow_json_to_list(load_json_str(response.text))
    assert len(data) == 1
    lib = narrow_json_to_dict(data[0])
    assert lib["name"] == "test-lib"
    assert lib["dependencies"] == ["fastapi", "httpx"]


def test_list_services(fake_container: ServiceContainer) -> None:
    """Test listing services."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/codebase/services")

    assert response.status_code == 200
    data = narrow_json_to_list(load_json_str(response.text))
    assert len(data) == 1
    svc = narrow_json_to_dict(data[0])
    assert svc["name"] == "test-service"
    assert svc["dependencies"] == ["flask"]
