"""Tests for health routes."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from opportunity_radar_api.api.container import ServiceContainer
from opportunity_radar_api.api.main import create_app


def test_healthz_endpoint(fake_container: ServiceContainer) -> None:
    """Test healthz endpoint returns ok status."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    data = narrow_json_to_dict(load_json_str(response.text))
    assert data["status"] == "ok"
