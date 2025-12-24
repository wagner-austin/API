"""Tests for grandma_api.api.routes.health module."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from grandma_api.api.main import create_app
from grandma_api.config import GrandmaApiSettings

from .conftest import set_fake_env


def _make_test_settings() -> GrandmaApiSettings:
    """Create test settings."""
    return GrandmaApiSettings(
        openai_api_key="sk-test",
        api_token="test-token",
        port=8080,
        log_level="INFO",
        log_format="json",
    )


def test_healthz_route_returns_ok() -> None:
    """Test /healthz route returns status ok."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    app = create_app(_make_test_settings())
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    body = narrow_json_to_dict(load_json_str(response.text))
    assert body.get("status") == "ok"
