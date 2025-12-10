from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from platform_core.health import HealthResponse
from platform_core.json_utils import JSONValue, load_json_str

from qr_api.api import _test_hooks
from qr_api.api.main import create_app
from qr_api.api.routes.health import _redis_client
from qr_api.health import healthz_endpoint
from qr_api.settings import load_default_options_from_env


def test_healthz_returns_ok() -> None:
    result: HealthResponse = healthz_endpoint()
    assert result == {"status": "ok"}


def test_healthz_route_via_client() -> None:
    """Test /healthz endpoint through the router."""
    env_values = {"REDIS_URL": "redis://ignored"}
    _test_hooks.get_env = lambda key: env_values.get(key)
    client = TestClient(create_app(load_default_options_from_env()))
    r = client.get("/healthz")
    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        pytest.fail("expected dict response body")
    body: dict[str, JSONValue] = body_raw
    assert body.get("status") == "ok"


def test_redis_client_raises_when_redis_url_missing() -> None:
    """Test _redis_client raises RuntimeError when REDIS_URL is not set."""
    _test_hooks.get_env = lambda key: None

    gen = _redis_client()
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        next(gen)
