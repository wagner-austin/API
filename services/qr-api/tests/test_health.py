from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from platform_core.health import HealthResponse
from platform_core.json_utils import JSONValue, load_json_str
from pytest import MonkeyPatch

from qr_api.app import create_app
from qr_api.health import healthz_endpoint
from qr_api.settings import load_default_options_from_env


def test_healthz_returns_ok() -> None:
    result: HealthResponse = healthz_endpoint()
    assert result == {"status": "ok"}


def test_healthz_route_via_client(monkeypatch: MonkeyPatch) -> None:
    """Test /healthz endpoint through the router."""
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    client = TestClient(create_app(load_default_options_from_env()))
    r = client.get("/healthz")
    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        pytest.fail("expected dict response body")
    body: dict[str, JSONValue] = body_raw
    assert body.get("status") == "ok"
