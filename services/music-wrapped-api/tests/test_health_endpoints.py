from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str

from music_wrapped_api.app import create_app


def test_healthz_ok() -> None:
    client = TestClient(create_app())
    r = client.get("/healthz")
    assert r.status_code == 200
    body = load_json_str(r.text)
    if not isinstance(body, dict):
        raise AssertionError("response must be an object")
    assert body.get("status") == "ok"
