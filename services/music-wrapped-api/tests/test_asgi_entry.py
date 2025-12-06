from __future__ import annotations

from fastapi.testclient import TestClient

from music_wrapped_api.asgi import app


def test_asgi_entry_exposes_app() -> None:
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
