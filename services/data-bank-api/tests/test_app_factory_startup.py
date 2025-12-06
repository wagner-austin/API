from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from data_bank_api.app import create_app


def test_app_factory_and_health_endpoints(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("API_UPLOAD_KEYS", "u1")
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    app = create_app()
    client: TestClient = TestClient(app)

    r1 = client.get("/healthz")
    assert r1.status_code == 200
    # Avoid JSON parsing here to satisfy strict typing policies in tests
    # while still validating the contract succinctly.
    assert '"status"' in r1.text
    assert '"ok"' in r1.text

    r2 = client.get("/readyz")
    assert r2.status_code in (200, 503)
    assert '"status"' in r2.text
