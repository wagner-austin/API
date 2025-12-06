from __future__ import annotations

from fastapi.testclient import TestClient

from model_trainer.api.main import create_app


def test_app_factory_and_health_endpoints() -> None:
    app = create_app()
    client: TestClient = TestClient(app)

    r1 = client.get("/healthz")
    assert r1.status_code == 200
    assert '"status"' in r1.text and '"ok"' in r1.text

    r2 = client.get("/readyz")
    # Readyz can be degraded locally if Redis/worker are not present.
    assert r2.status_code in (200, 503)
    assert '"status"' in r2.text
