from __future__ import annotations

from fastapi.testclient import TestClient
from platform_workers.testing import FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def test_cancel_endpoint_sets_flag() -> None:
    app = create_app(load_settings())
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake

    client = TestClient(app)
    run_id = "run-x"
    r = client.post(f"/runs/{run_id}/cancel")
    assert r.status_code == 200
    v: str | int | bool | None = fake.get(f"runs:{run_id}:cancelled")
    assert isinstance(v, str) and v == "1"
    fake.assert_only_called({"set", "get"})
