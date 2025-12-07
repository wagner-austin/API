from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def test_run_status_not_found_maps_to_app_error() -> None:
    from model_trainer.worker.trainer_job_store import TrainerJobStore

    app = create_app(load_settings())
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake
    container.training_orchestrator._redis = fake
    container.training_orchestrator._job_store = TrainerJobStore(fake)
    client = TestClient(app)

    r = client.get("/runs/nonexistent")
    assert r.status_code == 404

    obj_raw = load_json_str(r.text)
    if not isinstance(obj_raw, dict):
        raise AssertionError("Expected response to be a dict")
    assert "code" in obj_raw and "message" in obj_raw
    obj: dict[str, JSONValue] = obj_raw
    err_code = obj.get("code")
    msg = obj.get("message")
    assert err_code == "RUN_NOT_FOUND", f"Expected error code RUN_NOT_FOUND, got {err_code}"
    assert msg and "not found" in str(msg).lower(), (
        f"Expected message to contain 'not found', got {msg}"
    )
    fake.assert_only_called({"hgetall"})
