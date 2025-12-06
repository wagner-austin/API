from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str
from platform_workers.testing import FakeRedis
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def test_runs_artifact_pointer_404_then_200() -> None:
    app = create_app(load_settings())
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake
    container.training_orchestrator._redis = fake

    client = TestClient(app)
    run_id = "run-x"

    r1 = client.get(f"/runs/{run_id}/artifact")
    assert r1.status_code == 404

    fake.set(f"runs:artifact:{run_id}:file_id", "deadbeef")
    r2 = client.get(f"/runs/{run_id}/artifact")
    assert r2.status_code == 200

    class _Ptr(TypedDict):
        storage: str
        file_id: str

    obj_raw = load_json_str(r2.text)
    if not isinstance(obj_raw, dict):
        raise AssertionError("Expected response to be a dict")
    assert "storage" in obj_raw and "file_id" in obj_raw
    body: _Ptr = {
        "storage": str(obj_raw.get("storage", "")),
        "file_id": str(obj_raw.get("file_id", "")),
    }
    assert body["storage"] == "data-bank" and body["file_id"] == "deadbeef"
