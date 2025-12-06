from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_core.trainer_keys import eval_key
from platform_workers.testing import FakeRedis
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.infra.persistence.models import EvalCache


def test_get_eval_returns_artifact_pointer() -> None:
    app = create_app(load_settings())
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake
    container.training_orchestrator._redis = fake

    run_id = "run-eval"
    # Pre-populate evaluation cache to simulate a completed eval
    cache = EvalCache(
        status="completed",
        split="validation",
        loss=0.5,
        ppl=1.5,
        artifact="/data/artifacts/models/run-eval/eval/metrics.json",
    )
    fake.set(eval_key(run_id), dump_json_str(cache))

    client = TestClient(app)
    r = client.get(f"/runs/{run_id}/eval")
    assert r.status_code == 200

    class _EvalTD(TypedDict):
        run_id: str
        split: str
        status: str
        loss: float | None
        perplexity: float | None
        artifact_path: str | None

    obj_raw = load_json_str(r.text)
    if not isinstance(obj_raw, dict):
        raise AssertionError("Expected response to be a dict")
    assert "status" in obj_raw and "run_id" in obj_raw and "split" in obj_raw
    obj: dict[str, JSONValue] = obj_raw
    loss_v = obj.get("loss")
    ppl_v = obj.get("perplexity")
    art_v = obj.get("artifact_path")
    body: _EvalTD = {
        "run_id": str(obj.get("run_id", "")),
        "split": str(obj.get("split", "")),
        "status": str(obj.get("status", "")),
        "loss": float(loss_v) if isinstance(loss_v, int | float) else None,
        "perplexity": float(ppl_v) if isinstance(ppl_v, int | float) else None,
        "artifact_path": str(art_v) if isinstance(art_v, str) else None,
    }
    assert body["status"] == "completed"
    expected_suffix = "/models/run-eval/eval/metrics.json"
    assert body["artifact_path"] and body["artifact_path"].endswith(expected_suffix)
