from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
from model_trainer.api.schemas.runs import EvaluateRequest, EvaluateResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        runs_root: str | None = ...,
        logs_root: str | None = ...,
        data_root: str | None = ...,
        data_bank_api_url: str | None = ...,
        data_bank_api_key: str | None = ...,
        threads: int | None = ...,
        redis_url: str | None = ...,
        app_env: Literal["dev", "prod"] | None = ...,
        security_api_key: str | None = ...,
    ) -> Settings: ...


def test_runs_evaluate_and_eval_result_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    app = create_app(settings_factory(artifacts_root=str(tmp_path / "artifacts")))
    cont: ServiceContainer = app.state.container

    # Stub orchestrator methods
    def _enq(run_id: str, req: EvaluateRequest) -> EvaluateResponse:
        return {
            "run_id": run_id,
            "split": req["split"],
            "status": "queued",
            "loss": None,
            "perplexity": None,
            "artifact_path": None,
        }

    def _get(run_id: str) -> EvaluateResponse:
        return {
            "run_id": run_id,
            "split": "validation",
            "status": "completed",
            "loss": 1.0,
            "perplexity": 2.0,
            "artifact_path": None,
        }

    monkeypatch.setattr(cont.training_orchestrator, "enqueue_evaluation", _enq)
    monkeypatch.setattr(cont.training_orchestrator, "get_evaluation", _get)

    client = TestClient(app)
    run_id = "r-eval"
    payload: dict[str, str | None] = {"split": "validation", "path_override": None}
    r1 = client.post(f"/runs/{run_id}/evaluate", json=payload)
    assert r1.status_code == 200

    class _EvalTD(TypedDict):
        run_id: str
        split: str
        status: str
        loss: float | None
        perplexity: float | None
        artifact_path: str | None

    obj1_raw = load_json_str(r1.text)
    if not isinstance(obj1_raw, dict):
        raise AssertionError("Response must be a JSON object")
    obj1: dict[str, JSONValue] = obj1_raw
    assert "run_id" in obj1, "Response must contain 'run_id' key"
    assert obj1["run_id"] == run_id, f"Response run_id must be '{run_id}'"
    assert "split" in obj1, "Response must contain 'split' key"
    assert obj1["split"] == "validation", "Response split must be 'validation'"
    assert "status" in obj1, "Response must contain 'status' key"
    assert obj1["status"] == "queued", "Response status must be 'queued'"
    assert "loss" in obj1, "Response must contain 'loss' key"
    assert "perplexity" in obj1, "Response must contain 'perplexity' key"
    assert "artifact_path" in obj1, "Response must contain 'artifact_path' key"
    r2 = client.get(f"/runs/{run_id}/eval")
    assert r2.status_code == 200
    obj2_raw = load_json_str(r2.text)
    if not isinstance(obj2_raw, dict):
        raise AssertionError("Response must be a JSON object")
    obj2: dict[str, JSONValue] = obj2_raw
    assert "run_id" in obj2, "Response must contain 'run_id' key"
    assert obj2["run_id"] == run_id, f"Response run_id must be '{run_id}'"
    assert "split" in obj2, "Response must contain 'split' key"
    assert obj2["split"] == "validation", "Response split must be 'validation'"
    assert "status" in obj2, "Response must contain 'status' key"
    assert obj2["status"] == "completed", "Response status must be 'completed'"
    assert "loss" in obj2, "Response must contain 'loss' key"
    assert obj2["loss"] == 1.0, "Response loss must be 1.0"
    assert "perplexity" in obj2, "Response must contain 'perplexity' key"
    assert obj2["perplexity"] == 2.0, "Response perplexity must be 2.0"
    assert "artifact_path" in obj2, "Response must contain 'artifact_path' key"
