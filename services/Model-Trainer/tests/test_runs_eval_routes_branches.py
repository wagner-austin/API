from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from fastapi.testclient import TestClient
from platform_core.job_types import job_key
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_core.trainer_keys import eval_key
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings


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
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    # Set up FakeRedis via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    settings = settings_factory(artifacts_root=str(tmp_path / "artifacts"))
    app = create_app(settings)

    run_id = "r-eval"

    # Pre-populate job status so enqueue_evaluation can proceed (it checks run status)
    fake.hset(
        job_key("trainer", run_id),
        {
            "job_id": run_id,
            "user_id": "1",
            "status": "completed",
            "progress": "100",
            "message": "Training completed",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "error": "",
            "artifact_file_id": "",
        },
    )

    client = TestClient(app)
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
    assert obj1["loss"] is None, "Response loss must be None for queued status"
    assert "perplexity" in obj1, "Response must contain 'perplexity' key"
    assert obj1["perplexity"] is None, "Response perplexity must be None for queued status"
    assert "artifact_path" in obj1, "Response must contain 'artifact_path' key"
    assert obj1["artifact_path"] is None, "Response artifact_path must be None for queued status"

    # Set completed eval result in Redis for the GET request
    fake.set(
        eval_key(run_id),
        dump_json_str(
            {
                "status": "completed",
                "split": "validation",
                "loss": 1.0,
                "ppl": 2.0,
                "artifact": None,
            }
        ),
    )

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
    assert obj2["artifact_path"] is None, "Response artifact_path must be None"
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish"})
