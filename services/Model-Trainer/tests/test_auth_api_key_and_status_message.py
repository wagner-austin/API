from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.worker.trainer_job_store import TrainerJobStore


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


def test_api_key_unauthorized_and_authorized(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"), security_api_key="sekret"
    )
    app = create_app(settings)

    # Wire fake redis and stub enqueuer
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake
    container.training_orchestrator._redis = fake
    container.training_orchestrator._job_store = TrainerJobStore(fake)

    def _fake_enqueue_train(payload: dict[str, str | int | float | bool | None]) -> str:
        return "job-1"

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_train", _fake_enqueue_train)

    # Stub CorpusFetcher to map file id to local corpus path
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, file_id: str) -> Path:
            (tmp_path / "corpus").mkdir(exist_ok=True)
            return tmp_path / "corpus"

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    client = TestClient(app)

    body = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0005,
        "corpus_file_id": "deadbeef",
        "tokenizer_id": "tok-1",
        "user_id": 123,
    }
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_text("hello", encoding="utf-8")

    r1 = client.post("/runs/train", json=body)
    # Requires API key -> 401
    assert r1.status_code == 401

    obj_err_raw = load_json_str(r1.text)
    assert isinstance(obj_err_raw, dict) and "code" in obj_err_raw
    obj_err: dict[str, JSONValue] = obj_err_raw
    code_o: JSONValue = obj_err.get("code")
    rid_o: JSONValue = obj_err.get("request_id")
    assert isinstance(code_o, str) and code_o == "UNAUTHORIZED"
    assert isinstance(rid_o, str) and len(rid_o) > 0

    r2 = client.post("/runs/train", json=body, headers={"X-Api-Key": "sekret"})
    assert r2.status_code == 200
    obj_train_raw = load_json_str(r2.text)
    assert isinstance(obj_train_raw, dict) and "run_id" in obj_train_raw
    obj_train: dict[str, JSONValue] = obj_train_raw
    run_id = str(obj_train.get("run_id"))

    # Populate a status message and verify surface via GET /runs/{id}
    now = datetime.utcnow()
    TrainerJobStore(fake).save(
        {
            "job_id": run_id,
            "user_id": 123,
            "status": "failed",
            "progress": 100,
            "message": "boom",
            "created_at": now,
            "updated_at": now,
            "error": "boom",
            "artifact_file_id": None,
        },
    )

    r3 = client.get(f"/runs/{run_id}", headers={"X-Api-Key": "sekret"})
    assert r3.status_code == 200
    obj_status_raw = load_json_str(r3.text)
    assert isinstance(obj_status_raw, dict) and "status" in obj_status_raw
    obj_status: dict[str, JSONValue] = obj_status_raw
    status_o: JSONValue = obj_status.get("status")
    message_o: JSONValue = obj_status.get("message")
    assert isinstance(status_o, str) and status_o == "failed"
    assert isinstance(message_o, str) and message_o == "boom"
