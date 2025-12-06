from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
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


def test_runs_train_and_status_and_logs(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(artifacts_root=str(artifacts))
    app = create_app(settings)

    # Access container for patching
    container: ServiceContainer = app.state.container  # type narrowing

    # Swap redis client in orchestrator + container to _FakeRedis
    from model_trainer.worker.trainer_job_store import TrainerJobStore

    fake = FakeRedis()
    container.redis = fake
    container.training_orchestrator._redis = fake
    container.training_orchestrator._job_store = TrainerJobStore(fake)

    # Stub RQ enqueuer
    def _fake_enqueue_train(payload: dict[str, str | int | float | bool | None]) -> str:
        return "job-test"

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_train", _fake_enqueue_train)

    # Stub CorpusFetcher to map file id to local corpus path
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, file_id: str) -> Path:
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
        "tokenizer_id": "tok-abc",
        "user_id": 1,
    }
    # Create a minimal corpus to allow stats/logging paths to proceed
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_text("hello", encoding="utf-8")

    r = client.post("/runs/train", json=body)
    assert r.status_code == 200

    class _TrainResp(TypedDict):
        run_id: str
        job_id: str

    obj_raw = load_json_str(r.text)
    assert isinstance(obj_raw, dict) and "run_id" in obj_raw
    obj: dict[str, JSONValue] = obj_raw
    run_id = str(obj.get("run_id"))

    # Status should be queued
    r2 = client.get(f"/runs/{run_id}")
    assert r2.status_code == 200

    class _Status(TypedDict):
        run_id: str
        status: str
        last_heartbeat_ts: float | None
        message: str | None

    obj2_raw = load_json_str(r2.text)
    assert isinstance(obj2_raw, dict) and "status" in obj2_raw
    obj2: dict[str, JSONValue] = obj2_raw
    hb_v = obj2.get("last_heartbeat_ts")
    msg_v = obj2.get("message")
    st: _Status = {
        "run_id": str(obj2.get("run_id", "")),
        "status": str(obj2.get("status", "")),
        "last_heartbeat_ts": float(hb_v) if isinstance(hb_v, int | float) else None,
        "message": (str(msg_v) if isinstance(msg_v, str) else None),
    }
    assert st["status"] == "queued"

    # Create a per-run log with known content and verify GET returns it
    log_dir = artifacts / "models" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "logs.jsonl").write_text('{"msg":"hello"}\n', encoding="utf-8")
    logs = client.get(f"/runs/{run_id}/logs", params={"tail": 10})
    assert logs.status_code == 200
    assert "hello" in logs.text
