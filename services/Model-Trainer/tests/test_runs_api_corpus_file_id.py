from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from platform_workers.testing import FakeRedis

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


def test_runs_train_with_corpus_file_id(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(artifacts_root=str(artifacts))
    app = create_app(settings)

    container: ServiceContainer = app.state.container  # type narrowing

    # Swap redis to _FakeRedis
    from model_trainer.worker.trainer_job_store import TrainerJobStore

    fake = FakeRedis()
    container.redis = fake
    container.training_orchestrator._redis = fake
    container.training_orchestrator._job_store = TrainerJobStore(fake)

    # Stub RQ to capture payload
    from model_trainer.core.contracts.queue import TrainJobPayload, TrainRequestPayload

    captured: list[TrainJobPayload] = []

    def _fake_enqueue_train(payload: TrainJobPayload) -> str:
        captured.append(payload)
        return "job-cfid"

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_train", _fake_enqueue_train)

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

    r = client.post("/runs/train", json=body)
    assert r.status_code == 200
    assert captured, "payload should be captured"
    req: TrainRequestPayload = captured[0]["request"]
    assert req["corpus_file_id"] == "deadbeef"
    fake.assert_only_called({"hset"})
