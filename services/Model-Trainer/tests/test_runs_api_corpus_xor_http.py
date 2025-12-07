from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

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


def _mk_app(tmp: Path, settings_factory: _SettingsFactory) -> tuple[TestClient, FakeRedis]:
    settings = settings_factory(
        artifacts_root=str(tmp / "artifacts"),
        runs_root=str(tmp / "runs"),
    )
    app = create_app(settings)

    # Swap out redis with _FakeRedis
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake
    container.training_orchestrator._redis = fake
    return TestClient(app), fake


def test_runs_train_missing_corpus_file_id_returns_400(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    client, fake = _mk_app(tmp_path, settings_factory)
    body = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 5e-4,
        # Missing required corpus_file_id
        "tokenizer_id": "tok-1",
        "user_id": 1,
    }
    r = client.post("/runs/train", json=body)
    # Validator returns 400 for missing required fields
    assert r.status_code == 400
    assert "corpus_file_id" in r.text
    fake.assert_only_called(set())


def test_runs_train_extra_field_corpus_path_forbidden_returns_422(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    client, fake = _mk_app(tmp_path, settings_factory)
    body = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 5e-4,
        "corpus_file_id": "deadbeef",
        "corpus_path": str(tmp_path / "corpus.txt"),
        "tokenizer_id": "tok-1",
        "user_id": 1,
    }
    r = client.post("/runs/train", json=body)
    assert r.status_code == 422
    assert "extra fields not allowed" in r.text.lower()
    fake.assert_only_called(set())
