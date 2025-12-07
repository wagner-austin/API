from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.trainer_keys import artifact_file_id_key
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, TrainOutcome
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.services.data import corpus_fetcher as cf
from model_trainer.worker import train_job
from model_trainer.worker.trainer_job_store import TrainerJobStore

_CORPUS_PATH: Path | None = None


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


class _Backend:
    def prepare(
        self: _Backend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: str | Path,
    ) -> str:
        return "prepared"

    def train(
        self: _Backend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: str,
        cancelled: str,
        prepared: str,
        progress: str,
        wandb_publisher: WandbPublisher | None = None,
    ) -> TrainOutcome:
        # Exercise the worker's progress callback wrapper so that the
        # training_worker._progress closure is covered.
        if callable(progress):
            progress(1, 0, 0.5, 1.65, 0.1, 100.0, None, None)
        return TrainOutcome(
            cancelled=False,
            loss=0.9,
            perplexity=1.5,
            steps=10,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def save(self: _Backend, prepared: str, out_dir: str) -> str:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "weights.bin").write_bytes(b"\x00mock")
        return out_dir


class _Reg:
    def get(self: _Reg, name: str) -> _Backend:
        return _Backend()


class _C:
    def __init__(self: _C) -> None:
        self.model_registry = _Reg()

    @staticmethod
    def from_settings(_: Settings) -> _C:
        return _C()


class _CF:
    def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
        pass

    def fetch(self: _CF, fid: str) -> Path:
        assert _CORPUS_PATH is not None
        return _CORPUS_PATH


def test_training_worker_spm_artifact_and_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    # Environment roots
    artifacts = tmp_path / "artifacts"
    runs = tmp_path / "runs"
    logs = tmp_path / "logs"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(runs),
        logs_root=str(logs),
        data_root=str(tmp_path / "data"),
    )
    monkeypatch.setattr(train_job, "load_settings", lambda: settings)

    # Provide sentencepiece tokenizer artifact (spm) so worker loads tok_spm path
    tok_id = "tok-spm"
    tok_dir = artifacts / "tokenizers" / tok_id
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "tokenizer.model").write_bytes(b"\x00\x01mock")
    (tok_dir / "tokenizer.vocab").write_text("[UNK]\t0\nA\t1\n", encoding="utf-8")

    # Minimal corpus
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is a test\n", encoding="utf-8")

    # Fake redis client
    fake = FakeRedis()

    def _fake_redis(settings: Settings) -> FakeRedis:
        return fake

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)

    monkeypatch.setattr(train_job, "ServiceContainer", _C)

    payload: TrainJobPayload = {
        "run_id": "run-complete",
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "tokenizer_id": tok_id,
            "corpus_file_id": "deadbeef",
            "holdout_fraction": 0.01,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "data_num_workers": None,
            "data_pin_memory": None,
            "early_stopping_patience": 5,
            "test_split_ratio": 0.15,
            "finetune_lr_cap": 5e-5,
            "precision": "auto",
        },
    }

    # Stub fetcher to map file id to local corpus
    global _CORPUS_PATH
    _CORPUS_PATH = corpus
    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    # Stub artifact store to avoid real network, and assert pointer is stored
    class _FakeStore:
        def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
            pass

        def upload_artifact(
            self,
            dir_path: Path,
            *,
            artifact_name: str,
            request_id: str,
        ) -> dict[str, str | int | None]:
            return {
                "file_id": "deadbeef",
                "size": 1,
                "sha256": "x",
                "content_type": "application/gzip",
                "created_at": None,
            }

    monkeypatch.setattr("platform_ml.ArtifactStore", _FakeStore)
    settings_with_db = settings_factory(
        artifacts_root=settings["app"]["artifacts_root"],
        runs_root=settings["app"]["runs_root"],
        logs_root=settings["app"]["logs_root"],
        data_root=settings["app"]["data_root"],
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )
    monkeypatch.setattr(train_job, "load_settings", lambda: settings_with_db)

    train_job.process_train_job(payload)

    status = TrainerJobStore(fake).load("run-complete")
    assert status is not None and status["status"] == "completed"
    # Pointer persisted for inference service
    artifact_id = fake.get(artifact_file_id_key("run-complete"))
    out_dir = artifacts / "models" / "run-complete"

    assert status["message"] == "Training completed"
    assert artifact_id == "deadbeef"
    # Cleanup enabled by default: local artifact directory should be removed
    assert not out_dir.exists()
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish"})
