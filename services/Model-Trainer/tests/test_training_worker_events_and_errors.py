from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.worker import job_utils, train_job
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


def _configure_worker_roots(
    tmp_path: Path, settings_factory: _SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    artifacts_root = tmp_path / "artifacts"
    runs_root = tmp_path / "runs"
    logs_root = tmp_path / "logs"
    settings = settings_factory(
        artifacts_root=str(artifacts_root),
        runs_root=str(runs_root),
        logs_root=str(logs_root),
        data_root=str(tmp_path / "data"),
    )
    monkeypatch.setattr(train_job, "load_settings", lambda: settings)
    return artifacts_root


def test_emit_metrics_helpers_publish() -> None:
    # Use a FakeRedis to exercise publish path without failures
    r = FakeRedis()
    run_id = "r-1"
    cfg = ModelTrainConfig(
        model_family="gpt2",
        model_size="small",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-4,
        tokenizer_id="tok",
        corpus_path="/dev/null",
        holdout_fraction=0.01,
        seed=42,
        pretrained_run_id=None,
        freeze_embed=False,
        gradient_clipping=1.0,
        optimizer="adamw",
        device="cpu",
        data_num_workers=0,
        data_pin_memory=False,
        early_stopping_patience=5,
        test_split_ratio=0.15,
        finetune_lr_cap=5e-5,
        precision="fp32",
    )
    # Does not raise - metrics events (no failed event - handled by job_events)
    job_utils.emit_config_event(r, run_id, 123, cfg, threads=2)
    job_utils.emit_progress_metrics(
        r,
        run_id,
        123,
        epoch=1,
        total_epochs=1,
        step=1,
        train_loss=1.23,
        train_ppl=3.42,
        grad_norm=0.1,
        samples_per_sec=100.0,
    )
    job_utils.emit_completed_metrics(
        r, run_id, 123, test_loss=0.9, test_ppl=1.5, artifact_path="/a"
    )


def test_process_train_job_sets_status_message_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    _configure_worker_roots(tmp_path, settings_factory, monkeypatch)

    payload: TrainJobPayload = {
        "run_id": "run-exc",
        "user_id": 7,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "tokenizer_id": "tok",
            "corpus_file_id": "deadbeef",
            "holdout_fraction": 0.01,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "data_num_workers": 0,
            "data_pin_memory": False,
            "early_stopping_patience": 5,
            "test_split_ratio": 0.15,
            "finetune_lr_cap": 5e-5,
            "precision": "auto",
        },
    }
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            assert fid == "deadbeef"
            return corpus_root

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    (corpus_root / "a.txt").write_text("hello\n", encoding="utf-8")

    # Fake redis
    client = FakeRedis()

    def _fake_redis(settings: Settings) -> FakeRedis:
        return client

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)

    # Force an exception by making container/model registry lookups blow up
    class _C:
        @staticmethod
        def from_settings(_: Settings) -> None:
            raise RuntimeError("container creation failed")

    monkeypatch.setattr(train_job, "ServiceContainer", _C)

    with pytest.raises(RuntimeError):
        train_job.process_train_job(payload)
    status = TrainerJobStore(client).load("run-exc")
    assert status is not None and status["status"] == "failed"
    assert status["error"] is not None and "container creation failed" in status["error"]
