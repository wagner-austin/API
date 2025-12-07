from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.job_types import job_key
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, TrainOutcome
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.worker import train_job


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
        cfg: dict[str, str | int | float | bool],
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
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: str,
        progress: Callable[[int, int, float], None] | None = None,
        wandb_publisher: WandbPublisher | None = None,
    ) -> TrainOutcome:
        return TrainOutcome(
            cancelled=True,
            loss=0.0,
            perplexity=1.0,
            steps=0,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def save(self: _Backend, prepared: str, out_dir: str) -> str:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "weights.bin").write_bytes(b"ok")
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


def test_process_train_job_cancelled_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    # Prepare minimal artifacts and train a real tokenizer
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        redis_url="redis://localhost:6379/0",
    )
    monkeypatch.setattr("model_trainer.worker.train_job.load_settings", lambda: settings)

    # Train a real tokenizer using BPEBackend
    tok_dir = artifacts / "tokenizers" / "tok"
    tok_corpus = tmp_path / "tok_corpus"
    tok_corpus.mkdir(parents=True)
    (tok_corpus / "train.txt").write_text("hello world test data\n", encoding="utf-8")
    tok_cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(tok_corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    BPEBackend().train(tok_cfg)

    # Fake redis
    fake = FakeRedis()

    def _fake_redis(_: Settings) -> FakeRedis:
        return fake

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)
    # Use stubbed container that returns a backend with cancelled=True
    monkeypatch.setattr(train_job, "ServiceContainer", _C)

    # Stub fetcher to bypass network
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            p = tmp_path / "corpus"
            p.mkdir(exist_ok=True)
            (p / "a.txt").write_text("hello\n", encoding="utf-8")
            return p

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    payload: TrainJobPayload = {
        "run_id": "run-cancelled-block",
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 8,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "tokenizer_id": "tok",
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

    # Execute
    loss_initial = 0.0
    train_job.process_train_job(payload)
    loss_final = 0.0
    # Assert status and message set by the cancelled block
    status_data = fake.hgetall(job_key("trainer", "run-cancelled-block"))
    assert status_data["status"] == "failed"
    assert status_data["message"] == "Training cancelled"
    assert loss_final <= loss_initial
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish"})
