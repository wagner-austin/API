"""Tests for pretrained model loading in train_job.py (lines 223-230)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, TrainOutcome
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.data import corpus_fetcher as cf
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.worker import train_job
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


class _BackendWithLoad:
    """Backend that supports loading pretrained models."""

    def __init__(self: _BackendWithLoad) -> None:
        self.load_called = False
        self.prepare_called = False
        self.loaded_from: str | None = None

    def prepare(
        self: _BackendWithLoad,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: str | Path,
    ) -> str:
        self.prepare_called = True
        return "prepared"

    def load(
        self: _BackendWithLoad,
        pretrained_dir: str,
        settings: Settings,
        *,
        tokenizer: str | Path,
    ) -> str:
        self.load_called = True
        self.loaded_from = pretrained_dir
        return "loaded"

    def train(
        self: _BackendWithLoad,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: str,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
    ) -> TrainOutcome:
        # Simulate training progress with decreasing loss
        # Args: step, epoch, loss, train_ppl, grad_norm, samples_per_sec, val_loss, val_ppl
        if progress:
            progress(0, 0, 2.5, 12.2, 0.5, 10.0, None, None)
            progress(1, 0, 1.8, 6.0, 0.4, 10.0, None, None)
            progress(2, 0, 1.2, 3.3, 0.3, 10.0, None, None)
            progress(3, 0, 0.9, 2.5, 0.2, 10.0, None, None)
            progress(4, 0, 0.5, 1.6, 0.1, 10.0, None, None)
        return TrainOutcome(
            cancelled=False,
            loss=0.5,
            perplexity=1.2,
            steps=5,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def save(self: _BackendWithLoad, prepared: str, out_dir: str) -> str:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "weights.bin").write_bytes(b"\x00mock")
        return out_dir


_BACKEND_INSTANCE: _BackendWithLoad | None = None


class _Reg:
    def get(self: _Reg, name: str) -> _BackendWithLoad:
        global _BACKEND_INSTANCE
        if _BACKEND_INSTANCE is None:
            _BACKEND_INSTANCE = _BackendWithLoad()
        return _BACKEND_INSTANCE


class _C:
    def __init__(self: _C) -> None:
        self.model_registry = _Reg()

    @staticmethod
    def from_settings(_: Settings) -> _C:
        return _C()


class _CF:
    def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def fetch(self: _CF, fid: str) -> Path:
        return self._cache_dir


def test_training_worker_loads_pretrained_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    """Cover train_job.py lines 223-230 - pretrained model loading branch."""
    global _BACKEND_INSTANCE
    _BACKEND_INSTANCE = None

    # Environment roots
    artifacts = tmp_path / "artifacts"
    runs = tmp_path / "runs"
    logs = tmp_path / "logs"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(runs),
        logs_root=str(logs),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )
    monkeypatch.setattr(train_job, "load_settings", lambda: settings)

    # Minimal corpus for tokenizer training
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nfinetuning data\n", encoding="utf-8")

    # Train a real tokenizer using BPEBackend
    tok_id = "tok-pretrained-test"
    tok_dir = artifacts / "tokenizers" / tok_id
    tok_cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    BPEBackend().train(tok_cfg)

    # Create a pretrained model directory (simulating a previous training run)
    pretrained_run_id = "run-pretrained-base"
    pretrained_model_dir = artifacts / "models" / pretrained_run_id
    pretrained_model_dir.mkdir(parents=True, exist_ok=True)
    (pretrained_model_dir / "weights.bin").write_bytes(b"\x00pretrained")
    (pretrained_model_dir / "manifest.json").write_text(
        '{"model_family": "gpt2", "model_size": "small"}', encoding="utf-8"
    )

    # Fake redis client
    fake = FakeRedis()

    def _fake_redis(settings: Settings) -> FakeRedis:
        return fake

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)
    monkeypatch.setattr(train_job, "ServiceContainer", _C)

    # Stub fetcher
    class _CorpusFetcher:
        def __init__(self: _CorpusFetcher, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CorpusFetcher, fid: str) -> Path:
            return corpus

    monkeypatch.setattr(cf, "CorpusFetcher", _CorpusFetcher)

    # Stub artifact store
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
                "file_id": "finetuned-file-id",
                "size": 1,
                "sha256": "x",
                "content_type": "application/gzip",
                "created_at": None,
            }

    monkeypatch.setattr("platform_ml.ArtifactStore", _FakeStore)

    # Track training losses via progress callback
    train_losses: list[float] = []

    def track_loss(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
        val_loss: float | None,
        val_ppl: float | None,
    ) -> None:
        train_losses.append(loss)

    # Patch the backend train method to use our progress callback
    original_train = _BackendWithLoad.train

    def patched_train(
        self: _BackendWithLoad,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: str,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
    ) -> TrainOutcome:
        # Use our tracking callback
        return original_train(
            self,
            cfg,
            settings,
            run_id=run_id,
            heartbeat=heartbeat,
            cancelled=cancelled,
            prepared=prepared,
            progress=track_loss,
        )

    monkeypatch.setattr(_BackendWithLoad, "train", patched_train)

    # Build payload with pretrained_run_id set
    payload: TrainJobPayload = {
        "run_id": "run-finetune",
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
            "pretrained_run_id": pretrained_run_id,
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

    train_job.process_train_job(payload)

    # Verify backend.load() was called instead of backend.prepare()
    assert _BACKEND_INSTANCE is not None and _BACKEND_INSTANCE.load_called is True
    assert _BACKEND_INSTANCE.prepare_called is False
    assert _BACKEND_INSTANCE.loaded_from == str(pretrained_model_dir)

    # Verify training worked by checking loss decreased
    assert len(train_losses) == 5, "Training should have reported 5 loss values"
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before, (
        f"Loss should decrease during training: {loss_before} -> {loss_after}"
    )

    # Verify status is completed
    status = TrainerJobStore(fake).load("run-finetune")
    assert status is not None and status["status"] == "completed"
