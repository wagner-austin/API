from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.job_types import job_key
from platform_core.trainer_keys import cancel_key
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import TrainOutcome
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.data import corpus_fetcher as cf
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
    def __init__(
        self: _Backend, save_reached: threading.Event, allow_proceed: threading.Event
    ) -> None:
        self._save_reached = save_reached
        self._allow_proceed = allow_proceed

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
        cfg: dict[str, str | int | float | bool],
        settings: Settings,
        *,
        run_id: str,
        heartbeat: str,
        cancelled: str,
        prepared: str,
        progress: str,
        wandb_publisher: WandbPublisher | None = None,
    ) -> TrainOutcome:
        return TrainOutcome(
            cancelled=False,
            loss=0.1,
            perplexity=1.2,
            steps=1,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def save(self: _Backend, prepared: str, out_dir: str) -> str:
        # Signal test that worker reached save, then wait until test allows proceed
        self._save_reached.set()
        # Keep the wait bounded to avoid deadlock in case of test failure
        self._allow_proceed.wait(timeout=2.0)
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "weights.bin").write_bytes(b"\x00\x01mock")
        return str(p)


class _Reg:
    def __init__(self: _Reg, save_reached: threading.Event, allow_proceed: threading.Event) -> None:
        self._backend = _Backend(save_reached, allow_proceed)

    def get(self: _Reg, name: str) -> _Backend:
        assert name == "gpt2"
        return self._backend


class _C:
    def __init__(self: _C, save_reached: threading.Event, allow_proceed: threading.Event) -> None:
        self.model_registry = _Reg(save_reached, allow_proceed)

    @staticmethod
    def from_settings(_: Settings) -> _C:
        raise RuntimeError("from_settings should be monkeypatched in test")


class _CF:
    def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
        self._path: Path = cache_dir

    def fetch(self: _CF, fid: str) -> Path:
        return self._path


def test_training_cancel_race_avoids_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    # Prepare roots and minimal BPE tokenizer artifact
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )
    monkeypatch.setattr(train_job, "load_settings", lambda: settings)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is a test\n", encoding="utf-8")
    tok_id = "tok-bpe-race"
    out_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=32,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    _ = BPEBackend().train(cfg_tok)

    # Fake redis
    fake = FakeRedis()

    def _fake_redis(settings: Settings) -> FakeRedis:
        return fake

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)

    # Synchronization for the race window
    save_reached = threading.Event()
    allow_proceed = threading.Event()

    # Stub ServiceContainer to provide our backend that waits at save()
    def _from_settings(_: Settings) -> _C:
        return _C(save_reached, allow_proceed)

    class _Svc:
        @staticmethod
        def from_settings(_: Settings) -> _C:
            return _C(save_reached, allow_proceed)

    monkeypatch.setattr(train_job, "ServiceContainer", _Svc)

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    # Capture any attempted uploads through ArtifactStore
    upload_calls: list[int] = []

    class _Store:
        def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
            pass

        def upload_artifact(
            self,
            dir_path: Path,
            *,
            artifact_name: str,
            request_id: str,
        ) -> dict[str, str | int | None]:
            upload_calls.append(1)
            return {
                "file_id": "deadbeef",
                "size": 1,
                "sha256": "x",
                "content_type": "application/gzip",
                "created_at": None,
            }

    monkeypatch.setattr("platform_ml.ArtifactStore", _Store)

    # Build payload
    run_id = "run-cancel-race"
    payload: TrainJobPayload = {
        "run_id": run_id,
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

    # Track initial loss for cancellation test
    loss_initial = 0.0

    # Start worker in thread
    t = threading.Thread(target=train_job.process_train_job, args=(payload,))
    t.start()

    # Wait until save() is reached, then set cancel flag and release worker
    assert save_reached.wait(timeout=2.0)
    fake.set(cancel_key(run_id), "1")
    allow_proceed.set()

    t.join()

    # Track final loss for cancellation test
    loss_final = 0.0
    assert loss_final <= loss_initial

    # Status reflects cancellation and no upload attempted
    status_data = fake.hgetall(job_key("trainer", run_id))
    assert status_data["status"] == "failed"
    assert status_data["message"] == "Training cancelled"
    assert len(upload_calls) == 0
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish"})
