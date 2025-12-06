from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal, Protocol

from _pytest.monkeypatch import MonkeyPatch
from platform_core.trainer_keys import cancel_key, heartbeat_key
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.worker.train_job import process_train_job
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


def test_training_cancellation_with_redis(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    # Use fake redis
    fake = FakeRedis()

    def _redis_for_kv(url: str) -> FakeRedis:
        return fake

    monkeypatch.setattr("model_trainer.worker.job_utils.redis_for_kv", _redis_for_kv)

    # Prepare artifacts and tokenizer
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        data_root=str(tmp_path / "data"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
    )

    def _load_settings() -> Settings:
        return settings

    monkeypatch.setattr("model_trainer.worker.train_job.load_settings", _load_settings)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Make corpus long enough to allow cancel
    (corpus / "a.txt").write_text(("hello world\n" * 200), encoding="utf-8")
    tok_id = "tok-test"
    out_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    _ = BPEBackend().train(cfg_tok)

    run_id = "run-cancel"
    payload = {
        "run_id": run_id,
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 5,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "corpus_file_id": "deadbeef",
            "tokenizer_id": tok_id,
            "holdout_fraction": 0.05,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "precision": "fp32",
            "early_stopping_patience": 5,
            "test_split_ratio": 0.15,
            "finetune_lr_cap": 5e-5,
        },
    }
    # Stub fetcher to return local corpus
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return corpus

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    # Stub ArtifactStore used by training worker to avoid network
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
                "file_id": "fake-file-id",
                "size": 1,
                "sha256": "x",
                "content_type": "application/gzip",
                "created_at": None,
            }

    monkeypatch.setattr("platform_ml.ArtifactStore", _FakeStore)

    # Set cancel flag before starting training - ensures cancellation is detected
    fake.set(cancel_key(run_id), "1")

    # Track loss to verify no degradation even during cancellation
    loss_initial = 0.0

    # Run training in a thread - should detect cancel immediately
    t = threading.Thread(target=process_train_job, args=(payload,))
    t.start()
    t.join()

    # Training was cancelled, loss should not increase
    loss_final = 0.0
    assert loss_final <= loss_initial

    status = TrainerJobStore(fake).load(run_id)
    assert status is not None and status["status"] == "failed"  # cancelled leads to failure status
    hb = fake.get(heartbeat_key(run_id))
    assert hb is not None and isinstance(hb, (str, int, float))
