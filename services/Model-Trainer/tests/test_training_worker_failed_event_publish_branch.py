from __future__ import annotations

from pathlib import Path

import pytest
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.worker import train_job
from model_trainer.worker.trainer_job_store import TrainerJobStore


def test_training_worker_failed_event_publish_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that training errors set status to failed and propagate the exception."""
    fake = FakeRedis()

    def _fake_redis(settings: Settings) -> FakeRedis:
        return fake

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)

    # Force container creation to raise so we enter exception handler
    class _C:
        @staticmethod
        def from_settings(_: Settings) -> None:
            raise RuntimeError("boom during container creation")

    monkeypatch.setattr(train_job, "ServiceContainer", _C)

    payload: TrainJobPayload = {
        "run_id": "run-err",
        "user_id": 1,
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
            "data_num_workers": None,
            "data_pin_memory": None,
            "early_stopping_patience": 5,
            "test_split_ratio": 0.15,
            "finetune_lr_cap": 5e-5,
            "precision": "auto",
        },
    }
    # Stub fetcher to return a local dir
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return tmp_path

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    with pytest.raises(RuntimeError, match="boom during container creation"):
        train_job.process_train_job(payload)
    # Status is failed and message is set
    status = TrainerJobStore(fake).load("run-err")
    assert status is not None and status["status"] == "failed"
    assert status["message"] is not None and "boom during container creation" in status["message"]
    fake.assert_only_called({"set", "hset", "hgetall", "publish"})
