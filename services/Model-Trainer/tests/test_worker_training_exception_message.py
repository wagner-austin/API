from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.job_types import job_key
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.worker import train_job


def test_training_worker_sets_status_message_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that training errors set status to failed and message before propagating."""
    fake = FakeRedis()

    def _fake_redis(settings: Settings) -> FakeRedis:
        return fake

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)

    # Ensure artifacts root has no tokenizer so worker fails with AppError
    monkeypatch.setenv("APP__ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    # Build payload with nonexistent tokenizer
    payload: TrainJobPayload = {
        "run_id": "run-x",
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.0005,
            "corpus_file_id": "deadbeef",
            "tokenizer_id": "tok-missing",
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

    # Stub fetcher to point to local corpus dir
    (tmp_path / "corpus").mkdir()
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return tmp_path / "corpus"

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    with pytest.raises(AppError, match="Tokenizer artifact not found"):
        train_job.process_train_job(payload)  # raises due to missing tokenizer artifact

    # Status and message set in job store
    status_data = fake.hgetall(job_key("trainer", "run-x"))
    assert status_data["status"] == "failed"
    msg = status_data.get("error", "")
    assert "Tokenizer artifact not found" in msg
    fake.assert_only_called({"set", "hset", "hgetall", "publish"})
