from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.job_types import job_key
from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core._test_hooks import CorpusFetcherProto
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.worker import train_job


def test_training_worker_sets_status_message_on_exception(tmp_path: Path) -> None:
    """Test that training errors set status to failed and message before propagating."""
    fake = FakeRedis()

    def _fake_kv(url: str) -> FakeRedis:
        return fake

    _test_hooks.kv_store_factory = _fake_kv

    # Stub fetcher to point to local corpus dir
    (tmp_path / "corpus").mkdir()

    class _FakeCorpusFetcher:
        def __init__(self, api_url: str, api_key: str, cache_dir: Path) -> None:
            self._tmp = tmp_path

        def fetch(self, fid: str) -> Path:
            return self._tmp / "corpus"

    def _fake_fetcher_factory(api_url: str, api_key: str, cache_dir: Path) -> CorpusFetcherProto:
        return _FakeCorpusFetcher(api_url, api_key, cache_dir)

    _test_hooks.corpus_fetcher_factory = _fake_fetcher_factory

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

    with pytest.raises(AppError, match="Tokenizer artifact not found"):
        train_job.process_train_job(payload)  # raises due to missing tokenizer artifact

    # Status and message set in job store
    status_data = fake.hgetall(job_key("trainer", "run-x"))
    assert status_data["status"] == "failed"
    msg = status_data.get("error", "")
    assert "Tokenizer artifact not found" in msg
    fake.assert_only_called({"set", "hset", "hgetall", "publish"})
