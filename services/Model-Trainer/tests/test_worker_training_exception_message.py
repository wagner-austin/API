from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.job_types import job_key
from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core._hook_protocols_ml import CorpusFetcherProto
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.contracts.queue_encoding import encode_train_job_payload
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
        "resume": False,
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
            "loss_mask_prefix_separator": None,
            "precision": "auto",
            "hub_model_id": None,
            "finetuning_strategy": "full",
            "lora": None,
            "quantization": None,
            "gguf_export": None,
        },
    }

    with pytest.raises(AppError, match="No tokenizer artifacts found in"):
        train_job.process_train_job(encode_train_job_payload(payload))

    # Status and message set in job store
    status_data = fake.hgetall(job_key("trainer", "run-x"))
    assert status_data["status"] == "failed"
    msg = status_data.get("error", "")
    assert "No tokenizer artifacts found in" in msg
    fake.assert_only_called({"set", "hset", "hgetall", "publish", "expire"})


def test_training_worker_sets_status_failed_when_corpus_fetch_raises(tmp_path: Path) -> None:
    """A corpus-fetch failure must mark the run failed, not leave it running.

    Observed live 2026-08-18 (run hf_lm-small-1787078741-2d48a0c4): a data-bank
    NotFoundError escaped before the failure guard opened, so the job store
    advertised the run as running forever while the job itself was gone.
    """
    fake = FakeRedis()

    def _fake_kv(url: str) -> FakeRedis:
        return fake

    _test_hooks.kv_store_factory = _fake_kv

    class _RaisingCorpusFetcher:
        def __init__(self, api_url: str, api_key: str, cache_dir: Path) -> None:
            self._cache_dir = cache_dir

        def fetch(self, fid: str) -> Path:
            raise RuntimeError(f"corpus {fid} not found in data bank")

    def _raising_fetcher_factory(api_url: str, api_key: str, cache_dir: Path) -> CorpusFetcherProto:
        return _RaisingCorpusFetcher(api_url, api_key, cache_dir)

    _test_hooks.corpus_fetcher_factory = _raising_fetcher_factory

    payload: TrainJobPayload = {
        "run_id": "run-corpus-fetch-fails",
        "user_id": 1,
        "resume": False,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.0005,
            "corpus_file_id": "deadbeef",
            "tokenizer_id": "tok",
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
            "loss_mask_prefix_separator": None,
            "precision": "auto",
            "hub_model_id": None,
            "finetuning_strategy": "full",
            "lora": None,
            "quantization": None,
            "gguf_export": None,
        },
    }

    with pytest.raises(RuntimeError, match="corpus deadbeef not found in data bank"):
        train_job.process_train_job(encode_train_job_payload(payload))

    status_data = fake.hgetall(job_key("trainer", "run-corpus-fetch-fails"))
    assert status_data["status"] == "failed"
    assert "corpus deadbeef not found in data bank" in status_data.get("error", "")
    fake.assert_only_called({"set", "hset", "hgetall", "publish", "expire"})
