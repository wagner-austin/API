"""Pretrained training worker: setup and requests."""

from __future__ import annotations

from pathlib import Path

from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.contracts.queue_encoding import encode_train_job_payload
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.worker import train_job
from model_trainer.worker.trainer_job_store import TrainerJobStore
from tests._pretrained_worker_support import (
    _BackendWithLoad,
    _create_artifact_store_factory,
    _create_corpus_fetcher_factory,
    _create_service_container_factory,
    _SettingsFactory,
)


def test_training_worker_loads_pretrained_model(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Cover train_job.py lines 223-230 - pretrained model loading branch."""
    # Track backend instance and losses for assertions
    backend_instance_holder: list[_BackendWithLoad | None] = [None]
    train_losses: list[float] = []

    # Environment roots
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )

    # Settings via hook
    _test_hooks.load_settings = lambda: settings

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

    # Fake redis via hook
    fake = FakeRedis()
    _test_hooks.kv_store_factory = lambda url: fake

    # Set up hooks using extracted factory functions
    _test_hooks.service_container_from_settings = _create_service_container_factory(
        fake, backend_instance_holder, train_losses
    )
    _test_hooks.corpus_fetcher_factory = _create_corpus_fetcher_factory(corpus)
    _test_hooks.artifact_store_factory = _create_artifact_store_factory()

    # Build payload with pretrained_run_id set
    payload: TrainJobPayload = {
        "run_id": "run-finetune",
        "user_id": 1,
        "resume": False,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "tokenizer_id": tok_id,
            "corpus_file_id": "deadbeef",
            "corpus_format": "lines",
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
            "loss_mask_prefix_separator": None,
            "precision": "auto",
            "hub_model_id": None,
            "finetuning_strategy": "full",
            "lora": None,
            "quantization": None,
            "gguf_export": None,
        },
    }

    train_job.process_train_job(encode_train_job_payload(payload))

    # Verify backend.load() was called instead of backend.prepare()
    backend_instance = backend_instance_holder[0]
    assert backend_instance is not None and backend_instance.load_called is True
    assert backend_instance.prepare_called is False
    assert backend_instance.loaded_from == str(pretrained_model_dir)
    # A fresh enqueue trains from scratch: the payload's resume flag reaches
    # the backend as False.
    assert backend_instance.resume_seen is False

    # Verify loss decreases during training (ml-train-no-loss-check)
    assert len(train_losses) >= 2, "Should have at least 2 loss values"
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before, f"Loss should decrease: {loss_before} -> {loss_after}"

    # Verify status is completed
    status = TrainerJobStore(fake).load("run-finetune")
    assert status is not None and status["status"] == "completed"
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish", "expire"})
