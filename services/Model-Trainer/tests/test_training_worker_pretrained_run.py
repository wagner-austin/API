"""Pretrained training worker: execution and outcomes."""

from __future__ import annotations

from pathlib import Path

from platform_core.trainer_keys import artifact_file_id_key
from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core._hook_protocols import (
    ArtifactStoreProto,
)
from model_trainer.core.contracts.model import (
    LoraConfig,
)
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
    _create_hf_lm_service_container_factory,
    _create_service_container_factory,
    _HfLmBackend,
    _RecordingStore,
    _SettingsFactory,
)


def test_training_worker_hf_lm_with_tokenizer_id_none(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Cover train_job.py line 293 - tokenizer_id=None branch for hf_lm models."""
    backend_instance_holder: list[_HfLmBackend | None] = []
    train_losses: list[float] = []

    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )

    _test_hooks.load_settings = lambda: settings

    # Create corpus (no tokenizer needed for hf_lm)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nhf_lm training data\n", encoding="utf-8")

    fake = FakeRedis()
    _test_hooks.kv_store_factory = lambda url: fake

    _test_hooks.service_container_from_settings = _create_hf_lm_service_container_factory(
        fake, backend_instance_holder, train_losses
    )
    _test_hooks.corpus_fetcher_factory = _create_corpus_fetcher_factory(corpus)
    _test_hooks.artifact_store_factory = _create_artifact_store_factory()

    # Build payload with tokenizer_id=None (hf_lm uses HF tokenizer from hub_model_id)
    payload: TrainJobPayload = {
        "run_id": "run-hflm-no-tok",
        "user_id": 1,
        "resume": False,
        "request": {
            "model_family": "hf_lm",
            "model_size": "small",
            "max_seq_len": 128,
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "tokenizer_id": None,  # None for hf_lm - uses HF tokenizer from hub_model_id
            "corpus_file_id": "deadbeef",
            "holdout_fraction": 0.1,
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
            "precision": "fp32",
            "hub_model_id": "nghuyong/ernie-2.0-base-en",
            "finetuning_strategy": "lora",
            "lora": LoraConfig(
                enabled=True,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=("query", "value"),
                bias="none",
            ),
            "quantization": None,
            "gguf_export": None,
        },
    }

    train_job.process_train_job(encode_train_job_payload(payload))

    # Verify backend.prepare() was called with tokenizer=None
    assert len(backend_instance_holder) == 1
    backend_instance = backend_instance_holder[0]
    assert backend_instance is not None and backend_instance.prepare_called is True
    assert backend_instance.prepare_tokenizer_was_none is True

    # Verify loss decreases during training
    assert len(train_losses) >= 2, "Should have at least 2 loss values"
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before, f"Loss should decrease: {loss_before} -> {loss_after}"

    # Verify status is completed
    status = TrainerJobStore(fake).load("run-hflm-no-tok")
    assert status is not None and status["status"] == "completed"
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish", "expire"})


def test_continued_training_downloads_artifacts_when_absent_locally(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """A finished run's artifacts are deleted from disk once uploaded.

    ``ArtifactCleanupService`` removes a run's local directory after the
    upload, so for any source run that is not the one currently training, "not
    on disk" is the normal state. train_job read the directory straight off
    disk and every continued-training run failed with
    ``Metadata not found: .../hf_lm_metadata.json``. The pre-existing test for
    this path pre-created the directory, so it exercised a state production
    never reaches.
    """
    backend_instance_holder: list[_BackendWithLoad | None] = [None]
    train_losses: list[float] = []

    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )
    _test_hooks.load_settings = lambda: settings

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nfinetuning data\n", encoding="utf-8")

    tok_id = "tok-pretrained-download"
    tok_cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(artifacts / "tokenizers" / tok_id),
    )
    BPEBackend().train(tok_cfg)

    # Deliberately NOT created on disk: this is the production state.
    pretrained_run_id = "run-pretrained-cleaned-up"
    assert not (artifacts / "models" / pretrained_run_id).exists()

    fake = FakeRedis()
    _test_hooks.kv_store_factory = lambda url: fake
    fake.set(artifact_file_id_key(pretrained_run_id), "pretrained-file-id")

    recording = _RecordingStore("http://data-bank-api.local", "secret-key")

    def _store_factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return recording

    _test_hooks.service_container_from_settings = _create_service_container_factory(
        fake, backend_instance_holder, train_losses
    )
    _test_hooks.corpus_fetcher_factory = _create_corpus_fetcher_factory(corpus)
    _test_hooks.artifact_store_factory = _store_factory

    payload: TrainJobPayload = {
        "run_id": "run-cooldown",
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

    # The artifacts were fetched from the store, under the run's pointer.
    assert recording.downloads == [("pretrained-file-id", f"model-{pretrained_run_id}")]

    # And training continued from them rather than starting fresh.
    backend_instance = backend_instance_holder[0]
    assert backend_instance is not None and backend_instance.load_called is True
    assert backend_instance.prepare_called is False
    assert backend_instance.loaded_from == str(artifacts / "models" / pretrained_run_id)

    # Training ran on the downloaded weights rather than stopping at the load.
    assert len(train_losses) >= 2, "Should have at least 2 loss values"
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before, f"Loss should decrease: {loss_before} -> {loss_after}"

    status = TrainerJobStore(fake).load("run-cooldown")
    assert status is not None and status["status"] == "completed"
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish", "expire"})


def test_training_worker_passes_resume_flag_to_backend(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """A resume execution hands resume=True to the backend unchanged."""
    backend_instance_holder: list[_BackendWithLoad | None] = [None]
    train_losses: list[float] = []

    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )
    _test_hooks.load_settings = lambda: settings

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nresume data\n", encoding="utf-8")

    tok_id = "tok-resume-test"
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

    pretrained_run_id = "run-resume-base"
    pretrained_model_dir = artifacts / "models" / pretrained_run_id
    pretrained_model_dir.mkdir(parents=True, exist_ok=True)
    (pretrained_model_dir / "weights.bin").write_bytes(b"\x00pretrained")
    (pretrained_model_dir / "manifest.json").write_text(
        '{"model_family": "gpt2", "model_size": "small"}', encoding="utf-8"
    )

    fake = FakeRedis()
    _test_hooks.kv_store_factory = lambda url: fake
    _test_hooks.service_container_from_settings = _create_service_container_factory(
        fake, backend_instance_holder, train_losses
    )
    _test_hooks.corpus_fetcher_factory = _create_corpus_fetcher_factory(corpus)
    _test_hooks.artifact_store_factory = _create_artifact_store_factory()

    payload: TrainJobPayload = {
        "run_id": "run-resumed-exec",
        "user_id": 1,
        "resume": True,
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

    backend_instance = backend_instance_holder[0]
    assert backend_instance is not None and backend_instance.resume_seen is True
    # The fake backend emits a decreasing loss series; the guard-mandated
    # trajectory check keeps this meaningful rather than status-only.
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before
    status = TrainerJobStore(fake).load("run-resumed-exec")
    assert status is not None and status["status"] == "completed"
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish", "expire"})
