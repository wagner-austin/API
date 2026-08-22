"""GGUF export integration: failure paths."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from model_trainer.core.contracts.model import (
    GgufExportConfig,
)
from model_trainer.core.contracts.queue import TrainRequestPayload
from model_trainer.core.contracts.queue_encoding import (
    decode_train_request_payload,
    encode_train_request_payload,
)
from model_trainer.core.services.export import _test_hooks as export_hooks
from model_trainer.worker.train_job_lifecycle import _maybe_export_to_gguf
from tests._gguf_integration_support import (
    _create_artifact_store_factory,
    _create_corpus_fetcher_factory,
    _create_service_container_factory,
    _HfLmBackend,
)


class TestQueueEncodingRoundTrip:
    """Integration tests for queue encoding/decoding round-trip."""

    def test_encode_decode_with_gguf_export(self) -> None:
        """Round-trip encode/decode preserves gguf_export config."""
        payload: TrainRequestPayload = {
            "model_family": "hf_lm",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "tokenizer_id": None,
            "holdout_fraction": 0.1,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cuda",
            "precision": "fp16",
            "data_num_workers": 0,
            "data_pin_memory": True,
            "early_stopping_patience": 3,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 0.0,
            "loss_mask_prefix_separator": None,
            "hub_model_id": "meta-llama/Llama-2-7b-hf",
            "finetuning_strategy": "lora",
            "lora": {
                "enabled": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ("q_proj", "v_proj"),
                "bias": "none",
            },
            "quantization": None,
            "gguf_export": {
                "enabled": True,
                "output_type": "f16",
            },
        }

        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)

        gguf_export = decoded["gguf_export"]
        assert gguf_export == {"enabled": True, "output_type": "f16"}

    def test_encode_decode_without_gguf_export(self) -> None:
        """Round-trip encode/decode preserves gguf_export=None."""
        payload: TrainRequestPayload = {
            "model_family": "hf_lm",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "tokenizer_id": None,
            "holdout_fraction": 0.1,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cuda",
            "precision": "fp16",
            "data_num_workers": 0,
            "data_pin_memory": True,
            "early_stopping_patience": 3,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 0.0,
            "loss_mask_prefix_separator": None,
            "hub_model_id": "meta-llama/Llama-2-7b-hf",
            "finetuning_strategy": "lora",
            "lora": {
                "enabled": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ("q_proj", "v_proj"),
                "bias": "none",
            },
            "quantization": None,
            "gguf_export": None,
        }

        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)

        assert decoded["gguf_export"] is None

    def test_encode_decode_f32_output_type(self) -> None:
        """Round-trip encode/decode preserves f32 output type."""
        payload: TrainRequestPayload = {
            "model_family": "hf_lm",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "tokenizer_id": None,
            "holdout_fraction": 0.1,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cuda",
            "precision": "fp16",
            "data_num_workers": 0,
            "data_pin_memory": True,
            "early_stopping_patience": 3,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 0.0,
            "loss_mask_prefix_separator": None,
            "hub_model_id": "model",
            "finetuning_strategy": "lora",
            "lora": {
                "enabled": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ("q_proj",),
                "bias": "none",
            },
            "quantization": None,
            "gguf_export": {"enabled": True, "output_type": "f32"},
        }
        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)
        gguf_export = decoded["gguf_export"]
        assert gguf_export == {"enabled": True, "output_type": "f32"}

    def test_encode_decode_q8_0_output_type(self) -> None:
        """Round-trip encode/decode preserves q8_0 output type."""
        payload: TrainRequestPayload = {
            "model_family": "hf_lm",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "tokenizer_id": None,
            "holdout_fraction": 0.1,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cuda",
            "precision": "fp16",
            "data_num_workers": 0,
            "data_pin_memory": True,
            "early_stopping_patience": 3,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 0.0,
            "loss_mask_prefix_separator": None,
            "hub_model_id": "model",
            "finetuning_strategy": "lora",
            "lora": {
                "enabled": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ("q_proj",),
                "bias": "none",
            },
            "quantization": None,
            "gguf_export": {"enabled": True, "output_type": "q8_0"},
        }
        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)
        gguf_export = decoded["gguf_export"]
        assert gguf_export == {"enabled": True, "output_type": "q8_0"}


class TestMaybeExportToGguf:
    """Integration tests for _maybe_export_to_gguf function."""

    def test_export_disabled_returns_none(self) -> None:
        """Export returns None when disabled."""
        config: GgufExportConfig = {
            "enabled": False,
            "output_type": "f16",
        }
        result = _maybe_export_to_gguf(
            gguf_export=config,
            out_dir="/tmp/output",
            hub_model_id="model",
        )
        assert result is None

    def test_export_none_config_returns_none(self) -> None:
        """Export returns None when config is None."""
        result = _maybe_export_to_gguf(
            gguf_export=None,
            out_dir="/tmp/output",
            hub_model_id="model",
        )
        assert result is None

    def test_export_enabled_no_hub_model_id_raises(self) -> None:
        """Export raises when enabled but hub_model_id is None."""
        config: GgufExportConfig = {
            "enabled": True,
            "output_type": "f16",
        }
        with pytest.raises(RuntimeError) as exc:
            _maybe_export_to_gguf(
                gguf_export=config,
                out_dir="/tmp/output",
                hub_model_id=None,
            )
        assert "hub_model_id" in str(exc.value)

    def test_export_enabled_success(self) -> None:
        """Export succeeds when enabled with valid config."""
        config: GgufExportConfig = {
            "enabled": True,
            "output_type": "f16",
        }

        def fake_converter(
            adapter_dir: str,
            base_model_id: str,
            output_path: str,
            output_type: Literal["f32", "f16", "bf16", "q8_0"],
        ) -> int:
            return 54321

        export_hooks.gguf_converter = fake_converter

        result = _maybe_export_to_gguf(
            gguf_export=config,
            out_dir="/tmp/output",
            hub_model_id="meta-llama/Llama-2-7b-hf",
        )

        expected_result = {
            "output_path": str(Path("/tmp/output") / "adapter.gguf"),
            "output_size_bytes": 54321,
        }
        assert result == expected_result

        export_hooks.reset_hooks()


class TestProgressShowsExportingPhase:
    """Tests for exporting phase in progress."""

    def test_exporting_phase_is_valid(self) -> None:
        """Exporting is a valid training phase."""
        from model_trainer.core.contracts.progress import TrainingPhase

        # This will be a type error if "exporting" is not in TrainingPhase
        phase: TrainingPhase = "exporting"
        assert phase == "exporting"

    def test_progress_response_accepts_exporting_phase(self) -> None:
        """ProgressResponse schema accepts exporting phase."""
        from model_trainer.api.schemas.runs import ProgressResponse

        response: ProgressResponse = {
            "run_id": "run-123",
            "phase": "exporting",
            "epoch": 1,
            "total_epochs": 1,
            "step": 100,
            "total_steps": 100,
            "train_loss": 0.5,
            "train_ppl": 1.5,
            "grad_norm": 0.1,
            "samples_per_sec": 10.0,
            "val_loss": None,
            "val_ppl": None,
            "updated_at": "2024-01-01T00:00:00",
        }
        assert response["phase"] == "exporting"


class TestTrainingJobWithGgufExport:
    """Integration tests for training job with GGUF export phase."""

    def test_training_job_triggers_gguf_export_phase(self, tmp_path: Path) -> None:
        """Training job with gguf_export enabled triggers export phase (lines 527-541)."""
        from platform_workers.testing import FakeRedis

        from model_trainer.core import _test_hooks
        from model_trainer.core.config.settings import load_settings
        from model_trainer.core.contracts.model import LoraConfig
        from model_trainer.core.contracts.queue import TrainJobPayload
        from model_trainer.core.contracts.queue_encoding import (
            encode_train_job_payload,
        )
        from model_trainer.worker import train_job
        from model_trainer.worker.trainer_job_store import TrainerJobStore

        # Track GGUF export calls
        gguf_export_calls: list[tuple[str, str, str, str]] = []

        def fake_gguf_converter(
            adapter_dir: str,
            base_model_id: str,
            output_path: str,
            output_type: Literal["f32", "f16", "bf16", "q8_0"],
        ) -> int:
            gguf_export_calls.append((adapter_dir, base_model_id, output_path, output_type))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"GGUF_FAKE_DATA")
            return 14

        export_hooks.gguf_converter = fake_gguf_converter

        # Set up environment
        artifacts = tmp_path / "artifacts"
        settings = load_settings()
        settings["app"]["artifacts_root"] = str(artifacts)
        settings["app"]["runs_root"] = str(tmp_path / "runs")
        settings["app"]["logs_root"] = str(tmp_path / "logs")
        settings["app"]["data_root"] = str(tmp_path / "data")
        settings["app"]["data_bank_api_url"] = "http://data-bank.local"
        settings["app"]["data_bank_api_key"] = "secret"
        _test_hooks.load_settings = lambda: settings

        # Corpus
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("hello world\ntest data\n", encoding="utf-8")

        # Redis and hooks
        fake_redis = FakeRedis()
        _test_hooks.kv_store_factory = lambda url: fake_redis
        backend = _HfLmBackend()

        _test_hooks.service_container_from_settings = _create_service_container_factory(
            fake_redis, backend
        )
        _test_hooks.corpus_fetcher_factory = _create_corpus_fetcher_factory(corpus)
        _test_hooks.artifact_store_factory = _create_artifact_store_factory()

        # Build payload with GGUF export enabled
        payload: TrainJobPayload = {
            "run_id": "run-gguf-export-test",
            "user_id": 1,
            "resume": False,
            "request": {
                "model_family": "hf_lm",
                "model_size": "small",
                "max_seq_len": 64,
                "num_epochs": 1,
                "batch_size": 1,
                "learning_rate": 5e-5,
                "tokenizer_id": None,
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
                "hub_model_id": "meta-llama/Llama-2-7b-hf",
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
                "gguf_export": {"enabled": True, "output_type": "f16"},
            },
        }

        # Run training job
        train_job.process_train_job(encode_train_job_payload(payload))

        # Verify GGUF export was called
        assert len(gguf_export_calls) == 1, "GGUF export should have been called once"
        adapter_dir, base_model_id, output_path, output_type = gguf_export_calls[0]
        assert "run-gguf-export-test" in adapter_dir
        assert base_model_id == "meta-llama/Llama-2-7b-hf"
        assert output_path.endswith("adapter.gguf")
        assert output_type == "f16"

        # Verify status is completed
        status = TrainerJobStore(fake_redis).load("run-gguf-export-test")
        assert status is not None and status["status"] == "completed"

        # Verify Redis operations
        fake_redis.assert_only_called({"set", "get", "hset", "hgetall", "publish", "expire"})

        # Clean up
        export_hooks.reset_hooks()
