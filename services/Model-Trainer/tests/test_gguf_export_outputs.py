"""GGUF export: outputs and errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONObject, JSONValue

from model_trainer.api.validators.runs import _decode_train_request
from model_trainer.core.services.export import _test_hooks as export_hooks


class TestGgufExportConfigInvalid:
    """Tests for invalid GGUF export config in validators."""

    def test_gguf_export_invalid_type(self) -> None:
        """GGUF export with invalid type raises error."""
        payload: dict[str, JSONValue] = {
            "model_family": "hf_lm",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "user_id": 0,
            "hub_model_id": "meta-llama/Llama-2-7b-hf",
            "finetuning_strategy": "lora",
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
            },
            "gguf_export": "invalid",  # Should be an object
        }
        with pytest.raises(AppError) as exc:
            _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_gguf_export_invalid_output_type_value(self) -> None:
        """GGUF export with invalid output_type value raises error."""
        payload: dict[str, JSONValue] = {
            "model_family": "hf_lm",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "user_id": 0,
            "hub_model_id": "meta-llama/Llama-2-7b-hf",
            "finetuning_strategy": "lora",
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
            },
            "gguf_export": {
                "enabled": True,
                "output_type": "invalid_type",
            },
        }
        with pytest.raises(AppError) as exc:
            _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_gguf_export_enabled_not_boolean(self) -> None:
        """GGUF export with non-boolean enabled raises error."""
        payload: dict[str, JSONValue] = {
            "model_family": "hf_lm",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "user_id": 0,
            "hub_model_id": "meta-llama/Llama-2-7b-hf",
            "finetuning_strategy": "lora",
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
            },
            "gguf_export": {
                "enabled": "yes",  # Should be boolean
                "output_type": "f16",
            },
        }
        with pytest.raises(AppError) as exc:
            _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400


class TestFindConvertScript:
    """Tests for _find_convert_script function."""

    def test_find_script_returns_existing_path(self) -> None:
        """Find script returns path when script exists in known location."""
        from model_trainer.core.services.export._test_hooks import _find_convert_script

        # The script should exist at ~/PROJECTS/llama.cpp-src/convert_lora_to_gguf.py
        result = _find_convert_script()
        assert "convert_lora_to_gguf.py" in result
        assert Path(result).exists()

        export_hooks.reset_hooks()

    def test_find_script_raises_when_not_found(self, tmp_path: Path) -> None:
        """Find script raises RuntimeError when script not in any location."""
        from model_trainer.core.services.export._test_hooks import _find_convert_script

        # Set paths to non-existent locations
        def fake_paths() -> tuple[Path, ...]:
            return (
                tmp_path / "nonexistent1" / "convert_lora_to_gguf.py",
                tmp_path / "nonexistent2" / "convert_lora_to_gguf.py",
            )

        export_hooks.convert_script_paths = fake_paths

        with pytest.raises(RuntimeError) as exc:
            _find_convert_script()
        assert "script not found" in str(exc.value)

        export_hooks.reset_hooks()


class TestRealGgufConverter:
    """Tests for _real_gguf_converter function."""

    def test_real_converter_fails_on_invalid_adapter(self, tmp_path: Path) -> None:
        """Real converter raises RuntimeError when adapter directory is invalid."""
        from model_trainer.core.services.export._test_hooks import _real_gguf_converter

        adapter_dir = str(tmp_path / "fake_adapter")
        output_path = str(tmp_path / "output.gguf")

        with pytest.raises(RuntimeError) as exc:
            _real_gguf_converter(
                adapter_dir=adapter_dir,
                base_model_id="meta-llama/Llama-2-7b-hf",
                output_path=output_path,
                output_type="f16",
            )
        assert "GGUF conversion failed" in str(exc.value)

        export_hooks.reset_hooks()

    def test_real_converter_success_returns_file_size(self, tmp_path: Path) -> None:
        """Real converter returns output file size on success."""
        from model_trainer.core.services.export._test_hooks import _real_gguf_converter

        # Create a fake script that creates the output file
        script = tmp_path / "fake_convert.py"
        script.write_text(
            "import sys\n"
            "args = sys.argv[1:]\n"
            "outfile_idx = args.index('--outfile') + 1\n"
            "outfile = args[outfile_idx]\n"
            "with open(outfile, 'wb') as f:\n"
            "    f.write(b'FAKE_GGUF_DATA_12345')\n"
            "sys.exit(0)\n"
        )

        # Override paths to point to our fake script
        def fake_paths() -> tuple[Path, ...]:
            return (script,)

        export_hooks.convert_script_paths = fake_paths

        adapter_dir = str(tmp_path / "adapter")
        output_path = str(tmp_path / "output.gguf")

        result = _real_gguf_converter(
            adapter_dir=adapter_dir,
            base_model_id="model",
            output_path=output_path,
            output_type="f16",
        )

        # File should be created with 20 bytes ("FAKE_GGUF_DATA_12345")
        assert result == 20
        assert Path(output_path).exists()

        export_hooks.reset_hooks()


class TestQueueEncodingGgufExportInvalidType:
    """Tests for gguf_export decoding with invalid types."""

    def test_decode_gguf_export_not_dict_raises(self) -> None:
        """Decode raises when gguf_export is not dict or None."""
        from platform_core.json_utils import JSONTypeError

        from model_trainer.core.contracts.queue_encoding import (
            decode_train_request_payload,
        )

        # Create a minimal valid payload with invalid gguf_export type
        obj: JSONObject = {
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
            "finetuning_strategy": "full",
            "lora": None,
            "quantization": None,
            "gguf_export": "invalid_string",  # Should be dict or None
        }
        with pytest.raises(JSONTypeError):
            decode_train_request_payload(obj)


class TestManifestGgufExportDecoding:
    """Tests for manifest gguf_export decoding."""

    def test_decode_manifest_gguf_export_valid(self) -> None:
        """Loading manifest with valid gguf_export dict decodes correctly."""
        from platform_core.json_utils import dump_json_str

        from model_trainer.worker.manifest import load_manifest_from_text

        # Complete manifest with valid gguf_export
        manifest_data = {
            "run_id": "run-123",
            "model_family": "hf_lm",
            "model_size": "small",
            "tokenizer_id": None,
            "corpus_path": "/tmp/corpus.txt",
            "optimizer": "adamw",
            "device": "cuda",
            "precision": "fp16",
            "epochs": 1,
            "batch_size": 4,
            "max_seq_len": 512,
            "steps": 100,
            "seed": 42,
            "early_stopping_patience": 3,
            "loss": 0.5,
            "learning_rate": 1e-4,
            "holdout_fraction": 0.1,
            "gradient_clipping": 1.0,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 0.0,
            "loss_mask_prefix_separator": None,
            "freeze_embed": False,
            "early_stopped": False,
            "git_commit": None,
            "pretrained_run_id": None,
            "test_loss": None,
            "test_perplexity": None,
            "best_val_loss": None,
            "versions": {
                "torch": "2.0.0",
                "transformers": "4.30.0",
                "tokenizers": "0.13.0",
                "datasets": "2.14.0",
            },
            "system": {
                "cpu_count": 8,
                "platform": "Linux",
                "platform_release": "5.15.0",
                "machine": "x86_64",
            },
            "timing": {
                "training_duration_sec": 3600.0,
                "started_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T01:00:00",
            },
            "performance": {
                "peak_gpu_memory_mb": 8000.0,
                "avg_samples_per_sec": 10.0,
                "total_tokens_processed": 100000,
            },
            "model_info": {
                "param_count": 1000000,
                "model_size_mb": 10.5,
                "vocab_size": 32000,
            },
            # Valid gguf_export dict
            "gguf_export": {
                "output_type": "f16",
                "output_filename": "adapter.gguf",
                "output_size_bytes": 12345,
            },
        }
        text = dump_json_str(manifest_data, compact=False)
        result = load_manifest_from_text(text)
        gguf_export = result["gguf_export"]
        # Verify gguf_export decoded correctly with expected values
        assert gguf_export == {
            "output_type": "f16",
            "output_filename": "adapter.gguf",
            "output_size_bytes": 12345,
        }

    def test_decode_manifest_gguf_export_not_dict_raises(self) -> None:
        """Loading manifest raises when gguf_export is not dict or None."""
        from platform_core.json_utils import JSONTypeError, dump_json_str

        from model_trainer.worker.manifest import load_manifest_from_text

        # Complete manifest with nested sections plus invalid gguf_export
        manifest_data = {
            "run_id": "run-123",
            "model_family": "hf_lm",
            "model_size": "small",
            "tokenizer_id": None,
            "corpus_path": "/tmp/corpus.txt",
            "optimizer": "adamw",
            "device": "cuda",
            "precision": "fp16",
            "epochs": 1,
            "batch_size": 4,
            "max_seq_len": 512,
            "steps": 100,
            "seed": 42,
            "early_stopping_patience": 3,
            "loss": 0.5,
            "learning_rate": 1e-4,
            "holdout_fraction": 0.1,
            "gradient_clipping": 1.0,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 0.0,
            "loss_mask_prefix_separator": None,
            "freeze_embed": False,
            "early_stopped": False,
            "git_commit": None,
            "pretrained_run_id": None,
            "test_loss": None,
            "test_perplexity": None,
            "best_val_loss": None,
            # Nested versions section
            "versions": {
                "torch": "2.0.0",
                "transformers": "4.30.0",
                "tokenizers": "0.13.0",
                "datasets": "2.14.0",
            },
            # Nested system section
            "system": {
                "cpu_count": 8,
                "platform": "Linux",
                "platform_release": "5.15.0",
                "machine": "x86_64",
            },
            # Nested timing section
            "timing": {
                "training_duration_sec": 3600.0,
                "started_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T01:00:00",
            },
            # Nested performance section
            "performance": {
                "peak_gpu_memory_mb": 8000.0,
                "avg_samples_per_sec": 10.0,
                "total_tokens_processed": 100000,
            },
            # Nested model_info section
            "model_info": {
                "param_count": 1000000,
                "model_size_mb": 10.5,
                "vocab_size": 32000,
            },
            # Invalid gguf_export type
            "gguf_export": "invalid_string",  # Should be dict or None
        }
        text = dump_json_str(manifest_data, compact=False)
        with pytest.raises(JSONTypeError):
            load_manifest_from_text(text)
