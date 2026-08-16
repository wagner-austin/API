"""Unit tests for GGUF export functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONObject, JSONValue

from model_trainer.api.validators.runs import _decode_train_request
from model_trainer.core.contracts.model import GgufExportConfig
from model_trainer.core.contracts.queue_encoding import (
    decode_gguf_export_config,
    encode_gguf_export_config,
)
from model_trainer.core.services.export import _test_hooks as export_hooks
from model_trainer.core.services.export.gguf_export import (
    export_lora_to_gguf,
)


class TestEncodeGgufExportConfig:
    """Tests for encode_gguf_export_config function."""

    def test_encode_gguf_export_config_f16(self) -> None:
        """Encode config with f16 output type."""
        config: GgufExportConfig = {
            "enabled": True,
            "output_type": "f16",
        }
        result = encode_gguf_export_config(config)
        assert result["enabled"] is True
        assert result["output_type"] == "f16"

    def test_encode_gguf_export_config_q8_0(self) -> None:
        """Encode config with q8_0 output type."""
        config: GgufExportConfig = {
            "enabled": False,
            "output_type": "q8_0",
        }
        result = encode_gguf_export_config(config)
        assert result["enabled"] is False
        assert result["output_type"] == "q8_0"

    def test_encode_gguf_export_config_bf16(self) -> None:
        """Encode config with bf16 output type."""
        config: GgufExportConfig = {
            "enabled": True,
            "output_type": "bf16",
        }
        result = encode_gguf_export_config(config)
        assert result["enabled"] is True
        assert result["output_type"] == "bf16"

    def test_encode_gguf_export_config_f32(self) -> None:
        """Encode config with f32 output type."""
        config: GgufExportConfig = {
            "enabled": True,
            "output_type": "f32",
        }
        result = encode_gguf_export_config(config)
        assert result["enabled"] is True
        assert result["output_type"] == "f32"


class TestDecodeGgufExportConfig:
    """Tests for decode_gguf_export_config function."""

    def test_decode_gguf_export_config_success(self) -> None:
        """Decode valid config successfully."""
        obj: JSONObject = {
            "enabled": True,
            "output_type": "f16",
        }
        result = decode_gguf_export_config(obj)
        assert result["enabled"] is True
        assert result["output_type"] == "f16"

    def test_decode_gguf_export_config_disabled(self) -> None:
        """Decode config with enabled=false."""
        obj: JSONObject = {
            "enabled": False,
            "output_type": "q8_0",
        }
        result = decode_gguf_export_config(obj)
        assert result["enabled"] is False
        assert result["output_type"] == "q8_0"

    def test_decode_gguf_export_config_all_output_types(self) -> None:
        """Decode all valid output types."""
        for output_type in ("f32", "f16", "bf16", "q8_0"):
            obj: JSONObject = {
                "enabled": True,
                "output_type": output_type,
            }
            result = decode_gguf_export_config(obj)
            assert result["output_type"] == output_type

    def test_decode_gguf_export_config_invalid_output_type(self) -> None:
        """Decode with invalid output type raises error."""
        from platform_core.json_utils import JSONTypeError

        obj: JSONObject = {
            "enabled": True,
            "output_type": "invalid",
        }
        with pytest.raises(JSONTypeError):
            decode_gguf_export_config(obj)

    def test_decode_gguf_export_config_missing_enabled(self) -> None:
        """Decode with missing enabled field raises error."""
        from platform_core.json_utils import JSONTypeError

        obj: JSONObject = {
            "output_type": "f16",
        }
        with pytest.raises(JSONTypeError):
            decode_gguf_export_config(obj)

    def test_decode_gguf_export_config_missing_output_type(self) -> None:
        """Decode with missing output_type field raises error."""
        from platform_core.json_utils import JSONTypeError

        obj: JSONObject = {
            "enabled": True,
        }
        with pytest.raises(JSONTypeError):
            decode_gguf_export_config(obj)


class TestDecodeOptionalGgufExport:
    """Tests for optional GGUF export decoding in validators."""

    def test_decode_optional_gguf_export_none(self) -> None:
        """Decode request without gguf_export returns None."""
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
        }
        out = _decode_train_request(payload)
        assert out["gguf_export"] is None


class TestValidateGgufExportRequiresLoraStrategy:
    """Tests for gguf_export requiring LoRA strategy."""

    def test_validate_gguf_export_with_full_strategy_raises(self) -> None:
        """GGUF export with full strategy raises error."""
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
            "finetuning_strategy": "full",
            "gguf_export": {
                "enabled": True,
                "output_type": "f16",
            },
        }
        with pytest.raises(AppError) as exc:
            _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert "gguf_export" in str(err.message)

    def test_validate_gguf_export_with_lora_strategy_succeeds(self) -> None:
        """GGUF export with lora strategy succeeds."""
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
                "output_type": "f16",
            },
        }
        out = _decode_train_request(payload)
        gguf_export = out["gguf_export"]
        assert gguf_export == {"enabled": True, "output_type": "f16"}

    def test_validate_gguf_export_with_qlora_strategy_succeeds(self) -> None:
        """GGUF export with qlora strategy succeeds."""
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
            "finetuning_strategy": "qlora",
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
            },
            "quantization": {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
            },
            "gguf_export": {
                "enabled": True,
                "output_type": "q8_0",
            },
        }
        out = _decode_train_request(payload)
        gguf_export = out["gguf_export"]
        assert gguf_export == {"enabled": True, "output_type": "q8_0"}

    def test_validate_gguf_export_bf16_output_type(self) -> None:
        """GGUF export with bf16 output type succeeds."""
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
                "output_type": "bf16",
            },
        }
        out = _decode_train_request(payload)
        gguf_export = out["gguf_export"]
        assert gguf_export == {"enabled": True, "output_type": "bf16"}

    def test_validate_gguf_export_f32_output_type(self) -> None:
        """GGUF export with f32 output type succeeds."""
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
                "output_type": "f32",
            },
        }
        out = _decode_train_request(payload)
        gguf_export = out["gguf_export"]
        assert gguf_export == {"enabled": True, "output_type": "f32"}

    def test_validate_gguf_export_enabled_defaults_true(self) -> None:
        """GGUF export without enabled field defaults to True."""
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
                "output_type": "f16",
            },
        }
        out = _decode_train_request(payload)
        gguf_export = out["gguf_export"]
        assert gguf_export == {"enabled": True, "output_type": "f16"}


class TestExportLoraToGguf:
    """Tests for export_lora_to_gguf function."""

    def test_export_lora_to_gguf_success(self) -> None:
        """Export succeeds with fake hook."""
        recorded_calls: list[tuple[str, str, str, str]] = []

        def fake_converter(
            adapter_dir: str,
            base_model_id: str,
            output_path: str,
            output_type: Literal["f32", "f16", "bf16", "q8_0"],
        ) -> int:
            recorded_calls.append((adapter_dir, base_model_id, output_path, output_type))
            return 12345

        export_hooks.gguf_converter = fake_converter

        result = export_lora_to_gguf(
            adapter_dir="/tmp/adapter",
            base_model_id="meta-llama/Llama-2-7b-hf",
            output_dir="/tmp/output",
            output_type="f16",
        )

        expected_path = str(Path("/tmp/output") / "adapter.gguf")
        assert result["output_path"] == expected_path
        assert result["output_size_bytes"] == 12345
        assert len(recorded_calls) == 1
        assert recorded_calls[0] == (
            "/tmp/adapter",
            "meta-llama/Llama-2-7b-hf",
            expected_path,
            "f16",
        )

        export_hooks.reset_hooks()

    def test_export_lora_to_gguf_f32_output_type(self) -> None:
        """Export handles f32 output type."""

        def fake_converter(
            adapter_dir: str,
            base_model_id: str,
            output_path: str,
            output_type: Literal["f32", "f16", "bf16", "q8_0"],
        ) -> int:
            return 100

        export_hooks.gguf_converter = fake_converter

        result = export_lora_to_gguf(
            adapter_dir="/tmp/adapter",
            base_model_id="model",
            output_dir="/tmp/output",
            output_type="f32",
        )
        assert result["output_size_bytes"] == 100
        export_hooks.reset_hooks()

    def test_export_lora_to_gguf_bf16_output_type(self) -> None:
        """Export handles bf16 output type."""

        def fake_converter(
            adapter_dir: str,
            base_model_id: str,
            output_path: str,
            output_type: Literal["f32", "f16", "bf16", "q8_0"],
        ) -> int:
            return 100

        export_hooks.gguf_converter = fake_converter

        result = export_lora_to_gguf(
            adapter_dir="/tmp/adapter",
            base_model_id="model",
            output_dir="/tmp/output",
            output_type="bf16",
        )
        assert result["output_size_bytes"] == 100
        export_hooks.reset_hooks()

    def test_export_lora_to_gguf_q8_0_output_type(self) -> None:
        """Export handles q8_0 output type."""

        def fake_converter(
            adapter_dir: str,
            base_model_id: str,
            output_path: str,
            output_type: Literal["f32", "f16", "bf16", "q8_0"],
        ) -> int:
            return 100

        export_hooks.gguf_converter = fake_converter

        result = export_lora_to_gguf(
            adapter_dir="/tmp/adapter",
            base_model_id="model",
            output_dir="/tmp/output",
            output_type="q8_0",
        )
        assert result["output_size_bytes"] == 100
        export_hooks.reset_hooks()


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
        from model_trainer.core.services.export import _test_hooks as export_hooks
        from model_trainer.core.services.export._test_hooks import _find_convert_script

        # The script should exist at ~/PROJECTS/llama.cpp-src/convert_lora_to_gguf.py
        result = _find_convert_script()
        assert "convert_lora_to_gguf.py" in result
        assert Path(result).exists()

        export_hooks.reset_hooks()

    def test_find_script_raises_when_not_found(self, tmp_path: Path) -> None:
        """Find script raises RuntimeError when script not in any location."""
        from model_trainer.core.services.export import _test_hooks as export_hooks
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
        from model_trainer.core.services.export import _test_hooks as export_hooks
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
        from model_trainer.core.services.export import _test_hooks as export_hooks
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
            "unsloth": None,
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
