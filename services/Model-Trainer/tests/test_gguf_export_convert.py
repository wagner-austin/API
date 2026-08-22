"""GGUF export: conversion mechanics."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONObject, JSONValue

from model_trainer.api.validators.runs import _decode_train_request
from model_trainer.core.contracts.model import GgufExportConfig
from model_trainer.core.contracts.queue_encoding_configs import (
    decode_gguf_export_config,
    encode_gguf_export_config,
)
from model_trainer.core.services.export import _test_hooks as export_hooks
from model_trainer.core.services.export.gguf_export import export_lora_to_gguf


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
