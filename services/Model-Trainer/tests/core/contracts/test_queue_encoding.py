"""Tests for queue_encoding module.

Tests encode/decode functions for LoraConfig, QuantizationConfig,
and TrainRequestPayload with full branch coverage.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from model_trainer.core.contracts.model import (
    LoraConfig,
    QuantizationConfig,
)
from model_trainer.core.contracts.queue import TrainJobPayload, TrainRequestPayload
from model_trainer.core.contracts.queue_encoding import (
    decode_lora_config,
    decode_quantization_config,
    decode_train_job_payload,
    decode_train_request_payload,
    encode_lora_config,
    encode_quantization_config,
    encode_train_job_payload,
    encode_train_request_payload,
)


class TestLoraConfigEncoding:
    """Tests for LoraConfig encode/decode functions."""

    def test_encode_lora_config_roundtrip(self) -> None:
        """Test that encode and decode are inverse operations."""
        config: LoraConfig = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj", "k_proj"),
            "bias": "none",
        }
        encoded = encode_lora_config(config)
        decoded = decode_lora_config(encoded)

        assert decoded["enabled"] is True
        assert decoded["r"] == 16
        assert decoded["lora_alpha"] == 32
        assert decoded["lora_dropout"] == 0.1
        assert decoded["target_modules"] == ("q_proj", "v_proj", "k_proj")
        assert decoded["bias"] == "none"

    def test_encode_lora_config_converts_tuple_to_list(self) -> None:
        """Test that target_modules tuple is converted to list for JSON."""
        config: LoraConfig = {
            "enabled": True,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ("q_proj",),
            "bias": "all",
        }
        encoded = encode_lora_config(config)
        assert encoded["target_modules"] == ["q_proj"]

    def test_decode_lora_config_all_bias_values(self) -> None:
        """Test decoding all valid bias values."""
        for bias_val in ("none", "all", "lora_only"):
            encoded: JSONObject = {
                "enabled": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj"],
                "bias": bias_val,
            }
            decoded = decode_lora_config(encoded)
            assert decoded["bias"] == bias_val

    def test_decode_lora_config_missing_enabled(self) -> None:
        """Test that missing enabled field raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="Missing required field 'enabled'"):
            decode_lora_config(
                {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "target_modules": [],
                    "bias": "none",
                }
            )

    def test_decode_lora_config_missing_target_modules(self) -> None:
        """Test that missing target_modules field raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="Missing required field 'target_modules'"):
            decode_lora_config(
                {"enabled": True, "r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none"}
            )

    def test_decode_lora_config_target_modules_not_list(self) -> None:
        """Test that non-list target_modules raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="must be an array"):
            decode_lora_config(
                {
                    "enabled": True,
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "target_modules": "q_proj",
                    "bias": "none",
                }
            )

    def test_decode_lora_config_target_modules_item_not_string(self) -> None:
        """Test that non-string item in target_modules raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match=r"target_modules\[0\].*must be a string"):
            decode_lora_config(
                {
                    "enabled": True,
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "target_modules": [123],
                    "bias": "none",
                }
            )

    def test_decode_lora_config_invalid_bias(self) -> None:
        """Test that invalid bias value raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="must be 'none', 'all', or 'lora_only'"):
            decode_lora_config(
                {
                    "enabled": True,
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "target_modules": [],
                    "bias": "invalid",
                }
            )


class TestQuantizationConfigEncoding:
    """Tests for QuantizationConfig encode/decode functions."""

    def test_encode_quantization_config_roundtrip(self) -> None:
        """Test that encode and decode are inverse operations."""
        config: QuantizationConfig = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
        }
        encoded = encode_quantization_config(config)
        decoded = decode_quantization_config(encoded)

        assert decoded["load_in_4bit"] is True
        assert decoded["load_in_8bit"] is False
        assert decoded["bnb_4bit_compute_dtype"] == "float16"
        assert decoded["bnb_4bit_quant_type"] == "nf4"

    def test_decode_quantization_config_all_compute_dtypes(self) -> None:
        """Test decoding all valid compute dtype values."""
        for dtype in ("float16", "bfloat16", "float32"):
            encoded: JSONObject = {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_compute_dtype": dtype,
                "bnb_4bit_quant_type": "nf4",
            }
            decoded = decode_quantization_config(encoded)
            assert decoded["bnb_4bit_compute_dtype"] == dtype

    def test_decode_quantization_config_all_quant_types(self) -> None:
        """Test decoding all valid quant type values."""
        for qtype in ("nf4", "fp4"):
            encoded: JSONObject = {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": qtype,
            }
            decoded = decode_quantization_config(encoded)
            assert decoded["bnb_4bit_quant_type"] == qtype

    def test_decode_quantization_config_invalid_compute_dtype(self) -> None:
        """Test that invalid compute dtype raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match=r"bnb_4bit_compute_dtype.*must be"):
            decode_quantization_config(
                {
                    "load_in_4bit": True,
                    "load_in_8bit": False,
                    "bnb_4bit_compute_dtype": "invalid",
                    "bnb_4bit_quant_type": "nf4",
                }
            )

    def test_decode_quantization_config_invalid_quant_type(self) -> None:
        """Test that invalid quant type raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match=r"bnb_4bit_quant_type.*must be"):
            decode_quantization_config(
                {
                    "load_in_4bit": True,
                    "load_in_8bit": False,
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_quant_type": "invalid",
                }
            )


class TestTrainRequestPayloadEncoding:
    """Tests for TrainRequestPayload encode/decode functions."""

    def _make_minimal_payload(self) -> TrainRequestPayload:
        """Create a minimal valid TrainRequestPayload."""
        return {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "corpus_file_id": "test-corpus-id",
            "tokenizer_id": "tok-123",
            "holdout_fraction": 0.1,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "precision": "fp32",
            "data_num_workers": None,
            "data_pin_memory": None,
            "early_stopping_patience": 3,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 1e-4,
            "loss_mask_prefix_separator": None,
            "hub_model_id": None,
            "finetuning_strategy": "full",
            "lora": None,
            "quantization": None,
            "gguf_export": None,
        }

    def test_encode_decode_roundtrip_minimal(self) -> None:
        """Test roundtrip with minimal payload (no nested configs)."""
        payload = self._make_minimal_payload()
        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)

        assert decoded["model_family"] == "gpt2"
        assert decoded["model_size"] == "small"
        assert decoded["finetuning_strategy"] == "full"
        assert decoded["lora"] is None
        assert decoded["quantization"] is None

    def test_roundtrip_preserves_the_loss_mask_separator(self) -> None:
        """Whitespace in the separator is significant, so it must survive the
        queue hop byte for byte."""
        payload = self._make_minimal_payload()
        payload["loss_mask_prefix_separator"] = " | "
        decoded = decode_train_request_payload(encode_train_request_payload(payload))
        assert decoded["loss_mask_prefix_separator"] == " | "

    def test_decode_rejects_an_empty_loss_mask_separator(self) -> None:
        """An empty separator masks nothing while claiming a masked arm, so the
        queue boundary rejects it exactly as the API edge does."""
        payload = self._make_minimal_payload()
        encoded = encode_train_request_payload(payload)
        encoded["loss_mask_prefix_separator"] = ""
        with pytest.raises(JSONTypeError, match="must not be empty"):
            decode_train_request_payload(encoded)

    def test_encode_decode_roundtrip_with_lora(self) -> None:
        """Test roundtrip with LoRA config."""
        payload = self._make_minimal_payload()
        payload["finetuning_strategy"] = "lora"
        payload["lora"] = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }

        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)

        assert decoded["finetuning_strategy"] == "lora"
        lora = decoded["lora"]
        assert lora == {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }

    def test_encode_decode_roundtrip_with_quantization(self) -> None:
        """Test roundtrip with quantization config."""
        payload = self._make_minimal_payload()
        payload["finetuning_strategy"] = "qlora"
        payload["lora"] = {
            "enabled": True,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ("q_proj",),
            "bias": "none",
        }
        payload["quantization"] = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
        }

        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)

        quantization = decoded["quantization"]
        assert quantization == {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
        }

    def test_decode_all_model_families(self) -> None:
        """Test decoding all valid model family values."""
        for family in ("gpt2", "llama", "qwen", "char_lstm", "hf_lm"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["model_family"] = family
            decoded = decode_train_request_payload(encoded)
            assert decoded["model_family"] == family

    def test_decode_invalid_model_family(self) -> None:
        """Test that invalid model family raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["model_family"] = "invalid"
        with pytest.raises(JSONTypeError, match=r"model_family.*must be"):
            decode_train_request_payload(encoded)

    def test_decode_all_optimizers(self) -> None:
        """Test decoding all valid optimizer values."""
        for opt in ("adamw", "adam", "sgd"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["optimizer"] = opt
            decoded = decode_train_request_payload(encoded)
            assert decoded["optimizer"] == opt

    def test_decode_invalid_optimizer(self) -> None:
        """Test that invalid optimizer raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["optimizer"] = "invalid"
        with pytest.raises(JSONTypeError, match=r"optimizer.*must be"):
            decode_train_request_payload(encoded)

    def test_decode_all_devices(self) -> None:
        """Test decoding all valid device values."""
        for device in ("cpu", "cuda", "auto"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["device"] = device
            decoded = decode_train_request_payload(encoded)
            assert decoded["device"] == device

    def test_decode_invalid_device(self) -> None:
        """Test that invalid device raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["device"] = "tpu"
        with pytest.raises(JSONTypeError, match=r"device.*must be"):
            decode_train_request_payload(encoded)

    def test_decode_all_precisions(self) -> None:
        """Test decoding all valid precision values."""
        for precision in ("fp32", "fp16", "bf16", "auto"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["precision"] = precision
            decoded = decode_train_request_payload(encoded)
            assert decoded["precision"] == precision

    def test_decode_invalid_precision(self) -> None:
        """Test that invalid precision raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["precision"] = "int8"
        with pytest.raises(JSONTypeError, match=r"precision.*must be"):
            decode_train_request_payload(encoded)

    def test_decode_all_finetuning_strategies(self) -> None:
        """Test decoding all valid finetuning strategy values."""
        for strategy in ("full", "lora", "qlora"):
            encoded = encode_train_request_payload(self._make_minimal_payload())
            encoded["finetuning_strategy"] = strategy
            decoded = decode_train_request_payload(encoded)
            assert decoded["finetuning_strategy"] == strategy

    def test_decode_invalid_finetuning_strategy(self) -> None:
        """Test that invalid finetuning strategy raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["finetuning_strategy"] = "freeze"
        with pytest.raises(JSONTypeError, match=r"finetuning_strategy.*must be"):
            decode_train_request_payload(encoded)

    def test_decode_data_pin_memory_bool(self) -> None:
        """Test decoding data_pin_memory as bool."""
        payload = self._make_minimal_payload()
        payload["data_pin_memory"] = True
        encoded = encode_train_request_payload(payload)
        decoded = decode_train_request_payload(encoded)
        assert decoded["data_pin_memory"] is True

    def test_decode_data_pin_memory_invalid(self) -> None:
        """Test that non-bool data_pin_memory raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["data_pin_memory"] = "yes"
        with pytest.raises(JSONTypeError, match=r"data_pin_memory.*must be a boolean or null"):
            decode_train_request_payload(encoded)

    def test_decode_lora_not_dict(self) -> None:
        """Test that non-dict lora field raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["lora"] = "invalid"
        with pytest.raises(JSONTypeError, match=r"lora.*must be an object or null"):
            decode_train_request_payload(encoded)

    def test_decode_quantization_not_dict(self) -> None:
        """Test that non-dict quantization field raises JSONTypeError."""
        encoded = encode_train_request_payload(self._make_minimal_payload())
        encoded["quantization"] = []
        with pytest.raises(JSONTypeError, match=r"quantization.*must be an object or null"):
            decode_train_request_payload(encoded)

    def test_encode_preserves_all_fields(self) -> None:
        """Test that encoding preserves all fields."""
        payload = self._make_minimal_payload()
        payload["pretrained_run_id"] = "prev-run-123"
        payload["data_num_workers"] = 4
        payload["data_pin_memory"] = True
        payload["hub_model_id"] = "gpt2"

        encoded = encode_train_request_payload(payload)

        assert encoded["pretrained_run_id"] == "prev-run-123"
        assert encoded["data_num_workers"] == 4
        assert encoded["data_pin_memory"] is True
        assert encoded["hub_model_id"] == "gpt2"


class TestTrainJobPayloadEncoding:
    """Tests for TrainJobPayload encode/decode functions."""

    def _make_minimal_request(self) -> TrainRequestPayload:
        """Create a minimal valid TrainRequestPayload."""
        return {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 512,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "corpus_file_id": "test-corpus-id",
            "tokenizer_id": "tok-123",
            "holdout_fraction": 0.1,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "precision": "fp32",
            "data_num_workers": None,
            "data_pin_memory": None,
            "early_stopping_patience": 3,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 1e-4,
            "loss_mask_prefix_separator": None,
            "hub_model_id": None,
            "finetuning_strategy": "full",
            "lora": None,
            "quantization": None,
            "gguf_export": None,
        }

    def _make_job_payload(self) -> TrainJobPayload:
        """Create a valid TrainJobPayload."""
        return {
            "run_id": "run-12345",
            "user_id": 42,
            "request": self._make_minimal_request(),
        }

    def test_encode_decode_roundtrip(self) -> None:
        """Test that encode and decode are inverse operations."""
        payload = self._make_job_payload()
        encoded = encode_train_job_payload(payload)
        decoded = decode_train_job_payload(encoded)

        assert decoded["run_id"] == "run-12345"
        assert decoded["user_id"] == 42
        assert decoded["request"]["model_family"] == "gpt2"
        assert decoded["request"]["corpus_file_id"] == "test-corpus-id"

    def test_encode_produces_json_object(self) -> None:
        """Test that encoding produces a plain dict suitable for JSON."""
        payload = self._make_job_payload()
        encoded = encode_train_job_payload(payload)

        assert encoded["run_id"] == "run-12345"
        assert encoded["user_id"] == 42
        # Decode back to verify the nested request was properly encoded
        decoded = decode_train_job_payload(encoded)
        assert decoded["request"]["model_family"] == "gpt2"
        assert decoded["request"]["corpus_file_id"] == "test-corpus-id"

    def test_decode_missing_run_id(self) -> None:
        """Test that missing run_id raises JSONTypeError."""
        encoded: JSONObject = {
            "user_id": 42,
            "request": encode_train_request_payload(self._make_minimal_request()),
        }
        with pytest.raises(JSONTypeError, match=r"Missing required field 'run_id'"):
            decode_train_job_payload(encoded)

    def test_decode_missing_user_id(self) -> None:
        """Test that missing user_id raises JSONTypeError."""
        encoded: JSONObject = {
            "run_id": "run-123",
            "request": encode_train_request_payload(self._make_minimal_request()),
        }
        with pytest.raises(JSONTypeError, match=r"Missing required field 'user_id'"):
            decode_train_job_payload(encoded)

    def test_decode_missing_request(self) -> None:
        """Test that missing request raises JSONTypeError."""
        encoded: JSONObject = {
            "run_id": "run-123",
            "user_id": 42,
        }
        with pytest.raises(JSONTypeError, match=r"Missing required field 'request'"):
            decode_train_job_payload(encoded)

    def test_decode_request_not_dict(self) -> None:
        """Test that non-dict request raises JSONTypeError."""
        encoded: JSONObject = {
            "run_id": "run-123",
            "user_id": 42,
            "request": "invalid",
        }
        with pytest.raises(JSONTypeError, match=r"request.*must be an object"):
            decode_train_job_payload(encoded)

    def test_decode_with_nested_lora(self) -> None:
        """Test decoding a full payload with nested LoRA config."""
        request = self._make_minimal_request()
        request["finetuning_strategy"] = "lora"
        request["lora"] = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }
        payload: TrainJobPayload = {
            "run_id": "lora-run",
            "user_id": 99,
            "request": request,
        }
        encoded = encode_train_job_payload(payload)
        decoded = decode_train_job_payload(encoded)

        assert decoded["run_id"] == "lora-run"
        assert decoded["user_id"] == 99
        lora = decoded["request"]["lora"]
        assert lora == {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ("q_proj", "v_proj"),
            "bias": "none",
        }
