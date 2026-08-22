"""Queue payload codecs: encode side."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from model_trainer.core.contracts.model import LoraConfig, QuantizationConfig
from model_trainer.core.contracts.queue_encoding_configs import (
    decode_lora_config,
    decode_quantization_config,
    encode_lora_config,
    encode_quantization_config,
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
