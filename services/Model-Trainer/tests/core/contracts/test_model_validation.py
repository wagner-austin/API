"""Tests for model configuration validation functions."""

from __future__ import annotations

import pytest

from model_trainer.core.contracts.model_validation import (
    JSONObject,
    _decode_lora_config,
    _decode_optional_lora_config,
    _decode_optional_quantization_config,
    _decode_quantization_config,
    encode_lora_config,
    encode_quantization_config,
)


class TestDecodeLoraConfig:
    """Tests for _decode_lora_config."""

    def test_valid_lora_config(self) -> None:
        """Test decoding a valid LoRA config."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        }
        result = _decode_lora_config(data)
        assert result["enabled"] is True
        assert result["r"] == 16
        assert result["lora_alpha"] == 32
        assert result["lora_dropout"] == 0.1
        assert result["target_modules"] == ("q_proj", "v_proj")
        assert result["bias"] == "none"

    def test_missing_enabled_raises(self) -> None:
        """Test that missing enabled field raises TypeError."""
        data: JSONObject = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q"],
            "bias": "none",
        }
        with pytest.raises(TypeError, match="Missing required field 'enabled'"):
            _decode_lora_config(data)

    def test_invalid_r_type_raises(self) -> None:
        """Test that non-int r raises TypeError."""
        data: JSONObject = {
            "enabled": True,
            "r": "16",
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q"],
            "bias": "none",
        }
        with pytest.raises(TypeError, match="Field 'r' must be an integer"):
            _decode_lora_config(data)

    def test_r_less_than_one_raises(self) -> None:
        """Test that r < 1 raises ValueError."""
        data: JSONObject = {
            "enabled": True,
            "r": 0,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q"],
            "bias": "none",
        }
        with pytest.raises(ValueError, match="r must be >= 1"):
            _decode_lora_config(data)

    def test_invalid_lora_alpha_raises(self) -> None:
        """Test that invalid lora_alpha raises."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": "32",
            "lora_dropout": 0.1,
            "target_modules": ["q"],
            "bias": "none",
        }
        with pytest.raises(TypeError, match="Field 'lora_alpha' must be an integer"):
            _decode_lora_config(data)

    def test_lora_alpha_less_than_one_raises(self) -> None:
        """Test that lora_alpha < 1 raises ValueError."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 0,
            "lora_dropout": 0.1,
            "target_modules": ["q"],
            "bias": "none",
        }
        with pytest.raises(ValueError, match="lora_alpha must be >= 1"):
            _decode_lora_config(data)

    def test_invalid_lora_dropout_raises(self) -> None:
        """Test that invalid lora_dropout raises TypeError."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": "0.1",
            "target_modules": ["q"],
            "bias": "none",
        }
        with pytest.raises(TypeError, match="lora_dropout must be float"):
            _decode_lora_config(data)

    def test_lora_dropout_out_of_range_raises(self) -> None:
        """Test that lora_dropout > 1.0 raises ValueError."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 1.5,
            "target_modules": ["q"],
            "bias": "none",
        }
        with pytest.raises(ValueError, match="lora_dropout must be between"):
            _decode_lora_config(data)

    def test_target_modules_not_list_raises(self) -> None:
        """Test that non-list target_modules raises TypeError."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": "q",
            "bias": "none",
        }
        with pytest.raises(TypeError, match="target_modules must be list"):
            _decode_lora_config(data)

    def test_empty_target_modules_raises(self) -> None:
        """Test that empty target_modules raises ValueError."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": [],
            "bias": "none",
        }
        with pytest.raises(ValueError, match="target_modules must not be empty"):
            _decode_lora_config(data)

    def test_target_modules_non_string_element_raises(self) -> None:
        """Test that non-string elements in target_modules raises TypeError."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": [1, 2],
            "bias": "none",
        }
        with pytest.raises(TypeError, match="target_modules elements must be str"):
            _decode_lora_config(data)

    def test_invalid_bias_raises(self) -> None:
        """Test that invalid bias value raises ValueError."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q"],
            "bias": "invalid",
        }
        with pytest.raises(ValueError, match="bias must be"):
            _decode_lora_config(data)

    def test_bias_all_accepted(self) -> None:
        """Test that bias='all' is accepted."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q"],
            "bias": "all",
        }
        result = _decode_lora_config(data)
        assert result["bias"] == "all"

    def test_bias_lora_only_accepted(self) -> None:
        """Test that bias='lora_only' is accepted."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q"],
            "bias": "lora_only",
        }
        result = _decode_lora_config(data)
        assert result["bias"] == "lora_only"

    def test_int_dropout_converted_to_float(self) -> None:
        """Test that int lora_dropout is converted to float."""
        data: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0,
            "target_modules": ["q"],
            "bias": "none",
        }
        result = _decode_lora_config(data)
        assert result["lora_dropout"] == 0.0
        assert type(result["lora_dropout"]) is float


class TestEncodeLoraConfig:
    """Tests for encode_lora_config."""

    def test_encode_roundtrip(self) -> None:
        """Test encoding and decoding produces same values."""
        original: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        }
        decoded = _decode_lora_config(original)
        encoded = encode_lora_config(decoded)
        assert encoded["enabled"] == original["enabled"]
        assert encoded["r"] == original["r"]
        assert encoded["lora_alpha"] == original["lora_alpha"]
        assert encoded["lora_dropout"] == original["lora_dropout"]
        assert encoded["target_modules"] == original["target_modules"]
        assert encoded["bias"] == original["bias"]


class TestDecodeQuantizationConfig:
    """Tests for _decode_quantization_config."""

    def test_valid_4bit_config(self) -> None:
        """Test decoding a valid 4-bit quantization config."""
        data: JSONObject = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
        }
        result = _decode_quantization_config(data)
        assert result["load_in_4bit"] is True
        assert result["load_in_8bit"] is False
        assert result["bnb_4bit_compute_dtype"] == "float16"
        assert result["bnb_4bit_quant_type"] == "nf4"

    def test_valid_8bit_config(self) -> None:
        """Test decoding a valid 8-bit quantization config."""
        data: JSONObject = {
            "load_in_4bit": False,
            "load_in_8bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "fp4",
        }
        result = _decode_quantization_config(data)
        assert result["load_in_8bit"] is True
        assert result["bnb_4bit_quant_type"] == "fp4"

    def test_missing_load_in_4bit_raises(self) -> None:
        """Test that missing load_in_4bit raises TypeError."""
        data: JSONObject = {
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
        }
        with pytest.raises(TypeError, match="Missing required field 'load_in_4bit'"):
            _decode_quantization_config(data)

    def test_both_4bit_and_8bit_raises(self) -> None:
        """Test that both 4bit and 8bit enabled raises ValueError."""
        data: JSONObject = {
            "load_in_4bit": True,
            "load_in_8bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
        }
        with pytest.raises(ValueError, match="cannot have both"):
            _decode_quantization_config(data)

    def test_invalid_compute_dtype_raises(self) -> None:
        """Test that invalid compute_dtype raises ValueError."""
        data: JSONObject = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "float64",
            "bnb_4bit_quant_type": "nf4",
        }
        with pytest.raises(ValueError, match="compute_dtype must be"):
            _decode_quantization_config(data)

    def test_invalid_quant_type_raises(self) -> None:
        """Test that invalid quant_type raises ValueError."""
        data: JSONObject = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "int8",
        }
        with pytest.raises(ValueError, match="quant_type must be"):
            _decode_quantization_config(data)

    def test_float32_compute_dtype(self) -> None:
        """Test float32 compute dtype is accepted."""
        data: JSONObject = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "float32",
            "bnb_4bit_quant_type": "nf4",
        }
        result = _decode_quantization_config(data)
        assert result["bnb_4bit_compute_dtype"] == "float32"


class TestEncodeQuantizationConfig:
    """Tests for encode_quantization_config."""

    def test_encode_roundtrip(self) -> None:
        """Test encoding and decoding produces same values."""
        original: JSONObject = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
        }
        decoded = _decode_quantization_config(original)
        encoded = encode_quantization_config(decoded)
        assert encoded == original


class TestDecodeOptionalConfigs:
    """Tests for optional config decoders."""

    def test_optional_lora_none(self) -> None:
        """Test that missing lora key returns None."""
        data: JSONObject = {"other": "value"}
        result = _decode_optional_lora_config(data)
        assert result is None

    def test_optional_lora_null(self) -> None:
        """Test that null lora value returns None."""
        data: JSONObject = {"lora": None}
        result = _decode_optional_lora_config(data)
        assert result is None

    def test_optional_lora_valid(self) -> None:
        """Test that valid lora dict is decoded."""
        lora_obj: JSONObject = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q"],
            "bias": "none",
        }
        data: JSONObject = {"lora": lora_obj}
        result = _decode_optional_lora_config(data)
        # Compare against directly decoded lora to verify equality
        expected = _decode_lora_config(lora_obj)
        assert result == expected

    def test_optional_lora_invalid_type_raises(self) -> None:
        """Test that non-dict lora raises TypeError."""
        data: JSONObject = {"lora": "invalid"}
        with pytest.raises(TypeError, match="lora must be dict or null"):
            _decode_optional_lora_config(data)

    def test_optional_quantization_none(self) -> None:
        """Test that missing quantization key returns None."""
        data: JSONObject = {"other": "value"}
        result = _decode_optional_quantization_config(data)
        assert result is None

    def test_optional_quantization_valid(self) -> None:
        """Test that valid quantization dict is decoded."""
        quant_obj: JSONObject = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
        }
        data: JSONObject = {"quantization": quant_obj}
        result = _decode_optional_quantization_config(data)
        expected = _decode_quantization_config(quant_obj)
        assert result == expected

    def test_optional_quantization_invalid_type_raises(self) -> None:
        """Test that non-dict quantization raises TypeError."""
        data: JSONObject = {"quantization": [1, 2, 3]}
        with pytest.raises(TypeError, match="quantization must be dict or null"):
            _decode_optional_quantization_config(data)
