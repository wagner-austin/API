"""Tests for quantization configuration validation in runs.py."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_train_request


def _base_qlora_payload() -> dict[str, JSONValue]:
    """Return base payload for qlora tests."""
    return {
        "model_family": "hf_lm",
        "model_size": "base",
        "max_seq_len": 128,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "corpus_file_id": "cid",
        "hub_model_id": "bert-base",
        "finetuning_strategy": "qlora",
        "lora": {"r": 16},
        "user_id": 0,
    }


class TestQuantizationConfig:
    """Tests for quantization configuration validation."""

    def test_quantization_defaults(self) -> None:
        """Test quantization config defaults."""
        payload = _base_qlora_payload()
        payload["quantization"] = {}
        out = _decode_train_request(payload)
        quant = out["quantization"]
        assert (
            quant is not None
            and quant["load_in_4bit"] is True
            and quant["load_in_8bit"] is False
            and quant["bnb_4bit_compute_dtype"] == "float16"
            and quant["bnb_4bit_quant_type"] == "nf4"
        )

    def test_quantization_load_in_4bit_true(self) -> None:
        """Test quantization load_in_4bit=True explicit."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"load_in_4bit": True}
        out = _decode_train_request(payload)
        assert out["quantization"] is not None and out["quantization"]["load_in_4bit"] is True

    def test_quantization_load_in_4bit_false(self) -> None:
        """Test quantization load_in_4bit=False."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"load_in_4bit": False}
        out = _decode_train_request(payload)
        assert out["quantization"] is not None and out["quantization"]["load_in_4bit"] is False

    def test_quantization_load_in_4bit_type_error(self) -> None:
        """Test quantization load_in_4bit type error."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"load_in_4bit": "yes"}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "load_in_4bit must be a boolean" in str(err.message)

    def test_quantization_load_in_8bit_true(self) -> None:
        """Test quantization load_in_8bit=True."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"load_in_8bit": True}
        out = _decode_train_request(payload)
        assert out["quantization"] is not None and out["quantization"]["load_in_8bit"] is True

    def test_quantization_load_in_8bit_false(self) -> None:
        """Test quantization load_in_8bit=False explicit."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"load_in_8bit": False}
        out = _decode_train_request(payload)
        assert out["quantization"] is not None and out["quantization"]["load_in_8bit"] is False

    def test_quantization_load_in_8bit_type_error(self) -> None:
        """Test quantization load_in_8bit type error."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"load_in_8bit": "no"}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "load_in_8bit must be a boolean" in str(err.message)

    def test_quantization_compute_dtype_float16(self) -> None:
        """Test quantization compute_dtype=float16 default."""
        payload = _base_qlora_payload()
        payload["quantization"] = {}
        out = _decode_train_request(payload)
        assert (
            out["quantization"] is not None
            and out["quantization"]["bnb_4bit_compute_dtype"] == "float16"
        )

    def test_quantization_compute_dtype_bfloat16(self) -> None:
        """Test quantization compute_dtype=bfloat16."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"bnb_4bit_compute_dtype": "bfloat16"}
        out = _decode_train_request(payload)
        assert (
            out["quantization"] is not None
            and out["quantization"]["bnb_4bit_compute_dtype"] == "bfloat16"
        )

    def test_quantization_compute_dtype_float32(self) -> None:
        """Test quantization compute_dtype=float32."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"bnb_4bit_compute_dtype": "float32"}
        out = _decode_train_request(payload)
        assert (
            out["quantization"] is not None
            and out["quantization"]["bnb_4bit_compute_dtype"] == "float32"
        )

    def test_quantization_compute_dtype_invalid(self) -> None:
        """Test quantization compute_dtype invalid value."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"bnb_4bit_compute_dtype": "invalid"}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_quantization_quant_type_nf4(self) -> None:
        """Test quantization quant_type=nf4 default."""
        payload = _base_qlora_payload()
        payload["quantization"] = {}
        out = _decode_train_request(payload)
        assert (
            out["quantization"] is not None and out["quantization"]["bnb_4bit_quant_type"] == "nf4"
        )

    def test_quantization_quant_type_fp4(self) -> None:
        """Test quantization quant_type=fp4."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"bnb_4bit_quant_type": "fp4"}
        out = _decode_train_request(payload)
        assert (
            out["quantization"] is not None and out["quantization"]["bnb_4bit_quant_type"] == "fp4"
        )

    def test_quantization_quant_type_invalid(self) -> None:
        """Test quantization quant_type invalid value."""
        payload = _base_qlora_payload()
        payload["quantization"] = {"bnb_4bit_quant_type": "int8"}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_quantization_not_dict(self) -> None:
        """Test quantization config must be dict."""
        payload = _base_qlora_payload()
        payload["quantization"] = "not-a-dict"
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "quantization must be an object" in str(err.message)
