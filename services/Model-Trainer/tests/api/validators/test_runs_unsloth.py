"""Tests for Unsloth configuration validation in runs.py."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_train_request


def _base_unsloth_payload() -> dict[str, JSONValue]:
    """Return base payload for unsloth tests."""
    return {
        "model_family": "hf_lm",
        "model_size": "base",
        "max_seq_len": 128,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "corpus_file_id": "cid",
        "hub_model_id": "bert-base",
        "finetuning_strategy": "unsloth",
        "lora": {"r": 16},
        "user_id": 0,
    }


class TestUnslothConfig:
    """Tests for Unsloth configuration validation."""

    def test_unsloth_defaults(self) -> None:
        """Test unsloth config defaults."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {}
        out = _decode_train_request(payload)
        unsloth = out["unsloth"]
        assert (
            unsloth is not None
            and unsloth["enabled"] is True
            and unsloth["max_seq_length"] == 2048
            and unsloth["dtype"] is None
        )

    def test_unsloth_enabled_true(self) -> None:
        """Test unsloth enabled=True explicit."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"enabled": True}
        out = _decode_train_request(payload)
        assert out["unsloth"] is not None and out["unsloth"]["enabled"] is True

    def test_unsloth_enabled_false(self) -> None:
        """Test unsloth enabled=False."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"enabled": False}
        out = _decode_train_request(payload)
        assert out["unsloth"] is not None and out["unsloth"]["enabled"] is False

    def test_unsloth_enabled_type_error(self) -> None:
        """Test unsloth enabled type error."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"enabled": "yes"}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "unsloth.enabled" in str(err.message)

    def test_unsloth_max_seq_length_custom(self) -> None:
        """Test unsloth max_seq_length custom value."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"max_seq_length": 4096}
        out = _decode_train_request(payload)
        assert out["unsloth"] is not None and out["unsloth"]["max_seq_length"] == 4096

    def test_unsloth_max_seq_length_min(self) -> None:
        """Test unsloth max_seq_length minimum valid value."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"max_seq_length": 128}
        out = _decode_train_request(payload)
        assert out["unsloth"] is not None and out["unsloth"]["max_seq_length"] == 128

    def test_unsloth_max_seq_length_max(self) -> None:
        """Test unsloth max_seq_length maximum valid value."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"max_seq_length": 8192}
        out = _decode_train_request(payload)
        assert out["unsloth"] is not None and out["unsloth"]["max_seq_length"] == 8192

    def test_unsloth_max_seq_length_too_small(self) -> None:
        """Test unsloth max_seq_length below minimum."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"max_seq_length": 64}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_unsloth_max_seq_length_too_large(self) -> None:
        """Test unsloth max_seq_length above maximum."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"max_seq_length": 16384}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_unsloth_dtype_none(self) -> None:
        """Test unsloth dtype=None for auto-detect default."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {}
        out = _decode_train_request(payload)
        assert out["unsloth"] is not None and out["unsloth"]["dtype"] is None

    def test_unsloth_dtype_null_explicit(self) -> None:
        """Test unsloth dtype=null explicit for auto-detect."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"dtype": None}
        out = _decode_train_request(payload)
        assert out["unsloth"] is not None and out["unsloth"]["dtype"] is None

    def test_unsloth_dtype_float16(self) -> None:
        """Test unsloth dtype=float16."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"dtype": "float16"}
        out = _decode_train_request(payload)
        assert out["unsloth"] is not None and out["unsloth"]["dtype"] == "float16"

    def test_unsloth_dtype_bfloat16(self) -> None:
        """Test unsloth dtype=bfloat16."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"dtype": "bfloat16"}
        out = _decode_train_request(payload)
        assert out["unsloth"] is not None and out["unsloth"]["dtype"] == "bfloat16"

    def test_unsloth_dtype_type_error(self) -> None:
        """Test unsloth dtype type error."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"dtype": 16}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "unsloth.dtype" in str(err.message)

    def test_unsloth_dtype_invalid_value(self) -> None:
        """Test unsloth dtype invalid value (float32 not allowed)."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = {"dtype": "float32"}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_unsloth_not_dict(self) -> None:
        """Test unsloth config must be dict."""
        payload = _base_unsloth_payload()
        payload["unsloth"] = "not-a-dict"
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "unsloth must be an object" in str(err.message)
