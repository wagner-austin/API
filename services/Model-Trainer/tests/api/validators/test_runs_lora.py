"""Tests for LoRA configuration validation in runs.py."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_train_request


def _base_lora_payload() -> dict[str, JSONValue]:
    """Return base payload for LoRA tests."""
    return {
        "model_family": "hf_lm",
        "model_size": "base",
        "max_seq_len": 128,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "corpus_file_id": "cid",
        "corpus_format": "lines",
        "hub_model_id": "bert-base",
        "finetuning_strategy": "lora",
        "user_id": 0,
    }


class TestLoraConfig:
    """Tests for LoRA configuration validation."""

    def test_lora_defaults(self) -> None:
        """Test LoRA config defaults."""
        payload = _base_lora_payload()
        payload["lora"] = {}
        out = _decode_train_request(payload)
        lora = out["lora"]
        assert (
            lora is not None
            and lora["enabled"] is True
            and lora["r"] == 16
            and lora["lora_alpha"] == 16
            and lora["lora_dropout"] == 0.1
            and lora["target_modules"] == ("q_proj", "k_proj", "v_proj", "o_proj")
            and lora["bias"] == "none"
        )

    def test_lora_enabled_true(self) -> None:
        """Test LoRA enabled=True explicit."""
        payload = _base_lora_payload()
        payload["lora"] = {"enabled": True}
        out = _decode_train_request(payload)
        assert out["lora"] is not None and out["lora"]["enabled"] is True

    def test_lora_enabled_false(self) -> None:
        """Test LoRA enabled=False."""
        payload = _base_lora_payload()
        payload["lora"] = {"enabled": False}
        out = _decode_train_request(payload)
        assert out["lora"] is not None and out["lora"]["enabled"] is False

    def test_lora_enabled_type_error(self) -> None:
        """Test LoRA enabled type error."""
        payload = _base_lora_payload()
        payload["lora"] = {"enabled": "yes"}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "lora.enabled" in str(err.message)

    def test_lora_r_custom(self) -> None:
        """Test LoRA r custom value."""
        payload = _base_lora_payload()
        payload["lora"] = {"r": 64}
        out = _decode_train_request(payload)
        assert out["lora"] is not None and out["lora"]["r"] == 64

    def test_lora_r_too_small(self) -> None:
        """Test LoRA r below minimum."""
        payload = _base_lora_payload()
        payload["lora"] = {"r": 2}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_lora_r_too_large(self) -> None:
        """Test LoRA r above maximum."""
        payload = _base_lora_payload()
        payload["lora"] = {"r": 256}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_lora_alpha_custom(self) -> None:
        """Test LoRA alpha custom value."""
        payload = _base_lora_payload()
        payload["lora"] = {"lora_alpha": 32}
        out = _decode_train_request(payload)
        assert out["lora"] is not None and out["lora"]["lora_alpha"] == 32

    def test_lora_dropout_custom(self) -> None:
        """Test LoRA dropout custom value."""
        payload = _base_lora_payload()
        payload["lora"] = {"lora_dropout": 0.2}
        out = _decode_train_request(payload)
        assert out["lora"] is not None and out["lora"]["lora_dropout"] == 0.2

    def test_lora_dropout_too_high(self) -> None:
        """Test LoRA dropout above maximum."""
        payload = _base_lora_payload()
        payload["lora"] = {"lora_dropout": 0.8}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_lora_target_modules_custom(self) -> None:
        """Test LoRA target_modules custom value."""
        payload = _base_lora_payload()
        payload["lora"] = {"target_modules": ["query", "key"]}
        out = _decode_train_request(payload)
        assert out["lora"] is not None and out["lora"]["target_modules"] == ("query", "key")

    def test_lora_target_modules_type_error(self) -> None:
        """Test LoRA target_modules type error."""
        payload = _base_lora_payload()
        payload["lora"] = {"target_modules": "q_proj"}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "target_modules must be a list" in str(err.message)

    def test_lora_target_modules_element_type_error(self) -> None:
        """Test LoRA target_modules element type error."""
        payload = _base_lora_payload()
        payload["lora"] = {"target_modules": ["q_proj", 123]}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "target_modules[1] must be a string" in str(err.message)

    def test_lora_bias_none(self) -> None:
        """Test LoRA bias='none' default."""
        payload = _base_lora_payload()
        payload["lora"] = {}
        out = _decode_train_request(payload)
        assert out["lora"] is not None and out["lora"]["bias"] == "none"

    def test_lora_bias_all(self) -> None:
        """Test LoRA bias='all'."""
        payload = _base_lora_payload()
        payload["lora"] = {"bias": "all"}
        out = _decode_train_request(payload)
        assert out["lora"] is not None and out["lora"]["bias"] == "all"

    def test_lora_bias_lora_only(self) -> None:
        """Test LoRA bias='lora_only'."""
        payload = _base_lora_payload()
        payload["lora"] = {"bias": "lora_only"}
        out = _decode_train_request(payload)
        assert out["lora"] is not None and out["lora"]["bias"] == "lora_only"

    def test_lora_bias_invalid(self) -> None:
        """Test LoRA bias invalid value."""
        payload = _base_lora_payload()
        payload["lora"] = {"bias": "invalid"}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_lora_not_dict(self) -> None:
        """Test LoRA config must be dict."""
        payload = _base_lora_payload()
        payload["lora"] = "not-a-dict"
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "lora must be an object" in str(err.message)
