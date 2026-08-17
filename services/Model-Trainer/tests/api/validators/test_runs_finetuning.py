"""Tests for finetuning_strategy validation in runs.py."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_train_request


def _base_hf_lm_payload() -> dict[str, JSONValue]:
    """Return base payload for hf_lm tests."""
    return {
        "model_family": "hf_lm",
        "model_size": "base",
        "max_seq_len": 128,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "corpus_file_id": "cid",
        "hub_model_id": "bert-base",
        "user_id": 0,
    }


class TestFinetuningStrategy:
    """Tests for finetuning_strategy validation."""

    def test_strategy_full_default(self) -> None:
        """Test finetuning_strategy defaults to 'full'."""
        payload = _base_hf_lm_payload()
        out = _decode_train_request(payload)
        assert out["finetuning_strategy"] == "full"

    def test_strategy_full_explicit(self) -> None:
        """Test finetuning_strategy explicit 'full'."""
        payload = _base_hf_lm_payload()
        payload["finetuning_strategy"] = "full"
        out = _decode_train_request(payload)
        assert out["finetuning_strategy"] == "full"

    def test_strategy_lora_requires_lora_config(self) -> None:
        """Test lora strategy requires lora config."""
        payload = _base_hf_lm_payload()
        payload["finetuning_strategy"] = "lora"
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert "lora config is required" in str(err.message)

    def test_strategy_lora_with_config(self) -> None:
        """Test lora strategy with valid config."""
        payload = _base_hf_lm_payload()
        payload["finetuning_strategy"] = "lora"
        payload["lora"] = {"r": 16, "lora_alpha": 16, "lora_dropout": 0.1}
        out = _decode_train_request(payload)
        assert out["finetuning_strategy"] == "lora"
        assert out["lora"] is not None and out["lora"]["r"] == 16

    def test_strategy_qlora_requires_lora_config(self) -> None:
        """Test qlora strategy requires lora config."""
        payload = _base_hf_lm_payload()
        payload["finetuning_strategy"] = "qlora"
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert "lora config is required" in str(err.message)

    def test_strategy_qlora_requires_quantization_config(self) -> None:
        """Test qlora strategy requires quantization config."""
        payload = _base_hf_lm_payload()
        payload["finetuning_strategy"] = "qlora"
        payload["lora"] = {"r": 16}
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert "quantization config is required" in str(err.message)

    def test_strategy_qlora_with_configs(self) -> None:
        """Test qlora strategy with valid configs."""
        payload = _base_hf_lm_payload()
        payload["finetuning_strategy"] = "qlora"
        payload["lora"] = {"r": 16}
        payload["quantization"] = {"load_in_4bit": True}
        out = _decode_train_request(payload)
        assert out["finetuning_strategy"] == "qlora"
        assert out["lora"] is not None and out["quantization"] is not None

    def test_strategy_invalid_value(self) -> None:
        """Test invalid finetuning_strategy value."""
        payload = _base_hf_lm_payload()
        payload["finetuning_strategy"] = "invalid"
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
