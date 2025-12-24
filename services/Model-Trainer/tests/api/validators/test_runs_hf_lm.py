"""Tests for hf_lm model family validation in runs.py."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_train_request


class TestHfLmModelFamily:
    """Tests for hf_lm model family validation."""

    def test_accepts_hf_lm_with_hub_model_id(self) -> None:
        """Test hf_lm model family with required hub_model_id."""
        payload: dict[str, JSONValue] = {
            "model_family": "hf_lm",
            "model_size": "base",
            "max_seq_len": 128,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "hub_model_id": "nghuyong/ernie-2.0-base-en",
            "user_id": 0,
        }
        out = _decode_train_request(payload)
        assert out["model_family"] == "hf_lm"
        assert out["hub_model_id"] == "nghuyong/ernie-2.0-base-en"
        assert out["tokenizer_id"] is None

    def test_hf_lm_tokenizer_id_optional_none(self) -> None:
        """Test hf_lm allows tokenizer_id=None."""
        payload: dict[str, JSONValue] = {
            "model_family": "hf_lm",
            "model_size": "base",
            "max_seq_len": 128,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "hub_model_id": "bert-base",
            "tokenizer_id": None,
            "user_id": 0,
        }
        out = _decode_train_request(payload)
        assert out["tokenizer_id"] is None

    def test_hf_lm_tokenizer_id_optional_empty_string(self) -> None:
        """Test hf_lm treats empty string tokenizer_id as None."""
        payload: dict[str, JSONValue] = {
            "model_family": "hf_lm",
            "model_size": "base",
            "max_seq_len": 128,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "hub_model_id": "bert-base",
            "tokenizer_id": "",
            "user_id": 0,
        }
        out = _decode_train_request(payload)
        assert out["tokenizer_id"] is None

    def test_hf_lm_tokenizer_id_accepts_string(self) -> None:
        """Test hf_lm accepts explicit tokenizer_id string."""
        payload: dict[str, JSONValue] = {
            "model_family": "hf_lm",
            "model_size": "base",
            "max_seq_len": 128,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "hub_model_id": "bert-base",
            "tokenizer_id": "custom-tok",
            "user_id": 0,
        }
        out = _decode_train_request(payload)
        assert out["tokenizer_id"] == "custom-tok"

    def test_hf_lm_tokenizer_id_type_error(self) -> None:
        """Test hf_lm rejects non-string tokenizer_id."""
        payload: dict[str, JSONValue] = {
            "model_family": "hf_lm",
            "model_size": "base",
            "max_seq_len": 128,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "hub_model_id": "bert-base",
            "tokenizer_id": 123,
            "user_id": 0,
        }
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert "tokenizer_id" in str(err.message)

    def test_hf_lm_requires_hub_model_id(self) -> None:
        """Test hf_lm requires hub_model_id."""
        payload: dict[str, JSONValue] = {
            "model_family": "hf_lm",
            "model_size": "base",
            "max_seq_len": 128,
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "corpus_file_id": "cid",
            "user_id": 0,
        }
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert "hub_model_id" in str(err.message)

    def test_non_hf_lm_requires_tokenizer_id(self) -> None:
        """Test gpt2 requires tokenizer_id."""
        payload: dict[str, JSONValue] = {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "corpus_file_id": "cid",
            "user_id": 0,
        }
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert "tokenizer_id is required" in str(err.message)

    def test_non_hf_lm_rejects_empty_tokenizer_id(self) -> None:
        """Test gpt2 rejects empty string tokenizer_id."""
        payload: dict[str, JSONValue] = {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "corpus_file_id": "cid",
            "tokenizer_id": "",
            "user_id": 0,
        }
        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert "tokenizer_id is required" in str(err.message)


def test_decode_train_request_accepts_llama_family() -> None:
    """Cover runs.py llama branch in _narrow_model_family."""
    payload: dict[str, JSONValue] = {
        "model_family": "llama",
        "model_size": "7b",
        "max_seq_len": 256,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
    }
    out = _decode_train_request(payload)
    assert out["model_family"] == "llama"
