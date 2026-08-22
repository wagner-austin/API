"""Tests for HTTP request body parsing."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.api.decode import (
    parse_external_train_request,
)


class TestParseExternalTrainRequest:
    """Tests for parse_external_train_request."""

    def test_valid_xgboost_request_defaults_to_xgboost(self) -> None:
        """Test parsing valid XGBoost request with default backend."""
        body = b"""{
            "dataset": "taiwan",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "xgboost"
        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["max_depth"] == 6
        assert result["config"]["n_estimators"] == 100
        assert result["config"]["device"] == "auto"
        assert result["config"]["train_ratio"] == 0.7
        assert result["config"]["val_ratio"] == 0.15
        assert result["config"]["test_ratio"] == 0.15
        assert result["config"]["early_stopping_rounds"] == 10
        assert result["config"]["reg_alpha"] == 0.0
        assert result["config"]["reg_lambda"] == 1.0

    def test_valid_xgboost_request_explicit_backend(self) -> None:
        """Test parsing valid XGBoost request with explicit backend."""
        body = b"""{
            "dataset": "us",
            "backend": "xgboost",
            "learning_rate": 0.2,
            "max_depth": 4,
            "n_estimators": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "random_state": 7,
            "device": "cpu",
            "scale_pos_weight": 2.5
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "xgboost"
        assert result["dataset"] == "us"
        assert result["config"]["device"] == "cpu"
        assert result["config"]["scale_pos_weight"] == 2.5

    def test_valid_mlp_request(self) -> None:
        """Test parsing valid MLP request."""
        body = b"""{
            "dataset": "polish",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, 32],
            "precision": "fp32",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "mlp"
        assert result["dataset"] == "polish"
        assert result["config"]["learning_rate"] == 0.001
        assert result["config"]["batch_size"] == 32
        assert result["config"]["n_epochs"] == 100
        assert result["config"]["dropout"] == 0.2
        assert result["config"]["hidden_sizes"] == (64, 32)
        assert result["config"]["precision"] == "fp32"
        assert result["config"]["optimizer"] == "adamw"
        assert result["config"]["random_state"] == 42
        assert result["config"]["early_stopping_patience"] == 10
        assert result["config"]["device"] == "auto"
        assert result["config"]["train_ratio"] == 0.7

    def test_mlp_request_with_cuda_and_fp16(self) -> None:
        """Test parsing MLP request with CUDA device and fp16 precision."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.01,
            "batch_size": 64,
            "n_epochs": 50,
            "dropout": 0.1,
            "hidden_sizes": [128, 64, 32],
            "precision": "fp16",
            "optimizer": "adam",
            "random_state": 123,
            "early_stopping_patience": 5,
            "device": "cuda"
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "mlp"
        assert result["config"]["device"] == "cuda"
        assert result["config"]["precision"] == "fp16"
        assert result["config"]["optimizer"] == "adam"
        assert result["config"]["hidden_sizes"] == (128, 64, 32)

    def test_mlp_request_with_bf16_and_sgd(self) -> None:
        """Test parsing MLP request with bf16 precision and SGD optimizer."""
        body = b"""{
            "dataset": "us",
            "backend": "mlp",
            "learning_rate": 0.1,
            "batch_size": 16,
            "n_epochs": 200,
            "dropout": 0.0,
            "hidden_sizes": [32],
            "precision": "bf16",
            "optimizer": "sgd",
            "random_state": 0,
            "early_stopping_patience": 20
        }"""
        result = parse_external_train_request(body)

        # Use if for type narrowing (discriminated union)
        if result["backend"] != "mlp":
            raise AssertionError("Expected mlp backend")
        assert result["config"]["precision"] == "bf16"
        assert result["config"]["optimizer"] == "sgd"

    def test_mlp_request_with_auto_precision(self) -> None:
        """Test parsing MLP request with auto precision."""
        body = b"""{
            "dataset": "polish",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, 32],
            "precision": "auto",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        result = parse_external_train_request(body)

        # Use if for type narrowing (discriminated union)
        if result["backend"] != "mlp":
            raise AssertionError("Expected mlp backend")
        assert result["config"]["precision"] == "auto"

    def test_request_with_custom_split_ratios(self) -> None:
        """Test parsing request with custom split ratios."""
        body = b"""{
            "dataset": "taiwan",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2
        }"""
        result = parse_external_train_request(body)

        assert result["config"]["train_ratio"] == 0.6
        assert result["config"]["val_ratio"] == 0.2
        assert result["config"]["test_ratio"] == 0.2

    def test_valid_lstm_request(self) -> None:
        """Test parsing valid LSTM request."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lstm",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": true,
            "sequence_length": 5,
            "precision": "fp32",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "lstm"
        assert result["dataset"] == "taiwan"
        # Use if for type narrowing (discriminated union)
        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["learning_rate"] == 0.001
        assert result["config"]["batch_size"] == 32
        assert result["config"]["n_epochs"] == 100
        assert result["config"]["dropout"] == 0.2
        assert result["config"]["hidden_size"] == 64
        assert result["config"]["num_layers"] == 2
        assert result["config"]["bidirectional"] is True
        assert result["config"]["sequence_length"] == 5
        assert result["config"]["precision"] == "fp32"
        assert result["config"]["random_state"] == 42
        assert result["config"]["early_stopping_patience"] == 10
        assert result["config"]["device"] == "auto"

    def test_lstm_request_with_cuda_and_fp16(self) -> None:
        """Test parsing LSTM request with CUDA device and fp16 precision."""
        body = b"""{
            "dataset": "us",
            "backend": "lstm",
            "learning_rate": 0.01,
            "batch_size": 64,
            "n_epochs": 50,
            "dropout": 0.1,
            "hidden_size": 128,
            "num_layers": 3,
            "bidirectional": false,
            "sequence_length": 10,
            "precision": "fp16",
            "random_state": 123,
            "early_stopping_patience": 5,
            "device": "cuda"
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["device"] == "cuda"
        assert result["config"]["precision"] == "fp16"
        assert result["config"]["bidirectional"] is False

    def test_lstm_request_with_bf16_and_auto_precision(self) -> None:
        """Test parsing LSTM request with bf16 and auto precision modes."""
        body = b"""{
            "dataset": "polish",
            "backend": "lstm",
            "learning_rate": 0.1,
            "batch_size": 16,
            "n_epochs": 200,
            "dropout": 0.0,
            "hidden_size": 32,
            "num_layers": 1,
            "bidirectional": true,
            "sequence_length": 3,
            "precision": "bf16",
            "random_state": 0,
            "early_stopping_patience": 20
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["precision"] == "bf16"

    def test_lstm_request_auto_precision(self) -> None:
        """Test parsing LSTM request with auto precision."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lstm",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": true,
            "sequence_length": 5,
            "precision": "auto",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["precision"] == "auto"

    def test_lstm_missing_bidirectional_raises_error(self) -> None:
        """Test that missing bidirectional field raises error."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lstm",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_size": 64,
            "num_layers": 2,
            "sequence_length": 5,
            "precision": "fp32",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="bidirectional must be a boolean"):
            parse_external_train_request(body)

    def test_lstm_invalid_precision_raises_error(self) -> None:
        """Test that invalid precision for LSTM raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lstm",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": true,
            "sequence_length": 5,
            "precision": "invalid_precision",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="precision must be fp32, fp16, bf16, or auto"):
            parse_external_train_request(body)
