"""Tests for HTTP request body parsing."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.api.decode import (
    parse_external_regression_train_request,
    parse_external_train_request,
    parse_train_request,
)


class TestParseTrainRequest:
    """Tests for parse_train_request."""

    def test_valid_train_request(self) -> None:
        """Test parsing a valid train request."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "early_stopping_rounds": 10
        }"""
        result = parse_train_request(body)

        assert result["learning_rate"] == 0.1
        assert result["max_depth"] == 6
        assert result["n_estimators"] == 100
        assert result["subsample"] == 0.8
        assert result["colsample_bytree"] == 0.8
        assert result["random_state"] == 42
        assert result["train_ratio"] == 0.7
        assert result["val_ratio"] == 0.15
        assert result["test_ratio"] == 0.15
        assert result["early_stopping_rounds"] == 10
        # reg_alpha/reg_lambda default when not provided
        assert result["reg_alpha"] == 0.0
        assert result["reg_lambda"] == 1.0
        assert result["device"] == "auto"

    def test_request_with_defaults(self) -> None:
        """Test parsing with optional fields defaulted."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        result = parse_train_request(body)

        assert result["learning_rate"] == 0.1
        # Default values for optional fields
        assert result["train_ratio"] == 0.7
        assert result["val_ratio"] == 0.15
        assert result["test_ratio"] == 0.15
        assert result["early_stopping_rounds"] == 10
        assert result["reg_alpha"] == 0.0
        assert result["reg_lambda"] == 1.0
        assert result["device"] == "auto"

    def test_train_request_with_regularization_and_scale(self) -> None:
        """Test parsing reg params, device, and scale_pos_weight."""
        body = b"""{
            "learning_rate": 0.2,
            "max_depth": 4,
            "n_estimators": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "random_state": 7,
            "device": "cuda",
            "reg_alpha": 2.5,
            "reg_lambda": 3.5,
            "scale_pos_weight": 1.2
        }"""
        result = parse_train_request(body)

        assert result["device"] == "cuda"
        assert result["reg_alpha"] == 2.5
        assert result["reg_lambda"] == 3.5
        assert result["scale_pos_weight"] == 1.2
        assert result["n_estimators"] == 50

    def test_train_request_invalid_scale_pos_weight(self) -> None:
        """Test parsing rejects invalid scale_pos_weight type."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "scale_pos_weight": "heavy"
        }"""
        with pytest.raises(JSONTypeError, match="scale_pos_weight must be a number"):
            parse_train_request(body)

    def test_train_request_invalid_ratio_type(self) -> None:
        """Test parsing rejects non-numeric ratio values."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": "big"
        }"""
        with pytest.raises(JSONTypeError, match="Field 'train_ratio' must be a number"):
            parse_train_request(body)

    def test_train_request_invalid_device(self) -> None:
        """Test parsing rejects unsupported device value."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "tpu"
        }"""
        with pytest.raises(JSONTypeError, match="device must be one of: cpu, cuda, auto"):
            parse_train_request(body)

    def test_train_request_device_cpu(self) -> None:
        """Test parsing accepts explicit CPU device."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "cpu"
        }"""
        result = parse_train_request(body)
        assert result["device"] == "cpu"

    def test_train_request_device_auto_string(self) -> None:
        """Test parsing accepts explicit auto device string."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "auto"
        }"""
        result = parse_train_request(body)
        assert result["device"] == "auto"

    def test_train_request_non_string_device(self) -> None:
        """Test parsing rejects non-string device types."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": 123
        }"""
        with pytest.raises(JSONTypeError, match="device must be a string"):
            parse_train_request(body)

    def test_early_stopping_as_float(self) -> None:
        """Test parsing early_stopping_rounds as float (converts to int)."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 15.0
        }"""
        result = parse_train_request(body)

        assert result["early_stopping_rounds"] == 15

    def test_early_stopping_invalid_type(self) -> None:
        """Test parsing rejects non-numeric early_stopping_rounds."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": "fast"
        }"""
        with pytest.raises(JSONTypeError, match="Field 'early_stopping_rounds' must be a number"):
            parse_train_request(body)

    def test_missing_field_raises_json_type_error(self) -> None:
        """Test that missing field raises JSONTypeError."""
        body = b"""{"learning_rate": 0.1}"""
        with pytest.raises(JSONTypeError, match="Missing required field"):
            parse_train_request(body)


class TestParseExternalTrainRequestValidation:
    """parse_external_train_request: cross-backend validation errors."""

    def test_invalid_dataset_raises_value_error(self) -> None:
        """Test that invalid dataset raises ValueError."""
        body = b"""{
            "dataset": "invalid_dataset",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        with pytest.raises(ValueError, match="dataset must be one of"):
            parse_external_train_request(body)

    def test_missing_dataset_raises_json_type_error(self) -> None:
        """Test that missing dataset raises JSONTypeError."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'dataset'"):
            parse_external_train_request(body)

    def test_invalid_ratios_sum_raises_value_error(self) -> None:
        """Test that split ratios not summing to 1.0 raises ValueError."""
        body = b"""{
            "dataset": "taiwan",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": 0.5,
            "val_ratio": 0.2,
            "test_ratio": 0.2
        }"""
        with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
            parse_external_train_request(body)

    def test_invalid_precision_raises_json_type_error(self) -> None:
        """Test that invalid precision raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, 32],
            "precision": "invalid",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="precision must be fp32, fp16, bf16, or auto"):
            parse_external_train_request(body)

    def test_invalid_optimizer_raises_json_type_error(self) -> None:
        """Test that invalid optimizer raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, 32],
            "precision": "fp32",
            "optimizer": "invalid",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="optimizer must be adamw, adam, or sgd"):
            parse_external_train_request(body)

    def test_invalid_hidden_sizes_not_list_raises_json_type_error(self) -> None:
        """Test that hidden_sizes not being a list raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": 64,
            "precision": "fp32",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="hidden_sizes must be list of ints"):
            parse_external_train_request(body)

    def test_invalid_hidden_sizes_contains_non_int_raises_json_type_error(self) -> None:
        """Test that hidden_sizes containing non-int raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, "invalid"],
            "precision": "fp32",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="hidden_sizes must be list of ints"):
            parse_external_train_request(body)

    def test_xgboost_invalid_scale_pos_weight_raises_json_type_error(self) -> None:
        """Test that invalid scale_pos_weight raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "scale_pos_weight": "heavy"
        }"""
        with pytest.raises(JSONTypeError, match="scale_pos_weight must be a number"):
            parse_external_train_request(body)


class TestParseExternalRegressionTrainRequest:
    """Tests for parse_external_regression_train_request."""

    def test_minimal_xgboost_reg_request(self) -> None:
        """Minimal request defaults to xgboost_reg backend."""
        body = b"""{
            "dataset": "financial_distress",
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        result = parse_external_regression_train_request(body)

        assert result["backend"] == "xgboost_reg"
        assert result["dataset"] == "financial_distress"
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["max_depth"] == 3
        assert result["config"]["device"] == "auto"
        assert result["config"]["train_ratio"] == 0.7

    def test_explicit_lightgbm_reg_request(self) -> None:
        """Explicit lightgbm_reg backend is parsed correctly."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "lightgbm_reg",
            "device": "cpu",
            "learning_rate": 0.05,
            "max_depth": 5,
            "n_estimators": 100,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        result = parse_external_regression_train_request(body)

        assert result["backend"] == "lightgbm_reg"
        assert result["config"]["num_leaves"] == 31

    def test_invalid_dataset_raises_value_error(self) -> None:
        """Invalid regression dataset raises ValueError."""
        body = b"""{
            "dataset": "nonexistent",
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        with pytest.raises(ValueError, match="dataset must be one of"):
            parse_external_regression_train_request(body)

    def test_classifier_backend_raises_value_error(self) -> None:
        """Classifier backend name raises ValueError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "xgboost",
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_external_regression_train_request(body)

    def test_bad_split_ratios_raises_value_error(self) -> None:
        """Split ratios not summing to 1.0 raises ValueError."""
        body = b"""{
            "dataset": "financial_distress",
            "train_ratio": 0.5,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
            parse_external_regression_train_request(body)
