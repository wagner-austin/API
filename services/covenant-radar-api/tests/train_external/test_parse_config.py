"""Tests for _parse_external_train_config function."""

from __future__ import annotations

import pytest
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str

from covenant_radar_api.worker.train_external_job import _parse_external_train_config


class TestXGBoostConfig:
    """Tests for parsing XGBoost configuration."""

    def test_valid_taiwan(self) -> None:
        """Parse valid config for Taiwan dataset."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        result = _parse_external_train_config(config_json)

        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["device"] == "auto"

    def test_valid_us(self) -> None:
        """Parse valid config for US dataset."""
        config_json = dump_json_str(
            {
                "dataset": "us",
                "learning_rate": 0.2,
                "max_depth": 4,
                "n_estimators": 50,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 99,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["dataset"] == "us"

    def test_valid_polish(self) -> None:
        """Parse valid config for Polish dataset."""
        config_json = dump_json_str(
            {
                "dataset": "polish",
                "learning_rate": 0.15,
                "max_depth": 5,
                "n_estimators": 100,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 7,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["dataset"] == "polish"

    def test_with_scale_pos_weight(self) -> None:
        """Config with scale_pos_weight is parsed correctly."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "scale_pos_weight": 2.5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["backend"] == "xgboost"
        assert result["config"].get("scale_pos_weight") == 2.5

    def test_with_custom_ratios(self) -> None:
        """Config with custom split ratios is accepted."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["config"]["train_ratio"] == 0.8
        assert result["config"]["val_ratio"] == 0.1
        assert result["config"]["test_ratio"] == 0.1


class TestMLPConfig:
    """Tests for parsing MLP configuration."""

    def test_valid(self) -> None:
        """Parse valid config for MLP backend."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "taiwan",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64, 32],
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["backend"] == "mlp"
        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.001
        assert result["config"]["hidden_sizes"] == (64, 32)

    def test_precision_fp16(self) -> None:
        """Parse MLP config with fp16 precision."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "fp16",
                "optimizer": "adam",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["config"].get("precision") == "fp16"
        assert result["config"].get("optimizer") == "adam"

    def test_precision_bf16(self) -> None:
        """Parse MLP config with bf16 precision."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "bf16",
                "optimizer": "sgd",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["config"].get("precision") == "bf16"
        assert result["config"].get("optimizer") == "sgd"

    def test_precision_auto(self) -> None:
        """Parse MLP config with auto precision."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "auto",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["config"].get("precision") == "auto"

    def test_invalid_precision(self) -> None:
        """Invalid precision raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "invalid",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="precision must be"):
            _parse_external_train_config(config_json)

    def test_invalid_optimizer(self) -> None:
        """Invalid optimizer raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64],
                "precision": "fp32",
                "optimizer": "invalid",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="optimizer must be"):
            _parse_external_train_config(config_json)

    def test_invalid_hidden_sizes_not_list(self) -> None:
        """hidden_sizes not a list raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": "not_a_list",
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="hidden_sizes must be list"):
            _parse_external_train_config(config_json)

    def test_invalid_hidden_sizes_not_ints(self) -> None:
        """hidden_sizes with non-int elements raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "mlp",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_sizes": [64, "not_int"],
                "precision": "fp32",
                "optimizer": "adamw",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="hidden_sizes must be list"):
            _parse_external_train_config(config_json)


class TestLSTMConfig:
    """Tests for parsing LSTM configuration."""

    def test_valid(self) -> None:
        """Parse valid config for LSTM backend."""
        config_json = dump_json_str(
            {
                "backend": "lstm",
                "dataset": "taiwan",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_size": 64,
                "num_layers": 2,
                "bidirectional": True,
                "sequence_length": 5,
                "precision": "fp32",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["backend"] == "lstm"
        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.001
        assert result["config"].get("hidden_size") == 64
        assert result["config"].get("num_layers") == 2
        assert result["config"].get("bidirectional") is True
        assert result["config"].get("sequence_length") == 5

    def test_precision_fp16(self) -> None:
        """Parse LSTM config with fp16 precision."""
        config_json = dump_json_str(
            {
                "backend": "lstm",
                "dataset": "us",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_size": 64,
                "num_layers": 2,
                "bidirectional": False,
                "sequence_length": 5,
                "precision": "fp16",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["config"].get("precision") == "fp16"
        assert result["config"].get("bidirectional") is False

    def test_precision_bf16(self) -> None:
        """Parse LSTM config with bf16 precision."""
        config_json = dump_json_str(
            {
                "backend": "lstm",
                "dataset": "polish",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_size": 64,
                "num_layers": 2,
                "bidirectional": True,
                "sequence_length": 3,
                "precision": "bf16",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["config"].get("precision") == "bf16"

    def test_precision_auto(self) -> None:
        """Parse LSTM config with auto precision."""
        config_json = dump_json_str(
            {
                "backend": "lstm",
                "dataset": "taiwan",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_size": 64,
                "num_layers": 2,
                "bidirectional": True,
                "sequence_length": 5,
                "precision": "auto",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["config"].get("precision") == "auto"

    def test_invalid_precision(self) -> None:
        """Invalid LSTM precision raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "lstm",
                "dataset": "taiwan",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_size": 64,
                "num_layers": 2,
                "bidirectional": True,
                "sequence_length": 5,
                "precision": "invalid",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="precision must be"):
            _parse_external_train_config(config_json)

    def test_missing_bidirectional(self) -> None:
        """Missing bidirectional raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "lstm",
                "dataset": "taiwan",
                "learning_rate": 0.001,
                "batch_size": 32,
                "n_epochs": 10,
                "dropout": 0.2,
                "hidden_size": 64,
                "num_layers": 2,
                "sequence_length": 5,
                "precision": "fp32",
                "random_state": 42,
                "early_stopping_patience": 5,
            }
        )
        with pytest.raises(JSONTypeError, match="bidirectional must be a boolean"):
            _parse_external_train_config(config_json)


class TestLightGBMConfig:
    """Tests for parsing LightGBM configuration."""

    def test_valid(self) -> None:
        """Parse valid config for LightGBM backend."""
        config_json = dump_json_str(
            {
                "backend": "lightgbm",
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 100,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["backend"] == "lightgbm"
        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"].get("num_leaves") == 31
        assert result["config"].get("min_child_samples") == 20
        assert result["config"].get("early_stopping_rounds") == 10
        assert result["config"].get("reg_alpha") == 0.0
        assert result["config"].get("reg_lambda") == 1.0

    def test_with_regularization(self) -> None:
        """Parse LightGBM config with custom regularization."""
        config_json = dump_json_str(
            {
                "backend": "lightgbm",
                "dataset": "us",
                "learning_rate": 0.05,
                "max_depth": 8,
                "n_estimators": 200,
                "num_leaves": 63,
                "min_child_samples": 10,
                "subsample": 0.9,
                "colsample_bytree": 0.7,
                "random_state": 123,
                "device": "cuda",
                "early_stopping_rounds": 20,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            }
        )
        result = _parse_external_train_config(config_json)
        assert result["config"]["device"] == "cuda"
        assert result["config"].get("early_stopping_rounds") == 20
        assert result["config"].get("reg_alpha") == 1.0
        assert result["config"].get("reg_lambda") == 5.0


class TestConfigErrors:
    """Tests for configuration error handling."""

    def test_invalid_dataset(self) -> None:
        """Invalid dataset name raises ValueError."""
        config_json = dump_json_str(
            {
                "dataset": "invalid",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )
        with pytest.raises(ValueError, match="dataset must be one of"):
            _parse_external_train_config(config_json)

    def test_invalid_ratios(self) -> None:
        """Ratios that don't sum to 1.0 raise ValueError."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "train_ratio": 0.5,
                "val_ratio": 0.5,
                "test_ratio": 0.5,
            }
        )
        with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
            _parse_external_train_config(config_json)

    def test_not_object(self) -> None:
        """Non-object JSON raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_external_train_config("[1, 2, 3]")

    def test_invalid_json(self) -> None:
        """Invalid JSON raises InvalidJsonError."""
        with pytest.raises(InvalidJsonError):
            _parse_external_train_config("not json")

    def test_invalid_scale_pos_weight_type(self) -> None:
        """scale_pos_weight with non-number value raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "scale_pos_weight": "not_a_number",
            }
        )
        with pytest.raises(JSONTypeError, match="scale_pos_weight must be a number"):
            _parse_external_train_config(config_json)
