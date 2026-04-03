"""Tests for _parse_external_train_config function."""

from __future__ import annotations

import pytest
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str

from covenant_radar_api.worker._train_external_parsers import (
    parse_external_train_config as _parse_external_train_config,
)


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
        if result["backend"] != "xgboost":
            raise AssertionError("Expected xgboost backend")

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
        if result["backend"] != "xgboost":
            raise AssertionError("Expected xgboost backend")
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
        if result["backend"] != "mlp":
            raise AssertionError("Expected mlp backend")
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
        if result["backend"] != "mlp":
            raise AssertionError("Expected mlp backend")
        assert result["config"]["precision"] == "fp16"
        assert result["config"]["optimizer"] == "adam"

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
        if result["backend"] != "mlp":
            raise AssertionError("Expected mlp backend")
        assert result["config"]["precision"] == "bf16"
        assert result["config"]["optimizer"] == "sgd"

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
        if result["backend"] != "mlp":
            raise AssertionError("Expected mlp backend")
        assert result["config"]["precision"] == "auto"

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
        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.001
        assert result["config"]["hidden_size"] == 64
        assert result["config"]["num_layers"] == 2
        assert result["config"]["bidirectional"] is True
        assert result["config"]["sequence_length"] == 5

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
        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["precision"] == "fp16"
        assert result["config"]["bidirectional"] is False

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
        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["precision"] == "bf16"

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
        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["precision"] == "auto"

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
        if result["backend"] != "lightgbm":
            raise AssertionError("Expected lightgbm backend")
        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["num_leaves"] == 31
        assert result["config"]["min_child_samples"] == 20
        assert result["config"]["early_stopping_rounds"] == 10
        assert result["config"]["reg_alpha"] == 0.0
        assert result["config"]["reg_lambda"] == 1.0

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
        if result["backend"] != "lightgbm":
            raise AssertionError("Expected lightgbm backend")
        assert result["config"]["device"] == "cuda"
        assert result["config"]["early_stopping_rounds"] == 20
        assert result["config"]["reg_alpha"] == 1.0
        assert result["config"]["reg_lambda"] == 5.0


class TestClearGBMConfig:
    """Tests for parsing ClearGBM configuration."""

    def test_valid(self) -> None:
        """Parse valid config for ClearGBM backend."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": 0.8,
                "subsample": 0.8,
                "random_state": 42,
                "track_contributions": True,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["dataset"] == "taiwan"
        assert result["config"]["n_estimators"] == 100
        assert result["config"]["max_depth"] == 6
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["min_samples_split"] == 10
        assert result["config"]["min_samples_leaf"] == 5
        assert result["config"]["max_features"] == 0.8
        assert result["config"]["subsample"] == 0.8
        assert result["config"]["track_contributions"] is True
        assert result["config"]["max_bins"] == 64
        assert result["config"]["reg_alpha"] == 0.0
        assert result["config"]["reg_lambda"] == 1.0
        assert result["config"]["n_jobs"] == -1
        assert result["config"]["early_stopping_rounds"] == 10

    def test_with_monotonic_constraints(self) -> None:
        """Parse ClearGBM config with monotonic constraints."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "us",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 10,
                "subsample": 0.9,
                "random_state": 7,
                "track_contributions": False,
                "monotonic_constraints": {"feature_a": 1, "feature_b": -1},
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["monotonic_constraints"] == {
            "feature_a": 1,
            "feature_b": -1,
        }
        assert result["config"]["max_features"] == 10

    def test_with_null_max_features(self) -> None:
        """Parse ClearGBM config with null max_features."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "polish",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": None,
                "subsample": 0.9,
                "random_state": 7,
                "track_contributions": False,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["max_features"] is None

    def test_with_null_monotonic_constraints(self) -> None:
        """Parse ClearGBM config with null monotonic_constraints."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 0.5,
                "subsample": 0.9,
                "random_state": 7,
                "track_contributions": True,
                "monotonic_constraints": None,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["monotonic_constraints"] is None

    def test_with_custom_optional_fields(self) -> None:
        """Parse ClearGBM config with custom optional values."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.01,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": 0.5,
                "max_bins": 128,
                "subsample": 1.0,
                "random_state": 99,
                "track_contributions": False,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "n_jobs": 4,
                "early_stopping_rounds": 20,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["max_bins"] == 128
        assert result["config"]["reg_alpha"] == 0.5
        assert result["config"]["reg_lambda"] == 2.0
        assert result["config"]["n_jobs"] == 4
        assert result["config"]["early_stopping_rounds"] == 20

    def test_missing_track_contributions_raises(self) -> None:
        """Missing track_contributions raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 0.5,
                "subsample": 0.9,
                "random_state": 7,
            }
        )
        with pytest.raises(JSONTypeError, match="track_contributions must be a boolean"):
            _parse_external_train_config(config_json)

    def test_invalid_max_features_type_raises(self) -> None:
        """Invalid max_features type raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": "invalid",
                "subsample": 0.9,
                "random_state": 7,
                "track_contributions": True,
            }
        )
        with pytest.raises(JSONTypeError, match="max_features must be"):
            _parse_external_train_config(config_json)

    def test_invalid_monotonic_constraints_type_raises(self) -> None:
        """Invalid monotonic_constraints type raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 0.5,
                "subsample": 0.9,
                "random_state": 7,
                "track_contributions": True,
                "monotonic_constraints": "invalid",
            }
        )
        with pytest.raises(JSONTypeError, match="monotonic_constraints must be"):
            _parse_external_train_config(config_json)

    def test_monotonic_constraints_non_int_values_raises(self) -> None:
        """Monotonic constraints with non-int values raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "cleargbm",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": 0.5,
                "subsample": 0.9,
                "random_state": 7,
                "track_contributions": True,
                "monotonic_constraints": {"feature_a": "not_int"},
            }
        )
        with pytest.raises(JSONTypeError, match="monotonic_constraints values must be ints"):
            _parse_external_train_config(config_json)


class TestLogRegConfig:
    """Tests for parsing LogReg configuration."""

    def test_valid(self) -> None:
        """Parse valid config for LogReg backend."""
        config_json = dump_json_str(
            {
                "backend": "logreg",
                "dataset": "taiwan",
                "solver": "saga",
                "penalty": "elasticnet",
                "C": 1.0,
                "max_iter": 1000,
                "tol": 0.0001,
                "class_weight_balanced": True,
                "random_state": 42,
                "l1_ratio": 0.5,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "logreg":
            raise AssertionError("Expected logreg backend")
        assert result["dataset"] == "taiwan"
        assert result["config"]["solver"] == "saga"
        assert result["config"]["penalty"] == "elasticnet"
        assert result["config"]["C"] == 1.0
        assert result["config"]["max_iter"] == 1000
        assert result["config"]["tol"] == 0.0001
        assert result["config"]["class_weight_balanced"] is True
        assert result["config"]["random_state"] == 42
        assert result["config"]["l1_ratio"] == 0.5

    def test_all_solvers(self) -> None:
        """All 6 LogReg solvers are accepted."""
        solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
        for solver in solvers:
            config_json = dump_json_str(
                {
                    "backend": "logreg",
                    "dataset": "taiwan",
                    "solver": solver,
                    "penalty": "l2",
                    "C": 1.0,
                    "max_iter": 100,
                    "tol": 0.001,
                    "class_weight_balanced": False,
                    "random_state": 42,
                }
            )
            result = _parse_external_train_config(config_json)
            if result["backend"] != "logreg":
                raise AssertionError("Expected logreg backend")
            assert result["config"]["solver"] == solver

    def test_all_penalties(self) -> None:
        """All 4 LogReg penalties are accepted."""
        penalties = ["l1", "l2", "elasticnet", "none"]
        for penalty in penalties:
            config_json = dump_json_str(
                {
                    "backend": "logreg",
                    "dataset": "us",
                    "solver": "saga",
                    "penalty": penalty,
                    "C": 1.0,
                    "max_iter": 100,
                    "tol": 0.001,
                    "class_weight_balanced": False,
                    "random_state": 42,
                }
            )
            result = _parse_external_train_config(config_json)
            if result["backend"] != "logreg":
                raise AssertionError("Expected logreg backend")
            assert result["config"]["penalty"] == penalty

    def test_default_l1_ratio(self) -> None:
        """Default l1_ratio is 0.0 when not provided."""
        config_json = dump_json_str(
            {
                "backend": "logreg",
                "dataset": "polish",
                "solver": "lbfgs",
                "penalty": "l2",
                "C": 0.5,
                "max_iter": 500,
                "tol": 0.01,
                "class_weight_balanced": False,
                "random_state": 0,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "logreg":
            raise AssertionError("Expected logreg backend")
        assert result["config"]["l1_ratio"] == 0.0

    def test_invalid_solver_raises(self) -> None:
        """Invalid solver raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "logreg",
                "dataset": "taiwan",
                "solver": "invalid_solver",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 100,
                "tol": 0.001,
                "class_weight_balanced": False,
                "random_state": 42,
            }
        )
        with pytest.raises(JSONTypeError, match="solver must be one of"):
            _parse_external_train_config(config_json)

    def test_invalid_penalty_raises(self) -> None:
        """Invalid penalty raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "logreg",
                "dataset": "taiwan",
                "solver": "saga",
                "penalty": "invalid_penalty",
                "C": 1.0,
                "max_iter": 100,
                "tol": 0.001,
                "class_weight_balanced": False,
                "random_state": 42,
            }
        )
        with pytest.raises(JSONTypeError, match="penalty must be one of"):
            _parse_external_train_config(config_json)

    def test_missing_class_weight_balanced_raises(self) -> None:
        """Missing class_weight_balanced raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "logreg",
                "dataset": "taiwan",
                "solver": "saga",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 100,
                "tol": 0.001,
                "random_state": 42,
            }
        )
        with pytest.raises(JSONTypeError, match="class_weight_balanced must be a boolean"):
            _parse_external_train_config(config_json)


class TestRandomForestConfig:
    """Tests for parsing RandomForest configuration."""

    def test_valid(self) -> None:
        """Parse valid config for RandomForest backend."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight_balanced": True,
                "random_state": 42,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "random_forest":
            raise AssertionError("Expected random_forest backend")
        assert result["dataset"] == "taiwan"
        assert result["config"]["n_estimators"] == 100
        assert result["config"]["max_depth"] == 10
        assert result["config"]["min_samples_split"] == 5
        assert result["config"]["min_samples_leaf"] == 2
        assert result["config"]["max_features"] == "sqrt"
        assert result["config"]["bootstrap"] is True
        assert result["config"]["class_weight_balanced"] is True
        assert result["config"]["random_state"] == 42
        assert result["config"]["n_jobs"] == -1
        assert result["config"]["oob_score"] is False

    def test_with_null_max_depth(self) -> None:
        """Parse RF config with null max_depth (unlimited depth)."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "us",
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "log2",
                "bootstrap": True,
                "class_weight_balanced": False,
                "random_state": 7,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "random_forest":
            raise AssertionError("Expected random_forest backend")
        assert result["config"]["max_depth"] is None
        assert result["config"]["max_features"] == "log2"

    def test_with_float_max_features(self) -> None:
        """Parse RF config with float max_features."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "polish",
                "n_estimators": 50,
                "max_depth": 5,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "max_features": 0.7,
                "bootstrap": False,
                "class_weight_balanced": False,
                "random_state": 99,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "random_forest":
            raise AssertionError("Expected random_forest backend")
        assert result["config"]["max_features"] == 0.7
        assert result["config"]["bootstrap"] is False

    def test_with_int_max_features(self) -> None:
        """Parse RF config with int max_features."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 5,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "max_features": 10,
                "bootstrap": True,
                "class_weight_balanced": True,
                "random_state": 0,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "random_forest":
            raise AssertionError("Expected random_forest backend")
        assert result["config"]["max_features"] == 10

    def test_with_null_max_features(self) -> None:
        """Parse RF config with null max_features (use all features)."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 5,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "max_features": None,
                "bootstrap": True,
                "class_weight_balanced": True,
                "random_state": 0,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "random_forest":
            raise AssertionError("Expected random_forest backend")
        assert result["config"]["max_features"] is None

    def test_with_oob_score(self) -> None:
        """Parse RF config with oob_score enabled."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "taiwan",
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight_balanced": True,
                "random_state": 42,
                "oob_score": True,
            }
        )
        result = _parse_external_train_config(config_json)
        if result["backend"] != "random_forest":
            raise AssertionError("Expected random_forest backend")
        assert result["config"]["oob_score"] is True

    def test_invalid_max_features_string_raises(self) -> None:
        """Invalid max_features string raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 5,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "max_features": "invalid",
                "bootstrap": True,
                "class_weight_balanced": True,
                "random_state": 0,
            }
        )
        with pytest.raises(JSONTypeError, match="max_features must be"):
            _parse_external_train_config(config_json)

    def test_invalid_max_depth_type_raises(self) -> None:
        """Non-int, non-null max_depth raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": "deep",
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight_balanced": True,
                "random_state": 0,
            }
        )
        with pytest.raises(JSONTypeError, match="max_depth must be an int or null"):
            _parse_external_train_config(config_json)

    def test_missing_bootstrap_raises(self) -> None:
        """Missing bootstrap raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 5,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight_balanced": True,
                "random_state": 0,
            }
        )
        with pytest.raises(JSONTypeError, match="bootstrap must be a boolean"):
            _parse_external_train_config(config_json)

    def test_missing_class_weight_balanced_raises(self) -> None:
        """Missing class_weight_balanced raises JSONTypeError."""
        config_json = dump_json_str(
            {
                "backend": "random_forest",
                "dataset": "taiwan",
                "n_estimators": 50,
                "max_depth": 5,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "random_state": 0,
            }
        )
        with pytest.raises(JSONTypeError, match="class_weight_balanced must be a boolean"):
            _parse_external_train_config(config_json)


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
