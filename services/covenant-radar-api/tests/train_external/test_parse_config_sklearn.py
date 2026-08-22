"""Tests for _parse_external_train_config function."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str

from covenant_radar_api.worker._train_external_parsers import (
    parse_external_train_config as _parse_external_train_config,
)


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
