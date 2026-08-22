"""Tests for HTTP request body parsing."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.api.decode import (
    parse_external_train_request,
)


class TestParseExternalTrainRequestTreeBackends:
    """parse_external_train_request: lightgbm / cleargbm / logreg / random forest."""

    def test_valid_lightgbm_request(self) -> None:
        """Test parsing valid LightGBM request."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lightgbm",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "lightgbm"
        assert result["dataset"] == "taiwan"
        if result["backend"] != "lightgbm":
            raise AssertionError("Expected lightgbm backend")
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["max_depth"] == 6
        assert result["config"]["n_estimators"] == 100
        assert result["config"]["num_leaves"] == 31
        assert result["config"]["min_child_samples"] == 20
        assert result["config"]["subsample"] == 0.8
        assert result["config"]["colsample_bytree"] == 0.8
        assert result["config"]["random_state"] == 42
        assert result["config"]["device"] == "auto"
        assert result["config"]["early_stopping_rounds"] == 10
        assert result["config"]["reg_alpha"] == 0.0
        assert result["config"]["reg_lambda"] == 1.0

    def test_lightgbm_request_with_custom_regularization(self) -> None:
        """Test parsing LightGBM request with custom regularization."""
        body = b"""{
            "dataset": "us",
            "backend": "lightgbm",
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
            "reg_lambda": 5.0
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "lightgbm":
            raise AssertionError("Expected lightgbm backend")
        assert result["config"]["device"] == "cuda"
        assert result["config"]["early_stopping_rounds"] == 20
        assert result["config"]["reg_alpha"] == 1.0
        assert result["config"]["reg_lambda"] == 5.0

    def test_valid_cleargbm_request(self) -> None:
        """Test parsing valid ClearGBM request."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "cleargbm",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": 0.8,
            "subsample": 0.8,
            "random_state": 42,
            "track_contributions": true
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "cleargbm"
        assert result["dataset"] == "taiwan"
        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
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
        assert result["config"]["train_ratio"] == 0.7

    def test_cleargbm_with_monotonic_constraints(self) -> None:
        """Test ClearGBM with monotonic constraints dict."""
        body = b"""{
            "dataset": "us",
            "backend": "cleargbm",
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "max_features": null,
            "subsample": 0.9,
            "random_state": 7,
            "track_contributions": false,
            "monotonic_constraints": {"feature_a": 1, "feature_b": -1}
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "cleargbm":
            raise AssertionError("Expected cleargbm backend")
        assert result["config"]["max_features"] is None
        assert result["config"]["monotonic_constraints"] == {
            "feature_a": 1,
            "feature_b": -1,
        }
        assert result["config"]["track_contributions"] is False

    def test_valid_logreg_request(self) -> None:
        """Test parsing valid LogReg request."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "logreg",
            "solver": "saga",
            "penalty": "elasticnet",
            "C": 1.0,
            "max_iter": 1000,
            "tol": 0.0001,
            "class_weight_balanced": true,
            "random_state": 42,
            "l1_ratio": 0.5
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "logreg"
        assert result["dataset"] == "taiwan"
        if result["backend"] != "logreg":
            raise AssertionError("Expected logreg backend")
        assert result["config"]["solver"] == "saga"
        assert result["config"]["penalty"] == "elasticnet"
        assert result["config"]["C"] == 1.0
        assert result["config"]["max_iter"] == 1000
        assert result["config"]["tol"] == 0.0001
        assert result["config"]["class_weight_balanced"] is True
        assert result["config"]["l1_ratio"] == 0.5
        assert result["config"]["train_ratio"] == 0.7

    def test_logreg_invalid_solver_raises(self) -> None:
        """Test that invalid LogReg solver raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "logreg",
            "solver": "invalid_solver",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 100,
            "tol": 0.001,
            "class_weight_balanced": false,
            "random_state": 42
        }"""
        with pytest.raises(JSONTypeError, match="solver must be one of"):
            parse_external_train_request(body)

    def test_logreg_invalid_penalty_raises(self) -> None:
        """Test that invalid LogReg penalty raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "logreg",
            "solver": "saga",
            "penalty": "invalid_penalty",
            "C": 1.0,
            "max_iter": 100,
            "tol": 0.001,
            "class_weight_balanced": false,
            "random_state": 42
        }"""
        with pytest.raises(JSONTypeError, match="penalty must be one of"):
            parse_external_train_request(body)

    def test_valid_random_forest_request(self) -> None:
        """Test parsing valid RandomForest request."""
        body = b"""{
            "dataset": "polish",
            "backend": "random_forest",
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": true,
            "class_weight_balanced": true,
            "random_state": 42
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "random_forest"
        assert result["dataset"] == "polish"
        if result["backend"] != "random_forest":
            raise AssertionError("Expected random_forest backend")
        assert result["config"]["n_estimators"] == 100
        assert result["config"]["max_depth"] == 10
        assert result["config"]["max_features"] == "sqrt"
        assert result["config"]["bootstrap"] is True
        assert result["config"]["class_weight_balanced"] is True
        assert result["config"]["n_jobs"] == -1
        assert result["config"]["oob_score"] is False
        assert result["config"]["train_ratio"] == 0.7

    def test_random_forest_with_null_max_depth(self) -> None:
        """Test RF with null max_depth (unlimited)."""
        body = b"""{
            "dataset": "us",
            "backend": "random_forest",
            "n_estimators": 200,
            "max_depth": null,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "log2",
            "bootstrap": true,
            "class_weight_balanced": false,
            "random_state": 7
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "random_forest":
            raise AssertionError("Expected random_forest backend")
        assert result["config"]["max_depth"] is None
        assert result["config"]["max_features"] == "log2"

    def test_random_forest_with_float_max_features(self) -> None:
        """Test RF with float max_features."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "random_forest",
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": 0.7,
            "bootstrap": false,
            "class_weight_balanced": false,
            "random_state": 99
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "random_forest":
            raise AssertionError("Expected random_forest backend")
        assert result["config"]["max_features"] == 0.7

    def test_random_forest_invalid_max_features_raises(self) -> None:
        """Test that invalid RF max_features raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "random_forest",
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "invalid",
            "bootstrap": true,
            "class_weight_balanced": true,
            "random_state": 0
        }"""
        with pytest.raises(JSONTypeError, match="max_features must be"):
            parse_external_train_request(body)
