"""Tests for regression ensemble type definitions.

Tests cover:
- RegressionEnsembleOOFData TypedDict structure
- RegressionOptimizationConfig TypedDict structure
- RegressionOptimizationResult TypedDict structure
- make_default_regression_optimization_config factory
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.ensemble.regression_types import (
    RegressionEnsembleOOFData,
    RegressionOptimizationConfig,
    RegressionOptimizationResult,
    make_default_regression_optimization_config,
)
from covenant_ml.ensemble.types import EnsembleWeights, ModelOOFPredictions

# =============================================================================
# Helpers
# =============================================================================


def _float_array(values: tuple[float, ...]) -> NDArray[np.float64]:
    """Create float64 array from values."""
    result: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _int_array(values: tuple[int, ...]) -> NDArray[np.int64]:
    """Create int64 array from values."""
    result: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        result[i] = v
    return result


# =============================================================================
# Test: RegressionEnsembleOOFData
# =============================================================================


class TestRegressionEnsembleOOFData:
    """Tests for RegressionEnsembleOOFData TypedDict."""

    def test_can_create_with_float64_labels(self) -> None:
        """RegressionEnsembleOOFData accepts float64 continuous labels."""
        preds1 = _float_array((1.5, 2.3))
        preds2 = _float_array((1.8, 2.1))
        folds = _int_array((0, 1))
        labels = _float_array((1.6, 2.2))

        model1 = ModelOOFPredictions(
            model_name="model1",
            predictions=preds1,
            fold_indices=folds,
        )
        model2 = ModelOOFPredictions(
            model_name="model2",
            predictions=preds2,
            fold_indices=folds,
        )

        oof_data = RegressionEnsembleOOFData(
            model_predictions=(model1, model2),
            labels=labels,
            n_samples=2,
            n_models=2,
        )

        assert oof_data["n_samples"] == 2
        assert oof_data["n_models"] == 2
        assert len(oof_data["model_predictions"]) == 2
        assert oof_data["labels"].dtype == np.float64

    def test_supports_dict_access(self) -> None:
        """RegressionEnsembleOOFData supports dict-style key access."""
        labels = _float_array((1.0,))
        preds = _float_array((1.1,))
        folds = _int_array((0,))

        m1 = ModelOOFPredictions(model_name="m1", predictions=preds, fold_indices=folds)
        m2 = ModelOOFPredictions(model_name="m2", predictions=preds, fold_indices=folds)

        oof_data = RegressionEnsembleOOFData(
            model_predictions=(m1, m2),
            labels=labels,
            n_samples=1,
            n_models=2,
        )

        keys = list(oof_data.keys())
        assert "model_predictions" in keys
        assert "labels" in keys
        assert "n_samples" in keys
        assert "n_models" in keys

    def test_three_models(self) -> None:
        """RegressionEnsembleOOFData works with three models."""
        labels = _float_array((1.0, 2.0, 3.0))
        folds = _int_array((0, 1, 2))

        models = tuple(
            ModelOOFPredictions(
                model_name=f"model_{i}",
                predictions=_float_array((float(i), float(i + 1), float(i + 2))),
                fold_indices=folds,
            )
            for i in range(3)
        )

        oof_data = RegressionEnsembleOOFData(
            model_predictions=models,
            labels=labels,
            n_samples=3,
            n_models=3,
        )

        assert oof_data["n_models"] == 3
        assert len(oof_data["model_predictions"]) == 3


# =============================================================================
# Test: RegressionOptimizationConfig
# =============================================================================


class TestRegressionOptimizationConfig:
    """Tests for RegressionOptimizationConfig TypedDict."""

    def test_create_neg_rmse_config(self) -> None:
        """Can create config with neg_rmse metric."""
        config = RegressionOptimizationConfig(
            metric="neg_rmse",
            method="SLSQP",
            max_iterations=100,
            tolerance=1e-6,
            random_state=42,
        )

        assert config["metric"] == "neg_rmse"
        assert config["method"] == "SLSQP"
        assert config["max_iterations"] == 100
        assert config["tolerance"] == 1e-6
        assert config["random_state"] == 42

    def test_create_neg_mae_config(self) -> None:
        """Can create config with neg_mae metric."""
        config = RegressionOptimizationConfig(
            metric="neg_mae",
            method="SLSQP",
            max_iterations=500,
            tolerance=1e-8,
            random_state=0,
        )

        assert config["metric"] == "neg_mae"

    def test_create_r_squared_config(self) -> None:
        """Can create config with r_squared metric."""
        config = RegressionOptimizationConfig(
            metric="r_squared",
            method="trust-constr",
            max_iterations=200,
            tolerance=1e-7,
            random_state=99,
        )

        assert config["metric"] == "r_squared"
        assert config["method"] == "trust-constr"


# =============================================================================
# Test: RegressionOptimizationResult
# =============================================================================


class TestRegressionOptimizationResult:
    """Tests for RegressionOptimizationResult TypedDict."""

    def test_can_create_result(self) -> None:
        """Can create RegressionOptimizationResult with all fields."""
        weights = _float_array((0.6, 0.4))
        ew = EnsembleWeights(
            weights=weights,
            model_names=("model1", "model2"),
        )

        result = RegressionOptimizationResult(
            weights=ew,
            best_score=0.15,
            n_iterations=50,
            converged=True,
            initial_score=0.22,
        )

        assert result["best_score"] == 0.15
        assert result["n_iterations"] == 50
        assert result["converged"] is True
        assert result["initial_score"] == 0.22
        assert len(result["weights"]["weights"]) == 2

    def test_supports_dict_access(self) -> None:
        """RegressionOptimizationResult supports dict-style key access."""
        weights = _float_array((0.5, 0.5))
        ew = EnsembleWeights(weights=weights, model_names=("a", "b"))

        result = RegressionOptimizationResult(
            weights=ew,
            best_score=0.1,
            n_iterations=10,
            converged=False,
            initial_score=0.2,
        )

        keys = list(result.keys())
        assert "weights" in keys
        assert "best_score" in keys
        assert "n_iterations" in keys
        assert "converged" in keys
        assert "initial_score" in keys


# =============================================================================
# Test: make_default_regression_optimization_config
# =============================================================================


class TestMakeDefaultRegressionOptimizationConfig:
    """Tests for make_default_regression_optimization_config."""

    def test_default_config(self) -> None:
        """Default config uses neg_rmse metric and SLSQP."""
        config = make_default_regression_optimization_config()

        assert config["metric"] == "neg_rmse"
        assert config["method"] == "SLSQP"
        assert config["max_iterations"] == 1000
        assert config["tolerance"] == 1e-8
        assert config["random_state"] == 42

    def test_custom_random_state(self) -> None:
        """Custom random_state is accepted."""
        config = make_default_regression_optimization_config(random_state=123)

        assert config["random_state"] == 123
        # Other defaults unchanged
        assert config["metric"] == "neg_rmse"
        assert config["method"] == "SLSQP"
