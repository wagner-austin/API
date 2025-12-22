"""Tests for ensemble type definitions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.ensemble.types import (
    EnsembleOOFData,
    EnsemblePrediction,
    EnsembleWeights,
    ModelOOFPredictions,
    OptimizationConfig,
    OptimizationResult,
    make_default_optimization_config,
)


def _float_array(*values: float) -> NDArray[np.float64]:
    """Create typed float64 array from values.

    Args:
        *values: Float values for the array.

    Returns:
        NDArray of float64.
    """
    return np.array(values, dtype=np.float64)


def _float_matrix(rows: tuple[tuple[float, ...], ...]) -> NDArray[np.float64]:
    """Create typed float64 2D array from rows.

    Args:
        rows: Tuple of tuples representing matrix rows.

    Returns:
        NDArray of float64 with shape (n_rows, n_cols).
    """
    return np.array(rows, dtype=np.float64)


def _int_array(*values: int) -> NDArray[np.int64]:
    """Create typed int64 array from values.

    Args:
        *values: Int values for the array.

    Returns:
        NDArray of int64.
    """
    return np.array(values, dtype=np.int64)


class TestModelOOFPredictions:
    """Tests for ModelOOFPredictions TypedDict."""

    def test_create_model_oof_predictions(self) -> None:
        """ModelOOFPredictions can be created with required fields."""
        predictions: NDArray[np.float64] = _float_array(0.1, 0.9, 0.5)
        fold_indices: NDArray[np.int64] = _int_array(0, 0, 1)

        oof = ModelOOFPredictions(
            model_name="xgboost",
            predictions=predictions,
            fold_indices=fold_indices,
        )

        assert oof["model_name"] == "xgboost"
        assert len(oof["predictions"]) == 3
        assert len(oof["fold_indices"]) == 3


class TestEnsembleOOFData:
    """Tests for EnsembleOOFData TypedDict."""

    def test_create_ensemble_oof_data(self) -> None:
        """EnsembleOOFData can be created with required fields."""
        preds1: NDArray[np.float64] = _float_array(0.1, 0.9)
        preds2: NDArray[np.float64] = _float_array(0.2, 0.8)
        folds: NDArray[np.int64] = _int_array(0, 1)
        labels: NDArray[np.int64] = _int_array(0, 1)

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

        oof_data = EnsembleOOFData(
            model_predictions=(model1, model2),
            labels=labels,
            n_samples=2,
            n_models=2,
        )

        assert oof_data["n_samples"] == 2
        assert oof_data["n_models"] == 2
        assert len(oof_data["model_predictions"]) == 2


class TestEnsembleWeights:
    """Tests for EnsembleWeights TypedDict."""

    def test_create_ensemble_weights(self) -> None:
        """EnsembleWeights can be created with required fields."""
        weights: NDArray[np.float64] = _float_array(0.6, 0.4)

        ew = EnsembleWeights(
            weights=weights,
            model_names=("model1", "model2"),
        )

        assert len(ew["weights"]) == 2
        assert ew["model_names"] == ("model1", "model2")


class TestOptimizationConfig:
    """Tests for OptimizationConfig TypedDict."""

    def test_create_optimization_config(self) -> None:
        """OptimizationConfig can be created with required fields."""
        config = OptimizationConfig(
            metric="amex",
            method="SLSQP",
            max_iterations=100,
            tolerance=1e-6,
            random_state=42,
        )

        assert config["metric"] == "amex"
        assert config["method"] == "SLSQP"
        assert config["max_iterations"] == 100


class TestOptimizationResult:
    """Tests for OptimizationResult TypedDict."""

    def test_create_optimization_result(self) -> None:
        """OptimizationResult can be created with required fields."""
        weights: NDArray[np.float64] = _float_array(0.6, 0.4)
        ew = EnsembleWeights(
            weights=weights,
            model_names=("model1", "model2"),
        )

        result = OptimizationResult(
            weights=ew,
            best_score=0.81,
            n_iterations=50,
            converged=True,
            initial_score=0.79,
        )

        assert result["best_score"] == 0.81
        assert result["converged"] is True
        assert result["initial_score"] == 0.79


class TestEnsemblePrediction:
    """Tests for EnsemblePrediction TypedDict."""

    def test_create_ensemble_prediction(self) -> None:
        """EnsemblePrediction can be created with required fields."""
        predictions: NDArray[np.float64] = _float_array(0.15, 0.85)
        contributions: NDArray[np.float64] = _float_matrix(((0.06, 0.54), (0.09, 0.31)))
        weights: NDArray[np.float64] = _float_array(0.6, 0.4)
        ew = EnsembleWeights(
            weights=weights,
            model_names=("model1", "model2"),
        )

        pred = EnsemblePrediction(
            predictions=predictions,
            weights=ew,
            model_contributions=contributions,
        )

        assert len(pred["predictions"]) == 2
        assert pred["model_contributions"].shape == (2, 2)


class TestMakeDefaultOptimizationConfig:
    """Tests for make_default_optimization_config function."""

    def test_default_config(self) -> None:
        """make_default_optimization_config returns valid config."""
        config = make_default_optimization_config()

        assert config["metric"] == "amex"
        assert config["method"] == "SLSQP"
        assert config["max_iterations"] == 1000
        assert config["tolerance"] == 1e-8
        assert config["random_state"] == 42

    def test_custom_random_state(self) -> None:
        """make_default_optimization_config accepts custom random_state."""
        config = make_default_optimization_config(random_state=123)

        assert config["random_state"] == 123
