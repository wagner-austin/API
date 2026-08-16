"""Tests for ensemble weight optimization."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.ensemble import _hooks
from covenant_ml.ensemble.optimizer import (
    _compute_ensemble_score,
    _objective_function,
    optimize_ensemble_weights,
)
from covenant_ml.ensemble.testing import fake_minimize
from covenant_ml.ensemble.types import (
    EnsembleOOFData,
    ModelOOFPredictions,
    OptimizationConfig,
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


def _make_model_predictions(
    name: str,
    predictions: tuple[float, ...],
) -> ModelOOFPredictions:
    """Create ModelOOFPredictions for testing.

    Args:
        name: Model name.
        predictions: Prediction values.

    Returns:
        ModelOOFPredictions instance.
    """
    n_samples = len(predictions)
    return ModelOOFPredictions(
        model_name=name,
        predictions=np.array(predictions, dtype=np.float64),
        fold_indices=np.zeros(n_samples, dtype=np.int64),
    )


def _make_oof_data(
    model_preds: tuple[tuple[str, tuple[float, ...]], ...],
    labels: tuple[int, ...],
) -> EnsembleOOFData:
    """Create EnsembleOOFData for testing.

    Args:
        model_preds: Tuple of (model_name, predictions) tuples.
        labels: True labels.

    Returns:
        EnsembleOOFData instance.
    """
    preds = tuple(_make_model_predictions(name, vals) for name, vals in model_preds)
    return EnsembleOOFData(
        model_predictions=preds,
        labels=np.array(labels, dtype=np.int64),
        n_samples=len(labels),
        n_models=len(model_preds),
    )


def _make_test_config() -> OptimizationConfig:
    """Create test optimization config.

    Returns:
        OptimizationConfig for testing.
    """
    return OptimizationConfig(
        metric="amex",
        method="SLSQP",
        max_iterations=100,
        tolerance=1e-6,
        random_state=42,
    )


class TestMinimizeBinding:
    """Tests for the solver the module binds."""

    def test_the_seam_is_bound_to_scipy(self) -> None:
        """Callers reach scipy with nothing wired first."""
        assert _hooks.minimize is _hooks._real_minimize

    def test_the_bound_solver_delegates_to_scipy(self) -> None:
        """The binding imports scipy and returns what its solver returned."""

        def distance_from_quarter(weights: NDArray[np.float64]) -> float:
            diffs: NDArray[np.float64] = weights - 0.25
            return float(np.sum(diffs * diffs))

        result = _hooks.minimize(
            distance_from_quarter,
            np.full(4, 0.5, dtype=np.float64),
            "SLSQP",
            tuple((0.0, 1.0) for _ in range(4)),
            (),
            {"maxiter": 50, "ftol": 1e-8},
        )
        assert result.success
        assert result.x.shape == (4,)


class TestComputeEnsembleScore:
    """Tests for _compute_ensemble_score function."""

    def test_computes_amex_score(self) -> None:
        """_compute_ensemble_score returns AMEX metric score."""
        # Create simple test case
        pred_matrix = _float_matrix(((0.1, 0.9), (0.2, 0.8)))
        labels = _int_array(0, 1)
        weights = _float_array(0.5, 0.5)

        score = _compute_ensemble_score(weights, pred_matrix, labels)

        # Score should be reasonable for this case (perfect ranking)
        assert 0.0 <= score <= 1.0

    def test_weights_affect_score(self) -> None:
        """_compute_ensemble_score varies with different weights."""
        # Create predictions where models differ significantly
        # Row 0: Good model (low pred for class 0, high for class 1)
        # Row 1: Bad model (high pred for class 0, low for class 1)
        pred_matrix = _float_matrix(((0.1, 0.9), (0.9, 0.1)))
        labels = _int_array(0, 1)

        # Score with good model weighted higher
        weights_good = _float_array(0.9, 0.1)
        score_good = _compute_ensemble_score(weights_good, pred_matrix, labels)

        # Score with bad model weighted higher
        weights_bad = _float_array(0.1, 0.9)
        score_bad = _compute_ensemble_score(weights_bad, pred_matrix, labels)

        # Good weights should give better score
        assert score_good > score_bad


class TestObjectiveFunction:
    """Tests for _objective_function function."""

    def test_returns_negative_score(self) -> None:
        """_objective_function returns negative of AMEX score."""
        pred_matrix = _float_matrix(((0.1, 0.9), (0.2, 0.8)))
        labels = _int_array(0, 1)
        weights = _float_array(0.5, 0.5)

        obj_value = _objective_function(weights, pred_matrix, labels)
        score = _compute_ensemble_score(weights, pred_matrix, labels)

        assert obj_value == -score


class TestOptimizeEnsembleWeights:
    """Tests for optimize_ensemble_weights function."""

    def test_with_fake_scipy(self) -> None:
        """optimize_ensemble_weights works with fake scipy."""
        # Set up fake minimize
        _hooks.minimize = fake_minimize

        # Create test data
        oof_data = _make_oof_data(
            (("m1", (0.1, 0.9)), ("m2", (0.2, 0.8))),
            (0, 1),
        )
        config = _make_test_config()

        result = optimize_ensemble_weights(oof_data, config)

        # Access all fields directly to verify structure and values
        weights = result["weights"]
        best_score = result["best_score"]
        n_iterations = result["n_iterations"]
        initial_score = result["initial_score"]

        # Check weights structure
        assert len(weights["weights"]) == 2
        assert weights["model_names"] == ("m1", "m2")

        # Weights should sum to 1
        assert np.isclose(float(np.sum(weights["weights"])), 1.0)

        # Values should be sensible
        assert best_score >= 0.0
        assert n_iterations > 0
        assert initial_score >= 0.0
        assert result["converged"] in (True, False)

        # Weights should be non-negative
        weight_arr = result["weights"]["weights"]
        assert float(np.min(weight_arr)) >= 0.0

    def test_with_real_scipy(self) -> None:
        """optimize_ensemble_weights works with real scipy."""
        # Set up real scipy
        _hooks.minimize = _hooks._real_minimize

        # Create test data with more samples for meaningful optimization
        rng = np.random.default_rng(42)

        # Create labels: 70% negative, 30% positive (100 samples)
        labels_list: list[int] = [0] * 70 + [1] * 30
        rng.shuffle(labels_list)
        labels_tuple: tuple[int, ...] = tuple(labels_list)

        # Model 1: Good predictions (correlates with labels)
        m1_preds: tuple[float, ...] = tuple(
            float(rng.uniform(0.1, 0.3)) if label == 0 else float(rng.uniform(0.7, 0.9))
            for label in labels_list
        )

        # Model 2: Medium predictions (some correlation)
        m2_preds: tuple[float, ...] = tuple(
            float(rng.uniform(0.2, 0.5)) if label == 0 else float(rng.uniform(0.5, 0.8))
            for label in labels_list
        )

        oof_data = _make_oof_data(
            (("good_model", m1_preds), ("medium_model", m2_preds)),
            labels_tuple,
        )
        config = _make_test_config()

        result = optimize_ensemble_weights(oof_data, config)

        # Check convergence
        assert result["converged"] is True

        # Best score should be at least as good as initial
        assert result["best_score"] >= result["initial_score"] - 0.01  # Small tolerance

        # Good model should get more weight
        weight_arr = result["weights"]["weights"]
        good_model_weight = float(weight_arr.flat[0])
        assert good_model_weight > 0.3  # Good model should have significant weight

    def test_raises_on_invalid_oof_data(self) -> None:
        """optimize_ensemble_weights raises on invalid OOF data."""
        _hooks.minimize = fake_minimize

        # Create invalid data (single model)
        oof_data = EnsembleOOFData(
            model_predictions=(_make_model_predictions("m1", (0.1, 0.9)),),
            labels=_int_array(0, 1),
            n_samples=2,
            n_models=1,
        )
        config = _make_test_config()

        with pytest.raises(ValueError, match="at least 2 models"):
            optimize_ensemble_weights(oof_data, config)

    def test_three_models(self) -> None:
        """optimize_ensemble_weights works with 3 models."""
        _hooks.minimize = fake_minimize

        oof_data = _make_oof_data(
            (
                ("m1", (0.1, 0.9, 0.3)),
                ("m2", (0.2, 0.8, 0.4)),
                ("m3", (0.3, 0.7, 0.5)),
            ),
            (0, 1, 0),
        )
        config = _make_test_config()

        result = optimize_ensemble_weights(oof_data, config)

        assert len(result["weights"]["weights"]) == 3
        assert result["weights"]["model_names"] == ("m1", "m2", "m3")
        assert np.isclose(float(np.sum(result["weights"]["weights"])), 1.0)

    def test_result_values_valid(self) -> None:
        """optimize_ensemble_weights returns valid values with correct structure."""
        _hooks.minimize = fake_minimize

        oof_data = _make_oof_data(
            (("m1", (0.1, 0.9)), ("m2", (0.2, 0.8))),
            (0, 1),
        )
        config = _make_test_config()

        result = optimize_ensemble_weights(oof_data, config)

        # Check values are in valid ranges (implicitly checks types)
        assert 0.0 <= result["best_score"] <= 1.0
        assert 0.0 <= result["initial_score"] <= 1.0
        assert result["n_iterations"] >= 1
        assert result["converged"] in (True, False)

        # Check weights array structure
        assert result["weights"]["weights"].shape == (2,)
        assert result["weights"]["weights"].dtype == np.float64
        assert len(result["weights"]["model_names"]) == 2
