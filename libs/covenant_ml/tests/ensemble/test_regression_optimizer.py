"""Tests for regression ensemble weight optimization.

Tests cover:
- validate_regression_oof_data: regression-specific validation
- extract_regression_prediction_matrix: float64 matrix extraction
- create_regression_equal_weights: equal weight creation
- Scoring functions: neg_rmse, neg_mae, r_squared
- optimize_regression_ensemble_weights: full optimization pipeline
- Objective function behavior
- Integration with fake and real scipy
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.ensemble import _hooks
from covenant_ml.ensemble.regression_optimizer import (
    _compute_neg_mae,
    _compute_neg_r_squared,
    _compute_neg_rmse,
    _compute_regression_ensemble_score,
    _compute_weighted_preds,
    _objective_function,
    create_regression_equal_weights,
    extract_regression_prediction_matrix,
    optimize_regression_ensemble_weights,
    validate_regression_oof_data,
)
from covenant_ml.ensemble.regression_testing import make_regression_oof_data
from covenant_ml.ensemble.regression_types import (
    RegressionEnsembleOOFData,
    RegressionOptimizationConfig,
)
from covenant_ml.ensemble.testing import fake_minimize
from covenant_ml.ensemble.types import ModelOOFPredictions

# Metric literal type matching RegressionOptimizationConfig
_RegressionMetric = Literal["neg_rmse", "neg_mae", "r_squared"]

# Type aliases for scipy minimize interface (used by zero_minimize test)
_ObjectiveFnType = Callable[[NDArray[np.float64]], float]
_ConstraintDict = dict[str, str | _ObjectiveFnType]


# =============================================================================
# Helpers
# =============================================================================


def _float_array(values: tuple[float, ...]) -> NDArray[np.float64]:
    """Create float64 array from tuple."""
    result: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _int_array(values: tuple[int, ...]) -> NDArray[np.int64]:
    """Create int64 array from tuple."""
    result: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _make_model_predictions(
    name: str,
    predictions: tuple[float, ...],
) -> ModelOOFPredictions:
    """Create ModelOOFPredictions for testing."""
    n_samples = len(predictions)
    preds: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for i, v in enumerate(predictions):
        preds[i] = v

    fold_indices: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    return ModelOOFPredictions(
        model_name=name,
        predictions=preds,
        fold_indices=fold_indices,
    )


def _make_test_config(
    metric: _RegressionMetric = "neg_rmse",
) -> RegressionOptimizationConfig:
    """Create test optimization config."""
    return RegressionOptimizationConfig(
        metric=metric,
        method="SLSQP",
        max_iterations=100,
        tolerance=1e-6,
        random_state=42,
    )


# =============================================================================
# Test: validate_regression_oof_data
# =============================================================================


class TestValidateRegressionOOFData:
    """Tests for validate_regression_oof_data."""

    def test_valid_data_passes(self) -> None:
        """Valid regression OOF data passes validation."""
        oof_data = make_regression_oof_data(
            (("m1", (1.5, 2.3)), ("m2", (1.8, 2.1))),
            (1.6, 2.2),
        )
        # Should not raise
        validate_regression_oof_data(oof_data)

    def test_raises_on_single_model(self) -> None:
        """Raises ValueError when fewer than 2 models."""
        oof_data = RegressionEnsembleOOFData(
            model_predictions=(_make_model_predictions("m1", (1.0, 2.0)),),
            labels=_float_array((1.0, 2.0)),
            n_samples=2,
            n_models=1,
        )

        with pytest.raises(ValueError, match="at least 2 models"):
            validate_regression_oof_data(oof_data)

    def test_raises_on_labels_length_mismatch(self) -> None:
        """Raises when labels length doesn't match n_samples."""
        oof_data = RegressionEnsembleOOFData(
            model_predictions=(
                _make_model_predictions("m1", (1.0, 2.0)),
                _make_model_predictions("m2", (1.1, 2.1)),
            ),
            labels=_float_array((1.0,)),  # Wrong length
            n_samples=2,
            n_models=2,
        )

        with pytest.raises(ValueError, match="Labels length"):
            validate_regression_oof_data(oof_data)

    def test_raises_on_model_preds_count_mismatch(self) -> None:
        """Raises when model_predictions count doesn't match n_models."""
        oof_data = RegressionEnsembleOOFData(
            model_predictions=(
                _make_model_predictions("m1", (1.0, 2.0)),
                _make_model_predictions("m2", (1.1, 2.1)),
            ),
            labels=_float_array((1.0, 2.0)),
            n_samples=2,
            n_models=3,  # Says 3, only 2 provided
        )

        with pytest.raises(ValueError, match="model_predictions length"):
            validate_regression_oof_data(oof_data)

    def test_raises_on_prediction_length_mismatch(self) -> None:
        """Raises when a model's predictions length doesn't match n_samples."""
        oof_data = RegressionEnsembleOOFData(
            model_predictions=(
                _make_model_predictions("m1", (1.0,)),  # 1 sample, expects 2
                _make_model_predictions("m2", (1.1, 2.1)),
            ),
            labels=_float_array((1.0, 2.0)),
            n_samples=2,
            n_models=2,
        )

        with pytest.raises(ValueError, match="predictions"):
            validate_regression_oof_data(oof_data)

    def test_raises_on_fold_indices_length_mismatch(self) -> None:
        """Raises when fold_indices length doesn't match n_samples."""
        # Build manually to get mismatched fold_indices
        preds_ok: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        folds_bad: NDArray[np.int64] = np.zeros(1, dtype=np.int64)  # Wrong length
        folds_ok: NDArray[np.int64] = np.zeros(2, dtype=np.int64)

        m1 = ModelOOFPredictions(model_name="m1", predictions=preds_ok, fold_indices=folds_bad)
        m2 = ModelOOFPredictions(model_name="m2", predictions=preds_ok, fold_indices=folds_ok)

        oof_data = RegressionEnsembleOOFData(
            model_predictions=(m1, m2),
            labels=_float_array((1.0, 2.0)),
            n_samples=2,
            n_models=2,
        )

        with pytest.raises(ValueError, match="fold_indices"):
            validate_regression_oof_data(oof_data)


# =============================================================================
# Test: extract_regression_prediction_matrix
# =============================================================================


class TestExtractRegressionPredictionMatrix:
    """Tests for extract_regression_prediction_matrix."""

    def test_extracts_correct_matrix(self) -> None:
        """Extracts (n_models, n_samples) matrix from OOF data."""
        oof_data = make_regression_oof_data(
            (("m1", (1.0, 2.0, 3.0)), ("m2", (4.0, 5.0, 6.0))),
            (1.5, 2.5, 3.5),
        )

        matrix = extract_regression_prediction_matrix(oof_data)

        assert matrix.shape == (2, 3)
        assert matrix.dtype == np.float64
        assert float(matrix.item((0, 0))) == 1.0
        assert float(matrix.item((0, 2))) == 3.0
        assert float(matrix.item((1, 0))) == 4.0
        assert float(matrix.item((1, 2))) == 6.0

    def test_three_models(self) -> None:
        """Works with three models."""
        oof_data = make_regression_oof_data(
            (("a", (1.0, 2.0)), ("b", (3.0, 4.0)), ("c", (5.0, 6.0))),
            (1.5, 3.5),
        )

        matrix = extract_regression_prediction_matrix(oof_data)
        assert matrix.shape == (3, 2)


# =============================================================================
# Test: create_regression_equal_weights
# =============================================================================


class TestCreateRegressionEqualWeights:
    """Tests for create_regression_equal_weights."""

    def test_two_models(self) -> None:
        """Creates equal weights for two models."""
        weights = create_regression_equal_weights(("m1", "m2"))

        assert len(weights["weights"]) == 2
        assert weights["model_names"] == ("m1", "m2")
        assert abs(float(weights["weights"].item(0)) - 0.5) < 1e-10
        assert abs(float(weights["weights"].item(1)) - 0.5) < 1e-10

    def test_three_models(self) -> None:
        """Creates equal weights for three models."""
        weights = create_regression_equal_weights(("a", "b", "c"))

        assert len(weights["weights"]) == 3
        expected = 1.0 / 3.0
        for i in range(3):
            assert abs(float(weights["weights"].item(i)) - expected) < 1e-10

    def test_raises_on_single_model(self) -> None:
        """Raises ValueError for fewer than 2 models."""
        with pytest.raises(ValueError, match="at least 2 models"):
            create_regression_equal_weights(("only_one",))

    def test_weights_sum_to_one(self) -> None:
        """Weights sum to exactly 1.0."""
        weights = create_regression_equal_weights(("a", "b", "c", "d"))
        total = 0.0
        for i in range(4):
            total += float(weights["weights"].item(i))
        assert abs(total - 1.0) < 1e-10


# =============================================================================
# Test: Scoring functions
# =============================================================================


class TestComputeWeightedPreds:
    """Tests for _compute_weighted_preds."""

    def test_equal_weights(self) -> None:
        """Equal weights average predictions."""
        weights = _float_array((0.5, 0.5))
        # model1: [1.0, 3.0], model2: [2.0, 4.0]
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[0, 1] = 3.0
        pred_matrix[1, 0] = 2.0
        pred_matrix[1, 1] = 4.0

        result = _compute_weighted_preds(weights, pred_matrix)

        assert result.shape == (2,)
        assert abs(float(result.item(0)) - 1.5) < 1e-10
        assert abs(float(result.item(1)) - 3.5) < 1e-10

    def test_unequal_weights(self) -> None:
        """Unequal weights produce weighted average."""
        weights = _float_array((0.8, 0.2))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[0, 1] = 3.0
        pred_matrix[1, 0] = 5.0
        pred_matrix[1, 1] = 7.0

        result = _compute_weighted_preds(weights, pred_matrix)

        # sample 0: 0.8*1.0 + 0.2*5.0 = 0.8 + 1.0 = 1.8
        assert abs(float(result.item(0)) - 1.8) < 1e-10
        # sample 1: 0.8*3.0 + 0.2*7.0 = 2.4 + 1.4 = 3.8
        assert abs(float(result.item(1)) - 3.8) < 1e-10


class TestComputeNegRmse:
    """Tests for _compute_neg_rmse."""

    def test_returns_negative_value(self) -> None:
        """neg_rmse returns a negative float."""
        weights = _float_array((0.5, 0.5))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[0, 1] = 3.0
        pred_matrix[1, 0] = 2.0
        pred_matrix[1, 1] = 4.0
        labels = _float_array((1.5, 3.0))

        result = _compute_neg_rmse(weights, pred_matrix, labels)
        assert result < 0.0

    def test_perfect_prediction_is_zero(self) -> None:
        """Perfect predictions give neg_rmse of 0.0."""
        weights = _float_array((1.0, 0.0))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        pred_matrix[0, 0] = 1.5
        pred_matrix[0, 1] = 3.0
        pred_matrix[1, 0] = 0.0
        pred_matrix[1, 1] = 0.0
        labels = _float_array((1.5, 3.0))

        result = _compute_neg_rmse(weights, pred_matrix, labels)
        assert abs(result) < 1e-10


class TestComputeNegMae:
    """Tests for _compute_neg_mae."""

    def test_returns_negative_value(self) -> None:
        """neg_mae returns a negative float."""
        weights = _float_array((0.5, 0.5))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[0, 1] = 3.0
        pred_matrix[1, 0] = 2.0
        pred_matrix[1, 1] = 4.0
        labels = _float_array((1.5, 3.0))

        result = _compute_neg_mae(weights, pred_matrix, labels)
        assert result < 0.0

    def test_perfect_prediction_is_zero(self) -> None:
        """Perfect predictions give neg_mae of 0.0."""
        weights = _float_array((1.0, 0.0))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        pred_matrix[0, 0] = 2.0
        pred_matrix[0, 1] = 4.0
        pred_matrix[1, 0] = 0.0
        pred_matrix[1, 1] = 0.0
        labels = _float_array((2.0, 4.0))

        result = _compute_neg_mae(weights, pred_matrix, labels)
        assert abs(result) < 1e-10


class TestComputeNegRSquared:
    """Tests for _compute_neg_r_squared."""

    def test_returns_negative_for_good_fit(self) -> None:
        """Good fit gives negative value (high R² negated)."""
        weights = _float_array((1.0, 0.0))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 3), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[0, 1] = 2.0
        pred_matrix[0, 2] = 3.0
        labels = _float_array((1.0, 2.0, 3.0))

        result = _compute_neg_r_squared(weights, pred_matrix, labels)
        # Perfect R² = 1.0, so neg = -1.0
        assert abs(result - (-1.0)) < 1e-10


class TestComputeRegressionEnsembleScore:
    """Tests for _compute_regression_ensemble_score."""

    def test_neg_rmse_metric(self) -> None:
        """Returns RMSE on natural scale for neg_rmse metric."""
        weights = _float_array((0.5, 0.5))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[1, 0] = 2.0
        pred_matrix[0, 1] = 3.0
        pred_matrix[1, 1] = 4.0
        labels = _float_array((1.5, 3.0))

        score = _compute_regression_ensemble_score(weights, pred_matrix, labels, "neg_rmse")
        assert score >= 0.0  # RMSE is always non-negative

    def test_neg_mae_metric(self) -> None:
        """Returns MAE on natural scale for neg_mae metric."""
        weights = _float_array((0.5, 0.5))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[1, 0] = 2.0
        pred_matrix[0, 1] = 3.0
        pred_matrix[1, 1] = 4.0
        labels = _float_array((1.5, 3.0))

        score = _compute_regression_ensemble_score(weights, pred_matrix, labels, "neg_mae")
        assert score >= 0.0  # MAE is always non-negative

    def test_r_squared_metric(self) -> None:
        """Returns R² on natural scale for r_squared metric."""
        weights = _float_array((1.0, 0.0))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 3), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[0, 1] = 2.0
        pred_matrix[0, 2] = 3.0
        labels = _float_array((1.0, 2.0, 3.0))

        score = _compute_regression_ensemble_score(weights, pred_matrix, labels, "r_squared")
        assert abs(score - 1.0) < 1e-10  # Perfect R²

    def test_unknown_metric_raises(self) -> None:
        """Unknown metric raises ValueError."""
        weights = _float_array((0.5, 0.5))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        labels = _float_array((1.0, 2.0))

        with pytest.raises(ValueError, match="Unknown metric"):
            _compute_regression_ensemble_score(weights, pred_matrix, labels, "unknown")


class TestObjectiveFunction:
    """Tests for _objective_function."""

    def test_neg_rmse_returns_positive_rmse(self) -> None:
        """For neg_rmse, objective returns positive RMSE (to minimize)."""
        weights = _float_array((0.5, 0.5))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[0, 1] = 3.0
        pred_matrix[1, 0] = 2.0
        pred_matrix[1, 1] = 4.0
        labels = _float_array((1.5, 3.0))

        result = _objective_function(weights, pred_matrix, labels, "neg_rmse")
        # RMSE is positive; minimize RMSE
        assert result >= 0.0

    def test_r_squared_returns_negative_r2(self) -> None:
        """For r_squared, objective returns -R² (to minimize = maximize R²)."""
        weights = _float_array((1.0, 0.0))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 3), dtype=np.float64)
        pred_matrix[0, 0] = 1.0
        pred_matrix[0, 1] = 2.0
        pred_matrix[0, 2] = 3.0
        labels = _float_array((1.0, 2.0, 3.0))

        result = _objective_function(weights, pred_matrix, labels, "r_squared")
        # Perfect R² = 1.0, so objective = -1.0
        assert abs(result - (-1.0)) < 1e-10

    def test_unknown_metric_raises(self) -> None:
        """Unknown metric raises ValueError."""
        weights = _float_array((0.5, 0.5))
        pred_matrix: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        labels = _float_array((1.0, 2.0))

        with pytest.raises(ValueError, match="Unknown metric"):
            _objective_function(weights, pred_matrix, labels, "bad_metric")


# =============================================================================
# Test: optimize_regression_ensemble_weights
# =============================================================================


class TestOptimizeRegressionEnsembleWeights:
    """Tests for optimize_regression_ensemble_weights."""

    def test_with_fake_scipy_neg_rmse(self) -> None:
        """Optimization works with fake scipy and neg_rmse metric."""
        _hooks.minimize = fake_minimize

        oof_data = make_regression_oof_data(
            (("m1", (1.0, 2.0, 3.0)), ("m2", (1.5, 2.5, 3.5))),
            (1.2, 2.3, 3.1),
        )
        config = _make_test_config("neg_rmse")

        result = optimize_regression_ensemble_weights(oof_data, config)

        # Check structure
        assert len(result["weights"]["weights"]) == 2
        assert result["weights"]["model_names"] == ("m1", "m2")
        assert result["n_iterations"] > 0
        assert result["converged"] in (True, False)

        # Weights sum to 1
        weight_sum = 0.0
        for i in range(2):
            weight_sum += float(result["weights"]["weights"].item(i))
        assert abs(weight_sum - 1.0) < 1e-6

        # Weights are non-negative
        for i in range(2):
            assert float(result["weights"]["weights"].item(i)) >= 0.0

    def test_with_fake_scipy_neg_mae(self) -> None:
        """Optimization works with neg_mae metric."""
        _hooks.minimize = fake_minimize

        oof_data = make_regression_oof_data(
            (("m1", (1.0, 2.0, 3.0)), ("m2", (1.5, 2.5, 3.5))),
            (1.2, 2.3, 3.1),
        )
        config = _make_test_config("neg_mae")

        result = optimize_regression_ensemble_weights(oof_data, config)

        assert len(result["weights"]["weights"]) == 2
        assert result["best_score"] >= 0.0  # MAE is non-negative

    def test_with_fake_scipy_r_squared(self) -> None:
        """Optimization works with r_squared metric."""
        _hooks.minimize = fake_minimize

        oof_data = make_regression_oof_data(
            (("m1", (1.0, 2.0, 3.0)), ("m2", (1.5, 2.5, 3.5))),
            (1.2, 2.3, 3.1),
        )
        config = _make_test_config("r_squared")

        result = optimize_regression_ensemble_weights(oof_data, config)

        assert len(result["weights"]["weights"]) == 2

    def test_with_real_scipy(self) -> None:
        """Optimization works with real scipy.optimize.minimize."""
        _hooks.minimize = _hooks._real_minimize

        # Create test data with enough samples for meaningful optimization
        n_samples = 50
        labels_list: list[float] = []
        for i in range(n_samples):
            labels_list.append(float(i) * 0.1 + 1.0)
        labels_tuple = tuple(labels_list)

        # Model 1: Good predictions (close to labels)
        m1_preds = tuple(v + 0.05 for v in labels_list)

        # Model 2: Worse predictions (further from labels)
        m2_preds = tuple(v + 0.5 for v in labels_list)

        oof_data = make_regression_oof_data(
            (("good_model", m1_preds), ("worse_model", m2_preds)),
            labels_tuple,
        )
        config = _make_test_config("neg_rmse")

        result = optimize_regression_ensemble_weights(oof_data, config)

        # Should converge
        assert result["converged"] is True

        # Best score should be better (lower RMSE) than or equal to initial
        assert result["best_score"] <= result["initial_score"] + 0.01

        # Good model should get more weight
        good_weight = float(result["weights"]["weights"].item(0))
        assert good_weight > 0.5

    def test_three_models(self) -> None:
        """Optimization works with three models."""
        _hooks.minimize = fake_minimize

        oof_data = make_regression_oof_data(
            (
                ("m1", (1.0, 2.0, 3.0)),
                ("m2", (1.5, 2.5, 3.5)),
                ("m3", (0.8, 1.8, 2.8)),
            ),
            (1.2, 2.3, 3.1),
        )
        config = _make_test_config("neg_rmse")

        result = optimize_regression_ensemble_weights(oof_data, config)

        assert len(result["weights"]["weights"]) == 3
        assert result["weights"]["model_names"] == ("m1", "m2", "m3")

        # Weights sum to 1
        weight_sum = 0.0
        for i in range(3):
            weight_sum += float(result["weights"]["weights"].item(i))
        assert abs(weight_sum - 1.0) < 1e-6

    def test_raises_on_invalid_oof_data(self) -> None:
        """Raises on invalid OOF data (single model)."""
        _hooks.minimize = fake_minimize

        oof_data = RegressionEnsembleOOFData(
            model_predictions=(_make_model_predictions("m1", (1.0, 2.0)),),
            labels=_float_array((1.0, 2.0)),
            n_samples=2,
            n_models=1,
        )
        config = _make_test_config("neg_rmse")

        with pytest.raises(ValueError, match="at least 2 models"):
            optimize_regression_ensemble_weights(oof_data, config)

    def test_initial_score_is_computed(self) -> None:
        """Initial score is computed from equal weights."""
        _hooks.minimize = fake_minimize

        oof_data = make_regression_oof_data(
            (("m1", (1.0, 2.0, 3.0)), ("m2", (1.5, 2.5, 3.5))),
            (1.2, 2.3, 3.1),
        )
        config = _make_test_config("neg_rmse")

        result = optimize_regression_ensemble_weights(oof_data, config)

        # Initial score should be a valid RMSE (non-negative)
        assert result["initial_score"] >= 0.0

    def test_zero_weight_sum_skips_normalization(self) -> None:
        """When optimizer returns all-zero weights, normalization is skipped."""
        from covenant_ml.ensemble.testing import FakeOptimizeResult

        def zero_minimize(
            fun: _ObjectiveFnType,
            x0: NDArray[np.float64],
            method: str,
            bounds: tuple[tuple[float, float], ...],
            constraints: tuple[_ConstraintDict, ...],
            options: dict[str, int | float],
        ) -> FakeOptimizeResult:
            """Return all-zero weights to exercise normalization skip."""
            zero_weights: NDArray[np.float64] = np.zeros(len(x0), dtype=np.float64)
            return FakeOptimizeResult(
                x=zero_weights,
                fun=0.0,
                nit=1,
                success=True,
            )

        _hooks.minimize = zero_minimize

        oof_data = make_regression_oof_data(
            (("m1", (1.0, 2.0)), ("m2", (1.5, 2.5))),
            (1.2, 2.3),
        )
        config = _make_test_config("neg_rmse")

        result = optimize_regression_ensemble_weights(oof_data, config)

        # Weights should remain all-zero (no division by zero)
        for i in range(2):
            assert float(result["weights"]["weights"].item(i)) == 0.0

    def test_result_values_are_finite(self) -> None:
        """All result values are finite."""
        _hooks.minimize = fake_minimize

        oof_data = make_regression_oof_data(
            (("m1", (1.0, 2.0)), ("m2", (1.5, 2.5))),
            (1.2, 2.3),
        )
        config = _make_test_config("neg_rmse")

        result = optimize_regression_ensemble_weights(oof_data, config)

        assert math.isfinite(result["best_score"])
        assert math.isfinite(result["initial_score"])
        for i in range(2):
            assert math.isfinite(float(result["weights"]["weights"].item(i)))


# =============================================================================
# Test: regression testing utilities
# =============================================================================


class TestRegressionTestingUtilities:
    """Tests for regression ensemble testing utilities."""

    def test_make_regression_oof_data_structure(self) -> None:
        """make_regression_oof_data creates valid OOF data."""
        oof_data = make_regression_oof_data(
            (("m1", (1.0, 2.0, 3.0)), ("m2", (4.0, 5.0, 6.0))),
            (1.5, 2.5, 3.5),
        )

        assert oof_data["n_samples"] == 3
        assert oof_data["n_models"] == 2
        assert oof_data["labels"].dtype == np.float64
        assert len(oof_data["model_predictions"]) == 2

    def test_make_regression_oof_data_passes_validation(self) -> None:
        """make_regression_oof_data output passes validation."""
        oof_data = make_regression_oof_data(
            (("m1", (1.0, 2.0)), ("m2", (3.0, 4.0))),
            (1.5, 3.5),
        )

        # Should not raise
        validate_regression_oof_data(oof_data)

    def test_make_regression_oof_data_model_names(self) -> None:
        """make_regression_oof_data preserves model names."""
        oof_data = make_regression_oof_data(
            (("xgboost", (1.0,)), ("lightgbm", (2.0,))),
            (1.5,),
        )

        assert oof_data["model_predictions"][0]["model_name"] == "xgboost"
        assert oof_data["model_predictions"][1]["model_name"] == "lightgbm"
