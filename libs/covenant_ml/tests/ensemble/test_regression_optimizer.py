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

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.ensemble.regression_optimizer import (
    _compute_neg_mae,
    _compute_neg_r_squared,
    _compute_neg_rmse,
    _compute_regression_ensemble_score,
    _compute_weighted_preds,
    _objective_function,
    create_regression_equal_weights,
    extract_regression_prediction_matrix,
    validate_regression_oof_data,
)
from covenant_ml.ensemble.regression_testing import make_regression_oof_data
from covenant_ml.ensemble.regression_types import (
    RegressionEnsembleOOFData,
)
from covenant_ml.ensemble.types import ModelOOFPredictions
from tests.ensemble._regression_optimizer_fixtures import (
    _float_array,
    _make_model_predictions,
)


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
