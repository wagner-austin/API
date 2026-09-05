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

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.ensemble import _hooks
from covenant_ml.ensemble.regression_optimizer import (
    optimize_regression_ensemble_weights,
    validate_regression_oof_data,
)
from covenant_ml.ensemble.regression_testing import make_regression_oof_data
from covenant_ml.ensemble.regression_types import (
    RegressionEnsembleOOFData,
)
from covenant_ml.ensemble.testing import fake_minimize
from tests.ensemble._regression_optimizer_fixtures import (
    _float_array,
    _make_model_predictions,
    _make_test_config,
)


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
            fun: Callable[[NDArray[np.float64]], float],
            x0: NDArray[np.float64],
            method: str,
            bounds: tuple[tuple[float, float], ...],
            constraints: tuple[dict[str, str | Callable[[NDArray[np.float64]], float]], ...],
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
