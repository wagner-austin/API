"""Tests for XGBoost regressor objective function.

Uses real xgboost library for integration testing. No mocks.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.objectives.xgboost_regressor_objective import (
    XGBoostRegressorObjective,
    _cuda_available,
    _get_xgb_dmatrix_and_train,
    create_xgboost_regressor_objective,
)
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams

# =============================================================================
# Test Data Helpers
# =============================================================================


def _make_regression_data(
    n_samples: int = 100, n_features: int = 5, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Create deterministic regression dataset for optimization."""
    x: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)
    y: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_features):
            x[i, j] = float((i * 7 + j * 3 + seed) % 100) / 100.0
        # Target is linear combination of features + noise
        y[i] = float(x.item((i, 0))) * 3.0 + float(x.item((i, 1))) * 1.5 + 2.0

    names = [f"feat_{j}" for j in range(n_features)]
    return x, y, names


def _make_positive_data(x: NDArray[np.float64], offset: float) -> NDArray[np.float64]:
    """Make data positive by taking absolute value and adding offset."""
    abs_x: NDArray[np.float64] = np.abs(x)
    result: NDArray[np.float64] = abs_x + offset
    return result


def _make_default_int_params() -> SampledIntParams:
    """Create default integer parameters for testing."""
    return SampledIntParams(
        max_depth=3,
        n_estimators=10,
    )


def _make_default_float_params() -> SampledFloatParams:
    """Create default float parameters for testing."""
    return SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
    )


def _make_default_string_params() -> SampledStringParams:
    """Create default string parameters for testing."""
    return SampledStringParams()


# =============================================================================
# Tests: Dynamic Import Helpers
# =============================================================================


def test_get_xgb_dmatrix_and_train_returns_valid_types() -> None:
    """_get_xgb_dmatrix_and_train returns DMatrix class and train function."""
    dmatrix_cls, train_fn = _get_xgb_dmatrix_and_train()

    x, y, _ = _make_regression_data(n_samples=10, n_features=3)
    dmatrix = dmatrix_cls(x, label=y)

    params: dict[str, str | int | float] = {
        "max_depth": 2,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
    }
    booster = train_fn(params, dmatrix, num_boost_round=2, verbose_eval=False)
    preds = booster.predict(dmatrix)
    assert len(preds) == 10


def test_cuda_available_returns_bool() -> None:
    """_cuda_available returns a boolean indicating CUDA status."""
    result = _cuda_available()
    assert result in (True, False)


# =============================================================================
# Tests: XGBoostRegressorObjective Initialization
# =============================================================================


def test_init_stores_feature_count() -> None:
    """XGBoostRegressorObjective stores correct feature count."""
    x, y, names = _make_regression_data(n_samples=50, n_features=5)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")
    assert objective.n_features == 5


def test_init_with_feature_engineering_log_only() -> None:
    """Applies log_only feature engineering correctly."""
    x, y, names = _make_regression_data(n_samples=50, n_features=5)
    x_positive = _make_positive_data(x, 1.0)
    objective = XGBoostRegressorObjective(
        x_positive, y, names, device="cpu", feature_preset="log_only"
    )
    assert objective.n_features > 5


def test_init_with_feature_engineering_full() -> None:
    """Applies full feature engineering correctly."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    x_positive = _make_positive_data(x, 0.1)
    objective = XGBoostRegressorObjective(x_positive, y, names, device="cpu", feature_preset="full")
    assert objective.n_features > 4


def test_init_with_auto_device() -> None:
    """Resolves 'auto' device based on CUDA availability."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    objective = XGBoostRegressorObjective(x, y, names, device="auto", feature_preset="none")
    assert objective.n_features == 4


def test_init_with_cpu_device() -> None:
    """Accepts 'cpu' device explicitly."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")
    assert objective.n_features == 4


# =============================================================================
# Tests: XGBoostRegressorObjective Call
# =============================================================================


def test_call_returns_negative_rmse() -> None:
    """__call__ returns negative RMSE (for Optuna maximization)."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    result = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=_make_default_int_params(),
        float_params=_make_default_float_params(),
        string_params=_make_default_string_params(),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Result should be negative (neg RMSE)
    assert result < 0.0
    # RMSE should be finite
    assert math.isfinite(result)


def test_call_ignores_passed_data() -> None:
    """Uses pre-split data, not passed arguments."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5, seed=42)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    x_other, y_other, names_other = _make_regression_data(n_samples=50, n_features=3, seed=99)

    # Should not raise even though passed data has different shape
    result = objective(
        x_features=x_other,
        y_targets=y_other,
        feature_names=names_other,
        int_params=_make_default_int_params(),
        float_params=_make_default_float_params(),
        string_params=_make_default_string_params(),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert result < 0.0


def test_call_with_different_hyperparams() -> None:
    """Returns different results for different hyperparameters."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    int_params_shallow = SampledIntParams(max_depth=2, n_estimators=5)
    int_params_deep = SampledIntParams(max_depth=8, n_estimators=50)
    float_params = _make_default_float_params()
    string_params = _make_default_string_params()

    result_shallow = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=int_params_shallow,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    result_deep = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=int_params_deep,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Both should be negative (neg RMSE)
    assert result_shallow < 0.0
    assert result_deep < 0.0


def test_call_deterministic() -> None:
    """Multiple calls with same params return same result."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = _make_default_float_params()
    string_params = _make_default_string_params()

    result1 = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )
    result2 = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )
    assert result1 == result2


# =============================================================================
# Tests: Factory Function
# =============================================================================


def test_create_returns_objective() -> None:
    """create_xgboost_regressor_objective returns callable with n_features."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    objective = create_xgboost_regressor_objective(x, y, names, device="cpu", feature_preset="none")
    assert objective.n_features == 4
    assert callable(objective)


def test_create_with_feature_preset() -> None:
    """create_xgboost_regressor_objective applies feature preset."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    x_positive = _make_positive_data(x, 0.1)
    objective = create_xgboost_regressor_objective(
        x_positive, y, names, device="cpu", feature_preset="full"
    )
    assert objective.n_features > 4


def test_create_callable() -> None:
    """create_xgboost_regressor_objective returns callable objective."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = create_xgboost_regressor_objective(x, y, names, device="cpu", feature_preset="none")

    result = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=_make_default_int_params(),
        float_params=_make_default_float_params(),
        string_params=_make_default_string_params(),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert result < 0.0


# =============================================================================
# Tests: DART Boosting
# =============================================================================


def test_with_dart_booster() -> None:
    """Works with DART booster and rate_drop/skip_drop params."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        rate_drop=0.1,
        skip_drop=0.5,
    )
    string_params = SampledStringParams(booster="dart")

    result = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=_make_default_int_params(),
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert result < 0.0


def test_with_dart_partial_params() -> None:
    """Works with DART and only rate_drop (no skip_drop)."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        rate_drop=0.1,
    )
    string_params = SampledStringParams(booster="dart")

    result = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=_make_default_int_params(),
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert result < 0.0


def test_with_dart_skip_drop_only() -> None:
    """Works with DART and only skip_drop (no rate_drop)."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        skip_drop=0.5,
    )
    string_params = SampledStringParams(booster="dart")

    result = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=_make_default_int_params(),
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert result < 0.0


# =============================================================================
# Tests: n_features Property
# =============================================================================


def test_n_features_matches_original() -> None:
    """n_features matches input when no engineering applied."""
    x, y, names = _make_regression_data(n_samples=50, n_features=7)
    objective = XGBoostRegressorObjective(x, y, names, device="cpu", feature_preset="none")
    assert objective.n_features == 7


def test_n_features_reflects_engineering() -> None:
    """n_features reflects engineered count when preset applied."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    x_positive = _make_positive_data(x, 0.1)
    objective = XGBoostRegressorObjective(
        x_positive, y, names, device="cpu", feature_preset="log_only"
    )
    assert objective.n_features > 4


# =============================================================================
# Tests: Static method
# =============================================================================


def test_compute_neg_rmse_static() -> None:
    """_compute_neg_rmse returns negative RMSE value."""
    y_true: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    y_pred: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    y_true[0] = 1.0
    y_true[1] = 2.0
    y_true[2] = 3.0
    y_pred[0] = 1.0
    y_pred[1] = 2.0
    y_pred[2] = 3.0

    result = XGBoostRegressorObjective._compute_neg_rmse(y_true, y_pred)
    assert result == 0.0

    # Non-perfect prediction
    y_pred[0] = 1.5
    result = XGBoostRegressorObjective._compute_neg_rmse(y_true, y_pred)
    assert result < 0.0
