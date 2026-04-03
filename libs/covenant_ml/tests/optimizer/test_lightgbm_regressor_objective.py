"""Tests for LightGBM regressor objective function.

Uses real lightgbm library for integration testing. No mocks.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.objectives.lightgbm_regressor_objective import (
    LightGBMRegressorObjective,
    _get_lgb_dataset_and_train,
    _resolve_lightgbm_device,
    create_lightgbm_regressor_objective,
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
        y[i] = float(x.item((i, 0))) * 3.0 + float(x.item((i, 1))) * 1.5 + 2.0

    names = [f"feat_{j}" for j in range(n_features)]
    return x, y, names


def _make_positive_data(x: NDArray[np.float64], offset: float) -> NDArray[np.float64]:
    """Make data positive by taking absolute value and adding offset."""
    abs_x: NDArray[np.float64] = np.abs(x)
    result: NDArray[np.float64] = abs_x + offset
    return result


def _make_default_int_params() -> SampledIntParams:
    """Create default integer parameters."""
    return SampledIntParams(
        n_estimators=10,
        num_leaves=15,
        min_child_samples=5,
    )


def _make_default_float_params() -> SampledFloatParams:
    """Create default float parameters."""
    return SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
    )


def _make_default_string_params() -> SampledStringParams:
    """Create default string parameters."""
    return SampledStringParams()


def _make_dummy_labels(n: int) -> NDArray[np.int64]:
    """Create dummy int64 labels for ObjectiveProtocol compatibility.

    The objective ignores y_labels in __call__ (uses pre-split data from __init__).
    This satisfies the int64 type requirement of ObjectiveProtocol.
    """
    return np.zeros(n, dtype=np.int64)


# =============================================================================
# Tests: Dynamic Import Helpers
# =============================================================================


def test_get_lgb_dataset_and_train_returns_valid_types() -> None:
    """_get_lgb_dataset_and_train returns Dataset class, train func, early stopping."""
    dataset_cls, train_fn, early_stopping = _get_lgb_dataset_and_train()

    x, y, _ = _make_regression_data(n_samples=10, n_features=3)
    dataset = dataset_cls(x, label=y, free_raw_data=False)

    params: dict[str, str | int | float] = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "num_leaves": 4,
    }
    booster = train_fn(params, dataset, num_boost_round=2)
    preds = booster.predict(x, num_iteration=None)
    assert len(preds) == 10

    # Verify early stopping factory
    cb = early_stopping(stopping_rounds=5, verbose=False)
    assert cb.stopping_rounds == 5


# =============================================================================
# Tests: Device Resolution
# =============================================================================


def test_resolve_device_auto() -> None:
    """'auto' resolves to 'cpu'."""
    result = _resolve_lightgbm_device("auto")
    assert result == "cpu"


def test_resolve_device_cpu() -> None:
    """'cpu' resolves to 'cpu'."""
    result = _resolve_lightgbm_device("cpu")
    assert result == "cpu"


def test_resolve_device_cuda_on_linux() -> None:
    """'cuda' stays 'cuda' on non-Windows."""
    result = _resolve_lightgbm_device("cuda", platform="linux")
    assert result == "cuda"


def test_resolve_device_cuda_on_windows() -> None:
    """'cuda' resolves to 'gpu' (OpenCL) on Windows."""
    result = _resolve_lightgbm_device("cuda", platform="win32")
    assert result == "gpu"


# =============================================================================
# Tests: LightGBMRegressorObjective Initialization
# =============================================================================


def test_init_stores_feature_count() -> None:
    """Stores correct feature count."""
    x, y, names = _make_regression_data(n_samples=50, n_features=5)
    objective = LightGBMRegressorObjective(x, y, names, device="cpu", feature_preset="none")
    assert objective.n_features == 5


def test_init_with_feature_engineering_log_only() -> None:
    """Applies log_only feature engineering correctly."""
    x, y, names = _make_regression_data(n_samples=50, n_features=5)
    x_positive = _make_positive_data(x, 1.0)
    objective = LightGBMRegressorObjective(
        x_positive, y, names, device="cpu", feature_preset="log_only"
    )
    assert objective.n_features > 5


def test_init_with_feature_engineering_full() -> None:
    """Applies full feature engineering correctly."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    x_positive = _make_positive_data(x, 0.1)
    objective = LightGBMRegressorObjective(
        x_positive, y, names, device="cpu", feature_preset="full"
    )
    assert objective.n_features > 4


def test_init_with_auto_device() -> None:
    """Resolves 'auto' device."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    objective = LightGBMRegressorObjective(x, y, names, device="auto", feature_preset="none")
    assert objective.n_features == 4


def test_init_with_custom_early_stopping() -> None:
    """Accepts custom early_stopping_rounds."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    objective = LightGBMRegressorObjective(
        x, y, names, device="cpu", feature_preset="none", early_stopping_rounds=20
    )
    assert objective.n_features == 4


# =============================================================================
# Tests: LightGBMRegressorObjective Call
# =============================================================================


def test_call_returns_negative_rmse() -> None:
    """__call__ returns negative RMSE."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = LightGBMRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    result = objective(
        x_features=x,
        y_labels=_make_dummy_labels(100),
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
    assert math.isfinite(result)


def test_call_ignores_passed_data() -> None:
    """Uses pre-split data, not passed arguments."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5, seed=42)
    objective = LightGBMRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    x_other, _, names_other = _make_regression_data(n_samples=50, n_features=3, seed=99)

    result = objective(
        x_features=x_other,
        y_labels=_make_dummy_labels(50),
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


def test_call_deterministic() -> None:
    """Multiple calls with same params return same result."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = LightGBMRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = _make_default_float_params()
    string_params = _make_default_string_params()

    result1 = objective(
        x_features=x,
        y_labels=_make_dummy_labels(100),
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
        y_labels=_make_dummy_labels(100),
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
    """create_lightgbm_regressor_objective returns callable with n_features."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    objective = create_lightgbm_regressor_objective(
        x, y, names, device="cpu", feature_preset="none"
    )
    assert objective.n_features == 4
    assert callable(objective)


def test_create_with_feature_preset() -> None:
    """create_lightgbm_regressor_objective applies feature preset."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    x_positive = _make_positive_data(x, 0.1)
    objective = create_lightgbm_regressor_objective(
        x_positive, y, names, device="cpu", feature_preset="full"
    )
    assert objective.n_features > 4


def test_create_callable() -> None:
    """Factory-created objective is callable and returns neg RMSE."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = create_lightgbm_regressor_objective(
        x, y, names, device="cpu", feature_preset="none"
    )

    result = objective(
        x_features=x,
        y_labels=_make_dummy_labels(100),
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


def test_create_with_custom_options() -> None:
    """Factory accepts custom early_stopping_rounds and n_jobs."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    objective = create_lightgbm_regressor_objective(
        x,
        y,
        names,
        device="cpu",
        feature_preset="none",
        early_stopping_rounds=20,
        n_jobs=1,
    )
    assert objective.n_features == 4


# =============================================================================
# Tests: DART Boosting
# =============================================================================


def test_with_dart_boosting() -> None:
    """Works with DART boosting type."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = LightGBMRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        drop_rate=0.1,
        skip_drop=0.5,
        feature_fraction=0.7,
    )
    string_params = SampledStringParams(boosting_type="dart")

    result = objective(
        x_features=x,
        y_labels=_make_dummy_labels(100),
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
    """Works with DART and partial params (only drop_rate)."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = LightGBMRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        drop_rate=0.1,
    )
    string_params = SampledStringParams(boosting_type="dart")

    result = objective(
        x_features=x,
        y_labels=_make_dummy_labels(100),
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
    """Works with DART and only skip_drop (no drop_rate)."""
    x, y, names = _make_regression_data(n_samples=100, n_features=5)
    objective = LightGBMRegressorObjective(x, y, names, device="cpu", feature_preset="none")

    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        skip_drop=0.5,
    )
    string_params = SampledStringParams(boosting_type="dart")

    result = objective(
        x_features=x,
        y_labels=_make_dummy_labels(100),
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
    objective = LightGBMRegressorObjective(x, y, names, device="cpu", feature_preset="none")
    assert objective.n_features == 7


def test_n_features_reflects_engineering() -> None:
    """n_features reflects engineered count when preset applied."""
    x, y, names = _make_regression_data(n_samples=50, n_features=4)
    x_positive = _make_positive_data(x, 0.1)
    objective = LightGBMRegressorObjective(
        x_positive, y, names, device="cpu", feature_preset="log_only"
    )
    assert objective.n_features > 4
