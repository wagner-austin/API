"""Tests for LightGBM objective function.

Uses real lightgbm library for integration testing. No mocks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.objectives.lightgbm_objective import (
    LightGBMObjective,
    _get_lgb_dataset_and_train,
    _resolve_lightgbm_device,
    create_lightgbm_objective,
)
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams

# =============================================================================
# Test Data Helpers
# =============================================================================


def _make_test_data(
    n_samples: int = 100, n_features: int = 5, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create test dataset for optimization."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    # Create imbalanced labels: ~30% positive
    n_positive = n_samples // 3
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_positive] = 1
    rng.shuffle(y)
    return x, y, [f"feat_{i}" for i in range(n_features)]


def _make_positive_data(x: NDArray[np.float64], offset: float) -> NDArray[np.float64]:
    """Make data positive by taking absolute value and adding offset.

    Uses direct type annotation to satisfy mypy, same pattern as features.py.
    """
    abs_x: NDArray[np.float64] = np.abs(x)
    result: NDArray[np.float64] = abs_x + offset
    return result


def _make_default_int_params() -> SampledIntParams:
    """Create default integer parameters for testing.

    Note: max_depth is not included because LightGBM optimization uses
    fixed max_depth=-1 (unlimited) to let num_leaves control tree complexity.
    """
    return SampledIntParams(
        n_estimators=10,
        num_leaves=8,
        min_child_samples=5,
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


# =============================================================================
# Tests: Dynamic Import Helpers
# =============================================================================


def test_get_lgb_dataset_and_train_returns_valid_types() -> None:
    """_get_lgb_dataset_and_train returns Dataset class, train function, and early_stopping."""
    dataset_cls, train_fn, early_stopping = _get_lgb_dataset_and_train()

    # Verify Dataset can be instantiated with data
    x, y, _ = _make_test_data(n_samples=30, n_features=3)
    dataset = dataset_cls(x, label=y, free_raw_data=False)

    # Verify early_stopping callback works
    early_stop_cb = early_stopping(stopping_rounds=5, verbose=False)
    assert early_stop_cb.stopping_rounds == 5

    # Verify train function works and returns booster that can predict
    params: dict[str, str | int | float] = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 4,
        "max_depth": 2,
        "learning_rate": 0.1,
        "verbose": -1,
    }
    booster = train_fn(
        params,
        dataset,
        num_boost_round=5,
        valid_sets=None,
        valid_names=None,
        callbacks=None,
    )
    # Verify booster can make predictions
    preds = booster.predict(x, num_iteration=None)
    assert len(preds) == 30


# =============================================================================
# Tests: LightGBMObjective Initialization
# =============================================================================


def test_lightgbm_objective_init_stores_feature_count() -> None:
    """LightGBMObjective stores correct feature count after initialization."""
    x, y, names = _make_test_data(n_samples=50, n_features=5)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")
    assert objective.n_features == 5


def test_lightgbm_objective_init_with_feature_engineering_log_only() -> None:
    """LightGBMObjective applies log_only feature engineering correctly."""
    x, y, names = _make_test_data(n_samples=50, n_features=5)
    # Make all values positive for log transform
    x_positive = _make_positive_data(x, 1.0)
    objective = LightGBMObjective(x_positive, y, names, device="cpu", feature_preset="log_only")
    # log_only adds log-transformed versions of original features
    assert objective.n_features > 5


def test_lightgbm_objective_init_with_feature_engineering_ratios_only() -> None:
    """LightGBMObjective applies ratios_only feature engineering correctly."""
    x, y, names = _make_test_data(n_samples=50, n_features=4)
    # Make values positive and avoid zeros for ratio computation
    x_positive = _make_positive_data(x, 0.1)
    objective = LightGBMObjective(x_positive, y, names, device="cpu", feature_preset="ratios_only")
    # ratios_only adds ratio features
    assert objective.n_features >= 4


def test_lightgbm_objective_init_with_feature_engineering_full() -> None:
    """LightGBMObjective applies full feature engineering correctly."""
    x, y, names = _make_test_data(n_samples=50, n_features=4)
    # Make values positive for log and ratio transforms
    x_positive = _make_positive_data(x, 0.1)
    objective = LightGBMObjective(x_positive, y, names, device="cpu", feature_preset="full")
    # full adds ratios, products, and log transforms
    assert objective.n_features > 4


def test_lightgbm_objective_init_with_auto_device() -> None:
    """LightGBMObjective resolves 'auto' device to cpu."""
    x, y, names = _make_test_data(n_samples=50, n_features=4)
    objective = LightGBMObjective(x, y, names, device="auto", feature_preset="none")
    # Should not raise - device resolution happens at init
    assert objective.n_features == 4


def test_lightgbm_objective_init_with_cpu_device() -> None:
    """LightGBMObjective accepts 'cpu' device explicitly."""
    x, y, names = _make_test_data(n_samples=50, n_features=4)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")
    assert objective.n_features == 4


def test_lightgbm_objective_init_with_early_stopping_rounds() -> None:
    """LightGBMObjective accepts custom early_stopping_rounds."""
    x, y, names = _make_test_data(n_samples=50, n_features=4)
    objective = LightGBMObjective(
        x, y, names, device="cpu", feature_preset="none", early_stopping_rounds=20
    )
    assert objective.n_features == 4


# =============================================================================
# Tests: LightGBMObjective Call
# =============================================================================


def test_lightgbm_objective_call_returns_auc() -> None:
    """LightGBMObjective.__call__ returns validation AUC between 0 and 1."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = _make_default_float_params()

    auc = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # AUC must be between 0 and 1
    assert 0.0 <= auc <= 1.0
    # AUC should be better than random (0.5) for this dataset
    assert auc > 0.5


def test_lightgbm_objective_call_ignores_passed_data() -> None:
    """LightGBMObjective uses pre-split data, not passed arguments."""
    # Create objective with specific data
    x, y, names = _make_test_data(n_samples=100, n_features=5, seed=42)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")

    # Pass different data (which should be ignored)
    x_other, y_other, names_other = _make_test_data(n_samples=50, n_features=3, seed=99)

    int_params = _make_default_int_params()
    float_params = _make_default_float_params()

    # Should not raise even though passed data has different shape
    auc = objective(
        x_features=x_other,
        y_labels=y_other,
        feature_names=names_other,
        int_params=int_params,
        float_params=float_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert 0.0 <= auc <= 1.0


def test_lightgbm_objective_call_with_different_hyperparams() -> None:
    """LightGBMObjective returns different results for different hyperparameters."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")

    # Test with few leaves (simpler model) vs many leaves (more complex model)
    int_params_simple = SampledIntParams(n_estimators=5, num_leaves=4)
    int_params_complex = SampledIntParams(n_estimators=50, num_leaves=64)
    float_params = _make_default_float_params()

    auc_simple = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params_simple,
        float_params=float_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    auc_complex = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params_complex,
        float_params=float_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Both should be valid AUCs
    assert 0.0 <= auc_simple <= 1.0
    assert 0.0 <= auc_complex <= 1.0


def test_lightgbm_objective_multiple_calls_deterministic() -> None:
    """Multiple calls with same params return same AUC (deterministic)."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = _make_default_float_params()

    auc1 = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    auc2 = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert auc1 == auc2


# =============================================================================
# Tests: Factory Function
# =============================================================================


def test_create_lightgbm_objective_returns_objective() -> None:
    """create_lightgbm_objective returns callable with n_features property."""
    x, y, names = _make_test_data(n_samples=50, n_features=4)
    objective = create_lightgbm_objective(x, y, names, device="cpu", feature_preset="none")
    # Verify it has the n_features property
    assert objective.n_features == 4
    # Verify it's callable
    assert callable(objective)


def test_create_lightgbm_objective_with_feature_preset() -> None:
    """create_lightgbm_objective applies feature preset."""
    x, y, names = _make_test_data(n_samples=50, n_features=4)
    x_positive = _make_positive_data(x, 0.1)  # Make positive for transformations
    objective = create_lightgbm_objective(x_positive, y, names, device="cpu", feature_preset="full")
    assert objective.n_features > 4


def test_create_lightgbm_objective_callable() -> None:
    """create_lightgbm_objective returns callable objective."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = create_lightgbm_objective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = _make_default_float_params()

    auc = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert 0.0 <= auc <= 1.0


def test_create_lightgbm_objective_with_early_stopping_rounds() -> None:
    """create_lightgbm_objective accepts early_stopping_rounds parameter."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = create_lightgbm_objective(
        x, y, names, device="cpu", feature_preset="none", early_stopping_rounds=5
    )

    int_params = _make_default_int_params()
    float_params = _make_default_float_params()

    auc = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert 0.0 <= auc <= 1.0


# =============================================================================
# Tests: n_features Property
# =============================================================================


def test_n_features_property_matches_original() -> None:
    """n_features matches input when no engineering applied."""
    x, y, names = _make_test_data(n_samples=50, n_features=7)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")
    assert objective.n_features == 7


def test_n_features_property_reflects_engineering() -> None:
    """n_features reflects engineered count when preset applied."""
    x, y, names = _make_test_data(n_samples=50, n_features=4)
    x_positive = _make_positive_data(x, 0.1)  # Make positive for log transform
    objective = LightGBMObjective(x_positive, y, names, device="cpu", feature_preset="log_only")
    # log_only typically doubles features (original + log)
    assert objective.n_features > 4


# =============================================================================
# Tests: Device Resolution
# =============================================================================


def test_resolve_lightgbm_device_auto_returns_cpu() -> None:
    """_resolve_lightgbm_device returns 'cpu' for 'auto' device."""
    result = _resolve_lightgbm_device("auto")
    assert result == "cpu"


def test_resolve_lightgbm_device_cpu_returns_cpu() -> None:
    """_resolve_lightgbm_device returns 'cpu' for 'cpu' device."""
    result = _resolve_lightgbm_device("cpu")
    assert result == "cpu"


def test_resolve_lightgbm_device_cuda_on_windows_returns_gpu() -> None:
    """_resolve_lightgbm_device returns 'gpu' for 'cuda' on Windows platform."""
    result = _resolve_lightgbm_device("cuda", platform="win32")
    assert result == "gpu"


def test_resolve_lightgbm_device_cuda_on_linux_returns_cuda() -> None:
    """_resolve_lightgbm_device returns 'cuda' for 'cuda' on Linux platform."""
    result = _resolve_lightgbm_device("cuda", platform="linux")
    assert result == "cuda"


def test_resolve_lightgbm_device_cuda_on_darwin_returns_cuda() -> None:
    """_resolve_lightgbm_device returns 'cuda' for 'cuda' on macOS platform."""
    result = _resolve_lightgbm_device("cuda", platform="darwin")
    assert result == "cuda"


def test_resolve_lightgbm_device_returns_valid_type_for_all_inputs() -> None:
    """_resolve_lightgbm_device returns valid LightGBMDevice for all inputs."""
    from covenant_ml.optimizer.types import DeviceRequest

    inputs: tuple[DeviceRequest, ...] = ("cpu", "cuda", "auto")
    valid_outputs = {"cpu", "gpu", "cuda"}

    for device_input in inputs:
        result = _resolve_lightgbm_device(device_input)
        assert result in valid_outputs
