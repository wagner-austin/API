"""Tests for Optuna module hook management.

Tests set_optuna_module_hook, use_real_optuna, and hook error handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.optimizer.optuna_backend import (
    create_xgboost_optimizer,
    set_optuna_module_hook,
    use_real_optuna,
)
from covenant_ml.optimizer.search_spaces import make_xgboost_default_space
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)

from .conftest import (
    FakeObjective,
    get_fake_optuna_factories,
    make_optuna_config,
    make_optuna_test_data,
)

# =============================================================================
# Tests: Hook Management
# =============================================================================


def test_set_optuna_module_hook_can_be_cleared() -> None:
    """Hook can be set to None to clear it."""
    set_optuna_module_hook(get_fake_optuna_factories)
    set_optuna_module_hook(None)
    optimizer = create_xgboost_optimizer()
    x, y, names = make_optuna_test_data(n_samples=20)
    with pytest.raises(RuntimeError, match="hook not set"):
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=make_optuna_config(n_trials=1),
            objective=FakeObjective(),
        )


def test_optimizer_raises_when_hook_not_set() -> None:
    """Optimizer raises RuntimeError when hook is not set."""
    set_optuna_module_hook(None)
    optimizer = create_xgboost_optimizer()
    x, y, names = make_optuna_test_data(n_samples=20)
    with pytest.raises(RuntimeError, match="hook not set"):
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=make_optuna_config(n_trials=1),
            objective=FakeObjective(),
        )


def test_use_real_optuna_sets_hook() -> None:
    """use_real_optuna() sets the hook to use real Optuna."""
    set_optuna_module_hook(None)
    use_real_optuna()

    def simple_objective(
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        int_params: SampledIntParams,
        float_params: SampledFloatParams,
        string_params: SampledStringParams,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        _ = x_features, y_labels, feature_names, int_params, float_params, string_params
        _ = train_ratio, val_ratio, test_ratio, random_state
        return 0.75

    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = make_optuna_test_data(n_samples=30)
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=make_optuna_config(n_trials=1),
            objective=simple_objective,
        )
        assert summary["n_trials_complete"] == 1
        assert summary["best_value"] == 0.75
    finally:
        set_optuna_module_hook(None)
