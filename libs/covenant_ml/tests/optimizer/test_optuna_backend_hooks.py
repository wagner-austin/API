"""Tests for the optuna factories hook.

Covers the binding the module ships with and a rebinding to fakes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.optuna_backend import _hooks as _backend_hooks
from covenant_ml.optimizer.optuna_backend import (
    create_xgboost_optimizer,
)
from covenant_ml.optimizer.search_spaces import make_xgboost_default_space
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)

from .conftest import (
    make_optuna_config,
    make_optuna_test_data,
)

# =============================================================================
# Tests: Hook Management
# =============================================================================


def test_the_hook_is_bound_to_real_optuna() -> None:
    """The module binds real optuna, so an optimizer runs with nothing wired."""
    assert _backend_hooks.optuna_factories is _backend_hooks._real_optuna_factories

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
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories
