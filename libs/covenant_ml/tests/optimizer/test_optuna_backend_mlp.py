"""Tests for MLP Optuna backend optimizer.

Uses fake Optuna implementation for testing without mocks.
"""

from __future__ import annotations

from covenant_ml.optimizer.optuna_backend import (
    create_mlp_optimizer,
    set_optuna_module_hook,
)
from covenant_ml.optimizer.search_spaces import make_mlp_default_space
from covenant_ml.optimizer.types import TrialResult

from .conftest import (
    FakeObjective,
    get_fake_optuna_factories,
    make_optuna_config,
    make_optuna_test_data,
)

# =============================================================================
# Tests: MLP Optimizer
# =============================================================================


def test_mlp_optimizer_runs_trials() -> None:
    """MLP optimizer runs all trials and returns summary."""
    set_optuna_module_hook(get_fake_optuna_factories)
    try:
        optimizer = create_mlp_optimizer()
        x, y, names = make_optuna_test_data()
        objective = FakeObjective()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_mlp_default_space(),
            config=make_optuna_config(n_trials=5),
            objective=objective,
        )
        assert summary["n_trials_total"] == 5
        assert summary["n_trials_complete"] == 5
        assert objective.call_count == 5
        assert "learning_rate" in summary["best_float_params"]
    finally:
        set_optuna_module_hook(None)


def test_mlp_optimizer_with_pruning_disabled() -> None:
    """MLP optimizer works with pruning disabled."""
    set_optuna_module_hook(get_fake_optuna_factories)
    try:
        optimizer = create_mlp_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_mlp_default_space(),
            config=make_optuna_config(n_trials=3, pruning_enabled=False),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


def test_mlp_optimizer_with_trial_callback() -> None:
    """MLP optimizer calls trial_callback for each trial."""
    set_optuna_module_hook(get_fake_optuna_factories)
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_mlp_optimizer()
        x, y, names = make_optuna_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_mlp_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for result in callbacks:
            assert result["state"] == "complete"
    finally:
        set_optuna_module_hook(None)


def test_mlp_optimizer_with_timeout() -> None:
    """MLP optimizer accepts timeout_seconds."""
    set_optuna_module_hook(get_fake_optuna_factories)
    try:
        optimizer = create_mlp_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_mlp_default_space(),
            config=make_optuna_config(n_trials=3, timeout_seconds=60),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)
