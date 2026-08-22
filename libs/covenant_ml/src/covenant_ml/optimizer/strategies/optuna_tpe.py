"""Optuna TPE optimizer strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Wraps Optuna's TPE (Tree-structured Parzen Estimator) for Bayesian optimization.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.optimizer.strategies._tpe_params import (
    _extract_best_params,
    _sample_params,
)

from ..protocol import ObjectiveProtocol, TrialCallbackProtocol
from ..strategy_protocol import OptimizerStrategyCapabilities, OptimizerStrategyName
from ..types import (
    OptimizationConfig,
    OptimizationSummary,
    SearchSpace,
    TrialResult,
)
from . import _hooks
from ._hooks import (
    OptunaPrunerProtocol,
    OptunaTrialProtocol,
)

_log = get_logger(__name__)


# =============================================================================
# Optuna Protocol Definitions (minimal for module hook)
# =============================================================================


# =============================================================================
# Parameter Sampling
# =============================================================================


# =============================================================================
# Optuna TPE Optimizer
# =============================================================================


class OptunaTpeOptimizer:
    """Bayesian optimization using Optuna's TPE algorithm.

    Tree-structured Parzen Estimator (TPE) is an efficient Bayesian
    optimization algorithm that models p(x|y) instead of p(y|x).
    It works well for hyperparameter optimization with up to ~1000 trials.
    """

    def __init__(self) -> None:
        """Initialize optimizer with trial counters."""
        self._trials_complete = 0
        self._trials_pruned = 0
        self._trials_failed = 0

    def strategy_name(self) -> OptimizerStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'optuna_tpe'.
        """
        return "optuna_tpe"

    def capabilities(self) -> OptimizerStrategyCapabilities:
        """Return the capabilities of this strategy.

        Returns:
            Capabilities indicating TPE supports pruning and parallelism.
        """
        return OptimizerStrategyCapabilities(
            supports_pruning=True,
            supports_parallel=True,
            is_deterministic=False,
            requires_bounds=True,
        )

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: SearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization using Optuna TPE.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_labels: Binary labels (n_samples,).
            feature_names: Names for each feature column.
            search_space: Parameter ranges to search.
            config: Optimization settings.
            objective: Function to evaluate hyperparameters.
            trial_callback: Optional callback after each trial.

        Returns:
            Summary with best hyperparameters and trial statistics.
        """
        create_study, tpe_sampler, median_pruner = _hooks.optuna_factories()

        self._trials_complete = 0
        self._trials_pruned = 0
        self._trials_failed = 0

        start_time = time.perf_counter()

        _log.info(
            "Starting Optuna TPE optimization",
            extra={
                "n_trials": config["n_trials"],
                "n_startup_trials": config["n_startup_trials"],
                "direction": config["direction"],
                "pruning_enabled": config["pruning_enabled"],
            },
        )

        sampler = tpe_sampler(
            seed=config["random_state"],
            n_startup_trials=config["n_startup_trials"],
        )

        pruner: OptunaPrunerProtocol | None = None
        if config["pruning_enabled"]:
            pruner = median_pruner(n_startup_trials=5, n_warmup_steps=10)

        study = create_study(
            direction=config["direction"],
            sampler=sampler,
            pruner=pruner,
        )

        def optuna_objective(trial: OptunaTrialProtocol) -> float:
            trial_start = time.perf_counter()

            int_params, float_params, string_params = _sample_params(trial, search_space)

            val_auc = objective(
                x_features,
                y_labels,
                feature_names,
                int_params,
                float_params,
                string_params,
                config["train_ratio"],
                config["val_ratio"],
                config["test_ratio"],
                config["random_state"],
            )

            trial_duration = time.perf_counter() - trial_start
            self._trials_complete += 1

            result: TrialResult = {
                "trial_number": trial.number,
                "int_params": int_params,
                "float_params": float_params,
                "string_params": string_params,
                "value": val_auc,
                "state": "complete",
                "duration_seconds": trial_duration,
            }

            if trial_callback is not None:
                trial_callback(result)

            _log.info(
                "Trial complete",
                extra={
                    "trial": trial.number,
                    "val_auc": val_auc,
                    "duration_sec": trial_duration,
                },
            )

            return val_auc

        timeout_val: float | None = None
        if config["timeout_seconds"] is not None:
            timeout_val = float(config["timeout_seconds"])

        study.optimize(
            optuna_objective,
            n_trials=config["n_trials"],
            timeout=timeout_val,
        )

        total_duration = time.perf_counter() - start_time

        best_int_params, best_float_params, best_string_params = _extract_best_params(
            search_space, study.best_params
        )

        summary: OptimizationSummary = {
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_int_params": best_int_params,
            "best_float_params": best_float_params,
            "best_string_params": best_string_params,
            "n_trials_total": config["n_trials"],
            "n_trials_complete": self._trials_complete,
            "n_trials_pruned": self._trials_pruned,
            "n_trials_failed": self._trials_failed,
            "total_duration_seconds": total_duration,
        }

        _log.info(
            "Optuna TPE optimization complete",
            extra={
                "best_value": summary["best_value"],
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


def create_optuna_tpe_optimizer() -> OptunaTpeOptimizer:
    """Factory function to create an OptunaTpeOptimizer.

    Returns:
        A new OptunaTpeOptimizer instance.
    """
    return OptunaTpeOptimizer()


__all__ = [
    "OptunaTpeOptimizer",
    "create_optuna_tpe_optimizer",
]
