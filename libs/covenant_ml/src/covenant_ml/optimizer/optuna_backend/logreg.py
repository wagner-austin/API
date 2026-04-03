"""Logistic Regression Optuna optimizer.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ..protocol import ObjectiveProtocol, TrialCallbackProtocol
from ..types import (
    LogRegSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
)
from ._hooks import get_optuna_factories
from ._protocols import OptunaPrunerProtocol, OptunaTrialProtocol
from ._sampling import sample_param_float, sample_param_int, sample_param_str

_log = get_logger(__name__)


def _sample_logreg_optional_params(
    trial: OptunaTrialProtocol,
    search_space: LogRegSearchSpace,
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
) -> None:
    """Sample optional LogReg params if present in search space.

    Modifies float_params and string_params in place.

    Args:
        trial: Optuna trial object.
        search_space: LogReg search space configuration.
        float_params: Float params dict to update with l1_ratio.
        string_params: String params dict to update with penalty/solver.
    """
    if "penalty" in search_space:
        string_params["penalty"] = sample_param_str(trial, "penalty", search_space["penalty"])
    if "solver" in search_space:
        string_params["solver"] = sample_param_str(trial, "solver", search_space["solver"])
    if "l1_ratio" in search_space:
        float_params["l1_ratio"] = sample_param_float(trial, "l1_ratio", search_space["l1_ratio"])


class OptunaLogRegOptimizer:
    """Logistic Regression hyperparameter optimizer using Optuna TPE."""

    def __init__(self) -> None:
        """Initialize optimizer."""
        self._trials_complete = 0
        self._trials_pruned = 0
        self._trials_failed = 0

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: LogRegSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization using Optuna TPE."""
        create_study, tpe_sampler, median_pruner = get_optuna_factories()

        self._trials_complete = 0
        self._trials_pruned = 0
        self._trials_failed = 0

        start_time = time.perf_counter()

        _log.info(
            "Starting LogReg Optuna optimization",
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

            int_params: SampledIntParams = {
                "max_iter": sample_param_int(trial, "max_iter", search_space["max_iter"]),
            }

            float_params: SampledFloatParams = {
                "C": sample_param_float(trial, "C", search_space["C"]),
                "tol": sample_param_float(trial, "tol", search_space["tol"]),
            }

            # Sample optional params (penalty, solver, l1_ratio)
            string_params: SampledStringParams = {}
            _sample_logreg_optional_params(trial, search_space, float_params, string_params)

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
                    "C": float_params.get("C"),
                    "penalty": string_params.get("penalty"),
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

        best_params = study.best_params
        best_int_params: SampledIntParams = {
            "max_iter": int(best_params["max_iter"]),
        }
        best_float_params: SampledFloatParams = {
            "C": float(best_params["C"]),
            "tol": float(best_params["tol"]),
        }

        # Extract optional best params
        best_string_params: SampledStringParams = {}
        if "penalty" in search_space:
            best_string_params["penalty"] = str(best_params["penalty"])
        if "solver" in search_space:
            best_string_params["solver"] = str(best_params["solver"])
        if "l1_ratio" in search_space:
            best_float_params["l1_ratio"] = float(best_params["l1_ratio"])

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
            "Optimization complete",
            extra={
                "best_value": summary["best_value"],
                "best_C": best_float_params.get("C"),
                "best_penalty": best_string_params.get("penalty"),
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


def create_logreg_optimizer() -> OptunaLogRegOptimizer:
    """Create a Logistic Regression hyperparameter optimizer."""
    return OptunaLogRegOptimizer()


__all__ = [
    "OptunaLogRegOptimizer",
    "create_logreg_optimizer",
]
