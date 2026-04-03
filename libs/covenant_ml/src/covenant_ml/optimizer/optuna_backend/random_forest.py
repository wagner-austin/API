"""Random Forest Optuna optimizer.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ..protocol import ObjectiveProtocol, TrialCallbackProtocol
from ..types import (
    OptimizationConfig,
    OptimizationSummary,
    RandomForestSearchSpace,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
)
from ._hooks import get_optuna_factories
from ._protocols import OptunaPrunerProtocol, OptunaTrialProtocol
from ._sampling import sample_param_int, sample_param_str

_log = get_logger(__name__)


class OptunaRandomForestOptimizer:
    """Random Forest hyperparameter optimizer using Optuna TPE."""

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
        search_space: RandomForestSearchSpace,
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
            "Starting RandomForest Optuna optimization",
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
                "n_estimators": sample_param_int(
                    trial, "n_estimators", search_space["n_estimators"]
                ),
                "max_depth": sample_param_int(trial, "max_depth", search_space["max_depth"]),
                "min_samples_split": sample_param_int(
                    trial, "min_samples_split", search_space["min_samples_split"]
                ),
                "min_samples_leaf": sample_param_int(
                    trial, "min_samples_leaf", search_space["min_samples_leaf"]
                ),
            }

            # RandomForest has no float params in the search space
            float_params: SampledFloatParams = {}

            string_params: SampledStringParams = {
                "max_features": sample_param_str(
                    trial, "max_features", search_space["max_features"]
                ),
            }

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
                    "max_depth": int_params.get("max_depth"),
                    "n_estimators": int_params.get("n_estimators"),
                    "max_features": string_params.get("max_features"),
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
            "n_estimators": int(best_params["n_estimators"]),
            "max_depth": int(best_params["max_depth"]),
            "min_samples_split": int(best_params["min_samples_split"]),
            "min_samples_leaf": int(best_params["min_samples_leaf"]),
        }
        best_float_params: SampledFloatParams = {}
        best_string_params: SampledStringParams = {
            "max_features": str(best_params["max_features"]),
        }

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
                "best_max_depth": best_int_params.get("max_depth"),
                "best_n_estimators": best_int_params.get("n_estimators"),
                "best_max_features": best_string_params.get("max_features"),
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


def create_random_forest_optimizer() -> OptunaRandomForestOptimizer:
    """Create a Random Forest hyperparameter optimizer."""
    return OptunaRandomForestOptimizer()


__all__ = [
    "OptunaRandomForestOptimizer",
    "create_random_forest_optimizer",
]
