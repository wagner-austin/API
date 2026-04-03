"""XGBoost Optuna optimizer.

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
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
    XGBoostSearchSpace,
)
from ._hooks import get_optuna_factories
from ._protocols import OptunaPrunerProtocol, OptunaTrialProtocol
from ._sampling import sample_param_float, sample_param_int, sample_param_str

_log = get_logger(__name__)


def _sample_xgboost_dart_params(
    trial: OptunaTrialProtocol,
    search_space: XGBoostSearchSpace,
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
) -> None:
    """Sample DART params for XGBoost if present in search space.

    Modifies float_params and string_params in place.

    Args:
        trial: Optuna trial object.
        search_space: XGBoost search space configuration.
        float_params: Float params dict to update with DART params.
        string_params: String params dict to update with booster type.
    """
    if "booster" not in search_space:
        return

    booster = sample_param_str(trial, "booster", search_space["booster"])
    string_params["booster"] = booster

    # DART-specific params only when booster is "dart"
    if booster == "dart":
        if "rate_drop" in search_space:
            float_params["rate_drop"] = sample_param_float(
                trial, "rate_drop", search_space["rate_drop"]
            )
        if "skip_drop" in search_space:
            float_params["skip_drop"] = sample_param_float(
                trial, "skip_drop", search_space["skip_drop"]
            )


def _extract_xgboost_dart_best_params(
    search_space: XGBoostSearchSpace,
    best_params: dict[str, float | int | str],
    best_float_params: SampledFloatParams,
    best_string_params: SampledStringParams,
) -> None:
    """Extract best DART params for XGBoost from study results.

    Modifies best_float_params and best_string_params in place.

    Args:
        search_space: XGBoost search space configuration.
        best_params: Best params from Optuna study.
        best_float_params: Float params dict to update with DART params.
        best_string_params: String params dict to update with booster type.
    """
    if "booster" not in search_space:
        return

    best_booster = str(best_params["booster"])
    best_string_params["booster"] = best_booster
    if best_booster == "dart":
        if "rate_drop" in search_space:
            best_float_params["rate_drop"] = float(best_params["rate_drop"])
        if "skip_drop" in search_space:
            best_float_params["skip_drop"] = float(best_params["skip_drop"])


class OptunaXGBoostOptimizer:
    """XGBoost hyperparameter optimizer using Optuna TPE."""

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
        search_space: XGBoostSearchSpace,
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
            "Starting XGBoost Optuna optimization",
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
                "max_depth": sample_param_int(trial, "max_depth", search_space["max_depth"]),
                "n_estimators": sample_param_int(
                    trial, "n_estimators", search_space["n_estimators"]
                ),
            }

            float_params: SampledFloatParams = {
                "learning_rate": sample_param_float(
                    trial, "learning_rate", search_space["learning_rate"]
                ),
                "reg_alpha": sample_param_float(trial, "reg_alpha", search_space["reg_alpha"]),
                "reg_lambda": sample_param_float(trial, "reg_lambda", search_space["reg_lambda"]),
                "subsample": sample_param_float(trial, "subsample", search_space["subsample"]),
                "colsample_bytree": sample_param_float(
                    trial, "colsample_bytree", search_space["colsample_bytree"]
                ),
            }

            # Sample optional DART params if present in search space
            string_params: SampledStringParams = {}
            _sample_xgboost_dart_params(trial, search_space, float_params, string_params)

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
                    "learning_rate": float_params.get("learning_rate"),
                    "booster": string_params.get("booster"),
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
            "max_depth": int(best_params["max_depth"]),
            "n_estimators": int(best_params["n_estimators"]),
        }
        best_float_params: SampledFloatParams = {
            "learning_rate": float(best_params["learning_rate"]),
            "reg_alpha": float(best_params["reg_alpha"]),
            "reg_lambda": float(best_params["reg_lambda"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
        }

        # Extract best string params and conditional DART float params
        best_string_params: SampledStringParams = {}
        _extract_xgboost_dart_best_params(
            search_space, best_params, best_float_params, best_string_params
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
            "Optimization complete",
            extra={
                "best_value": summary["best_value"],
                "best_max_depth": best_int_params.get("max_depth"),
                "best_learning_rate": best_float_params.get("learning_rate"),
                "best_booster": best_string_params.get("booster"),
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


def create_xgboost_optimizer() -> OptunaXGBoostOptimizer:
    """Create an XGBoost hyperparameter optimizer."""
    return OptunaXGBoostOptimizer()


__all__ = [
    "OptunaXGBoostOptimizer",
    "create_xgboost_optimizer",
]
