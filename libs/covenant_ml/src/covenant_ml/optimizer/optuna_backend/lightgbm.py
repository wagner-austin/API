"""LightGBM Optuna optimizer.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ..protocol import ObjectiveProtocol, TrialCallbackProtocol
from ..types import (
    LightGBMSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
)
from . import _hooks
from ._protocols import OptunaPrunerProtocol, OptunaTrialProtocol
from ._sampling import sample_param_float, sample_param_int, sample_param_str

_log = get_logger(__name__)


def _sample_lightgbm_dart_params(
    trial: OptunaTrialProtocol,
    search_space: LightGBMSearchSpace,
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
) -> None:
    """Sample DART params for LightGBM if present in search space.

    Modifies float_params and string_params in place.

    Args:
        trial: Optuna trial object.
        search_space: LightGBM search space configuration.
        float_params: Float params dict to update with DART params.
        string_params: String params dict to update with boosting type.
    """
    if "boosting_type" not in search_space:
        return

    boosting_type = sample_param_str(trial, "boosting_type", search_space["boosting_type"])
    string_params["boosting_type"] = boosting_type

    # DART-specific params only when boosting_type is "dart"
    if boosting_type == "dart":
        if "drop_rate" in search_space:
            float_params["drop_rate"] = sample_param_float(
                trial, "drop_rate", search_space["drop_rate"]
            )
        if "skip_drop" in search_space:
            float_params["skip_drop"] = sample_param_float(
                trial, "skip_drop", search_space["skip_drop"]
            )
        if "feature_fraction" in search_space:
            float_params["feature_fraction"] = sample_param_float(
                trial, "feature_fraction", search_space["feature_fraction"]
            )


def _extract_lightgbm_dart_best_params(
    search_space: LightGBMSearchSpace,
    best_params: dict[str, float | int | str],
    best_float_params: SampledFloatParams,
    best_string_params: SampledStringParams,
) -> None:
    """Extract best DART params for LightGBM from study results.

    Modifies best_float_params and best_string_params in place.

    Args:
        search_space: LightGBM search space configuration.
        best_params: Best params from Optuna study.
        best_float_params: Float params dict to update with DART params.
        best_string_params: String params dict to update with boosting type.
    """
    if "boosting_type" not in search_space:
        return

    best_boosting_type = str(best_params["boosting_type"])
    best_string_params["boosting_type"] = best_boosting_type
    if best_boosting_type == "dart":
        if "drop_rate" in search_space:
            best_float_params["drop_rate"] = float(best_params["drop_rate"])
        if "skip_drop" in search_space:
            best_float_params["skip_drop"] = float(best_params["skip_drop"])
        if "feature_fraction" in search_space:
            best_float_params["feature_fraction"] = float(best_params["feature_fraction"])


class OptunaLightGBMOptimizer:
    """LightGBM hyperparameter optimizer using Optuna TPE."""

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
        search_space: LightGBMSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization using Optuna TPE."""
        create_study, tpe_sampler, median_pruner = _hooks.optuna_factories()

        self._trials_complete = 0
        self._trials_pruned = 0
        self._trials_failed = 0

        start_time = time.perf_counter()

        _log.info(
            "Starting LightGBM Optuna optimization",
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

            # Note: max_depth is not sampled for LightGBM. The objective uses
            # max_depth=-1 (unlimited) to let num_leaves control tree complexity.
            int_params: SampledIntParams = {
                "n_estimators": sample_param_int(
                    trial, "n_estimators", search_space["n_estimators"]
                ),
                "num_leaves": sample_param_int(trial, "num_leaves", search_space["num_leaves"]),
            }

            float_params: SampledFloatParams = {
                "learning_rate": sample_param_float(
                    trial, "learning_rate", search_space["learning_rate"]
                ),
                "subsample": sample_param_float(trial, "subsample", search_space["subsample"]),
                "colsample_bytree": sample_param_float(
                    trial, "colsample_bytree", search_space["colsample_bytree"]
                ),
                "reg_alpha": sample_param_float(trial, "reg_alpha", search_space["reg_alpha"]),
                "reg_lambda": sample_param_float(trial, "reg_lambda", search_space["reg_lambda"]),
            }

            # Sample optional DART params if present in search space
            string_params: SampledStringParams = {}
            _sample_lightgbm_dart_params(trial, search_space, float_params, string_params)

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
                    "num_leaves": int_params.get("num_leaves"),
                    "learning_rate": float_params.get("learning_rate"),
                    "boosting_type": string_params.get("boosting_type"),
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
            "num_leaves": int(best_params["num_leaves"]),
        }
        best_float_params: SampledFloatParams = {
            "learning_rate": float(best_params["learning_rate"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "reg_alpha": float(best_params["reg_alpha"]),
            "reg_lambda": float(best_params["reg_lambda"]),
        }

        # Extract best string params and conditional DART float params
        best_string_params: SampledStringParams = {}
        _extract_lightgbm_dart_best_params(
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
                "best_num_leaves": best_int_params.get("num_leaves"),
                "best_learning_rate": best_float_params.get("learning_rate"),
                "best_boosting_type": best_string_params.get("boosting_type"),
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


def create_lightgbm_optimizer() -> OptunaLightGBMOptimizer:
    """Create a LightGBM hyperparameter optimizer."""
    return OptunaLightGBMOptimizer()


__all__ = [
    "OptunaLightGBMOptimizer",
    "create_lightgbm_optimizer",
]
