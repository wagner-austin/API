"""LSTM Optuna optimizer.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ..protocol import ObjectiveProtocol, TrialCallbackProtocol
from ..types import (
    LSTMSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
)
from . import _hooks
from ._protocols import OptunaPrunerProtocol, OptunaTrialProtocol
from ._sampling import sample_param_float, sample_param_int

_log = get_logger(__name__)


class OptunaLSTMOptimizer:
    """LSTM hyperparameter optimizer using Optuna TPE."""

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
        search_space: LSTMSearchSpace,
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
            "Starting LSTM Optuna optimization",
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
                "hidden_size": sample_param_int(trial, "hidden_size", search_space["hidden_size"]),
                "num_layers": sample_param_int(trial, "num_layers", search_space["num_layers"]),
                "batch_size": sample_param_int(trial, "batch_size", search_space["batch_size"]),
            }

            float_params: SampledFloatParams = {
                "learning_rate": sample_param_float(
                    trial, "learning_rate", search_space["learning_rate"]
                ),
                "dropout": sample_param_float(trial, "dropout", search_space["dropout"]),
            }

            # LSTM has no string params
            string_params: SampledStringParams = {}

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
                    "hidden_size": int_params.get("hidden_size"),
                    "num_layers": int_params.get("num_layers"),
                    "learning_rate": float_params.get("learning_rate"),
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
            "hidden_size": int(best_params["hidden_size"]),
            "num_layers": int(best_params["num_layers"]),
            "batch_size": int(best_params["batch_size"]),
        }
        best_float_params: SampledFloatParams = {
            "learning_rate": float(best_params["learning_rate"]),
            "dropout": float(best_params["dropout"]),
        }

        # LSTM has no string params
        best_string_params: SampledStringParams = {}

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
                "best_hidden_size": best_int_params.get("hidden_size"),
                "best_num_layers": best_int_params.get("num_layers"),
                "best_learning_rate": best_float_params.get("learning_rate"),
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


def create_lstm_optimizer() -> OptunaLSTMOptimizer:
    """Create an LSTM hyperparameter optimizer."""
    return OptunaLSTMOptimizer()


__all__ = [
    "OptunaLSTMOptimizer",
    "create_lstm_optimizer",
]
