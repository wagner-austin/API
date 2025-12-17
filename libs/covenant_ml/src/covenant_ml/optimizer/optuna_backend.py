"""Optuna-based hyperparameter optimizer.

Strict typing only: no Any, no casts, no stubs.
Uses Optuna's TPE (Tree-structured Parzen Estimator) for Bayesian optimization.
Supports XGBoost, MLP, LSTM, and LightGBM search spaces.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from .protocol import ObjectiveProtocol, TrialCallbackProtocol
from .types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    TrialResult,
    XGBoostSearchSpace,
)

_log = get_logger(__name__)


# =============================================================================
# Optuna Protocol Definitions
# =============================================================================


class OptunaSamplerProtocol(Protocol):
    """Protocol for Optuna sampler."""

    ...


class OptunaTrialProtocol(Protocol):
    """Protocol for Optuna trial object."""

    @property
    def number(self) -> int: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
    ) -> int: ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float: ...

    def suggest_categorical(
        self,
        name: str,
        choices: tuple[float, ...] | tuple[int, ...],
    ) -> float | int: ...

    def report(self, value: float, step: int) -> None: ...

    def should_prune(self) -> bool: ...


class OptunaStudyProtocol(Protocol):
    """Protocol for Optuna study object."""

    @property
    def best_trial(self) -> OptunaTrialProtocol: ...

    @property
    def best_value(self) -> float: ...

    @property
    def best_params(self) -> dict[str, float | int]: ...

    def optimize(
        self,
        func: Callable[[OptunaTrialProtocol], float],
        n_trials: int,
        timeout: float | None = None,
        callbacks: list[Callable[[OptunaStudyProtocol, OptunaTrialProtocol], None]] | None = None,
    ) -> None: ...

    def get_trials(
        self,
        deepcopy: bool = True,
        states: tuple[str, ...] | None = None,
    ) -> list[OptunaTrialProtocol]: ...


class OptunaCreateStudyProtocol(Protocol):
    """Protocol for optuna.create_study function."""

    def __call__(
        self,
        *,
        direction: str,
        sampler: OptunaSamplerProtocol,
        pruner: OptunaPrunerProtocol | None = None,
    ) -> OptunaStudyProtocol: ...


class OptunaTPESamplerProtocol(Protocol):
    """Protocol for TPESampler constructor."""

    def __call__(
        self,
        *,
        seed: int,
        n_startup_trials: int,
    ) -> OptunaSamplerProtocol: ...


class OptunaPrunerProtocol(Protocol):
    """Protocol for Optuna pruner."""

    ...


class OptunaMedianPrunerProtocol(Protocol):
    """Protocol for MedianPruner constructor."""

    def __call__(
        self,
        *,
        n_startup_trials: int,
        n_warmup_steps: int,
    ) -> OptunaPrunerProtocol: ...


class OptunaModuleProtocol(Protocol):
    """Protocol for optuna module."""

    @property
    def create_study(self) -> OptunaCreateStudyProtocol: ...


# =============================================================================
# Optuna Module Hook
# =============================================================================

_optuna_module_hook: (
    Callable[
        [],
        tuple[
            OptunaCreateStudyProtocol,
            OptunaTPESamplerProtocol,
            OptunaMedianPrunerProtocol,
        ],
    ]
    | None
) = None


def set_optuna_module_hook(
    hook: Callable[
        [],
        tuple[
            OptunaCreateStudyProtocol,
            OptunaTPESamplerProtocol,
            OptunaMedianPrunerProtocol,
        ],
    ]
    | None,
) -> None:
    """Set hook for Optuna module access.

    Production code sets this to real Optuna at startup.
    Tests can set a fake implementation.

    Args:
        hook: Callable returning (create_study, TPESampler, MedianPruner)
    """
    global _optuna_module_hook
    _optuna_module_hook = hook


def _get_optuna_factories() -> tuple[
    OptunaCreateStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaMedianPrunerProtocol,
]:
    """Get Optuna factories via hook.

    The hook MUST be set before calling this function.

    Returns:
        Tuple of (create_study, TPESampler, MedianPruner) factories

    Raises:
        RuntimeError: If hook is not set
    """
    if _optuna_module_hook is None:
        raise RuntimeError(
            "Optuna module hook not set. "
            "Call set_optuna_module_hook() or use_real_optuna() before optimization."
        )
    return _optuna_module_hook()


def _real_optuna_factories() -> tuple[
    OptunaCreateStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaMedianPrunerProtocol,
]:
    """Get real Optuna factories via dynamic import.

    Uses __import__ with Protocol type assignment to avoid Any types.

    Returns:
        Tuple of (create_study, TPESampler, MedianPruner) factories
    """
    optuna_mod: OptunaModuleProtocol = __import__("optuna")
    create_study: OptunaCreateStudyProtocol = optuna_mod.create_study

    samplers_submod = __import__("optuna.samplers", fromlist=["TPESampler"])
    tpe_sampler: OptunaTPESamplerProtocol = samplers_submod.TPESampler

    pruners_submod = __import__("optuna.pruners", fromlist=["MedianPruner"])
    median_pruner: OptunaMedianPrunerProtocol = pruners_submod.MedianPruner

    return create_study, tpe_sampler, median_pruner


def use_real_optuna() -> None:
    """Set the hook to use real Optuna.

    Call this at application startup before running optimization.
    """
    set_optuna_module_hook(_real_optuna_factories)


# =============================================================================
# Parameter Sampling Functions
# =============================================================================


def _sample_param_int(
    trial: OptunaTrialProtocol,
    name: str,
    spec: IntRangeSpec | CategoricalIntSpec,
) -> int:
    """Sample integer parameter from trial."""
    if spec["param_type"] == "int":
        int_spec: IntRangeSpec = spec
        return trial.suggest_int(
            name,
            int_spec["low"],
            int_spec["high"],
            log=int_spec["log_scale"],
        )
    cat_spec: CategoricalIntSpec = spec
    result = trial.suggest_categorical(name, cat_spec["choices"])
    return int(result)


def _sample_param_float(
    trial: OptunaTrialProtocol,
    name: str,
    spec: FloatRangeSpec | CategoricalFloatSpec,
) -> float:
    """Sample float parameter from trial."""
    if spec["param_type"] == "float":
        float_spec: FloatRangeSpec = spec
        return trial.suggest_float(
            name,
            float_spec["low"],
            float_spec["high"],
            log=float_spec["log_scale"],
        )
    cat_spec: CategoricalFloatSpec = spec
    result = trial.suggest_categorical(name, cat_spec["choices"])
    return float(result)


# =============================================================================
# XGBoost Optimizer
# =============================================================================


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
        create_study, tpe_sampler, median_pruner = _get_optuna_factories()

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
                "max_depth": _sample_param_int(trial, "max_depth", search_space["max_depth"]),
                "n_estimators": _sample_param_int(
                    trial, "n_estimators", search_space["n_estimators"]
                ),
            }

            float_params: SampledFloatParams = {
                "learning_rate": _sample_param_float(
                    trial, "learning_rate", search_space["learning_rate"]
                ),
                "reg_alpha": _sample_param_float(trial, "reg_alpha", search_space["reg_alpha"]),
                "reg_lambda": _sample_param_float(trial, "reg_lambda", search_space["reg_lambda"]),
                "subsample": _sample_param_float(trial, "subsample", search_space["subsample"]),
                "colsample_bytree": _sample_param_float(
                    trial, "colsample_bytree", search_space["colsample_bytree"]
                ),
            }

            val_auc = objective(
                x_features,
                y_labels,
                feature_names,
                int_params,
                float_params,
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

        summary: OptimizationSummary = {
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_int_params": best_int_params,
            "best_float_params": best_float_params,
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
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


# =============================================================================
# MLP Optimizer
# =============================================================================


class OptunaMLPOptimizer:
    """MLP hyperparameter optimizer using Optuna TPE."""

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
        search_space: MLPSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization using Optuna TPE."""
        create_study, tpe_sampler, median_pruner = _get_optuna_factories()

        self._trials_complete = 0
        self._trials_pruned = 0
        self._trials_failed = 0

        start_time = time.perf_counter()

        _log.info(
            "Starting MLP Optuna optimization",
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
                "n_layers": _sample_param_int(trial, "n_layers", search_space["n_layers"]),
                "hidden_size": _sample_param_int(trial, "hidden_size", search_space["hidden_size"]),
                "batch_size": _sample_param_int(trial, "batch_size", search_space["batch_size"]),
            }

            float_params: SampledFloatParams = {
                "learning_rate": _sample_param_float(
                    trial, "learning_rate", search_space["learning_rate"]
                ),
                "dropout": _sample_param_float(trial, "dropout", search_space["dropout"]),
            }

            val_auc = objective(
                x_features,
                y_labels,
                feature_names,
                int_params,
                float_params,
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
                    "n_layers": int_params.get("n_layers"),
                    "hidden_size": int_params.get("hidden_size"),
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
            "n_layers": int(best_params["n_layers"]),
            "hidden_size": int(best_params["hidden_size"]),
            "batch_size": int(best_params["batch_size"]),
        }
        best_float_params: SampledFloatParams = {
            "learning_rate": float(best_params["learning_rate"]),
            "dropout": float(best_params["dropout"]),
        }

        summary: OptimizationSummary = {
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_int_params": best_int_params,
            "best_float_params": best_float_params,
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
                "best_n_layers": best_int_params.get("n_layers"),
                "best_hidden_size": best_int_params.get("hidden_size"),
                "best_learning_rate": best_float_params.get("learning_rate"),
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


# =============================================================================
# LSTM Optimizer
# =============================================================================


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
        create_study, tpe_sampler, median_pruner = _get_optuna_factories()

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
                "hidden_size": _sample_param_int(trial, "hidden_size", search_space["hidden_size"]),
                "num_layers": _sample_param_int(trial, "num_layers", search_space["num_layers"]),
                "batch_size": _sample_param_int(trial, "batch_size", search_space["batch_size"]),
            }

            float_params: SampledFloatParams = {
                "learning_rate": _sample_param_float(
                    trial, "learning_rate", search_space["learning_rate"]
                ),
                "dropout": _sample_param_float(trial, "dropout", search_space["dropout"]),
            }

            val_auc = objective(
                x_features,
                y_labels,
                feature_names,
                int_params,
                float_params,
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

        summary: OptimizationSummary = {
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_int_params": best_int_params,
            "best_float_params": best_float_params,
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


# =============================================================================
# LightGBM Optimizer
# =============================================================================


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
        create_study, tpe_sampler, median_pruner = _get_optuna_factories()

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

            int_params: SampledIntParams = {
                "max_depth": _sample_param_int(trial, "max_depth", search_space["max_depth"]),
                "n_estimators": _sample_param_int(
                    trial, "n_estimators", search_space["n_estimators"]
                ),
                "num_leaves": _sample_param_int(trial, "num_leaves", search_space["num_leaves"]),
            }

            float_params: SampledFloatParams = {
                "learning_rate": _sample_param_float(
                    trial, "learning_rate", search_space["learning_rate"]
                ),
                "subsample": _sample_param_float(trial, "subsample", search_space["subsample"]),
                "colsample_bytree": _sample_param_float(
                    trial, "colsample_bytree", search_space["colsample_bytree"]
                ),
                "reg_alpha": _sample_param_float(trial, "reg_alpha", search_space["reg_alpha"]),
                "reg_lambda": _sample_param_float(trial, "reg_lambda", search_space["reg_lambda"]),
            }

            val_auc = objective(
                x_features,
                y_labels,
                feature_names,
                int_params,
                float_params,
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
                    "num_leaves": int_params.get("num_leaves"),
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
            "max_depth": int(best_params["max_depth"]),
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

        summary: OptimizationSummary = {
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_int_params": best_int_params,
            "best_float_params": best_float_params,
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
                "best_num_leaves": best_int_params.get("num_leaves"),
                "best_learning_rate": best_float_params.get("learning_rate"),
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


# =============================================================================
# Factory Functions
# =============================================================================


def create_xgboost_optimizer() -> OptunaXGBoostOptimizer:
    """Create an XGBoost hyperparameter optimizer."""
    return OptunaXGBoostOptimizer()


def create_mlp_optimizer() -> OptunaMLPOptimizer:
    """Create an MLP hyperparameter optimizer."""
    return OptunaMLPOptimizer()


def create_lstm_optimizer() -> OptunaLSTMOptimizer:
    """Create an LSTM hyperparameter optimizer."""
    return OptunaLSTMOptimizer()


def create_lightgbm_optimizer() -> OptunaLightGBMOptimizer:
    """Create a LightGBM hyperparameter optimizer."""
    return OptunaLightGBMOptimizer()


__all__ = [
    "OptunaLSTMOptimizer",
    "OptunaLightGBMOptimizer",
    "OptunaMLPOptimizer",
    "OptunaXGBoostOptimizer",
    "create_lightgbm_optimizer",
    "create_lstm_optimizer",
    "create_mlp_optimizer",
    "create_xgboost_optimizer",
    "set_optuna_module_hook",
    "use_real_optuna",
]
