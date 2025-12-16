"""Optuna-based hyperparameter optimizer for XGBoost.

Strict typing only: no Any, no casts, no stubs.
Uses Optuna's TPE (Tree-structured Parzen Estimator) for Bayesian optimization.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from .protocol import TrialCallbackProtocol, XGBoostObjectiveProtocol
from .types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    OptimizationConfig,
    OptimizationSummary,
    TrialResult,
    XGBoostSearchSpace,
)

_log = get_logger(__name__)


# Optuna protocol definitions (avoid importing optuna at module level)
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
    """Protocol for optuna module.

    We only define create_study here; samplers and pruners are accessed
    via getattr with direct Protocol type assignment to avoid naming issues.
    """

    @property
    def create_study(self) -> OptunaCreateStudyProtocol: ...


# Hook for optuna module access (set at runtime)
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
    Production code should call set_optuna_module_hook() with the real
    Optuna factories at startup.
    Tests should set a fake implementation.

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


def _sample_param_int(
    trial: OptunaTrialProtocol,
    name: str,
    spec: IntRangeSpec | CategoricalIntSpec,
) -> int:
    """Sample integer parameter from trial.

    Args:
        trial: Optuna trial object
        name: Parameter name
        spec: Parameter specification

    Returns:
        Sampled integer value
    """
    if spec["param_type"] == "int":
        int_spec: IntRangeSpec = spec
        return trial.suggest_int(
            name,
            int_spec["low"],
            int_spec["high"],
            log=int_spec["log_scale"],
        )
    # categorical_int
    cat_spec: CategoricalIntSpec = spec
    result = trial.suggest_categorical(name, cat_spec["choices"])
    # suggest_categorical returns float | int, we know it's int for int choices
    return int(result)


def _sample_param_float(
    trial: OptunaTrialProtocol,
    name: str,
    spec: FloatRangeSpec | CategoricalFloatSpec,
) -> float:
    """Sample float parameter from trial.

    Args:
        trial: Optuna trial object
        name: Parameter name
        spec: Parameter specification

    Returns:
        Sampled float value
    """
    if spec["param_type"] == "float":
        float_spec: FloatRangeSpec = spec
        return trial.suggest_float(
            name,
            float_spec["low"],
            float_spec["high"],
            log=float_spec["log_scale"],
        )
    # categorical_float
    cat_spec: CategoricalFloatSpec = spec
    result = trial.suggest_categorical(name, cat_spec["choices"])
    return float(result)


class OptunaXGBoostOptimizer:
    """XGBoost hyperparameter optimizer using Optuna TPE.

    Uses Tree-structured Parzen Estimator for efficient Bayesian optimization.
    Supports early pruning of unpromising trials.
    """

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
        objective: XGBoostObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Run hyperparameter optimization using Optuna TPE.

        Args:
            x_features: Feature matrix (n_samples, n_features)
            y_labels: Binary labels (n_samples,)
            feature_names: Names for each feature column
            search_space: Parameter ranges to search
            config: Optimization settings
            objective: Function to evaluate hyperparameters
            trial_callback: Optional callback after each trial

        Returns:
            Summary with best hyperparameters and trial statistics
        """
        create_study, tpe_sampler, median_pruner = _get_optuna_factories()

        # Reset counters
        self._trials_complete = 0
        self._trials_pruned = 0
        self._trials_failed = 0

        start_time = time.perf_counter()

        _log.info(
            "Starting Optuna optimization",
            extra={
                "n_trials": config["n_trials"],
                "n_startup_trials": config["n_startup_trials"],
                "direction": config["direction"],
                "pruning_enabled": config["pruning_enabled"],
            },
        )

        # Create sampler
        sampler = tpe_sampler(
            seed=config["random_state"],
            n_startup_trials=config["n_startup_trials"],
        )

        # Create pruner if enabled
        pruner: OptunaPrunerProtocol | None = None
        if config["pruning_enabled"]:
            pruner = median_pruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            )

        # Create study
        study = create_study(
            direction=config["direction"],
            sampler=sampler,
            pruner=pruner,
        )

        # Define objective function wrapper
        def optuna_objective(trial: OptunaTrialProtocol) -> float:
            trial_start = time.perf_counter()

            # Sample hyperparameters
            max_depth = _sample_param_int(trial, "max_depth", search_space["max_depth"])
            n_estimators = _sample_param_int(trial, "n_estimators", search_space["n_estimators"])
            learning_rate = _sample_param_float(
                trial, "learning_rate", search_space["learning_rate"]
            )
            reg_alpha = _sample_param_float(trial, "reg_alpha", search_space["reg_alpha"])
            reg_lambda = _sample_param_float(trial, "reg_lambda", search_space["reg_lambda"])
            subsample = _sample_param_float(trial, "subsample", search_space["subsample"])
            colsample_bytree = _sample_param_float(
                trial, "colsample_bytree", search_space["colsample_bytree"]
            )

            # Evaluate objective
            val_auc = objective(
                x_features,
                y_labels,
                feature_names,
                max_depth,
                n_estimators,
                learning_rate,
                reg_alpha,
                reg_lambda,
                subsample,
                colsample_bytree,
                config["random_state"],
                config["train_ratio"],
                config["val_ratio"],
                config["test_ratio"],
            )

            trial_duration = time.perf_counter() - trial_start
            self._trials_complete += 1

            # Build trial result
            result: TrialResult = {
                "trial_number": trial.number,
                "params_max_depth": max_depth,
                "params_n_estimators": n_estimators,
                "params_learning_rate": learning_rate,
                "params_reg_alpha": reg_alpha,
                "params_reg_lambda": reg_lambda,
                "params_subsample": subsample,
                "params_colsample_bytree": colsample_bytree,
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
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "duration_sec": trial_duration,
                },
            )

            return val_auc

        # Run optimization
        timeout_val: float | None = None
        if config["timeout_seconds"] is not None:
            timeout_val = float(config["timeout_seconds"])

        study.optimize(
            optuna_objective,
            n_trials=config["n_trials"],
            timeout=timeout_val,
        )

        total_duration = time.perf_counter() - start_time

        # Extract best parameters
        best_params = study.best_params
        best_max_depth = int(best_params["max_depth"])
        best_n_estimators = int(best_params["n_estimators"])
        best_learning_rate = float(best_params["learning_rate"])
        best_reg_alpha = float(best_params["reg_alpha"])
        best_reg_lambda = float(best_params["reg_lambda"])
        best_subsample = float(best_params["subsample"])
        best_colsample_bytree = float(best_params["colsample_bytree"])

        summary: OptimizationSummary = {
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_max_depth": best_max_depth,
            "best_n_estimators": best_n_estimators,
            "best_learning_rate": best_learning_rate,
            "best_reg_alpha": best_reg_alpha,
            "best_reg_lambda": best_reg_lambda,
            "best_subsample": best_subsample,
            "best_colsample_bytree": best_colsample_bytree,
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
                "best_max_depth": summary["best_max_depth"],
                "best_learning_rate": summary["best_learning_rate"],
                "n_trials_complete": summary["n_trials_complete"],
                "total_duration_sec": summary["total_duration_seconds"],
            },
        )

        return summary


def create_xgboost_optimizer() -> OptunaXGBoostOptimizer:
    """Create an XGBoost hyperparameter optimizer.

    Returns:
        OptunaXGBoostOptimizer instance ready for optimization
    """
    return OptunaXGBoostOptimizer()


__all__ = [
    "OptunaXGBoostOptimizer",
    "create_xgboost_optimizer",
    "set_optuna_module_hook",
    "use_real_optuna",
]
