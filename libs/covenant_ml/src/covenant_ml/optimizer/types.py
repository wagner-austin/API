"""Type definitions for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
Defines search space configurations, trial results, and optimization summaries.
"""

from __future__ import annotations

from typing import Literal, TypedDict


# Parameter specification types
class FloatRangeSpec(TypedDict, total=True):
    """Continuous float parameter range for sampling."""

    param_type: Literal["float"]
    low: float
    high: float
    log_scale: bool  # If True, sample in log scale (for learning rates, etc.)


class IntRangeSpec(TypedDict, total=True):
    """Discrete integer parameter range for sampling."""

    param_type: Literal["int"]
    low: int
    high: int
    log_scale: bool


class CategoricalFloatSpec(TypedDict, total=True):
    """Categorical parameter with fixed float choices."""

    param_type: Literal["categorical_float"]
    choices: tuple[float, ...]


class CategoricalIntSpec(TypedDict, total=True):
    """Categorical parameter with fixed int choices."""

    param_type: Literal["categorical_int"]
    choices: tuple[int, ...]


# Union of all parameter specification types
ParamSpec = FloatRangeSpec | IntRangeSpec | CategoricalFloatSpec | CategoricalIntSpec


class XGBoostSearchSpace(TypedDict, total=True):
    """Search space for XGBoost hyperparameters.

    Each field maps to a ParamSpec defining the sampling range.
    """

    max_depth: IntRangeSpec | CategoricalIntSpec
    n_estimators: IntRangeSpec | CategoricalIntSpec
    learning_rate: FloatRangeSpec | CategoricalFloatSpec
    reg_alpha: FloatRangeSpec | CategoricalFloatSpec
    reg_lambda: FloatRangeSpec | CategoricalFloatSpec
    subsample: FloatRangeSpec | CategoricalFloatSpec
    colsample_bytree: FloatRangeSpec | CategoricalFloatSpec


class MLPSearchSpace(TypedDict, total=True):
    """Search space for MLP hyperparameters."""

    n_layers: IntRangeSpec | CategoricalIntSpec
    hidden_size: IntRangeSpec | CategoricalIntSpec
    learning_rate: FloatRangeSpec | CategoricalFloatSpec
    dropout: FloatRangeSpec | CategoricalFloatSpec
    batch_size: IntRangeSpec | CategoricalIntSpec


# Trial state enumeration
TrialState = Literal["complete", "pruned", "failed", "running"]


class TrialResult(TypedDict, total=True):
    """Result of a single optimization trial."""

    trial_number: int
    params_max_depth: int
    params_n_estimators: int
    params_learning_rate: float
    params_reg_alpha: float
    params_reg_lambda: float
    params_subsample: float
    params_colsample_bytree: float
    value: float  # The objective value (validation AUC to maximize)
    state: TrialState
    duration_seconds: float


class OptimizationSummary(TypedDict, total=True):
    """Summary of a completed optimization study."""

    best_trial_number: int
    best_value: float
    best_max_depth: int
    best_n_estimators: int
    best_learning_rate: float
    best_reg_alpha: float
    best_reg_lambda: float
    best_subsample: float
    best_colsample_bytree: float
    n_trials_total: int
    n_trials_complete: int
    n_trials_pruned: int
    n_trials_failed: int
    total_duration_seconds: float


class OptimizationConfig(TypedDict, total=True):
    """Configuration for an optimization run."""

    n_trials: int
    timeout_seconds: int | None  # None = no timeout
    n_startup_trials: int  # Random trials before TPE kicks in
    random_state: int
    direction: Literal["maximize", "minimize"]
    pruning_enabled: bool
    train_ratio: float
    val_ratio: float
    test_ratio: float


__all__ = [
    "CategoricalFloatSpec",
    "CategoricalIntSpec",
    "FloatRangeSpec",
    "IntRangeSpec",
    "MLPSearchSpace",
    "OptimizationConfig",
    "OptimizationSummary",
    "ParamSpec",
    "TrialResult",
    "TrialState",
    "XGBoostSearchSpace",
]
