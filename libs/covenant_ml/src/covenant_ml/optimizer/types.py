"""Type definitions for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
Defines search space configurations, trial results, and optimization summaries.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# =============================================================================
# Parameter Specification Types
# =============================================================================


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


# =============================================================================
# Sampled Parameter Value Types
# =============================================================================


class SampledIntParams(TypedDict, total=False):
    """Sampled integer parameter values.

    All fields optional - only present params are included.
    """

    max_depth: int
    n_estimators: int
    n_layers: int
    hidden_size: int
    num_layers: int
    batch_size: int
    num_leaves: int
    min_child_samples: int


class SampledFloatParams(TypedDict, total=False):
    """Sampled float parameter values.

    All fields optional - only present params are included.
    """

    learning_rate: float
    reg_alpha: float
    reg_lambda: float
    subsample: float
    colsample_bytree: float
    dropout: float


# =============================================================================
# Search Space Types
# =============================================================================


class XGBoostSearchSpace(TypedDict, total=True):
    """Search space for XGBoost hyperparameters."""

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


class LSTMSearchSpace(TypedDict, total=True):
    """Search space for LSTM hyperparameters."""

    hidden_size: IntRangeSpec | CategoricalIntSpec
    num_layers: IntRangeSpec | CategoricalIntSpec
    dropout: FloatRangeSpec | CategoricalFloatSpec
    learning_rate: FloatRangeSpec | CategoricalFloatSpec
    batch_size: IntRangeSpec | CategoricalIntSpec


class LightGBMSearchSpace(TypedDict, total=True):
    """Search space for LightGBM hyperparameters."""

    max_depth: IntRangeSpec | CategoricalIntSpec
    n_estimators: IntRangeSpec | CategoricalIntSpec
    num_leaves: IntRangeSpec | CategoricalIntSpec
    learning_rate: FloatRangeSpec | CategoricalFloatSpec
    subsample: FloatRangeSpec | CategoricalFloatSpec
    colsample_bytree: FloatRangeSpec | CategoricalFloatSpec
    reg_alpha: FloatRangeSpec | CategoricalFloatSpec
    reg_lambda: FloatRangeSpec | CategoricalFloatSpec


# Union of all backend-specific search spaces for generic optimizer interfaces
SearchSpace = XGBoostSearchSpace | MLPSearchSpace | LSTMSearchSpace | LightGBMSearchSpace


# =============================================================================
# Trial State and Results
# =============================================================================

TrialState = Literal["complete", "pruned", "failed", "running"]


class TrialResult(TypedDict, total=True):
    """Result of a single optimization trial."""

    trial_number: int
    int_params: SampledIntParams
    float_params: SampledFloatParams
    value: float  # The objective value (validation AUC to maximize)
    state: TrialState
    duration_seconds: float


class OptimizationSummary(TypedDict, total=True):
    """Summary of a completed optimization study."""

    best_trial_number: int
    best_value: float
    best_int_params: SampledIntParams
    best_float_params: SampledFloatParams
    n_trials_total: int
    n_trials_complete: int
    n_trials_pruned: int
    n_trials_failed: int
    total_duration_seconds: float


# =============================================================================
# Optimization Configuration
# =============================================================================


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
    "LSTMSearchSpace",
    "LightGBMSearchSpace",
    "MLPSearchSpace",
    "OptimizationConfig",
    "OptimizationSummary",
    "SampledFloatParams",
    "SampledIntParams",
    "SearchSpace",
    "TrialResult",
    "TrialState",
    "XGBoostSearchSpace",
]
