"""Type definitions for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
Defines search space configurations, trial results, and optimization summaries.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# =============================================================================
# Device Types
# =============================================================================

# User-facing device request (common across all backends)
DeviceRequest = Literal["cpu", "cuda", "auto"]

# LightGBM-specific device parameter
# Note: "cuda" is Linux-only; "gpu" uses OpenCL and works on all platforms
LightGBMDevice = Literal["cpu", "gpu", "cuda"]

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


class CategoricalStringSpec(TypedDict, total=True):
    """Categorical parameter with fixed string choices.

    Used for parameters like boosting_type that take string values.
    """

    param_type: Literal["categorical_str"]
    choices: tuple[str, ...]


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
    # ClearGBM-specific params
    min_samples_split: int
    min_samples_leaf: int
    max_bins: int


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
    # DART-specific params (LightGBM and XGBoost)
    drop_rate: float
    skip_drop: float
    # XGBoost DART-specific
    rate_drop: float
    # LightGBM DART-specific: very low feature fraction for strong regularization
    feature_fraction: float


class SampledStringParams(TypedDict, total=False):
    """Sampled string parameter values.

    All fields optional - only present params are included.
    """

    # LightGBM boosting type: "gbdt" or "dart"
    boosting_type: str
    # XGBoost booster: "gbtree" or "dart"
    booster: str


# =============================================================================
# Search Space Types
# =============================================================================


class _XGBoostSearchSpaceRequired(TypedDict, total=True):
    """Required XGBoost search space parameters."""

    max_depth: IntRangeSpec | CategoricalIntSpec
    n_estimators: IntRangeSpec | CategoricalIntSpec
    learning_rate: FloatRangeSpec | CategoricalFloatSpec
    reg_alpha: FloatRangeSpec | CategoricalFloatSpec
    reg_lambda: FloatRangeSpec | CategoricalFloatSpec
    subsample: FloatRangeSpec | CategoricalFloatSpec
    colsample_bytree: FloatRangeSpec | CategoricalFloatSpec


class XGBoostSearchSpace(_XGBoostSearchSpaceRequired, total=False):
    """Search space for XGBoost hyperparameters with optional DART support.

    Required params are inherited from _XGBoostSearchSpaceRequired.
    DART params are optional and only used when booster includes "dart".
    """

    # Optional: booster type ("gbtree" or "dart")
    booster: CategoricalStringSpec
    # DART-specific params (only used when booster is "dart")
    rate_drop: FloatRangeSpec | CategoricalFloatSpec
    skip_drop: FloatRangeSpec | CategoricalFloatSpec


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


class _LightGBMSearchSpaceRequired(TypedDict, total=True):
    """Required LightGBM search space parameters.

    Note: max_depth is intentionally excluded. LightGBM uses leaf-wise growth
    where num_leaves is the primary complexity control. Using max_depth=-1
    (unlimited) with num_leaves avoids constraint conflicts that can cause
    training failures when num_leaves > 2^max_depth.
    """

    n_estimators: IntRangeSpec | CategoricalIntSpec
    num_leaves: IntRangeSpec | CategoricalIntSpec
    learning_rate: FloatRangeSpec | CategoricalFloatSpec
    subsample: FloatRangeSpec | CategoricalFloatSpec
    colsample_bytree: FloatRangeSpec | CategoricalFloatSpec
    reg_alpha: FloatRangeSpec | CategoricalFloatSpec
    reg_lambda: FloatRangeSpec | CategoricalFloatSpec


class LightGBMSearchSpace(_LightGBMSearchSpaceRequired, total=False):
    """Search space for LightGBM hyperparameters with optional DART support.

    Required params are inherited from _LightGBMSearchSpaceRequired.
    DART params are optional and only used when boosting_type includes "dart".
    """

    # Optional: boosting algorithm choice ("gbdt" or "dart")
    boosting_type: CategoricalStringSpec
    # DART-specific params (only used when boosting_type is "dart")
    drop_rate: FloatRangeSpec | CategoricalFloatSpec
    skip_drop: FloatRangeSpec | CategoricalFloatSpec
    # DART-specific: very low feature fraction (0.02-0.1) for strong regularization
    feature_fraction: FloatRangeSpec | CategoricalFloatSpec


class ClearGBMSearchSpace(TypedDict, total=True):
    """Search space for ClearGBM hyperparameters.

    ClearGBM is a pure Python from-scratch gradient boosting implementation
    with built-in interpretability features. The search space includes:
    - n_estimators: Number of boosting rounds
    - max_depth: Maximum tree depth
    - learning_rate: Shrinkage factor
    - min_samples_split: Minimum samples to split a node
    - min_samples_leaf: Minimum samples in a leaf
    - max_bins: Histogram bins for split finding
    - subsample: Row subsampling ratio
    """

    n_estimators: IntRangeSpec | CategoricalIntSpec
    max_depth: IntRangeSpec | CategoricalIntSpec
    learning_rate: FloatRangeSpec | CategoricalFloatSpec
    min_samples_split: IntRangeSpec | CategoricalIntSpec
    min_samples_leaf: IntRangeSpec | CategoricalIntSpec
    max_bins: IntRangeSpec | CategoricalIntSpec
    subsample: FloatRangeSpec | CategoricalFloatSpec


# Union of all backend-specific search spaces for generic optimizer interfaces
SearchSpace = (
    XGBoostSearchSpace
    | MLPSearchSpace
    | LSTMSearchSpace
    | LightGBMSearchSpace
    | ClearGBMSearchSpace
)


# =============================================================================
# Trial State and Results
# =============================================================================

TrialState = Literal["complete", "pruned", "failed", "running"]


class TrialResult(TypedDict, total=True):
    """Result of a single optimization trial."""

    trial_number: int
    int_params: SampledIntParams
    float_params: SampledFloatParams
    string_params: SampledStringParams
    value: float  # The objective value (validation AUC to maximize)
    state: TrialState
    duration_seconds: float


class OptimizationSummary(TypedDict, total=True):
    """Summary of a completed optimization study."""

    best_trial_number: int
    best_value: float
    best_int_params: SampledIntParams
    best_float_params: SampledFloatParams
    best_string_params: SampledStringParams
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
    timeout_seconds: float | None  # None = no timeout, float for fractional seconds
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
    "CategoricalStringSpec",
    "ClearGBMSearchSpace",
    "DeviceRequest",
    "FloatRangeSpec",
    "IntRangeSpec",
    "LSTMSearchSpace",
    "LightGBMDevice",
    "LightGBMSearchSpace",
    "MLPSearchSpace",
    "OptimizationConfig",
    "OptimizationSummary",
    "SampledFloatParams",
    "SampledIntParams",
    "SampledStringParams",
    "SearchSpace",
    "TrialResult",
    "TrialState",
    "XGBoostSearchSpace",
]
