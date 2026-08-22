"""Shared fixtures and helpers for test_strategies splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.finetuning.types import (
    FineTuningConfig,
    StageConfig,
)
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    XGBoostSearchSpace,
)


def _make_features(n_samples: int, n_features: int) -> NDArray[np.float64]:
    """Create feature matrix."""
    rng = np.random.default_rng(42)
    result: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    return result


def _make_labels(n_samples: int) -> NDArray[np.int64]:
    """Create binary label array."""
    rng = np.random.default_rng(42)
    result: NDArray[np.int64] = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    return result


def _make_xgboost_search_space() -> XGBoostSearchSpace:
    """Create a simple XGBoost search space."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=6, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=100, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.1, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
    )


def _make_finetuning_config(n_stages: int = 2, trials_per_stage: int = 3) -> FineTuningConfig:
    """Create fine-tuning config with specified stages."""
    stages: list[StageConfig] = []

    if n_stages >= 1:
        stages.append(
            StageConfig(
                stage_name="exploration",
                n_trials=trials_per_stage,
                search_radius=1.0,
                use_previous_best=False,
            )
        )

    if n_stages >= 2:
        stages.append(
            StageConfig(
                stage_name="refinement",
                n_trials=trials_per_stage,
                search_radius=0.5,
                use_previous_best=True,
            )
        )

    if n_stages >= 3:
        stages.append(
            StageConfig(
                stage_name="final",
                n_trials=trials_per_stage,
                search_radius=0.25,
                use_previous_best=True,
            )
        )

    return FineTuningConfig(
        stages=tuple(stages),
        random_state=42,
        early_stop_threshold=0.001,
        max_total_trials=n_stages * trials_per_stage + 10,
    )


def _make_prior_summary(best_value: float = 0.80) -> OptimizationSummary:
    """Create prior optimization summary for warm-start testing."""
    return OptimizationSummary(
        best_trial_number=0,
        best_value=best_value,
        best_int_params=SampledIntParams(max_depth=5, n_estimators=80),
        best_float_params=SampledFloatParams(
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.5,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        best_string_params=SampledStringParams(),
        n_trials_total=10,
        n_trials_complete=10,
        n_trials_pruned=0,
        n_trials_failed=0,
        total_duration_seconds=5.0,
    )


def _dummy_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> float:
    """Dummy objective that returns a random value."""
    rng = np.random.default_rng(random_state + int_params.get("max_depth", 0))
    return float(rng.uniform(0.5, 0.9))
