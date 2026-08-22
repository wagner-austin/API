"""Shared fixtures and helpers for test_optimize_types splits."""

from __future__ import annotations

from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types import BackendName

from covenant_radar_api.worker.optimize_types import (
    UnifiedOptimizationResult,
    UnifiedOptimizeParseResult,
)


def _make_parse_result(
    backend: BackendName = "xgboost",
    dataset: str = "taiwan",
) -> UnifiedOptimizeParseResult:
    """Create a valid UnifiedOptimizeParseResult for testing.

    Args:
        backend: Backend name.
        dataset: Dataset name.

    Returns:
        UnifiedOptimizeParseResult with all fields populated.
    """
    return UnifiedOptimizeParseResult(
        backend=backend,
        dataset=dataset,
        n_trials=50,
        timeout_seconds=3600,
        device="auto",
        feature_preset="none",
        random_state=42,
        early_stopping_rounds=10,
        n_jobs=-1,
        precision="fp32",
        nn_optimizer="adamw",
        n_epochs=50,
        early_stopping_patience=10,
        sequence_length=5,
        bidirectional=False,
    )


def _make_optimization_result(
    backend: BackendName = "xgboost",
) -> UnifiedOptimizationResult:
    """Create a valid UnifiedOptimizationResult for testing.

    Args:
        backend: Backend name.

    Returns:
        UnifiedOptimizationResult with all fields populated.
    """
    return UnifiedOptimizationResult(
        backend=backend,
        status="complete",
        dataset="taiwan",
        n_samples=6819,
        n_features=95,
        feature_preset="none",
        n_trials_complete=50,
        n_trials_pruned=3,
        n_trials_failed=1,
        best_trial_number=42,
        best_value=0.8765,
        best_int_params=SampledIntParams(max_depth=6, n_estimators=200),
        best_float_params=SampledFloatParams(learning_rate=0.05, reg_alpha=0.1),
        best_string_params=SampledStringParams(booster="gbtree"),
        duration_seconds=120.5,
    )
