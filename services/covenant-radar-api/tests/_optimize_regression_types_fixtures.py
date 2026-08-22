"""Shared fixtures and helpers for test_optimize_regression_types splits."""

from __future__ import annotations

from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types import RegressorBackendName

from covenant_radar_api.worker.optimize_regression_results import (
    UnifiedRegressionOptimizationResult,
)
from covenant_radar_api.worker.optimize_regression_types import (
    UnifiedRegressionOptimizeParseResult,
)


def _make_regression_parse_result(
    backend: RegressorBackendName = "xgboost_reg",
    dataset: str = "us_bankruptcy",
) -> UnifiedRegressionOptimizeParseResult:
    """Create a valid UnifiedRegressionOptimizeParseResult for testing.

    Args:
        backend: Regressor backend name.
        dataset: Dataset name.

    Returns:
        Valid UnifiedRegressionOptimizeParseResult.
    """
    return UnifiedRegressionOptimizeParseResult(
        backend=backend,
        dataset=dataset,
        n_trials=50,
        timeout_seconds=None,
        device="cpu",
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


def _make_regression_optimization_result(
    backend: RegressorBackendName = "xgboost_reg",
) -> UnifiedRegressionOptimizationResult:
    """Create a valid UnifiedRegressionOptimizationResult for testing.

    Args:
        backend: Regressor backend name.

    Returns:
        Valid UnifiedRegressionOptimizationResult.
    """
    return UnifiedRegressionOptimizationResult(
        backend=backend,
        status="complete",
        dataset="us_bankruptcy",
        n_samples=1000,
        n_features=18,
        feature_preset="none",
        n_trials_complete=50,
        n_trials_pruned=5,
        n_trials_failed=0,
        best_trial_number=37,
        best_value=-0.123,
        best_int_params=SampledIntParams(max_depth=6, n_estimators=100),
        best_float_params=SampledFloatParams(
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=1.5,
            subsample=0.9,
            colsample_bytree=0.7,
        ),
        best_string_params=SampledStringParams(),
        duration_seconds=120.5,
    )
