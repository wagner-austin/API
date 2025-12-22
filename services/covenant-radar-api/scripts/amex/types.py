"""Type definitions for AMEX competition pipeline.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from covenant_ml.types import BackendName
from numpy.typing import NDArray


class AMEXPipelineConfig(TypedDict, total=True):
    """Configuration for AMEX competition pipeline.

    Attributes:
        backends: List of ML backends to train and ensemble.
        n_folds: Number of cross-validation folds.
        n_estimators: Number of boosting rounds (tree backends) or epochs (neural).
        learning_rate: Learning rate.
        aggregation: Time-series aggregation strategy.
        include_rank_features: Whether to compute rank features.
        include_diff_features: Whether to compute diff features.
        include_window_features: Whether to compute window features.
        window_sizes: Window sizes for window features.
        random_state: Random seed for reproducibility.
    """

    backends: tuple[BackendName, ...]
    n_folds: int
    n_estimators: int
    learning_rate: float
    aggregation: Literal["last", "first", "mean", "statistics"]
    include_rank_features: bool
    include_diff_features: bool
    include_window_features: bool
    window_sizes: tuple[int, ...]
    random_state: int


class ModelOOFResult(TypedDict, total=True):
    """Out-of-fold predictions from a single model.

    Attributes:
        model_name: Name of the model (backend name).
        oof_predictions: OOF predictions for all samples, shape (n_samples,).
        fold_indices: Fold index for each sample, shape (n_samples,).
        cv_scores: AMEX score for each fold.
        mean_cv_score: Mean AMEX score across folds.
    """

    model_name: str
    oof_predictions: NDArray[np.float64]
    fold_indices: NDArray[np.int64]
    cv_scores: tuple[float, ...]
    mean_cv_score: float


class EnsembleResult(TypedDict, total=True):
    """Result of ensemble weight optimization.

    Attributes:
        model_names: Names of models in ensemble.
        weights: Optimized weights for each model.
        initial_score: AMEX score with equal weights.
        optimized_score: AMEX score with optimized weights.
        improvement: Score improvement from optimization.
    """

    model_names: tuple[str, ...]
    weights: tuple[float, ...]
    initial_score: float
    optimized_score: float
    improvement: float


class PipelineResult(TypedDict, total=True):
    """Complete result from AMEX pipeline.

    Attributes:
        n_samples_train: Number of training samples.
        n_samples_test: Number of test samples.
        n_features: Number of features.
        model_results: OOF results for each model.
        ensemble_result: Ensemble optimization result.
        submission_path: Path to submission CSV.
    """

    n_samples_train: int
    n_samples_test: int
    n_features: int
    model_results: tuple[ModelOOFResult, ...]
    ensemble_result: EnsembleResult
    submission_path: str


def make_default_config() -> AMEXPipelineConfig:
    """Create default AMEX pipeline configuration.

    Returns:
        Default configuration matching 1st place solution approach.
    """
    return AMEXPipelineConfig(
        backends=("lightgbm", "xgboost"),
        n_folds=5,
        n_estimators=1000,
        learning_rate=0.05,
        aggregation="statistics",
        include_rank_features=True,
        include_diff_features=True,
        include_window_features=True,
        window_sizes=(3, 6),
        random_state=42,
    )


__all__ = [
    "AMEXPipelineConfig",
    "EnsembleResult",
    "ModelOOFResult",
    "PipelineResult",
    "make_default_config",
]
