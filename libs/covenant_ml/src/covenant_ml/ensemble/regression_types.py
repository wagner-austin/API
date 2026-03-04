"""Type definitions for regression ensemble models.

Parallel to types.py (classification). Key differences:
- labels: NDArray[np.float64] (continuous, not int64 binary)
- Metrics: neg_rmse, neg_mae, r_squared (not amex)
- No class-based assumptions

All types are immutable TypedDicts with strict typing.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from covenant_ml.ensemble.types import EnsembleWeights, ModelOOFPredictions


class RegressionEnsembleOOFData(TypedDict):
    """Combined OOF predictions from multiple regression models.

    Attributes:
        model_predictions: Tuple of OOF predictions from each model.
        labels: True continuous labels for all samples, shape (n_samples,).
        n_samples: Total number of samples.
        n_models: Number of models in ensemble.
    """

    model_predictions: tuple[ModelOOFPredictions, ...]
    labels: NDArray[np.float64]
    n_samples: int
    n_models: int


class RegressionOptimizationConfig(TypedDict):
    """Configuration for regression weight optimization.

    Attributes:
        metric: Metric to optimize.
        method: Scipy optimization method.
        max_iterations: Maximum optimization iterations.
        tolerance: Convergence tolerance.
        random_state: Random seed for reproducibility.
    """

    metric: Literal["neg_rmse", "neg_mae", "r_squared"]
    method: Literal["SLSQP", "trust-constr"]
    max_iterations: int
    tolerance: float
    random_state: int


class RegressionOptimizationResult(TypedDict):
    """Result of regression ensemble weight optimization.

    Attributes:
        weights: Optimized ensemble weights.
        best_score: Best metric score achieved (on internal scale).
        n_iterations: Number of optimization iterations.
        converged: Whether optimization converged.
        initial_score: Score before optimization (equal weights).
    """

    weights: EnsembleWeights
    best_score: float
    n_iterations: int
    converged: bool
    initial_score: float


def make_default_regression_optimization_config(
    random_state: int = 42,
) -> RegressionOptimizationConfig:
    """Create default regression optimization configuration.

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        Default RegressionOptimizationConfig with neg_rmse metric and SLSQP.
    """
    return RegressionOptimizationConfig(
        metric="neg_rmse",
        method="SLSQP",
        max_iterations=1000,
        tolerance=1e-8,
        random_state=random_state,
    )


__all__ = [
    "RegressionEnsembleOOFData",
    "RegressionOptimizationConfig",
    "RegressionOptimizationResult",
    "make_default_regression_optimization_config",
]
