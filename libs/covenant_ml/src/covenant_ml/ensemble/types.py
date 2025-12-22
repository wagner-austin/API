"""Type definitions for ensemble models.

Provides TypedDicts for ensemble configuration, OOF predictions, and
optimization results. All types are immutable and strictly typed.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray


class ModelOOFPredictions(TypedDict):
    """Out-of-fold predictions from a single model.

    Attributes:
        model_name: Identifier for this model.
        predictions: OOF probability predictions, shape (n_samples,).
        fold_indices: Which fold each sample was validated in, shape (n_samples,).
    """

    model_name: str
    predictions: NDArray[np.float64]
    fold_indices: NDArray[np.int64]


class EnsembleOOFData(TypedDict):
    """Combined OOF predictions from multiple models.

    Attributes:
        model_predictions: Tuple of OOF predictions from each model.
        labels: True labels for all samples, shape (n_samples,).
        n_samples: Total number of samples.
        n_models: Number of models in ensemble.
    """

    model_predictions: tuple[ModelOOFPredictions, ...]
    labels: NDArray[np.int64]
    n_samples: int
    n_models: int


class EnsembleWeights(TypedDict):
    """Optimized weights for ensemble models.

    Attributes:
        weights: Weight for each model, shape (n_models,). Sums to 1.0.
        model_names: Names of models in weight order.
    """

    weights: NDArray[np.float64]
    model_names: tuple[str, ...]


class OptimizationConfig(TypedDict):
    """Configuration for weight optimization.

    Attributes:
        metric: Metric to optimize. Currently only "amex" supported.
        method: Scipy optimization method.
        max_iterations: Maximum optimization iterations.
        tolerance: Convergence tolerance.
        random_state: Random seed for reproducibility.
    """

    metric: Literal["amex"]
    method: Literal["SLSQP", "trust-constr"]
    max_iterations: int
    tolerance: float
    random_state: int


class OptimizationResult(TypedDict):
    """Result of ensemble weight optimization.

    Attributes:
        weights: Optimized ensemble weights.
        best_score: Best metric score achieved.
        n_iterations: Number of optimization iterations.
        converged: Whether optimization converged.
        initial_score: Score before optimization (equal weights).
    """

    weights: EnsembleWeights
    best_score: float
    n_iterations: int
    converged: bool
    initial_score: float


class EnsemblePrediction(TypedDict):
    """Ensemble prediction result.

    Attributes:
        predictions: Weighted ensemble predictions, shape (n_samples,).
        weights: Weights used for this prediction.
        model_contributions: Per-model weighted contributions, shape (n_models, n_samples).
    """

    predictions: NDArray[np.float64]
    weights: EnsembleWeights
    model_contributions: NDArray[np.float64]


def make_default_optimization_config(random_state: int = 42) -> OptimizationConfig:
    """Create default optimization configuration.

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        Default OptimizationConfig with SLSQP method.
    """
    return OptimizationConfig(
        metric="amex",
        method="SLSQP",
        max_iterations=1000,
        tolerance=1e-8,
        random_state=random_state,
    )


__all__ = [
    "EnsembleOOFData",
    "EnsemblePrediction",
    "EnsembleWeights",
    "ModelOOFPredictions",
    "OptimizationConfig",
    "OptimizationResult",
    "make_default_optimization_config",
]
