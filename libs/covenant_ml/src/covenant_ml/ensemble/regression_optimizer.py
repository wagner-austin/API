"""Regression ensemble weight optimization using scipy.

Parallel to optimizer.py (classification). Key differences:
- labels: NDArray[np.float64] (continuous, not int64 binary)
- Objectives: neg_rmse, neg_mae, r_squared (not amex)
- Validation: regression-specific (float64 labels)

Reuses the scipy minimize hook from optimizer.py.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.ensemble import _hooks
from covenant_ml.ensemble.regression_types import (
    RegressionEnsembleOOFData,
    RegressionOptimizationConfig,
    RegressionOptimizationResult,
)
from covenant_ml.ensemble.types import EnsembleWeights
from covenant_ml.metrics import compute_mae, compute_r_squared, compute_rmse

_log = get_logger(__name__)

# Type aliases for scipy minimize interface
_ObjectiveFnType = Callable[[NDArray[np.float64]], float]
_ConstraintDict = dict[str, str | _ObjectiveFnType]
_OptionsDict = dict[str, int | float]


# =============================================================================
# Validation
# =============================================================================


def validate_regression_oof_data(oof_data: RegressionEnsembleOOFData) -> None:
    """Validate regression OOF data structure and consistency.

    Args:
        oof_data: OOF data to validate.

    Raises:
        ValueError: If data is invalid or inconsistent.
    """
    n_samples = oof_data["n_samples"]
    n_models = oof_data["n_models"]
    labels = oof_data["labels"]
    model_preds = oof_data["model_predictions"]

    if n_models < 2:
        raise ValueError(f"Ensemble requires at least 2 models, got {n_models}")

    if len(labels) != n_samples:
        raise ValueError(f"Labels length ({len(labels)}) does not match n_samples ({n_samples})")

    if len(model_preds) != n_models:
        raise ValueError(
            f"model_predictions length ({len(model_preds)}) does not match n_models ({n_models})"
        )

    for pred in model_preds:
        if len(pred["predictions"]) != n_samples:
            raise ValueError(
                f"Model {pred['model_name']} has {len(pred['predictions'])} predictions, "
                f"expected {n_samples}"
            )
        if len(pred["fold_indices"]) != n_samples:
            raise ValueError(
                f"Model {pred['model_name']} has {len(pred['fold_indices'])} fold_indices, "
                f"expected {n_samples}"
            )


# =============================================================================
# Prediction matrix
# =============================================================================


def extract_regression_prediction_matrix(
    oof_data: RegressionEnsembleOOFData,
) -> NDArray[np.float64]:
    """Extract predictions as a matrix for optimization.

    Args:
        oof_data: OOF data from all models.

    Returns:
        Prediction matrix of shape (n_models, n_samples).
    """
    n_samples = oof_data["n_samples"]
    n_models = oof_data["n_models"]

    pred_matrix: NDArray[np.float64] = np.zeros((n_models, n_samples), dtype=np.float64)
    for i, pred in enumerate(oof_data["model_predictions"]):
        pred_matrix[i, :] = pred["predictions"]

    return pred_matrix


# =============================================================================
# Scoring functions
# =============================================================================


def _compute_weighted_preds(
    weights: NDArray[np.float64],
    pred_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute weighted ensemble predictions.

    Args:
        weights: Model weights, shape (n_models,).
        pred_matrix: Prediction matrix, shape (n_models, n_samples).

    Returns:
        Weighted predictions, shape (n_samples,).
    """
    result: NDArray[np.float64] = np.dot(weights, pred_matrix)
    return result


def _compute_neg_rmse(
    weights: NDArray[np.float64],
    pred_matrix: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> float:
    """Compute negative RMSE for given weights.

    Negative because scipy minimizes, and we want lower RMSE to be better.

    Args:
        weights: Model weights, shape (n_models,).
        pred_matrix: Prediction matrix, shape (n_models, n_samples).
        labels: True continuous labels, shape (n_samples,).

    Returns:
        Negative RMSE (higher is better).
    """
    preds = _compute_weighted_preds(weights, pred_matrix)
    rmse = compute_rmse(labels, preds)
    return -rmse


def _compute_neg_mae(
    weights: NDArray[np.float64],
    pred_matrix: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> float:
    """Compute negative MAE for given weights.

    Args:
        weights: Model weights, shape (n_models,).
        pred_matrix: Prediction matrix, shape (n_models, n_samples).
        labels: True continuous labels, shape (n_samples,).

    Returns:
        Negative MAE (higher is better).
    """
    preds = _compute_weighted_preds(weights, pred_matrix)
    mae = compute_mae(labels, preds)
    return -mae


def _compute_neg_r_squared(
    weights: NDArray[np.float64],
    pred_matrix: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> float:
    """Compute negative R-squared for given weights.

    Negative because scipy minimizes, and we want higher R² to be better.

    Args:
        weights: Model weights, shape (n_models,).
        pred_matrix: Prediction matrix, shape (n_models, n_samples).
        labels: True continuous labels, shape (n_samples,).

    Returns:
        Negative R-squared (minimized to maximize R²).
    """
    preds = _compute_weighted_preds(weights, pred_matrix)
    r2 = compute_r_squared(labels, preds)
    return -r2


def _compute_regression_ensemble_score(
    weights: NDArray[np.float64],
    pred_matrix: NDArray[np.float64],
    labels: NDArray[np.float64],
    metric: str,
) -> float:
    """Compute regression score for given weights on the natural scale.

    Returns the metric on its natural scale (e.g., RMSE as positive value,
    R² as positive value). Used for reporting, not for optimization.

    Args:
        weights: Model weights, shape (n_models,).
        pred_matrix: Prediction matrix, shape (n_models, n_samples).
        labels: True continuous labels, shape (n_samples,).
        metric: Metric name (neg_rmse, neg_mae, r_squared).

    Returns:
        Metric value on natural scale.
    """
    preds = _compute_weighted_preds(weights, pred_matrix)

    if metric == "neg_rmse":
        return compute_rmse(labels, preds)
    if metric == "neg_mae":
        return compute_mae(labels, preds)
    if metric == "r_squared":
        return compute_r_squared(labels, preds)

    raise ValueError(f"Unknown metric: {metric}")


def _objective_function(
    weights: NDArray[np.float64],
    pred_matrix: NDArray[np.float64],
    labels: NDArray[np.float64],
    metric: str,
) -> float:
    """Objective function for minimization.

    For neg_rmse and neg_mae: returns positive value (minimize = lower error).
    For r_squared: returns -R² (minimize = maximize R²).

    Args:
        weights: Model weights, shape (n_models,).
        pred_matrix: Prediction matrix, shape (n_models, n_samples).
        labels: True continuous labels, shape (n_samples,).
        metric: Metric name.

    Returns:
        Value to minimize.
    """
    if metric == "neg_rmse":
        preds = _compute_weighted_preds(weights, pred_matrix)
        return compute_rmse(labels, preds)
    if metric == "neg_mae":
        preds = _compute_weighted_preds(weights, pred_matrix)
        return compute_mae(labels, preds)
    if metric == "r_squared":
        return _compute_neg_r_squared(weights, pred_matrix, labels)

    raise ValueError(f"Unknown metric: {metric}")


# =============================================================================
# Equal weights helper
# =============================================================================


def create_regression_equal_weights(
    model_names: tuple[str, ...],
) -> EnsembleWeights:
    """Create equal weights for all regression models.

    Args:
        model_names: Names of models in the ensemble.

    Returns:
        EnsembleWeights with equal weight for each model.

    Raises:
        ValueError: If fewer than 2 model names provided.
    """
    n_models = len(model_names)
    if n_models < 2:
        raise ValueError(f"Ensemble requires at least 2 models, got {n_models}")

    weight_value = 1.0 / n_models
    weights: NDArray[np.float64] = np.full(n_models, weight_value, dtype=np.float64)

    return EnsembleWeights(
        weights=weights,
        model_names=model_names,
    )


# =============================================================================
# Optimizer
# =============================================================================


def optimize_regression_ensemble_weights(
    oof_data: RegressionEnsembleOOFData,
    config: RegressionOptimizationConfig,
) -> RegressionOptimizationResult:
    """Optimize regression ensemble weights.

    Uses scipy.optimize.minimize to find optimal model weights that
    minimize error (RMSE/MAE) or maximize R² on OOF predictions.

    Constraints:
    - Weights sum to 1.0
    - Weights are non-negative

    Args:
        oof_data: Out-of-fold predictions from all regression models.
        config: Optimization configuration.

    Returns:
        RegressionOptimizationResult with optimized weights and metrics.

    Raises:
        ValueError: If OOF data is invalid.
        RuntimeError: If scipy hook not configured.
    """
    validate_regression_oof_data(oof_data)

    n_models = oof_data["n_models"]
    labels = oof_data["labels"]
    metric = config["metric"]
    model_names = tuple(p["model_name"] for p in oof_data["model_predictions"])

    # Extract prediction matrix
    pred_matrix = extract_regression_prediction_matrix(oof_data)

    # Compute initial score with equal weights
    equal_weights = create_regression_equal_weights(model_names)
    initial_score = _compute_regression_ensemble_score(
        equal_weights["weights"], pred_matrix, labels, metric
    )

    _log.info(
        "Starting regression ensemble weight optimization",
        extra={
            "n_models": n_models,
            "n_samples": oof_data["n_samples"],
            "initial_score": initial_score,
            "method": config["method"],
            "metric": metric,
        },
    )

    # Set up optimization
    minimize_fn = _hooks.minimize

    # Initial guess: equal weights
    x0: NDArray[np.float64] = np.full(n_models, 1.0 / n_models, dtype=np.float64)

    # Bounds: weights in [0, 1]
    bounds: tuple[tuple[float, float], ...] = tuple((0.0, 1.0) for _ in range(n_models))

    # Constraint: weights sum to 1
    sum_constraint: _ConstraintDict = {
        "type": "eq",
        "fun": lambda w: float(np.sum(w)) - 1.0,
    }
    constraints: tuple[_ConstraintDict, ...] = (sum_constraint,)

    # Options
    options: _OptionsDict = {
        "maxiter": config["max_iterations"],
        "ftol": config["tolerance"],
    }

    # Run optimization
    result = minimize_fn(
        fun=lambda w: _objective_function(w, pred_matrix, labels, metric),
        x0=x0,
        method=config["method"],
        bounds=bounds,
        constraints=constraints,
        options=options,
    )

    # Extract optimized weights
    opt_weights: NDArray[np.float64] = np.asarray(result.x, dtype=np.float64)

    # Normalize to ensure exact sum to 1.0 (numerical precision)
    weight_sum = float(np.sum(opt_weights))
    if weight_sum > 0.0:
        opt_weights = opt_weights / weight_sum

    # Compute final score on natural scale
    best_score = _compute_regression_ensemble_score(opt_weights, pred_matrix, labels, metric)

    _log.info(
        "Regression ensemble weight optimization complete",
        extra={
            "initial_score": initial_score,
            "best_score": best_score,
            "n_iterations": result.nit,
            "converged": result.success,
            "metric": metric,
        },
    )

    return RegressionOptimizationResult(
        weights=EnsembleWeights(
            weights=opt_weights,
            model_names=model_names,
        ),
        best_score=best_score,
        n_iterations=result.nit,
        converged=result.success,
        initial_score=initial_score,
    )


__all__ = [
    "create_regression_equal_weights",
    "extract_regression_prediction_matrix",
    "optimize_regression_ensemble_weights",
    "validate_regression_oof_data",
]
