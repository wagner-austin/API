"""Ensemble weight optimization using scipy.

Optimizes model weights to maximize the AMEX competition metric
using out-of-fold predictions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.ensemble.types import (
    EnsembleOOFData,
    EnsembleWeights,
    OptimizationConfig,
    OptimizationResult,
)
from covenant_ml.ensemble.weighted import (
    create_equal_weights,
    extract_prediction_matrix,
    validate_oof_data,
)
from covenant_ml.metrics import compute_amex_metric

_log = get_logger(__name__)

# Type aliases for scipy minimize interface
_ObjectiveFnType = Callable[[NDArray[np.float64]], float]
_ConstraintDict = dict[str, str | _ObjectiveFnType]
_OptionsDict = dict[str, int | float]


class _OptimizeResultProtocol(Protocol):
    """Protocol for scipy.optimize.OptimizeResult."""

    x: NDArray[np.float64]
    fun: float
    nit: int
    success: bool


class _MinimizeFnProtocol(Protocol):
    """Protocol for scipy.optimize.minimize function."""

    def __call__(
        self,
        fun: _ObjectiveFnType,
        x0: NDArray[np.float64],
        method: str,
        bounds: tuple[tuple[float, float], ...],
        constraints: tuple[_ConstraintDict, ...],
        options: _OptionsDict,
    ) -> _OptimizeResultProtocol:
        """Minimize a function."""
        ...


# Hook for scipy.optimize.minimize - set at startup
_minimize_hook: _MinimizeFnProtocol | None = None


def set_minimize_hook(hook: _MinimizeFnProtocol) -> None:
    """Set the scipy minimize hook.

    Called at application startup to inject the real scipy implementation.
    Tests can inject a fake implementation.

    Args:
        hook: The minimize function to use.
    """
    global _minimize_hook
    _minimize_hook = hook


def use_real_scipy() -> None:
    """Configure to use real scipy.optimize.minimize.

    Call this at application startup.
    """
    scipy_opt = __import__("scipy.optimize", fromlist=["minimize"])
    minimize_fn: _MinimizeFnProtocol = scipy_opt.minimize
    set_minimize_hook(minimize_fn)


def _get_minimize() -> _MinimizeFnProtocol:
    """Get the minimize function.

    Returns:
        The configured minimize function.

    Raises:
        RuntimeError: If no hook has been set.
    """
    if _minimize_hook is None:
        raise RuntimeError(
            "Scipy minimize hook not set. Call use_real_scipy() at startup "
            "or set_minimize_hook() with a fake for tests."
        )
    return _minimize_hook


def _compute_ensemble_score(
    weights: NDArray[np.float64],
    pred_matrix: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    """Compute AMEX score for given weights.

    Args:
        weights: Model weights, shape (n_models,).
        pred_matrix: Prediction matrix, shape (n_models, n_samples).
        labels: True labels, shape (n_samples,).

    Returns:
        AMEX metric score (higher is better).
    """
    # Compute weighted ensemble predictions
    ensemble_preds: NDArray[np.float64] = np.dot(weights, pred_matrix)

    # Compute AMEX metric
    result = compute_amex_metric(labels, ensemble_preds)
    return result["score"]


def _objective_function(
    weights: NDArray[np.float64],
    pred_matrix: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    """Objective function for minimization (negative score).

    Args:
        weights: Model weights, shape (n_models,).
        pred_matrix: Prediction matrix, shape (n_models, n_samples).
        labels: True labels, shape (n_samples,).

    Returns:
        Negative AMEX score (for minimization).
    """
    score = _compute_ensemble_score(weights, pred_matrix, labels)
    return -score  # Negate for minimization


def optimize_ensemble_weights(
    oof_data: EnsembleOOFData,
    config: OptimizationConfig,
) -> OptimizationResult:
    """Optimize ensemble weights to maximize AMEX metric.

    Uses scipy.optimize.minimize with SLSQP to find optimal weights
    that maximize the AMEX competition metric on OOF predictions.

    Constraints:
    - Weights sum to 1.0
    - Weights are non-negative

    Args:
        oof_data: Out-of-fold predictions from all models.
        config: Optimization configuration.

    Returns:
        OptimizationResult with optimized weights and metrics.

    Raises:
        ValueError: If OOF data is invalid.
        RuntimeError: If scipy hook not configured.
    """
    validate_oof_data(oof_data)

    n_models = oof_data["n_models"]
    labels = oof_data["labels"]
    model_names = tuple(p["model_name"] for p in oof_data["model_predictions"])

    # Extract prediction matrix
    pred_matrix = extract_prediction_matrix(oof_data)

    # Compute initial score with equal weights
    equal_weights = create_equal_weights(model_names)
    initial_score = _compute_ensemble_score(equal_weights["weights"], pred_matrix, labels)

    _log.info(
        "Starting ensemble weight optimization",
        extra={
            "n_models": n_models,
            "n_samples": oof_data["n_samples"],
            "initial_score": initial_score,
            "method": config["method"],
        },
    )

    # Set up optimization
    minimize_fn = _get_minimize()

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
    result: _OptimizeResultProtocol = minimize_fn(
        fun=lambda w: _objective_function(w, pred_matrix, labels),
        x0=x0,
        method=config["method"],
        bounds=bounds,
        constraints=constraints,
        options=options,
    )

    # Extract optimized weights
    opt_weights: NDArray[np.float64] = np.asarray(result.x, dtype=np.float64)

    # Normalize to ensure exact sum to 1.0 (numerical precision)
    opt_weights = opt_weights / float(np.sum(opt_weights))

    # Compute final score
    best_score = _compute_ensemble_score(opt_weights, pred_matrix, labels)

    _log.info(
        "Ensemble weight optimization complete",
        extra={
            "initial_score": initial_score,
            "best_score": best_score,
            "improvement": best_score - initial_score,
            "n_iterations": result.nit,
            "converged": result.success,
        },
    )

    return OptimizationResult(
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
    "optimize_ensemble_weights",
    "set_minimize_hook",
    "use_real_scipy",
]
