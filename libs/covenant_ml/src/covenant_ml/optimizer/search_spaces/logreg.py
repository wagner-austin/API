"""Logistic Regression search space definitions for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from ..types import (
    CategoricalStringSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LogRegSearchSpace,
)


def make_logreg_default_space() -> LogRegSearchSpace:
    """Create default Logistic Regression search space.

    Based on empirical testing for tabular classification:
    - C 1e-4 to 1e4 in log scale (inverse regularization strength)
    - max_iter 100-1000 (solver convergence budget)
    - tol 1e-6 to 1e-3 in log scale (stopping criteria)
    - penalty l2 and l1 (most common regularization types)
    - solver saga (supports all penalties including l1)
    - l1_ratio 0.0-1.0 (only for elasticnet, mixing of L1/L2)

    Returns:
        LogRegSearchSpace with sensible default ranges.
    """
    c_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 1e-4,
        "high": 1e4,
        "log_scale": True,
    }
    max_iter_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 100,
        "high": 1000,
        "log_scale": False,
    }
    tol_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 1e-6,
        "high": 1e-3,
        "log_scale": True,
    }
    penalty_spec: CategoricalStringSpec = {
        "param_type": "categorical_str",
        "choices": ("l2", "l1"),
    }
    solver_spec: CategoricalStringSpec = {
        "param_type": "categorical_str",
        "choices": ("saga",),
    }
    l1_ratio_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 1.0,
        "log_scale": False,
    }

    space: LogRegSearchSpace = {
        "C": c_spec,
        "max_iter": max_iter_spec,
        "tol": tol_spec,
        "penalty": penalty_spec,
        "solver": solver_spec,
        "l1_ratio": l1_ratio_spec,
    }
    return space


def make_logreg_focused_space(
    *,
    best_c: float,
    best_tol: float,
) -> LogRegSearchSpace:
    """Create focused LogReg search space around known good values.

    Args:
        best_c: Best C value from initial search.
        best_tol: Best tol value from initial search.

    Returns:
        LogRegSearchSpace with narrowed ranges around best values.
    """
    c_low = max(1e-6, best_c * 0.1)
    c_high = min(1e6, best_c * 10.0)

    c_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": c_low,
        "high": c_high,
        "log_scale": True,
    }
    max_iter_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 200,
        "high": 500,
        "log_scale": False,
    }

    tol_low = max(1e-8, best_tol * 0.1)
    tol_high = min(1e-2, best_tol * 10.0)

    tol_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": tol_low,
        "high": tol_high,
        "log_scale": True,
    }

    space: LogRegSearchSpace = {
        "C": c_spec,
        "max_iter": max_iter_spec,
        "tol": tol_spec,
    }
    return space


__all__ = [
    "make_logreg_default_space",
    "make_logreg_focused_space",
]
