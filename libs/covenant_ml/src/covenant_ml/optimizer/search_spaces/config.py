"""Optimization configuration factories.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from ..types import OptimizationConfig


def make_default_optimization_config(
    *,
    n_trials: int = 100,
    timeout_seconds: int | None = None,
    random_state: int = 42,
) -> OptimizationConfig:
    """Create default optimization configuration.

    Args:
        n_trials: Number of trials to run (default 100).
        timeout_seconds: Optional timeout in seconds (None = no timeout).
        random_state: Random seed for reproducibility.

    Returns:
        OptimizationConfig with sensible defaults.
    """
    config: OptimizationConfig = {
        "n_trials": n_trials,
        "timeout_seconds": timeout_seconds,
        "n_startup_trials": 10,
        "random_state": random_state,
        "direction": "maximize",
        "pruning_enabled": True,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }
    return config


__all__ = [
    "make_default_optimization_config",
]
