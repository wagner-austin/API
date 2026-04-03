"""MLP search space definitions for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from ..types import (
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    MLPSearchSpace,
)


def make_mlp_default_space() -> MLPSearchSpace:
    """Create default MLP search space for bankruptcy prediction.

    Based on empirical testing for tabular data:
    - n_layers 1-4 (deeper not usually better for tabular)
    - hidden_size 64-512 (common hidden layer sizes)
    - learning_rate 1e-5 to 1e-2 in log scale
    - dropout 0.0-0.5 for regularization
    - batch_size 32-256 for stable gradients

    Returns:
        MLPSearchSpace with sensible default ranges.
    """
    n_layers_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 1,
        "high": 4,
        "log_scale": False,
    }
    hidden_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (64, 128, 256, 512),
    }
    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "log_scale": True,
    }
    dropout_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.5,
        "log_scale": False,
    }
    batch_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (32, 64, 128, 256),
    }

    space: MLPSearchSpace = {
        "n_layers": n_layers_spec,
        "hidden_size": hidden_size_spec,
        "learning_rate": learning_rate_spec,
        "dropout": dropout_spec,
        "batch_size": batch_size_spec,
    }
    return space


def make_mlp_focused_space(
    *,
    best_n_layers: int,
    best_hidden_size: int,
    best_learning_rate: float,
) -> MLPSearchSpace:
    """Create focused MLP search space around known good values.

    Args:
        best_n_layers: Best n_layers from initial search.
        best_hidden_size: Best hidden_size from initial search.
        best_learning_rate: Best learning_rate from initial search.

    Returns:
        MLPSearchSpace with narrowed ranges around best values.
    """
    layers_low = max(1, best_n_layers - 1)
    layers_high = min(5, best_n_layers + 1)

    n_layers_spec: IntRangeSpec = {
        "param_type": "int",
        "low": layers_low,
        "high": layers_high,
        "log_scale": False,
    }

    hidden_choices: list[int] = []
    for size in [32, 64, 128, 256, 512]:
        if abs(size - best_hidden_size) <= best_hidden_size:
            hidden_choices.append(size)
    if not hidden_choices:
        hidden_choices = [best_hidden_size]

    hidden_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": tuple(hidden_choices),
    }

    lr_low = max(1e-6, best_learning_rate * 0.1)
    lr_high = min(1e-1, best_learning_rate * 10.0)

    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": lr_low,
        "high": lr_high,
        "log_scale": True,
    }
    dropout_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.4,
        "log_scale": False,
    }
    batch_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (32, 64, 128),
    }

    space: MLPSearchSpace = {
        "n_layers": n_layers_spec,
        "hidden_size": hidden_size_spec,
        "learning_rate": learning_rate_spec,
        "dropout": dropout_spec,
        "batch_size": batch_size_spec,
    }
    return space


__all__ = [
    "make_mlp_default_space",
    "make_mlp_focused_space",
]
