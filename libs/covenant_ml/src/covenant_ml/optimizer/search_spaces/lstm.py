"""LSTM search space definitions for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from ..types import (
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LSTMSearchSpace,
)


def make_lstm_default_space() -> LSTMSearchSpace:
    """Create default LSTM search space for temporal bankruptcy prediction.

    Based on empirical testing for sequential financial data:
    - hidden_size 64-256 (smaller than NLP tasks)
    - num_layers 1-3 (deeper LSTMs often overfit on financial data)
    - learning_rate 1e-5 to 1e-2 in log scale
    - dropout 0.0-0.5 for regularization
    - batch_size 16-64 (smaller batches for sequential data)

    Returns:
        LSTMSearchSpace with sensible default ranges.
    """
    hidden_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (64, 128, 256),
    }
    num_layers_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 1,
        "high": 3,
        "log_scale": False,
    }
    dropout_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.5,
        "log_scale": False,
    }
    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "log_scale": True,
    }
    batch_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (16, 32, 64),
    }

    space: LSTMSearchSpace = {
        "hidden_size": hidden_size_spec,
        "num_layers": num_layers_spec,
        "dropout": dropout_spec,
        "learning_rate": learning_rate_spec,
        "batch_size": batch_size_spec,
    }
    return space


def make_lstm_focused_space(
    *,
    best_hidden_size: int,
    best_num_layers: int,
    best_learning_rate: float,
) -> LSTMSearchSpace:
    """Create focused LSTM search space around known good values.

    Args:
        best_hidden_size: Best hidden_size from initial search.
        best_num_layers: Best num_layers from initial search.
        best_learning_rate: Best learning_rate from initial search.

    Returns:
        LSTMSearchSpace with narrowed ranges around best values.
    """
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

    layers_low = max(1, best_num_layers - 1)
    layers_high = min(4, best_num_layers + 1)

    num_layers_spec: IntRangeSpec = {
        "param_type": "int",
        "low": layers_low,
        "high": layers_high,
        "log_scale": False,
    }
    dropout_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.4,
        "log_scale": False,
    }

    lr_low = max(1e-6, best_learning_rate * 0.1)
    lr_high = min(1e-1, best_learning_rate * 10.0)

    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": lr_low,
        "high": lr_high,
        "log_scale": True,
    }
    batch_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (16, 32),
    }

    space: LSTMSearchSpace = {
        "hidden_size": hidden_size_spec,
        "num_layers": num_layers_spec,
        "dropout": dropout_spec,
        "learning_rate": learning_rate_spec,
        "batch_size": batch_size_spec,
    }
    return space


__all__ = [
    "make_lstm_default_space",
    "make_lstm_focused_space",
]
