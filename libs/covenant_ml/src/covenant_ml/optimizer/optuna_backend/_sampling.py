"""Shared parameter sampling functions for Optuna optimizers.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from ..types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    CategoricalStringSpec,
    FloatRangeSpec,
    IntRangeSpec,
)
from ._protocols import OptunaTrialProtocol


def sample_param_int(
    trial: OptunaTrialProtocol,
    name: str,
    spec: IntRangeSpec | CategoricalIntSpec,
) -> int:
    """Sample integer parameter from trial.

    Args:
        trial: Optuna trial object.
        name: Parameter name.
        spec: Integer range or categorical specification.

    Returns:
        Sampled integer value.
    """
    if spec["param_type"] == "int":
        int_spec: IntRangeSpec = spec
        return trial.suggest_int(
            name,
            int_spec["low"],
            int_spec["high"],
            log=int_spec["log_scale"],
        )
    cat_spec: CategoricalIntSpec = spec
    result = trial.suggest_categorical(name, cat_spec["choices"])
    return int(result)


def sample_param_float(
    trial: OptunaTrialProtocol,
    name: str,
    spec: FloatRangeSpec | CategoricalFloatSpec,
) -> float:
    """Sample float parameter from trial.

    Args:
        trial: Optuna trial object.
        name: Parameter name.
        spec: Float range or categorical specification.

    Returns:
        Sampled float value.
    """
    if spec["param_type"] == "float":
        float_spec: FloatRangeSpec = spec
        return trial.suggest_float(
            name,
            float_spec["low"],
            float_spec["high"],
            log=float_spec["log_scale"],
        )
    cat_spec: CategoricalFloatSpec = spec
    result = trial.suggest_categorical(name, cat_spec["choices"])
    return float(result)


def sample_param_str(
    trial: OptunaTrialProtocol,
    name: str,
    spec: CategoricalStringSpec,
) -> str:
    """Sample string parameter from trial.

    Args:
        trial: Optuna trial object.
        name: Parameter name.
        spec: Categorical string specification with choices.

    Returns:
        Sampled string value from the choices.
    """
    result = trial.suggest_categorical(name, spec["choices"])
    return str(result)


__all__ = [
    "sample_param_float",
    "sample_param_int",
    "sample_param_str",
]
