"""Type definitions for probability calibration.

Strict typing only. No Any, casts, or stubs.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import JSONValue

# Calibration method types
CalibrationMethod = Literal["isotonic", "platt"]


class CalibratorConfig(TypedDict, total=True):
    """Configuration for probability calibrator.

    Args:
        method: Calibration method to use.
            "isotonic" - Non-parametric monotonic regression.
            "platt" - Parametric sigmoid (logistic) regression.
        clip_proba: Whether to clip output probabilities to [eps, 1-eps]
            to avoid log(0) issues. Default True.
        eps: Small epsilon for probability clipping. Default 1e-10.
    """

    method: CalibrationMethod
    clip_proba: bool
    eps: float


class IsotonicParams(TypedDict, total=True):
    """Learned parameters for isotonic regression calibrator.

    Isotonic regression fits a piecewise constant monotonically increasing
    function. The calibration curve is defined by breakpoints (X_thresholds)
    and their corresponding calibrated values (y_values).

    Args:
        X_thresholds: Sorted array of input probability thresholds as list.
        y_values: Corresponding calibrated probability values as list.
    """

    X_thresholds: list[float]
    y_values: list[float]


class PlattParams(TypedDict, total=True):
    """Learned parameters for Platt scaling calibrator.

    Platt scaling fits a sigmoid: P_calibrated = 1 / (1 + exp(A*P + B))
    This is equivalent to logistic regression on the raw probabilities.

    Args:
        A: Slope parameter (typically negative).
        B: Intercept parameter.
    """

    A: float
    B: float


class IsotonicState(TypedDict, total=True):
    """Serializable state for isotonic calibrator.

    Args:
        method: Discriminator field, always "isotonic".
        config: Calibrator configuration.
        params: Learned isotonic parameters.
    """

    method: Literal["isotonic"]
    config: CalibratorConfig
    params: IsotonicParams


class PlattState(TypedDict, total=True):
    """Serializable state for Platt scaling calibrator.

    Args:
        method: Discriminator field, always "platt".
        config: Calibrator configuration.
        params: Learned Platt parameters.
    """

    method: Literal["platt"]
    config: CalibratorConfig
    params: PlattParams


# Union of calibrator states
CalibratorState = IsotonicState | PlattState


def _require_str(value: JSONValue, field: str) -> str:
    """Validate and return string value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated string value.

    Raises:
        ValueError: If value is not a string.
    """
    if not isinstance(value, str):
        raise ValueError(f"Field '{field}' must be str, got {type(value).__name__}")
    return value


def _require_float(value: JSONValue, field: str) -> float:
    """Validate and return float value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated float value.

    Raises:
        ValueError: If value is not a number.
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"Field '{field}' must be number, got {type(value).__name__}")
    return float(value)


def _require_bool(value: JSONValue, field: str) -> bool:
    """Validate and return bool value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated bool value.

    Raises:
        ValueError: If value is not a bool.
    """
    if not isinstance(value, bool):
        raise ValueError(f"Field '{field}' must be bool, got {type(value).__name__}")
    return value


def _require_list_float(value: JSONValue, field: str) -> list[float]:
    """Validate and return list of floats.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated list of floats.

    Raises:
        ValueError: If value is not a list of numbers.
    """
    if not isinstance(value, list):
        raise ValueError(f"Field '{field}' must be list, got {type(value).__name__}")
    result: list[float] = []
    for i, item in enumerate(value):
        if not isinstance(item, (int, float)):
            raise ValueError(f"Field '{field}[{i}]' must be number, got {type(item).__name__}")
        result.append(float(item))
    return result


def _require_dict(value: JSONValue, field: str) -> dict[str, JSONValue]:
    """Validate and return dict value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated dict value.

    Raises:
        ValueError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise ValueError(f"Field '{field}' must be dict, got {type(value).__name__}")
    # Value is confirmed to be dict; cast to proper type
    result: dict[str, JSONValue] = {}
    for k, v in value.items():
        result[k] = v
    return result


def decode_calibrator_state(data: dict[str, JSONValue]) -> CalibratorState:
    """Decode calibrator state from dictionary.

    Args:
        data: Raw dictionary from JSON deserialization.

    Returns:
        Validated CalibratorState.

    Raises:
        ValueError: If data is invalid or missing required fields.
    """
    method = _require_str(data.get("method"), "method")

    config_raw = _require_dict(data.get("config"), "config")
    config_method_raw = config_raw.get("method")
    config_method = _require_str(config_method_raw, "config.method")

    if config_method not in ("isotonic", "platt"):
        raise ValueError(f"Invalid config.method: {config_method}")

    config: CalibratorConfig = {
        "method": "isotonic" if config_method == "isotonic" else "platt",
        "clip_proba": _require_bool(config_raw.get("clip_proba"), "config.clip_proba"),
        "eps": _require_float(config_raw.get("eps"), "config.eps"),
    }

    params_raw = _require_dict(data.get("params"), "params")

    if method == "isotonic":
        iso_params: IsotonicParams = {
            "X_thresholds": _require_list_float(
                params_raw.get("X_thresholds"), "params.X_thresholds"
            ),
            "y_values": _require_list_float(params_raw.get("y_values"), "params.y_values"),
        }
        iso_state: IsotonicState = {
            "method": "isotonic",
            "config": config,
            "params": iso_params,
        }
        return iso_state

    if method == "platt":
        platt_params: PlattParams = {
            "A": _require_float(params_raw.get("A"), "params.A"),
            "B": _require_float(params_raw.get("B"), "params.B"),
        }
        platt_state: PlattState = {
            "method": "platt",
            "config": config,
            "params": platt_params,
        }
        return platt_state

    raise ValueError(f"Unknown calibration method: {method}")


def _encode_isotonic_state(state: IsotonicState) -> dict[str, JSONValue]:
    """Encode isotonic state to dictionary.

    Args:
        state: Isotonic calibrator state.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    config = state["config"]
    params = state["params"]
    config_dict: dict[str, JSONValue] = {
        "method": config["method"],
        "clip_proba": config["clip_proba"],
        "eps": config["eps"],
    }
    # Convert list[float] to list[JSONValue] explicitly for type compatibility
    x_thresholds_json: list[JSONValue] = list(params["X_thresholds"])
    y_values_json: list[JSONValue] = list(params["y_values"])
    params_dict: dict[str, JSONValue] = {
        "X_thresholds": x_thresholds_json,
        "y_values": y_values_json,
    }
    result: dict[str, JSONValue] = {
        "method": "isotonic",
        "config": config_dict,
        "params": params_dict,
    }
    return result


def _encode_platt_state(state: PlattState) -> dict[str, JSONValue]:
    """Encode platt state to dictionary.

    Args:
        state: Platt calibrator state.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    config = state["config"]
    params = state["params"]
    config_dict: dict[str, JSONValue] = {
        "method": config["method"],
        "clip_proba": config["clip_proba"],
        "eps": config["eps"],
    }
    params_dict: dict[str, JSONValue] = {
        "A": params["A"],
        "B": params["B"],
    }
    result: dict[str, JSONValue] = {
        "method": "platt",
        "config": config_dict,
        "params": params_dict,
    }
    return result


def encode_calibrator_state(state: CalibratorState) -> dict[str, JSONValue]:
    """Encode calibrator state to dictionary for JSON serialization.

    Args:
        state: Calibrator state to encode.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    if state["method"] == "isotonic":
        return _encode_isotonic_state(state)
    return _encode_platt_state(state)


__all__ = [
    "CalibrationMethod",
    "CalibratorConfig",
    "CalibratorState",
    "IsotonicParams",
    "IsotonicState",
    "PlattParams",
    "PlattState",
    "decode_calibrator_state",
    "encode_calibrator_state",
]
