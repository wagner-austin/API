"""JSON type foundations for ClearGBM.

Provides the recursive JSON type aliases, type-safe extraction helpers, and
numeric validation functions used by all other ``_types_*`` sub-modules.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

# Recursive JSON type for strict typing
JSONValue = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None
JSONDict = dict[str, JSONValue]


class JSONTypeError(TypeError):
    """Raised when JSON value has unexpected type during decoding."""


# =============================================================================
# Validation Helpers
# =============================================================================


def require_positive_int(value: int, name: str) -> int:
    """Validate that value is a positive integer.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_non_negative_int(value: int, name: str) -> int:
    """Validate that value is a non-negative integer.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def require_positive_float(value: float, name: str) -> float:
    """Validate that value is a positive float.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_unit_float(value: float, name: str) -> float:
    """Validate that value is in (0, 1].

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is not in (0, 1].
    """
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def require_non_negative_float(value: float, name: str) -> float:
    """Validate that value is a non-negative float.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def require_n_jobs(value: int, name: str) -> int:
    """Validate n_jobs: must be -1 (all cores) or positive.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is not -1 or positive.
    """
    if value != -1 and value <= 0:
        raise ValueError(f"{name} must be -1 or positive, got {value}")
    return value


# =============================================================================
# Raw Dict Extraction Helpers
# =============================================================================


def _require_str(raw: JSONDict, key: str) -> str:
    """Extract and validate string from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        String value.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value is not a string.
    """
    value = raw[key]
    if not isinstance(value, str):
        raise JSONTypeError(f"{key} must be str, got {type(value).__name__}")
    return value


def _require_int(raw: JSONDict, key: str) -> int:
    """Extract and validate int from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Integer value.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value is not an int.
    """
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise JSONTypeError(f"{key} must be int, got {type(value).__name__}")
    return value


def _require_float(raw: JSONDict, key: str) -> float:
    """Extract and validate float from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Float value.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value is not a float or int.
    """
    value = raw[key]
    if isinstance(value, bool):
        raise JSONTypeError(f"{key} must be float, got bool")
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise JSONTypeError(f"{key} must be float, got {type(value).__name__}")
    return value


def _require_bool(raw: JSONDict, key: str) -> bool:
    """Extract and validate bool from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Boolean value.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value is not a bool.
    """
    value = raw[key]
    if not isinstance(value, bool):
        raise JSONTypeError(f"{key} must be bool, got {type(value).__name__}")
    return value


def _get_optional_int(raw: JSONDict, key: str) -> int | None:
    """Extract optional int from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Integer value or None if key not present or value is None.

    Raises:
        JSONTypeError: If value is present but not int or None.
    """
    if key not in raw:
        return None
    value = raw[key]
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise JSONTypeError(f"{key} must be int or None, got {type(value).__name__}")
    return value


def _get_optional_float(raw: JSONDict, key: str) -> float | None:
    """Extract optional float from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Float value or None if key not present or value is None.

    Raises:
        JSONTypeError: If value is present but not float/int or None.
    """
    if key not in raw:
        return None
    value = raw[key]
    if value is None:
        return None
    if isinstance(value, bool):
        raise JSONTypeError(f"{key} must be float or None, got bool")
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise JSONTypeError(f"{key} must be float or None, got {type(value).__name__}")
    return value


def _get_optional_str(raw: JSONDict, key: str) -> str | None:
    """Extract optional string from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        String value or None if key not present or value is None.

    Raises:
        JSONTypeError: If value is present but not str or None.
    """
    if key not in raw:
        return None
    value = raw[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise JSONTypeError(f"{key} must be str or None, got {type(value).__name__}")
    return value


def _as_json_dict(value: JSONValue, context: str) -> JSONDict:
    """Convert JSONValue to JSONDict with validation.

    Args:
        value: Value to convert.
        context: Context for error messages.

    Returns:
        The value as a JSONDict.

    Raises:
        JSONTypeError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be dict, got {type(value).__name__}")
    return value


__all__ = [
    "JSONDict",
    "JSONTypeError",
    "JSONValue",
    "require_n_jobs",
    "require_non_negative_float",
    "require_non_negative_int",
    "require_positive_float",
    "require_positive_int",
    "require_unit_float",
]
