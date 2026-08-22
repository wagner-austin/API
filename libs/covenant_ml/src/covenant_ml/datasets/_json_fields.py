"""Typed field extractors for decoding dataset-type JSON payloads."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONValue

SeasonDefinition = Literal["warm", "cold", "full_year"]


def _require_positive_int(value: JSONValue, field: str) -> int:
    """Validate and return positive integer value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated positive integer.

    Raises:
        ValueError: If value is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be positive integer")
    return value


def _require_percentile(value: JSONValue, field: str) -> float:
    """Validate and return percentile value strictly between 0 and 100.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated percentile as float.

    Raises:
        ValueError: If value is not a number in (0, 100).
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    f = float(value)
    if not (0.0 < f < 100.0):
        raise ValueError(f"{field} must be between 0 and 100 exclusive")
    return f


def _require_numeric(value: JSONValue, field: str) -> float:
    """Validate and return numeric value as float.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated float value.

    Raises:
        ValueError: If value is not numeric.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _require_season(value: JSONValue, field: str) -> SeasonDefinition:
    """Validate and return season definition.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated SeasonDefinition literal.

    Raises:
        ValueError: If value is not a valid season.
    """
    if not isinstance(value, str) or value not in ("warm", "cold", "full_year"):
        raise ValueError(f"{field} must be 'warm', 'cold', or 'full_year'")
    if value == "warm":
        return "warm"
    if value == "cold":
        return "cold"
    return "full_year"


def _require_month_tuple(value: JSONValue, field: str) -> tuple[int, ...]:
    """Validate and return tuple of month numbers (1-12).

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated tuple of month integers.

    Raises:
        ValueError: If value is not a non-empty sequence of valid months.
    """
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{field} must be non-empty tuple of ints")
    result: list[int] = []
    for i, m in enumerate(value):
        if not isinstance(m, int) or isinstance(m, bool) or not (1 <= m <= 12):
            raise ValueError(f"{field}[{i}] must be int in 1..12")
        result.append(m)
    return tuple(result)


def _require_float_tuple(value: JSONValue, field: str, expected_len: int) -> tuple[float, ...]:
    """Validate and return tuple of floats with expected length.

    Args:
        value: Value to validate.
        field: Field name for error message.
        expected_len: Required number of elements.

    Returns:
        Validated tuple of floats.

    Raises:
        ValueError: If value is not a sequence of numerics with correct length.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be tuple of floats")
    if len(value) != expected_len:
        raise ValueError(f"{field} length {len(value)} != expected {expected_len}")
    result: list[float] = []
    for i, v in enumerate(value):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{field}[{i}] must be numeric")
        result.append(float(v))
    return tuple(result)


def _require_str_field(value: JSONValue, field: str) -> str:
    """Validate and return string value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated string.

    Raises:
        ValueError: If value is not a string.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be string")
    return value


def _require_str_tuple(value: JSONValue, field: str) -> tuple[str, ...]:
    """Validate and return tuple of strings.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated tuple of strings.

    Raises:
        ValueError: If value is not a sequence of strings.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be tuple of strings")
    result: list[str] = []
    for i, v in enumerate(value):
        if not isinstance(v, str):
            raise ValueError(f"{field}[{i}] must be string")
        result.append(v)
    return tuple(result)


def _require_nested_float_tuple(
    value: JSONValue,
    field: str,
    expected_outer: int,
    expected_inner: int,
) -> tuple[tuple[float, ...], ...]:
    """Validate and return nested tuple of floats with expected dimensions.

    Args:
        value: Value to validate.
        field: Field name for error message.
        expected_outer: Required outer length (e.g. n_harmonics).
        expected_inner: Required inner length (e.g. n_locations).

    Returns:
        Validated nested tuple of floats.

    Raises:
        ValueError: If value dimensions or element types are wrong.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be nested tuple of floats")
    if len(value) != expected_outer:
        raise ValueError(f"{field} outer length {len(value)} != expected {expected_outer}")
    result: list[tuple[float, ...]] = []
    for i, row in enumerate(value):
        result.append(_require_float_tuple(row, f"{field}[{i}]", expected_inner))
    return tuple(result)


def _require_bool_field(value: JSONValue, field: str) -> bool:
    """Validate and return bool value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated bool.

    Raises:
        ValueError: If value is not a bool.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be bool")
    return value


def _require_json_dict(value: JSONValue, field: str) -> dict[str, JSONValue]:
    """Validate and return JSON dictionary.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated dictionary.

    Raises:
        ValueError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be dictionary")
    return value


def _require_non_negative_int(value: JSONValue, field: str) -> int:
    """Validate and return non-negative integer value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated non-negative integer.

    Raises:
        ValueError: If value is not a non-negative integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be non-negative integer")
    return value
