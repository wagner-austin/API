"""Decoder functions for Excel data via polars/openpyxl.

Converts Excel cell data to typed structures using the JSON bridge.
"""

from __future__ import annotations

from instrument_io._exceptions import DecodingError
from instrument_io._json_bridge import CellValue, JSONValue, _json_value_to_cell


def _decode_cell_value(value: JSONValue) -> CellValue:
    """Decode a JSON value to an Excel cell value.

    Args:
        value: Value from JSON-parsed data.

    Returns:
        CellValue (str, int, float, bool, or None).
    """
    return _json_value_to_cell(value)


def _decode_row_dict(
    row_dict: dict[str, JSONValue],
    columns: list[str],
) -> dict[str, CellValue]:
    """Decode a row dictionary to typed cell values.

    Args:
        row_dict: Dictionary from JSON-parsed row.
        columns: List of column names to extract.

    Returns:
        Dictionary mapping column names to cell values.
    """
    result: dict[str, CellValue] = {}
    for col in columns:
        raw_value = row_dict.get(col)
        result[col] = _json_value_to_cell(raw_value)
    return result


def _decode_rows(
    rows: list[dict[str, JSONValue]],
    columns: list[str],
) -> list[dict[str, CellValue]]:
    """Decode multiple rows to typed cell values.

    Args:
        rows: List of row dictionaries from JSON-parsed data.
        columns: List of column names to extract.

    Returns:
        List of typed row dictionaries.
    """
    return [_decode_row_dict(row, columns) for row in rows]


def _extract_string(value: JSONValue, field: str) -> str:
    """Extract string value with validation.

    Args:
        value: JSON value to extract.
        field: Field name for error messages.

    Returns:
        String value.

    Raises:
        DecodingError: If value is not a string.
    """
    if value is None:
        raise DecodingError(field, "Value is None, expected string")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    raise DecodingError(field, f"Cannot convert {type(value).__name__} to string")


def _extract_string_or_none(value: JSONValue) -> str | None:
    """Extract optional string value.

    Args:
        value: JSON value to extract.

    Returns:
        String value or None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)


def _extract_int(value: JSONValue, field: str) -> int:
    """Extract integer value with validation.

    Args:
        value: JSON value to extract.
        field: Field name for error messages.

    Returns:
        Integer value.

    Raises:
        DecodingError: If value cannot be converted to int.
    """
    if value is None:
        raise DecodingError(field, "Value is None, expected int")
    if isinstance(value, bool):
        raise DecodingError(field, "Value is bool, expected int")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            raise DecodingError(field, f"Cannot parse '{value}' as int") from None
    raise DecodingError(field, f"Cannot convert {type(value).__name__} to int")


def _extract_int_or_none(value: JSONValue) -> int | None:
    """Extract optional integer value.

    Args:
        value: JSON value to extract.

    Returns:
        Integer value or None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _extract_float(value: JSONValue, field: str) -> float:
    """Extract float value with validation.

    Args:
        value: JSON value to extract.
        field: Field name for error messages.

    Returns:
        Float value.

    Raises:
        DecodingError: If value cannot be converted to float.
    """
    if value is None:
        raise DecodingError(field, "Value is None, expected float")
    if isinstance(value, bool):
        raise DecodingError(field, "Value is bool, expected float")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            raise DecodingError(field, f"Cannot parse '{value}' as float") from None
    raise DecodingError(field, f"Cannot convert {type(value).__name__} to float")


def _extract_float_or_none(value: JSONValue) -> float | None:
    """Extract optional float value.

    Args:
        value: JSON value to extract.

    Returns:
        Float value or None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_bool(value: JSONValue, field: str) -> bool:
    """Extract boolean value with validation.

    Args:
        value: JSON value to extract.
        field: Field name for error messages.

    Returns:
        Boolean value.

    Raises:
        DecodingError: If value cannot be converted to bool.
    """
    if value is None:
        raise DecodingError(field, "Value is None, expected bool")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in ("true", "yes", "1", "y"):
            return True
        if lower in ("false", "no", "0", "n"):
            return False
        raise DecodingError(field, f"Cannot parse '{value}' as bool")
    if isinstance(value, (int, float)):
        return bool(value)
    raise DecodingError(field, f"Cannot convert {type(value).__name__} to bool")


__all__ = [
    "_decode_cell_value",
    "_decode_row_dict",
    "_decode_rows",
    "_extract_bool",
    "_extract_float",
    "_extract_float_or_none",
    "_extract_int",
    "_extract_int_or_none",
    "_extract_string",
    "_extract_string_or_none",
]
