"""JSON serialization bridge for type-safe data extraction.

This module provides helper functions for extracting typed values from
JSON-serialized data. Uses platform_core.json_utils for JSON loading.
Used as a bridge between untyped external library returns and our strict type system.
"""

from __future__ import annotations

# Import JSON utilities from platform_core (the only allowed json.loads location)
from platform_core.json_utils import JSONValue
from platform_core.json_utils import load_json_str as _load_json_str

# Cell value type for Excel (primitives only, no nested structures)
CellValue = str | int | float | bool | None


def _json_value_to_cell(value: JSONValue) -> CellValue:
    """Convert JSONValue to Excel cell value.

    Only primitive types are supported. Complex types (dict, list) return None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    # dict or list - unsupported, return None
    return None


def _extract_row_from_dict(row_dict: dict[str, JSONValue], columns: list[str]) -> list[CellValue]:
    """Extract values from row dict in column order."""
    return [_json_value_to_cell(row_dict.get(col)) for col in columns]


def _json_col_to_str_list(json_str: str, col: str) -> list[str]:
    """Extract typed string list from JSON-serialized single-column DataFrame.

    Use: df.select(col).write_json() -> pass result and col name here.
    """
    value: JSONValue = _load_json_str(json_str)
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        item: JSONValue = row.get(col)
        if item is None:
            result.append("")
        elif isinstance(item, str):
            result.append(item)
        else:
            result.append(str(item))
    return result


def _json_col_to_opt_str_list(json_str: str, col: str) -> list[str | None]:
    """Extract typed optional string list from JSON-serialized single-column DataFrame.

    Use: df.select(col).write_json() -> pass result and col name here.
    """
    value: JSONValue = _load_json_str(json_str)
    if not isinstance(value, list):
        return []
    result: list[str | None] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        item: JSONValue = row.get(col)
        if item is None:
            result.append(None)
        elif isinstance(item, str):
            result.append(item)
        else:
            result.append(str(item))
    return result


def _json_col_to_int_list(json_str: str, col: str) -> list[int]:
    """Extract typed int list from JSON-serialized single-column DataFrame.

    Use: df.select(col).write_json() -> pass result and col name here.
    """
    value: JSONValue = _load_json_str(json_str)
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        item: JSONValue = row.get(col)
        if isinstance(item, int) and not isinstance(item, bool):
            result.append(item)
        elif isinstance(item, float):
            result.append(int(item))
    return result


def _json_col_to_float_list(json_str: str, col: str) -> list[float]:
    """Extract typed float list from JSON-serialized single-column DataFrame.

    Use: df.select(col).write_json() -> pass result and col name here.
    Skips null values.
    """
    value: JSONValue = _load_json_str(json_str)
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        item: JSONValue = row.get(col)
        if isinstance(item, float):
            result.append(item)
        elif isinstance(item, int) and not isinstance(item, bool):
            result.append(float(item))
    return result


def _json_col_to_opt_float_list(json_str: str, col: str) -> list[float | None]:
    """Extract typed optional float list from JSON-serialized single-column DataFrame.

    Use: df.select(col).write_json() -> pass result and col name here.
    """
    value: JSONValue = _load_json_str(json_str)
    if not isinstance(value, list):
        return []
    result: list[float | None] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        item: JSONValue = row.get(col)
        if item is None:
            result.append(None)
        elif isinstance(item, float):
            result.append(item)
        elif isinstance(item, int) and not isinstance(item, bool):
            result.append(float(item))
        else:
            result.append(None)
    return result


__all__ = [
    "CellValue",
    "JSONValue",
    "_extract_row_from_dict",
    "_json_col_to_float_list",
    "_json_col_to_int_list",
    "_json_col_to_opt_float_list",
    "_json_col_to_opt_str_list",
    "_json_col_to_str_list",
    "_json_value_to_cell",
    "_load_json_str",
]
