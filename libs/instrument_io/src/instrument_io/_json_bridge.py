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


def _df_json_to_row_dicts(json_str: str) -> list[dict[str, JSONValue]]:
    """Convert JSON-serialized DataFrame to list of typed row dicts.

    Use: df.write_json() -> pass result here to get typed iteration.
    This avoids iter_rows(named=True) which returns dict[str, Any].
    """
    value: JSONValue = _load_json_str(json_str)
    if not isinstance(value, list):
        return []
    result: list[dict[str, JSONValue]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
    return result


def _get_json_str_value(row: dict[str, JSONValue], key: str) -> str | None:
    """Get string value from JSON row dict, handling type narrowing.

    Converts non-string values to strings. Returns None for missing/null.
    """
    val: JSONValue = row.get(key)
    if val is None:
        return None
    if isinstance(val, str):
        stripped = val.strip()
        return stripped if stripped else None
    # Convert other types to string
    return str(val)


def _get_json_opt_str_value(row: dict[str, JSONValue], key: str) -> str | None:
    """Get optional string value from JSON row dict without stripping empty strings.

    Returns None only for missing/null values.
    """
    val: JSONValue = row.get(key)
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return str(val)


def _df_get_row_values(json_str: str, row_idx: int) -> list[str]:
    """Get all values from a single row as strings.

    Use: df.write_json() -> pass result and row index here.
    Returns list of string values for the row, converting nulls to empty strings.
    """
    rows = _df_json_to_row_dicts(json_str)
    if row_idx < 0 or row_idx >= len(rows):
        return []
    row = rows[row_idx]
    result: list[str] = []
    for val in row.values():
        if val is None:
            result.append("")
        elif isinstance(val, str):
            result.append(val)
        else:
            result.append(str(val))
    return result


def _df_get_headers_from_row(json_str: str, row_idx: int, columns: list[str]) -> list[str]:
    """Get header values from a specific row, using column positions.

    Use: df.write_json() -> pass result, row index, and column names.
    Returns list of string values to use as headers, with fallback for empty/null.
    """
    rows = _df_json_to_row_dicts(json_str)
    if row_idx < 0 or row_idx >= len(rows):
        return [f"col_{i}" for i in range(len(columns))]
    row = rows[row_idx]
    result: list[str] = []
    for i, col in enumerate(columns):
        val = row.get(col)
        if val is None:
            result.append(f"col_{i}")
        elif isinstance(val, str):
            stripped = val.strip()
            result.append(stripped if stripped else f"col_{i}")
        else:
            result.append(str(val).strip() or f"col_{i}")
    return result


def _df_get_cell_str(json_str: str, row_idx: int, col: str) -> str | None:
    """Get a single cell value as string from JSON-serialized DataFrame.

    Use: df.write_json() -> pass result, row index, and column name.
    Returns string value or None for null/missing.
    """
    rows = _df_json_to_row_dicts(json_str)
    if row_idx < 0 or row_idx >= len(rows):
        return None
    return _get_json_str_value(rows[row_idx], col)


def _df_slice_to_rows(json_str: str, start: int) -> list[list[str]]:
    """Get DataFrame rows from start index as lists of strings.

    Use: df.write_json() -> pass result and start index.
    Returns list of rows, each row being a list of string values.
    """
    rows = _df_json_to_row_dicts(json_str)
    if start >= len(rows):
        return []
    result: list[list[str]] = []
    for row in rows[start:]:
        row_vals: list[str] = []
        for val in row.values():
            if val is None:
                row_vals.append("")
            elif isinstance(val, str):
                row_vals.append(val)
            else:
                row_vals.append(str(val))
        result.append(row_vals)
    return result


__all__ = [
    "CellValue",
    "JSONValue",
    "_df_get_cell_str",
    "_df_get_headers_from_row",
    "_df_get_row_values",
    "_df_json_to_row_dicts",
    "_df_slice_to_rows",
    "_extract_row_from_dict",
    "_get_json_opt_str_value",
    "_get_json_str_value",
    "_json_col_to_float_list",
    "_json_col_to_int_list",
    "_json_col_to_opt_float_list",
    "_json_col_to_opt_str_list",
    "_json_col_to_str_list",
    "_json_value_to_cell",
    "_load_json_str",
]
