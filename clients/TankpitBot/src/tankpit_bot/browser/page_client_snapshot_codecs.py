"""Codecs for the nested blocks of a page-client snapshot.

The field map and collection blocks, their JSON narrowing helpers, and
the runtime-value extractor. The snapshot record that composes them is
:mod:`tankpit_bot.browser.page_client_snapshot`.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_dict,
)


def _extract_runtime_value(result: JSONObject) -> JSONValue:
    """Return the ``Runtime.evaluate`` value field.

    Args:
        result: Raw CDP result object returned by ``cdp.send``.

    Returns:
        The evaluated JavaScript value.

    Raises:
        ValueError: If the CDP result is missing the value field.
    """
    result_obj = require_dict(result, "result")
    if "value" not in result_obj:
        raise ValueError(f"Runtime.evaluate result missing value: {result_obj}")
    return result_obj["value"]


def _require_optional_int(data: JSONObject, field: str) -> int | None:
    """Return an optional integer field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Integer value or None when the field is null.

    Raises:
        JSONTypeError: If the field is present but not an integer.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise JSONTypeError(f"Field '{field}' must be an integer or null")
    return raw


def _require_optional_bool(data: JSONObject, field: str) -> bool | None:
    """Return an optional boolean field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Boolean value or None when the field is null.

    Raises:
        JSONTypeError: If the field is present but not a boolean.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise JSONTypeError(f"Field '{field}' must be a boolean or null")
    return raw


def _require_optional_str(data: JSONObject, field: str) -> str | None:
    """Return an optional string field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        String value or None when the field is null.

    Raises:
        JSONTypeError: If the field is present but not a string.
    """
    raw = data.get(field)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise JSONTypeError(f"Field '{field}' must be a string or null")
    return raw


def _require_bool_field(data: JSONObject, field: str) -> bool:
    """Return a required boolean field from a JSON object.

    Args:
        data: JSON object to inspect.
        field: Field name to validate.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If the field is not a boolean.
    """
    raw = data.get(field)
    if not isinstance(raw, bool):
        raise JSONTypeError(f"Field '{field}' must be a boolean")
    return raw


def _require_client_field_value(
    value: JSONValue,
    field: str,
    key: str,
) -> int | float | bool | str | None:
    """Validate one entry of a client field map as a JSON primitive.

    Args:
        value: Raw value associated with ``key``.
        field: Outer field name (for error reporting).
        key: Inner key inside the field map (for error reporting).

    Returns:
        Validated primitive value.

    Raises:
        JSONTypeError: If ``value`` is a list or dict (only primitives are
            accepted in field maps; the JS-side primitivesOnly filter is
            the source of truth, this is the strict re-check).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    raise JSONTypeError(
        f"Field '{field}' entry '{key}' must be a JSON primitive (got {type(value).__name__})"
    )


def decode_client_field_map(
    raw: JSONObject,
    *,
    field: str,
) -> dict[str, int | float | bool | str | None]:
    """Decode a raw JSON object as a minified-key field map.

    Args:
        raw: JSON object whose entries are all expected to be primitives.
        field: Outer field name used in error messages.

    Returns:
        Validated mapping of minified key names to primitive scalars.

    Raises:
        JSONTypeError: If any entry holds a non-primitive value.
    """
    result: dict[str, int | float | bool | str | None] = {}
    for key, value in raw.items():
        result[key] = _require_client_field_value(value, field, key)
    return result


def _require_client_field_map(
    data: JSONObject,
    field: str,
) -> dict[str, int | float | bool | str | None]:
    """Decode a discovery field map (minified key -> primitive value).

    Args:
        data: JSON object to inspect.
        field: Field name to read and validate.

    Returns:
        Validated mapping of minified key names to primitive scalars.

    Raises:
        JSONTypeError: If the field is missing, not an object, or contains
            non-primitive values.
    """
    raw = data.get(field)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object")
    return decode_client_field_map(raw, field=field)


def encode_client_field_map(
    field_map: dict[str, int | float | bool | str | None],
) -> JSONObject:
    """Encode a discovery field map as a JSON object.

    Args:
        field_map: Mapping of minified key names to primitive scalars.

    Returns:
        JSON-ready object that preserves insertion order.
    """
    result: JSONObject = {}
    for key, value in field_map.items():
        result[key] = value
    return result


def decode_client_collections(
    raw: JSONObject,
    *,
    field: str,
) -> dict[str, list[dict[str, int | float | bool | str | None]]]:
    """Decode a raw JSON object as a minified-key collection map.

    Args:
        raw: JSON object whose values are lists of primitive field maps.
        field: Outer field name used in error messages.

    Returns:
        Validated mapping of minified property names to item lists.

    Raises:
        JSONTypeError: If any value is not a list, any item is not an
            object, or any item field holds a non-primitive value.
    """
    result: dict[str, list[dict[str, int | float | bool | str | None]]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            raise JSONTypeError(f"Field '{field}' entry '{key}' must be a list")
        items: list[dict[str, int | float | bool | str | None]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise JSONTypeError(f"Field '{field}' entry '{key}[{index}]' must be an object")
            items.append(decode_client_field_map(item, field=f"{field}.{key}[{index}]"))
        result[key] = items
    return result


def _require_client_collections(
    data: JSONObject,
    field: str,
) -> dict[str, list[dict[str, int | float | bool | str | None]]]:
    """Decode a required collection map field from a snapshot payload.

    Args:
        data: JSON object to inspect.
        field: Field name to read and validate.

    Returns:
        Validated mapping of minified property names to item lists.

    Raises:
        JSONTypeError: If the field is missing, not an object, or fails
            item validation.
    """
    raw = data.get(field)
    if not isinstance(raw, dict):
        raise JSONTypeError(f"Field '{field}' must be an object")
    return decode_client_collections(raw, field=field)


def encode_client_collections(
    collections: dict[str, list[dict[str, int | float | bool | str | None]]],
) -> JSONObject:
    """Encode a collection map as a JSON object.

    Args:
        collections: Mapping of minified property names to item lists.

    Returns:
        JSON-ready object that preserves insertion order.
    """
    result: JSONObject = {}
    for key, items in collections.items():
        encoded_items: list[JSONValue] = [encode_client_field_map(item) for item in items]
        result[key] = encoded_items
    return result


__all__ = [
    "decode_client_collections",
    "decode_client_field_map",
    "encode_client_collections",
    "encode_client_field_map",
]
