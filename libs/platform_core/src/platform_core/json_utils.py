from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from json import JSONDecodeError
from typing import Protocol

from fastapi import FastAPI
from fastapi.responses import Response
from starlette.requests import Request

JSONValue = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None

# Input type for dump_json_str - broad enough to accept:
# 1. TypedDict subclasses (via Mapping)
# 2. Dict literals with mixed value types
# 3. Primitives and sequences
# Using object allows any JSON-serializable value since we only serialize (read) the data
_JSONInputValue = str | int | float | bool | None | Mapping[str, object] | Sequence[object]


class InvalidJsonError(ValueError):
    """Raised when JSON parsing fails."""


class _JsonLoads(Protocol):
    def __call__(self, s: str) -> JSONValue: ...


class _JsonDumps(Protocol):
    def __call__(
        self,
        obj: _JSONInputValue,
        *,
        separators: tuple[str, str] | None = ...,
        indent: int | None = ...,
    ) -> str: ...


def dump_json_str(
    value: _JSONInputValue, *, compact: bool = True, indent: int | None = None
) -> str:
    """Serialize a JSON-compatible value to a JSON string.

    Args:
        value: JSON-serializable value
        compact: If True (default), produce compact JSON without extra whitespace.
                 Ignored if indent is specified.
        indent: If specified, pretty-print with this many spaces of indentation.
    """
    module = __import__("json")
    dumps: _JsonDumps = module.dumps
    if indent is not None:
        return dumps(value, separators=None, indent=indent)
    if compact:
        return dumps(value, separators=(",", ":"), indent=None)
    return dumps(value, separators=None, indent=None)


def load_json_str(raw: str) -> JSONValue:
    module = __import__("json")
    loads: _JsonLoads = module.loads
    try:
        value = loads(raw)
    except JSONDecodeError as exc:
        raise InvalidJsonError("Invalid JSON payload") from exc
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    raise InvalidJsonError("Invalid JSON payload")


def load_json_bytes(raw: bytes) -> JSONValue:
    text = raw.decode("utf-8")
    return load_json_str(text)


class JSONTypeError(TypeError):
    """Raised when JSON value has unexpected type during narrowing."""


def narrow_json_to_dict(value: JSONValue) -> dict[str, JSONValue]:
    """Narrow JSONValue to dict.

    Raises JSONTypeError if value is not a dict.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"Expected JSON object, got {type(value).__name__}")
    return value


def narrow_json_to_list(value: JSONValue) -> list[JSONValue]:
    """Narrow JSONValue to list.

    Raises JSONTypeError if value is not a list.
    """
    if not isinstance(value, list):
        raise JSONTypeError(f"Expected JSON array, got {type(value).__name__}")
    return value


def narrow_json_to_str(value: JSONValue) -> str:
    """Narrow JSONValue to str.

    Raises JSONTypeError if value is not a str.
    """
    if not isinstance(value, str):
        raise JSONTypeError(f"Expected JSON string, got {type(value).__name__}")
    return value


def narrow_json_to_int(value: JSONValue) -> int:
    """Narrow JSONValue to int.

    Raises JSONTypeError if value is not an int (excludes bool).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise JSONTypeError(f"Expected JSON integer, got {type(value).__name__}")
    return value


def narrow_json_to_float(value: JSONValue) -> float:
    """Narrow JSONValue to float (accepts int as well).

    Raises JSONTypeError if value is not a number.
    """
    if isinstance(value, bool):
        raise JSONTypeError(f"Expected JSON number, got {type(value).__name__}")
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    raise JSONTypeError(f"Expected JSON number, got {type(value).__name__}")


def narrow_json_to_bool(value: JSONValue) -> bool:
    """Narrow JSONValue to bool.

    Raises JSONTypeError if value is not a bool.
    """
    if not isinstance(value, bool):
        raise JSONTypeError(f"Expected JSON boolean, got {type(value).__name__}")
    return value


# -----------------------------------------------------------------------------
# Field extraction helpers - extract and validate fields from JSON objects
# -----------------------------------------------------------------------------

JSONObject = dict[str, JSONValue]


def require_str(obj: JSONObject, key: str) -> str:
    """Extract required string field from JSON object.

    Raises JSONTypeError if field is missing or not a string.
    """
    value = obj.get(key)
    if value is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(value, str):
        raise JSONTypeError(f"Field '{key}' must be a string, got {type(value).__name__}")
    return value


def require_int(obj: JSONObject, key: str) -> int:
    """Extract required int field from JSON object.

    Raises JSONTypeError if field is missing or not an int (excludes bool).
    """
    value = obj.get(key)
    if value is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if isinstance(value, bool) or not isinstance(value, int):
        raise JSONTypeError(f"Field '{key}' must be an integer, got {type(value).__name__}")
    return value


def require_float(obj: JSONObject, key: str) -> float:
    """Extract required float field from JSON object (accepts int as well).

    Raises JSONTypeError if field is missing or not a number.
    """
    value = obj.get(key)
    if value is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if isinstance(value, bool):
        raise JSONTypeError(f"Field '{key}' must be a number, got {type(value).__name__}")
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    raise JSONTypeError(f"Field '{key}' must be a number, got {type(value).__name__}")


def require_bool(obj: JSONObject, key: str) -> bool:
    """Extract required bool field from JSON object.

    Raises JSONTypeError if field is missing or not a bool.
    """
    value = obj.get(key)
    if value is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(value, bool):
        raise JSONTypeError(f"Field '{key}' must be a boolean, got {type(value).__name__}")
    return value


def require_list(obj: JSONObject, key: str) -> list[JSONValue]:
    """Extract required list field from JSON object.

    Raises JSONTypeError if field is missing or not a list.
    """
    value = obj.get(key)
    if value is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(value, list):
        raise JSONTypeError(f"Field '{key}' must be an array, got {type(value).__name__}")
    return value


def require_dict(obj: JSONObject, key: str) -> JSONObject:
    """Extract required dict field from JSON object.

    Raises JSONTypeError if field is missing or not a dict.
    """
    value = obj.get(key)
    if value is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(value, dict):
        raise JSONTypeError(f"Field '{key}' must be an object, got {type(value).__name__}")
    return value


# -----------------------------------------------------------------------------
# Optional field extraction helpers
# -----------------------------------------------------------------------------


def optional_str(obj: JSONObject, key: str) -> str | None:
    """Extract optional string field from JSON object.

    Returns None if field is missing. Raises JSONTypeError if present but not a string.
    """
    if key not in obj:
        return None
    value = obj[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise JSONTypeError(f"Field '{key}' must be a string, got {type(value).__name__}")
    return value


def optional_int(obj: JSONObject, key: str) -> int | None:
    """Extract optional int field from JSON object.

    Returns None if field is missing. Raises JSONTypeError if present but not an int.
    """
    if key not in obj:
        return None
    value = obj[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise JSONTypeError(f"Field '{key}' must be an integer, got {type(value).__name__}")
    return value


def optional_float(obj: JSONObject, key: str) -> float | None:
    """Extract optional float field from JSON object (accepts int as well).

    Returns None if field is missing. Raises JSONTypeError if present but not a number.
    """
    if key not in obj:
        return None
    value = obj[key]
    if value is None:
        return None
    if isinstance(value, bool):
        raise JSONTypeError(f"Field '{key}' must be a number, got {type(value).__name__}")
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    raise JSONTypeError(f"Field '{key}' must be a number, got {type(value).__name__}")


def register_json_error_handler(
    app: FastAPI, *, detail: str = "Invalid JSON body"
) -> Callable[[Request, Exception], Response | Awaitable[Response]]:
    handler_type = Callable[[Request, Exception], Response | Awaitable[Response]]
    from platform_core.errors import AppError, ErrorCode

    def _handler(_: Request, exc: Exception) -> Response:
        if isinstance(exc, (InvalidJsonError, JSONDecodeError)):
            raise AppError(code=ErrorCode.INVALID_INPUT, message=detail, http_status=400) from exc
        raise exc

    handler: handler_type = _handler
    app.add_exception_handler(InvalidJsonError, handler)
    app.add_exception_handler(JSONDecodeError, handler)
    return handler


__all__ = [
    "InvalidJsonError",
    "JSONObject",
    "JSONTypeError",
    "JSONValue",
    "dump_json_str",
    "load_json_bytes",
    "load_json_str",
    "narrow_json_to_bool",
    "narrow_json_to_dict",
    "narrow_json_to_float",
    "narrow_json_to_int",
    "narrow_json_to_list",
    "narrow_json_to_str",
    "optional_float",
    "optional_int",
    "optional_str",
    "register_json_error_handler",
    "require_bool",
    "require_dict",
    "require_float",
    "require_int",
    "require_list",
    "require_str",
]
