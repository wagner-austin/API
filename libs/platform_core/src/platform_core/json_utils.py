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
    "JSONValue",
    "dump_json_str",
    "load_json_bytes",
    "load_json_str",
    "register_json_error_handler",
]
