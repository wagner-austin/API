from __future__ import annotations

from platform_core.errors import AppError, ErrorCode, ErrorCodeBase
from platform_core.json_utils import JSONValue


def _load_json_dict(
    obj: JSONValue,
    *,
    error_code: ErrorCodeBase = ErrorCode.INVALID_INPUT,
    message: str = "Invalid request body",
    http_status: int = 400,
) -> dict[str, JSONValue]:
    """Ensure object is a dict, otherwise raise AppError."""
    if isinstance(obj, dict):
        return obj
    raise AppError(code=error_code, message=message, http_status=http_status)


def _decode_optional_literal(
    value: JSONValue,
    field: str,
    allowed: frozenset[str],
    *,
    error_code: ErrorCodeBase = ErrorCode.INVALID_INPUT,
    http_status: int = 400,
) -> str | None:
    """Validate optional string literal; returns None when value is missing."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AppError(
            code=error_code,
            message=f"{field} must be a string",
            http_status=http_status,
        )
    if value not in allowed:
        opts = ", ".join(sorted(allowed))
        raise AppError(
            code=error_code,
            message=f"{field} must be one of: {opts}",
            http_status=http_status,
        )
    return value


def _decode_required_literal(
    value: JSONValue,
    field: str,
    allowed: frozenset[str],
    *,
    error_code: ErrorCodeBase = ErrorCode.INVALID_INPUT,
    http_status: int = 400,
) -> str:
    """Validate required string literal."""
    if value is None:
        raise AppError(
            code=error_code,
            message=f"{field} is required",
            http_status=http_status,
        )
    if not isinstance(value, str):
        raise AppError(
            code=error_code,
            message=f"{field} must be a string",
            http_status=http_status,
        )
    if value not in allowed:
        opts = ", ".join(sorted(allowed))
        raise AppError(
            code=error_code,
            message=f"{field} must be one of: {opts}",
            http_status=http_status,
        )
    return value


def _decode_int_range(
    value: JSONValue,
    field: str,
    *,
    ge: int | None = None,
    le: int | None = None,
    default: int | None = None,
    error_code: ErrorCodeBase = ErrorCode.INVALID_INPUT,
    http_status: int = 400,
) -> int:
    """Validate integer within inclusive bounds."""
    if value is None:
        if default is not None:
            return default
        raise AppError(
            code=error_code,
            message=f"{field} is required",
            http_status=http_status,
        )
    if not isinstance(value, int):
        raise AppError(
            code=error_code,
            message=f"{field} must be an integer",
            http_status=http_status,
        )
    if ge is not None and value < ge:
        raise AppError(
            code=error_code,
            message=f"{field} must be >= {ge}",
            http_status=http_status,
        )
    if le is not None and value > le:
        raise AppError(
            code=error_code,
            message=f"{field} must be <= {le}",
            http_status=http_status,
        )
    return value


def _decode_float_range(
    value: JSONValue,
    field: str,
    *,
    ge: float | None = None,
    le: float | None = None,
    default: float | None = None,
    error_code: ErrorCodeBase = ErrorCode.INVALID_INPUT,
    http_status: int = 400,
) -> float:
    """Validate float within inclusive bounds."""
    if value is None:
        if default is not None:
            return default
        raise AppError(
            code=error_code,
            message=f"{field} is required",
            http_status=http_status,
        )
    if not isinstance(value, (int, float)):
        raise AppError(
            code=error_code,
            message=f"{field} must be a number",
            http_status=http_status,
        )
    fval = float(value)
    if ge is not None and fval < ge:
        raise AppError(
            code=error_code,
            message=f"{field} must be >= {ge}",
            http_status=http_status,
        )
    if le is not None and fval > le:
        raise AppError(
            code=error_code,
            message=f"{field} must be <= {le}",
            http_status=http_status,
        )
    return fval


def _decode_bool(
    value: JSONValue,
    field: str,
    *,
    default: bool | None = None,
    error_code: ErrorCodeBase = ErrorCode.INVALID_INPUT,
    http_status: int = 400,
) -> bool:
    """Validate boolean value."""
    if value is None:
        if default is not None:
            return default
        raise AppError(
            code=error_code,
            message=f"{field} is required",
            http_status=http_status,
        )
    if not isinstance(value, bool):
        raise AppError(
            code=error_code,
            message=f"{field} must be a boolean",
            http_status=http_status,
        )
    return value


def _decode_str(
    value: JSONValue,
    field: str,
    *,
    default: str | None = None,
    error_code: ErrorCodeBase = ErrorCode.INVALID_INPUT,
    http_status: int = 400,
) -> str:
    """Validate string value."""
    if value is None:
        if default is not None:
            return default
        raise AppError(
            code=error_code,
            message=f"{field} is required",
            http_status=http_status,
        )
    if not isinstance(value, str):
        raise AppError(
            code=error_code,
            message=f"{field} must be a string",
            http_status=http_status,
        )
    return value


load_json_dict = _load_json_dict
validate_optional_literal = _decode_optional_literal
validate_required_literal = _decode_required_literal
validate_int_range = _decode_int_range
validate_float_range = _decode_float_range
validate_bool = _decode_bool
validate_str = _decode_str


__all__ = [
    "JSONValue",
    "load_json_dict",
    "validate_bool",
    "validate_float_range",
    "validate_int_range",
    "validate_optional_literal",
    "validate_required_literal",
    "validate_str",
]
