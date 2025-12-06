"""Validation helpers for API request parsing."""

from __future__ import annotations

from platform_core.errors import ErrorCode
from platform_core.json_utils import JSONValue
from platform_core.validators import (
    load_json_dict,
    validate_bool,
    validate_float_range,
    validate_int_range,
    validate_optional_literal,
    validate_required_literal,
    validate_str,
)

_ERR = ErrorCode.INVALID_INPUT
_STATUS = 400


def _load_json(value: JSONValue) -> dict[str, JSONValue]:
    return load_json_dict(
        value,
        error_code=_ERR,
        message="Invalid request body",
        http_status=_STATUS,
    )


def _decode_optional(
    value: JSONValue,
    field: str,
    allowed: frozenset[str],
) -> str | None:
    return validate_optional_literal(
        value,
        field,
        allowed,
        error_code=_ERR,
        http_status=_STATUS,
    )


def _decode_required(
    value: JSONValue,
    field: str,
    allowed: frozenset[str],
) -> str:
    return validate_required_literal(
        value,
        field,
        allowed,
        error_code=_ERR,
        http_status=_STATUS,
    )


def _decode_int(
    value: JSONValue,
    field: str,
    *,
    ge: int | None = None,
    le: int | None = None,
    default: int | None = None,
) -> int:
    return validate_int_range(
        value,
        field,
        ge=ge,
        le=le,
        default=default,
        error_code=_ERR,
        http_status=_STATUS,
    )


def _decode_float(
    value: JSONValue,
    field: str,
    *,
    ge: float | None = None,
    le: float | None = None,
    default: float | None = None,
) -> float:
    return validate_float_range(
        value,
        field,
        ge=ge,
        le=le,
        default=default,
        error_code=_ERR,
        http_status=_STATUS,
    )


def _decode_flag(
    value: JSONValue,
    field: str,
    *,
    default: bool | None = None,
) -> bool:
    return validate_bool(
        value,
        field,
        default=default,
        error_code=_ERR,
        http_status=_STATUS,
    )


def _decode_text(
    value: JSONValue,
    field: str,
    *,
    default: str | None = None,
) -> str:
    return validate_str(
        value,
        field,
        default=default,
        error_code=_ERR,
        http_status=_STATUS,
    )


_load_json_dict = _load_json
_decode_optional_literal = _decode_optional
_decode_required_literal = _decode_required
_decode_int_range = _decode_int
_decode_float_range = _decode_float
_decode_bool = _decode_flag
_decode_str = _decode_text


__all__: list[str] = [
    "_decode_bool",
    "_decode_float_range",
    "_decode_int_range",
    "_decode_optional_literal",
    "_decode_required_literal",
    "_decode_str",
    "_load_json_dict",
]
