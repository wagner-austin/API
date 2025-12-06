from __future__ import annotations

import pytest

from platform_core.errors import AppError, ErrorCode
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


def test_load_json_dict_accepts_dict() -> None:
    payload: JSONValue = {"a": 1}
    assert load_json_dict(payload) == {"a": 1}


def test_load_json_dict_rejects_non_dict() -> None:
    with pytest.raises(AppError) as excinfo:
        load_json_dict(["x"])
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.http_status == 400


def test_validate_optional_literal_allows_none_and_valid_values() -> None:
    allowed = frozenset({"A", "B"})
    assert validate_optional_literal(None, "opt", allowed) is None
    assert validate_optional_literal("A", "opt", allowed) == "A"


def test_validate_optional_literal_rejects_bad_value() -> None:
    allowed = frozenset({"A", "B"})
    with pytest.raises(AppError) as excinfo:
        validate_optional_literal("C", "level", allowed)
    assert "one of" in excinfo.value.message
    with pytest.raises(AppError):
        validate_optional_literal(123, "level", allowed)


def test_validate_optional_literal_rejects_non_str_type() -> None:
    allowed = frozenset({"yes", "no"})
    with pytest.raises(AppError):
        validate_optional_literal(True, "flag", allowed)


def test_validate_required_literal_enforces_presence_and_allowed() -> None:
    allowed = frozenset({"L1", "L2"})
    assert validate_required_literal("L1", "level", allowed) == "L1"
    with pytest.raises(AppError):
        validate_required_literal(None, "level", allowed)
    with pytest.raises(AppError):
        validate_required_literal("X", "level", allowed)
    with pytest.raises(AppError):
        validate_required_literal(5, "level", allowed)


def test_validate_int_range_bounds_and_default() -> None:
    assert validate_int_range(5, "count", ge=1, le=10) == 5
    assert validate_int_range(None, "count", default=7) == 7
    with pytest.raises(AppError):
        validate_int_range("x", "count")
    with pytest.raises(AppError):
        validate_int_range(0, "count", ge=1)
    with pytest.raises(AppError):
        validate_int_range(11, "count", le=10)
    with pytest.raises(AppError):
        validate_int_range(None, "count")


def test_validate_float_range_bounds_and_default() -> None:
    assert validate_float_range(1.5, "ratio", ge=0.5, le=2.0) == 1.5
    assert validate_float_range(None, "ratio", default=0.75) == 0.75
    with pytest.raises(AppError):
        validate_float_range("bad", "ratio")
    with pytest.raises(AppError):
        validate_float_range(0.25, "ratio", ge=0.5)
    with pytest.raises(AppError):
        validate_float_range(3.0, "ratio", le=2.5)
    with pytest.raises(AppError):
        validate_float_range(None, "ratio")


def test_validate_bool_enforces_type_and_default() -> None:
    assert validate_bool(True, "flag") is True
    assert validate_bool(None, "flag", default=False) is False
    with pytest.raises(AppError):
        validate_bool("no", "flag")
    with pytest.raises(AppError):
        validate_bool(None, "flag")


def test_validate_str_enforces_type_and_default() -> None:
    assert validate_str("ok", "name") == "ok"
    assert validate_str(None, "name", default="d") == "d"
    with pytest.raises(AppError):
        validate_str(1, "name")
    with pytest.raises(AppError):
        validate_str(None, "name")


def test_validate_functions_can_use_custom_error_code() -> None:
    code = ErrorCode.UNAUTHORIZED
    with pytest.raises(AppError) as excinfo:
        validate_required_literal(None, "api_key", frozenset(), error_code=code, http_status=401)
    assert excinfo.value.code is code
    assert excinfo.value.http_status == 401
