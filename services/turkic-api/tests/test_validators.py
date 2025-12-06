"""Tests for API validators to reach 100% coverage."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError

from turkic_api.api.validators import (
    _decode_bool,
    _decode_float_range,
    _decode_int_range,
    _decode_optional_literal,
    _decode_required_literal,
    _decode_str,
    _load_json_dict,
)


def test_load_json_dict_valid() -> None:
    result = _load_json_dict({"key": "value"})
    assert result == {"key": "value"}


def test_load_json_dict_invalid() -> None:
    with pytest.raises(AppError) as exc_info:
        _load_json_dict("not a dict")
    assert exc_info.value.http_status == 400
    assert "Invalid request body" in exc_info.value.message


def test_decode_optional_literal_none() -> None:
    allowed = frozenset(["a", "b"])
    result = _decode_optional_literal(None, "field", allowed)
    assert result is None


def test_decode_optional_literal_not_string() -> None:
    allowed = frozenset(["a", "b"])
    with pytest.raises(AppError) as exc_info:
        _decode_optional_literal(123, "field", allowed)
    assert exc_info.value.http_status == 400
    assert "field must be a string" in exc_info.value.message


def test_decode_optional_literal_not_in_allowed() -> None:
    allowed = frozenset(["a", "b"])
    with pytest.raises(AppError) as exc_info:
        _decode_optional_literal("c", "field", allowed)
    assert exc_info.value.http_status == 400
    assert "field must be one of" in exc_info.value.message


def test_decode_optional_literal_valid() -> None:
    allowed = frozenset(["a", "b"])
    result = _decode_optional_literal("a", "field", allowed)
    assert result == "a"


def test_decode_required_literal_none() -> None:
    allowed = frozenset(["a", "b"])
    with pytest.raises(AppError) as exc_info:
        _decode_required_literal(None, "field", allowed)
    assert exc_info.value.http_status == 400
    assert "field is required" in exc_info.value.message


def test_decode_required_literal_not_string() -> None:
    allowed = frozenset(["a", "b"])
    with pytest.raises(AppError) as exc_info:
        _decode_required_literal(123, "field", allowed)
    assert exc_info.value.http_status == 400
    assert "field must be a string" in exc_info.value.message


def test_decode_required_literal_not_in_allowed() -> None:
    allowed = frozenset(["a", "b"])
    with pytest.raises(AppError) as exc_info:
        _decode_required_literal("c", "field", allowed)
    assert exc_info.value.http_status == 400
    assert "field must be one of" in exc_info.value.message


def test_decode_required_literal_valid() -> None:
    allowed = frozenset(["a", "b"])
    result = _decode_required_literal("a", "field", allowed)
    assert result == "a"


def test_decode_int_range_with_default() -> None:
    result = _decode_int_range(None, "field", default=10)
    assert result == 10


def test_decode_int_range_none_no_default() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_int_range(None, "field")
    assert exc_info.value.http_status == 400
    assert "field is required" in exc_info.value.message


def test_decode_int_range_not_int() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_int_range("123", "field")
    assert exc_info.value.http_status == 400
    assert "field must be an integer" in exc_info.value.message


def test_decode_int_range_less_than_ge() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_int_range(5, "field", ge=10)
    assert exc_info.value.http_status == 400
    assert "field must be >= 10" in exc_info.value.message


def test_decode_int_range_greater_than_le() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_int_range(15, "field", le=10)
    assert exc_info.value.http_status == 400
    assert "field must be <= 10" in exc_info.value.message


def test_decode_int_range_valid() -> None:
    result = _decode_int_range(5, "field", ge=0, le=10)
    assert result == 5


def test_decode_float_range_with_default() -> None:
    result = _decode_float_range(None, "field", default=1.5)
    assert result == 1.5


def test_decode_float_range_none_no_default() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_float_range(None, "field")
    assert exc_info.value.http_status == 400
    assert "field is required" in exc_info.value.message


def test_decode_float_range_not_number() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_float_range("abc", "field")
    assert exc_info.value.http_status == 400
    assert "field must be a number" in exc_info.value.message


def test_decode_float_range_less_than_ge() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_float_range(0.5, "field", ge=1.0)
    assert exc_info.value.http_status == 400
    assert "field must be >= 1.0" in exc_info.value.message


def test_decode_float_range_greater_than_le() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_float_range(1.5, "field", le=1.0)
    assert exc_info.value.http_status == 400
    assert "field must be <= 1.0" in exc_info.value.message


def test_decode_float_range_valid() -> None:
    result = _decode_float_range(0.5, "field", ge=0.0, le=1.0)
    assert result == 0.5


def test_decode_bool_with_default() -> None:
    result = _decode_bool(None, "field", default=True)
    assert result is True


def test_decode_bool_none_no_default() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_bool(None, "field")
    assert exc_info.value.http_status == 400
    assert "field is required" in exc_info.value.message


def test_decode_bool_not_bool() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_bool("true", "field")
    assert exc_info.value.http_status == 400
    assert "field must be a boolean" in exc_info.value.message


def test_decode_bool_valid() -> None:
    result = _decode_bool(True, "field")
    assert result is True


def test_decode_str_with_default() -> None:
    result = _decode_str(None, "field", default="default_value")
    assert result == "default_value"


def test_decode_str_none_no_default() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_str(None, "field")
    assert exc_info.value.http_status == 400
    assert "field is required" in exc_info.value.message


def test_decode_str_not_str() -> None:
    with pytest.raises(AppError) as exc_info:
        _decode_str(123, "field")
    assert exc_info.value.http_status == 400
    assert "field must be a string" in exc_info.value.message


def test_decode_str_valid() -> None:
    result = _decode_str("value", "field")
    assert result == "value"
