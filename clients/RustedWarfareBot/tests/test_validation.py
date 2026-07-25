"""Field-validator behaviour, including every rejection path."""

from __future__ import annotations

import pytest

from rw_bot.validation import (
    DecodeError,
    require_absolute_path,
    require_bool,
    require_int,
    require_non_empty_str,
    require_positive_int,
    require_str,
)


def test_require_str_returns_value() -> None:
    assert require_str({"name": "engine"}, "name") == "engine"


def test_require_str_rejects_absent_field() -> None:
    with pytest.raises(DecodeError) as caught:
        require_str({}, "name")
    assert caught.value.code == "RW-DECODE-001"
    assert caught.value.message == "required field 'name' is absent"


def test_require_str_rejects_wrong_type() -> None:
    with pytest.raises(DecodeError) as caught:
        require_str({"name": 7}, "name")
    assert caught.value.code == "RW-DECODE-002"
    assert caught.value.message == "field 'name' must be str, got int"


def test_require_non_empty_str_returns_value() -> None:
    assert require_non_empty_str({"dir": ".game"}, "dir") == ".game"


def test_require_non_empty_str_rejects_whitespace_only() -> None:
    with pytest.raises(DecodeError) as caught:
        require_non_empty_str({"dir": "   "}, "dir")
    assert caught.value.code == "RW-DECODE-003"
    assert caught.value.message == "field 'dir' must not be blank"


def test_require_int_returns_value() -> None:
    assert require_int({"code": 176}, "code") == 176


def test_require_int_rejects_bool() -> None:
    with pytest.raises(DecodeError) as caught:
        require_int({"code": True}, "code")
    assert caught.value.code == "RW-DECODE-002"
    assert caught.value.message == "field 'code' must be int, got bool"


def test_require_int_rejects_numeric_string_without_coercing() -> None:
    with pytest.raises(DecodeError) as caught:
        require_int({"code": "176"}, "code")
    assert caught.value.code == "RW-DECODE-002"


def test_require_positive_int_returns_value() -> None:
    assert require_positive_int({"width": 800}, "width") == 800


@pytest.mark.parametrize("value", [0, -1])
def test_require_positive_int_rejects_non_positive(value: int) -> None:
    with pytest.raises(DecodeError) as caught:
        require_positive_int({"width": value}, "width")
    assert caught.value.code == "RW-DECODE-004"
    assert caught.value.message == f"field 'width' must be > 0, got {value}"


def test_require_bool_returns_value() -> None:
    assert require_bool({"sandbox": False}, "sandbox") is False


def test_require_bool_rejects_int() -> None:
    with pytest.raises(DecodeError) as caught:
        require_bool({"sandbox": 1}, "sandbox")
    assert caught.value.code == "RW-DECODE-002"
    assert caught.value.message == "field 'sandbox' must be bool, got int"


@pytest.mark.parametrize(
    "value",
    [
        "C:/runs/boot.log",
        "C:\\runs\\boot.log",
        "\\\\server\\share\\boot.log",
    ],
)
def test_require_absolute_path_accepts_absolute_windows_paths(value: str) -> None:
    assert require_absolute_path({"log_path": value}, "log_path") == value


@pytest.mark.parametrize(
    "value",
    [
        "runs/boot.log",
        "./boot.log",
        "/runs/boot.log",
    ],
)
def test_require_absolute_path_rejects_relative_paths(value: str) -> None:
    """``/runs/boot.log`` is drive-rooted but not absolute under Windows rules."""
    with pytest.raises(DecodeError) as caught:
        require_absolute_path({"log_path": value}, "log_path")
    assert caught.value.code == "RW-DECODE-005"
    assert "must be an absolute path" in caught.value.message


def test_require_absolute_path_rejects_blank_before_testing_shape() -> None:
    with pytest.raises(DecodeError) as caught:
        require_absolute_path({"log_path": "  "}, "log_path")
    assert caught.value.code == "RW-DECODE-003"


def test_require_absolute_path_rejects_wrong_type() -> None:
    with pytest.raises(DecodeError) as caught:
        require_absolute_path({"log_path": 5}, "log_path")
    assert caught.value.code == "RW-DECODE-002"
