"""Field-validator behaviour, including every rejection path."""

from __future__ import annotations

import pytest

from rw_bot.platform_id import WINDOWS
from rw_bot.validation import (
    DecodeError,
    require_absolute_path,
    require_bool,
    require_finite_float,
    require_int,
    require_non_empty_str,
    require_positive_int,
    require_str,
)

#: The platform the cluster runs, so the POSIX half of every rule below is
#: exercised from the Windows workstation the suite runs on.
LINUX = "linux"


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
    assert require_absolute_path({"log_path": value}, "log_path", WINDOWS) == value


@pytest.mark.parametrize("value", ["/runs/boot.log", "/pub/rw/runs/boot.log"])
def test_require_absolute_path_accepts_absolute_posix_paths(value: str) -> None:
    assert require_absolute_path({"log_path": value}, "log_path", LINUX) == value


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
        require_absolute_path({"log_path": value}, "log_path", WINDOWS)
    assert caught.value.code == "RW-DECODE-005"
    assert "must be an absolute path" in caught.value.message


@pytest.mark.parametrize("value", ["runs/boot.log", "./boot.log", "C:/runs/boot.log"])
def test_require_absolute_path_rejects_relative_posix_paths(value: str) -> None:
    """``C:/runs/boot.log`` has no root at all on POSIX -- it is a directory
    literally named ``C:``, relative to wherever the process happens to be."""
    with pytest.raises(DecodeError) as caught:
        require_absolute_path({"log_path": value}, "log_path", LINUX)
    assert caught.value.code == "RW-DECODE-005"


def test_the_two_families_disagree_about_the_case_that_matters() -> None:
    """Reading a path under the wrong family does not mis-parse it -- it
    INVERTS this check, accepting exactly what it exists to reject. Both
    directions, because a rule that is only wrong one way is a rule half
    checked."""
    drive_rooted = {"log_path": "/runs/boot.log"}
    lettered = {"log_path": "C:/runs/boot.log"}
    assert require_absolute_path(drive_rooted, "log_path", LINUX) == "/runs/boot.log"
    assert require_absolute_path(lettered, "log_path", WINDOWS) == "C:/runs/boot.log"
    with pytest.raises(DecodeError):
        require_absolute_path(drive_rooted, "log_path", WINDOWS)
    with pytest.raises(DecodeError):
        require_absolute_path(lettered, "log_path", LINUX)


def test_require_absolute_path_rejects_blank_before_testing_shape() -> None:
    with pytest.raises(DecodeError) as caught:
        require_absolute_path({"log_path": "  "}, "log_path", WINDOWS)
    assert caught.value.code == "RW-DECODE-003"


def test_require_absolute_path_rejects_wrong_type() -> None:
    with pytest.raises(DecodeError) as caught:
        require_absolute_path({"log_path": 5}, "log_path", WINDOWS)
    assert caught.value.code == "RW-DECODE-002"


def test_require_finite_float_returns_a_float() -> None:
    assert require_finite_float({"x": 4250.5}, "x") == 4250.5


def test_require_finite_float_widens_an_int() -> None:
    """A whole-numbered coordinate is emitted as 4250, not 4250.0."""
    # repr distinguishes the int 4250 from the float 4250.0; == does not.
    assert repr(require_finite_float({"x": 4250}, "x")) == "4250.0"


def test_require_finite_float_accepts_zero_and_negatives() -> None:
    assert require_finite_float({"x": 0}, "x") == 0.0
    assert require_finite_float({"x": -2610.75}, "x") == -2610.75


def test_require_finite_float_rejects_an_absent_field() -> None:
    with pytest.raises(DecodeError) as caught:
        require_finite_float({}, "x")
    assert caught.value.code == "RW-DECODE-001"
    assert caught.value.message == "required field 'x' is absent"


def test_require_finite_float_rejects_bool() -> None:
    with pytest.raises(DecodeError) as caught:
        require_finite_float({"x": True}, "x")
    assert caught.value.code == "RW-DECODE-002"
    assert caught.value.message == "field 'x' must be a number, got bool"


def test_require_finite_float_rejects_a_numeric_string_without_coercing() -> None:
    with pytest.raises(DecodeError) as caught:
        require_finite_float({"x": "4250.0"}, "x")
    assert caught.value.code == "RW-DECODE-002"
    assert caught.value.message == "field 'x' must be a number, got str"


@pytest.mark.parametrize("literal", ["nan", "inf", "-inf"])
def test_require_finite_float_rejects_non_finite(literal: str) -> None:
    """JSON cannot carry these, so their presence is a producer bug."""
    with pytest.raises(DecodeError) as caught:
        require_finite_float({"x": float(literal)}, "x")
    assert caught.value.code == "RW-DECODE-006"
    assert caught.value.message.startswith("field 'x' must be finite, got ")
