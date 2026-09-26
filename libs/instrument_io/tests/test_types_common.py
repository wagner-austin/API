"""Tests for types.common module."""

from __future__ import annotations

from instrument_io.types.common import (
    ErrorResult,
    MSLevel,
    OperationResult,
    Polarity,
    SignalType,
    SuccessResult,
    make_error,
    make_success,
)


def test_make_success() -> None:
    result: SuccessResult = make_success()
    assert result["status"] == "success"


def test_make_error_with_details() -> None:
    result: ErrorResult = make_error("ValueError", "Something went wrong", "/path/to/file")
    assert result["status"] == "error"
    assert result["error_type"] == "ValueError"
    assert result["message"] == "Something went wrong"
    assert result["path"] == "/path/to/file"


def test_make_error_empty_message() -> None:
    result: ErrorResult = make_error("EmptyError", "", "/empty")
    assert result["status"] == "error"
    assert result["error_type"] == "EmptyError"
    assert result["message"] == ""
    assert result["path"] == "/empty"


def test_operation_result_success_type() -> None:
    result: OperationResult = make_success()
    assert result["status"] == "success"


def test_operation_result_error_type() -> None:
    result: OperationResult = make_error("TestError", "test", "/test")
    assert result["status"] == "error"


def test_signal_types_are_spelled_as_the_instruments_spell_them() -> None:
    assert [signal.value for signal in SignalType] == ["TIC", "EIC", "DAD", "UV", "FID", "MS"]


def test_polarity_values() -> None:
    assert [polarity.value for polarity in Polarity] == ["positive", "negative", "unknown"]


def test_ms_level_values() -> None:
    assert [level.value for level in MSLevel] == [1, 2, 3]
