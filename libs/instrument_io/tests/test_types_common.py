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


def test_signal_type_values() -> None:
    # Verify SignalType literal values
    tic: SignalType = "TIC"
    eic: SignalType = "EIC"
    dad: SignalType = "DAD"
    uv: SignalType = "UV"
    fid: SignalType = "FID"
    ms: SignalType = "MS"

    assert tic == "TIC"
    assert eic == "EIC"
    assert dad == "DAD"
    assert uv == "UV"
    assert fid == "FID"
    assert ms == "MS"


def test_polarity_values() -> None:
    pos: Polarity = "positive"
    neg: Polarity = "negative"
    unk: Polarity = "unknown"

    assert pos == "positive"
    assert neg == "negative"
    assert unk == "unknown"


def test_ms_level_values() -> None:
    ms1: MSLevel = 1
    ms2: MSLevel = 2
    ms3: MSLevel = 3

    assert ms1 == 1
    assert ms2 == 2
    assert ms3 == 3
