"""Common type definitions for instrument_io library.

Provides shared types used across all modules including signal types,
result wrappers, and re-exports from _json_bridge.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Literal, TypedDict

# Re-export from json bridge
from instrument_io._json_bridge import CellValue, JSONValue


class SignalType(StrEnum):
    """A signal an analytical instrument records a chromatogram from."""

    TIC = "TIC"
    EIC = "EIC"
    DAD = "DAD"
    UV = "UV"
    FID = "FID"
    MS = "MS"


class Polarity(StrEnum):
    """The ion polarity a mass spectrometer scanned in."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class MSLevel(IntEnum):
    """How many stages of mass spectrometry produced a spectrum."""

    MS1 = 1
    MS2 = 2
    MS3 = 3


class SuccessResult(TypedDict):
    """Result wrapper for successful operations."""

    status: Literal["success"]


class ErrorResult(TypedDict):
    """Result wrapper for failed operations.

    Attributes:
        status: Always "error".
        error_type: Exception class name.
        message: Human-readable error description.
        path: File path that caused the error.
    """

    status: Literal["error"]
    error_type: str
    message: str
    path: str


# Union type for operation results
OperationResult = SuccessResult | ErrorResult


def make_success() -> SuccessResult:
    """Create a success result."""
    return SuccessResult(status="success")


def make_error(error_type: str, message: str, path: str) -> ErrorResult:
    """Create an error result.

    Args:
        error_type: The exception class name.
        message: Human-readable description.
        path: File path that caused the error.

    Returns:
        ErrorResult TypedDict.
    """
    return ErrorResult(
        status="error",
        error_type=error_type,
        message=message,
        path=path,
    )


__all__ = [
    "CellValue",
    "ErrorResult",
    "JSONValue",
    "MSLevel",
    "OperationResult",
    "Polarity",
    "SignalType",
    "SuccessResult",
    "make_error",
    "make_success",
]
