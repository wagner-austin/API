"""Container decoding helper functions and error types.

This module provides validation functions and the ContainerDecodeError
exception for container message decoding.
"""

from __future__ import annotations


class ContainerDecodeError(Exception):
    """Raised when container message decoding fails validation.

    Args:
        message: Description of the validation failure.
    """

    def __init__(self, message: str) -> None:
        """Initialize with error message.

        Args:
            message: Description of the validation failure.
        """
        super().__init__(message)
        self.message = message


def require_min_length(data: bytes, min_len: int, context: str) -> None:
    """Validate data meets minimum length requirement.

    Args:
        data: Bytes to validate.
        min_len: Minimum required length.
        context: Context string for error message.

    Raises:
        ContainerDecodeError: If data is too short.
    """
    if len(data) < min_len:
        raise ContainerDecodeError(f"{context}: need at least {min_len} bytes, got {len(data)}")


def require_exact_length(data: bytes, exact_len: int, context: str) -> None:
    """Validate data is exactly the expected length.

    Args:
        data: Bytes to validate.
        exact_len: Expected length.
        context: Context string for error message.

    Raises:
        ContainerDecodeError: If length doesn't match.
    """
    if len(data) != exact_len:
        raise ContainerDecodeError(f"{context}: expected {exact_len} bytes, got {len(data)}")


def require_length_range(data: bytes, min_len: int, max_len: int, context: str) -> None:
    """Validate data length is within expected range.

    Args:
        data: Bytes to validate.
        min_len: Minimum length (inclusive).
        max_len: Maximum length (inclusive).
        context: Context string for error message.

    Raises:
        ContainerDecodeError: If length is outside range.
    """
    if not (min_len <= len(data) <= max_len):
        raise ContainerDecodeError(
            f"{context}: expected {min_len}-{max_len} bytes, got {len(data)}"
        )


def extract_uint16_le(data: bytes, offset: int, context: str) -> int:
    """Extract little-endian uint16 from bytes at offset.

    Args:
        data: Source bytes.
        offset: Byte offset to read from.
        context: Context string for error message.

    Returns:
        Extracted uint16 value.

    Raises:
        ContainerDecodeError: If offset is out of bounds.
    """
    if offset + 2 > len(data):
        raise ContainerDecodeError(
            f"{context}: cannot read uint16 at offset {offset}, data length {len(data)}"
        )
    return data[offset] | (data[offset + 1] << 8)


__all__ = [
    "ContainerDecodeError",
    "extract_uint16_le",
    "require_exact_length",
    "require_length_range",
    "require_min_length",
]
