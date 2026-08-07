"""Byte arithmetic and length validation for the wire codecs.

Shared by :mod:`tankpit_bot.protocol` and
:mod:`tankpit_bot.container`, which is why it lives below both rather
than inside protocol -- see :mod:`tankpit_bot.wire`.

Holds the two byte-combining primitives named after their JS client
counterparts (:func:`x16`, :func:`x24`), their pack inverses, the two
codec exceptions, and the ``require_*`` validators that raise
:class:`DecodeError` with a message naming the frame being decoded.
"""

from __future__ import annotations


class DecodeError(Exception):
    """Raised when message decoding fails."""


class EncodeError(Exception):
    """Raised when message encoding fails (unsupported message shape)."""


def x16(low: int, high: int) -> int:
    """Combine two bytes into 16-bit value (JS X function).

    Args:
        low: Low byte (0-255).
        high: High byte (0-255).

    Returns:
        Combined 16-bit unsigned value.
    """
    return (low & 255) + 256 * (high & 255)


def x24(a: int, b: int, c: int) -> int:
    """Combine three bytes into 24-bit value (big-endian).

    Args:
        a: High byte.
        b: Middle byte.
        c: Low byte.

    Returns:
        Combined 24-bit unsigned value.
    """
    return 256 * (256 * a + b) + c


def pack16(value: int) -> bytes:
    """Pack a 16-bit unsigned value into two bytes (inverse of :func:`x16`).

    Args:
        value: 16-bit unsigned value.

    Returns:
        Two bytes, low byte first.
    """
    return bytes([value & 255, (value >> 8) & 255])


def pack24(value: int) -> bytes:
    """Pack a 24-bit unsigned value into three bytes (inverse of :func:`x24`).

    Args:
        value: 24-bit unsigned value.

    Returns:
        Three bytes, high byte first (big-endian).
    """
    return bytes([(value >> 16) & 255, (value >> 8) & 255, value & 255])


def require_min_length(data: bytes, min_len: int, msg_name: str) -> None:
    """Validate minimum data length.

    Args:
        data: Raw bytes to validate.
        min_len: Minimum required length.
        msg_name: Message type name for error messages.

    Raises:
        DecodeError: If data is too short.
    """
    if len(data) < min_len:
        raise DecodeError(f"{msg_name}: expected >= {min_len} bytes, got {len(data)}")


def require_exact_length(data: bytes, exact_len: int, msg_name: str) -> None:
    """Validate exact data length.

    Args:
        data: Raw bytes to validate.
        exact_len: Required exact length.
        msg_name: Message type name for error messages.

    Raises:
        DecodeError: If data length doesn't match.
    """
    if len(data) != exact_len:
        raise DecodeError(f"{msg_name}: expected {exact_len} bytes, got {len(data)}")


def require_prefix(text: str, prefix: str, msg_name: str) -> None:
    """Validate text starts with expected prefix.

    Args:
        text: Text to validate.
        prefix: Required prefix.
        msg_name: Message type name for error messages.

    Raises:
        DecodeError: If prefix is missing.
    """
    if not text.startswith(prefix):
        raise DecodeError(f"{msg_name}: expected prefix '{prefix}'")


def require_parts(parts: list[str], min_parts: int, msg_name: str) -> None:
    """Validate minimum number of pipe-separated parts.

    Args:
        parts: Split parts to validate.
        min_parts: Minimum required parts.
        msg_name: Message type name for error messages.

    Raises:
        DecodeError: If not enough parts.
    """
    if len(parts) < min_parts:
        raise DecodeError(f"{msg_name}: expected >= {min_parts} parts, got {len(parts)}")


__all__ = [
    "DecodeError",
    "EncodeError",
    "pack16",
    "pack24",
    "require_exact_length",
    "require_min_length",
    "require_parts",
    "require_prefix",
    "x16",
    "x24",
]
