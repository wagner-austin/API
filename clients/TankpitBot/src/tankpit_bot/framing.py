"""WebSocket message framing for Tankpit protocol.

All messages have a 2-byte little-endian length header followed by the body.
This module handles encoding and decoding the framing layer.

Wire format: [length_lo] [length_hi] [body...]
"""

from __future__ import annotations


class FramingError(Exception):
    """Error in message framing."""


def encode_frame(body: bytes) -> bytes:
    """Encode a message body with 2-byte length header.

    Args:
        body: Message body bytes.

    Returns:
        Framed message: 2-byte LE length + body.

    Raises:
        FramingError: If body exceeds maximum length (65535 bytes).
    """
    length = len(body)
    if length > 0xFFFF:
        raise FramingError(f"Body too long: {length} bytes, max 65535")

    # 2-byte little-endian length prefix
    header = bytes([length & 0xFF, (length >> 8) & 0xFF])
    return header + body


def decode_frame_header(data: bytes) -> int:
    """Decode the 2-byte length header from frame data.

    Args:
        data: At least 2 bytes of frame data.

    Returns:
        Body length from header.

    Raises:
        FramingError: If data is less than 2 bytes.
    """
    if len(data) < 2:
        raise FramingError(f"Need at least 2 bytes for header, got {len(data)}")

    return data[0] | (data[1] << 8)


def decode_frame(data: bytes) -> tuple[bytes, bytes]:
    """Decode a complete frame from data.

    Args:
        data: Frame data (header + body).

    Returns:
        Tuple of (body, remaining_data).

    Raises:
        FramingError: If data is incomplete.
    """
    if len(data) < 2:
        raise FramingError(f"Need at least 2 bytes for header, got {len(data)}")

    body_length = decode_frame_header(data)
    total_length = 2 + body_length

    if len(data) < total_length:
        raise FramingError(f"Incomplete frame: need {total_length} bytes, got {len(data)}")

    body = data[2:total_length]
    remaining = data[total_length:]
    return body, remaining


def split_frames(data: bytes) -> list[bytes]:
    """Split data into individual frame bodies.

    Args:
        data: Buffer containing one or more complete frames.

    Returns:
        List of frame bodies (without headers).

    Raises:
        FramingError: If data contains incomplete frames.
    """
    frames: list[bytes] = []
    remaining = data

    while len(remaining) > 0:
        body, remaining = decode_frame(remaining)
        frames.append(body)

    return frames


__all__ = [
    "FramingError",
    "decode_frame",
    "decode_frame_header",
    "encode_frame",
    "split_frames",
]
