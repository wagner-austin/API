"""Tests for tankpit_bot.framing module."""

from __future__ import annotations

import pytest

from tankpit_bot.protocol.framing import (
    FramingError,
    decode_frame,
    decode_frame_header,
    encode_frame,
    split_frames,
)

# =============================================================================
# encode_frame Tests
# =============================================================================


def test_encode_frame_empty_body() -> None:
    """Test encoding empty body."""
    result = encode_frame(b"")

    assert result == bytes([0x00, 0x00])


def test_encode_frame_short_body() -> None:
    """Test encoding short body."""
    result = encode_frame(b"AB")

    # Length 2 = 0x02 0x00 in little-endian
    assert result == bytes([0x02, 0x00, 0x41, 0x42])


def test_encode_frame_length_256() -> None:
    """Test encoding body with length requiring both bytes."""
    body = b"X" * 256

    result = encode_frame(body)

    # Length 256 = 0x00 0x01 in little-endian
    assert result[:2] == bytes([0x00, 0x01])
    assert result[2:] == body


def test_encode_frame_length_65535() -> None:
    """Test encoding maximum length body."""
    body = b"Y" * 65535

    result = encode_frame(body)

    # Length 65535 = 0xFF 0xFF in little-endian
    assert result[:2] == bytes([0xFF, 0xFF])
    assert len(result) == 65537


def test_encode_frame_too_long_raises() -> None:
    """Test encoding body exceeding max length raises."""
    body = b"Z" * 65536

    with pytest.raises(FramingError, match="Body too long"):
        encode_frame(body)


# =============================================================================
# decode_frame_header Tests
# =============================================================================


def test_decode_frame_header_zero() -> None:
    """Test decoding zero length header."""
    result = decode_frame_header(bytes([0x00, 0x00]))

    assert result == 0


def test_decode_frame_header_short() -> None:
    """Test decoding short length header."""
    result = decode_frame_header(bytes([0x05, 0x00]))

    assert result == 5


def test_decode_frame_header_256() -> None:
    """Test decoding length 256 header."""
    result = decode_frame_header(bytes([0x00, 0x01]))

    assert result == 256


def test_decode_frame_header_max() -> None:
    """Test decoding maximum length header."""
    result = decode_frame_header(bytes([0xFF, 0xFF]))

    assert result == 65535


def test_decode_frame_header_too_short_raises() -> None:
    """Test decoding with insufficient data raises."""
    with pytest.raises(FramingError, match="Need at least 2 bytes"):
        decode_frame_header(bytes([0x01]))


def test_decode_frame_header_empty_raises() -> None:
    """Test decoding empty data raises."""
    with pytest.raises(FramingError, match="Need at least 2 bytes"):
        decode_frame_header(b"")


# =============================================================================
# decode_frame Tests
# =============================================================================


def test_decode_frame_empty_body() -> None:
    """Test decoding frame with empty body."""
    data = bytes([0x00, 0x00])

    body, remaining = decode_frame(data)

    assert body == b""
    assert remaining == b""


def test_decode_frame_with_body() -> None:
    """Test decoding frame with body."""
    data = bytes([0x03, 0x00, 0x41, 0x42, 0x43])

    body, remaining = decode_frame(data)

    assert body == b"ABC"
    assert remaining == b""


def test_decode_frame_with_remaining() -> None:
    """Test decoding frame with extra data."""
    data = bytes([0x02, 0x00, 0x58, 0x59, 0xFF, 0xFE])

    body, remaining = decode_frame(data)

    assert body == b"XY"
    assert remaining == bytes([0xFF, 0xFE])


def test_decode_frame_incomplete_header_raises() -> None:
    """Test decoding with incomplete header raises."""
    with pytest.raises(FramingError, match="Need at least 2 bytes"):
        decode_frame(bytes([0x01]))


def test_decode_frame_incomplete_body_raises() -> None:
    """Test decoding with incomplete body raises."""
    # Header says 10 bytes, but only 5 provided
    data = bytes([0x0A, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05])

    with pytest.raises(FramingError, match="Incomplete frame"):
        decode_frame(data)


# =============================================================================
# split_frames Tests
# =============================================================================


def test_split_frames_single() -> None:
    """Test splitting single frame."""
    data = bytes([0x02, 0x00, 0x41, 0x42])

    frames = split_frames(data)

    assert len(frames) == 1
    assert frames[0] == b"AB"


def test_split_frames_multiple() -> None:
    """Test splitting multiple frames."""
    # Frame 1: length 2, body "AB"
    # Frame 2: length 3, body "XYZ"
    data = bytes([0x02, 0x00, 0x41, 0x42, 0x03, 0x00, 0x58, 0x59, 0x5A])

    frames = split_frames(data)

    assert len(frames) == 2
    assert frames[0] == b"AB"
    assert frames[1] == b"XYZ"


def test_split_frames_empty() -> None:
    """Test splitting empty data."""
    frames = split_frames(b"")

    assert frames == []


def test_split_frames_incomplete_raises() -> None:
    """Test splitting incomplete data raises."""
    # First frame complete, second incomplete
    data = bytes([0x01, 0x00, 0x41, 0x05, 0x00, 0x42])

    with pytest.raises(FramingError, match="Incomplete frame"):
        split_frames(data)


# =============================================================================
# Round-trip Tests
# =============================================================================


def test_encode_decode_roundtrip() -> None:
    """Test encode then decode returns original body."""
    original = b"Hello, Tankpit!"

    encoded = encode_frame(original)
    decoded, remaining = decode_frame(encoded)

    assert decoded == original
    assert remaining == b""


def test_encode_decode_roundtrip_binary() -> None:
    """Test encode then decode with binary data."""
    original = bytes(range(256))

    encoded = encode_frame(original)
    decoded, remaining = decode_frame(encoded)

    assert decoded == original
    assert remaining == b""


# =============================================================================
# Error Class Tests
# =============================================================================


def test_framing_error_is_exception() -> None:
    """Test FramingError is an Exception."""
    assert issubclass(FramingError, Exception)
    err = FramingError("test")
    assert str(err) == "test"
