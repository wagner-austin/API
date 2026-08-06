"""Tests for the archive-analysis value types.

Every TypedDict here round-trips through its encode/decode pair, and
every ``require_*`` validator is driven to each of its rejection
branches, so a malformed record cannot reach a miner as a silently
half-decoded value.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.analysis.types import (
    SESSION_SKIP_REASONS,
    DecodedFrameDict,
    SkippedSessionDict,
    decode_decoded_frame,
    decode_skipped_session,
    encode_decoded_frame,
    encode_skipped_session,
    require_frame_direction,
    require_hex_bytes,
    require_session_skip_reason,
)


def test_skip_reason_vocabulary_is_exactly_the_documented_set() -> None:
    """The closed vocabulary is two reasons, and the tuple says so."""
    assert SESSION_SKIP_REASONS == ("no_magic", "unframed_payload")


def test_skipped_session_round_trips() -> None:
    """Encode then decode returns an equal record."""
    original = SkippedSessionDict(path="runs/bot/a.capture_session.json", reason="no_magic")
    assert decode_skipped_session(encode_skipped_session(original)) == original


def test_skipped_session_encodes_both_fields() -> None:
    """The wire form carries exactly path and reason."""
    encoded = encode_skipped_session(SkippedSessionDict(path="p", reason="no_magic"))
    assert encoded == {"path": "p", "reason": "no_magic"}


def test_decode_skipped_session_rejects_unknown_reason() -> None:
    """An invented reason names itself and the closed vocabulary."""
    with pytest.raises(JSONTypeError) as excinfo:
        decode_skipped_session({"path": "p", "reason": "because"})
    message = str(excinfo.value)
    assert "unknown session skip reason 'because'" in message
    assert "no_magic" in message


def test_decode_skipped_session_rejects_missing_path() -> None:
    """A missing required field is a decode failure, not a default."""
    with pytest.raises(JSONTypeError):
        decode_skipped_session({"reason": "no_magic"})


def test_require_session_skip_reason_narrows_a_valid_value() -> None:
    """A known reason returns unchanged and is usable as the Literal."""
    assert require_session_skip_reason("no_magic") == "no_magic"


def test_decoded_frame_round_trips_including_body_bytes() -> None:
    """Bytes survive the hex hop exactly, including high bytes and NUL."""
    original = DecodedFrameDict(
        timestamp_ms=1_700_000_000_123,
        direction="received",
        msg_type=0x53,
        body=bytes([0x00, 0x01, 0x7F, 0x80, 0xFF]),
    )
    assert decode_decoded_frame(encode_decoded_frame(original)) == original


def test_decoded_frame_encodes_body_as_hex() -> None:
    """The encoded body is lowercase hex of the raw bytes."""
    encoded = encode_decoded_frame(
        DecodedFrameDict(timestamp_ms=7, direction="sent", msg_type=0x41, body=bytes([0xDE, 0xAD]))
    )
    assert encoded == {
        "timestamp_ms": 7,
        "direction": "sent",
        "msg_type": 0x41,
        "body": "dead",
    }


def test_decoded_frame_round_trips_an_empty_body() -> None:
    """A zero-length body encodes to an empty string and back."""
    original = DecodedFrameDict(timestamp_ms=1, direction="received", msg_type=2, body=b"")
    assert decode_decoded_frame(encode_decoded_frame(original)) == original


def test_require_hex_bytes_rejects_odd_length() -> None:
    """Half a byte is not a byte."""
    with pytest.raises(JSONTypeError) as excinfo:
        require_hex_bytes("abc")
    assert "odd length 3" in str(excinfo.value)


def test_require_hex_bytes_rejects_non_hex_character() -> None:
    """The message names the offending character and its index."""
    with pytest.raises(JSONTypeError) as excinfo:
        require_hex_bytes("00zz")
    message = str(excinfo.value)
    assert "non-hex character 'z'" in message
    assert "index 2" in message


def test_require_hex_bytes_accepts_uppercase() -> None:
    """Uppercase hex decodes identically to lowercase."""
    assert require_hex_bytes("DEAD") == bytes([0xDE, 0xAD])


def test_decode_decoded_frame_rejects_bad_body_hex() -> None:
    """A malformed body fails the record, not just the field."""
    with pytest.raises(JSONTypeError):
        decode_decoded_frame(
            {"timestamp_ms": 1, "direction": "received", "msg_type": 2, "body": "xy"}
        )


def test_require_frame_direction_narrows_and_rejects() -> None:
    """Both directions narrow; anything else names the vocabulary."""
    assert require_frame_direction("received") == "received"
    assert require_frame_direction("sent") == "sent"
    with pytest.raises(JSONTypeError) as excinfo:
        require_frame_direction("sideways")
    message = str(excinfo.value)
    assert "unknown frame direction 'sideways'" in message
    assert "received" in message and "sent" in message
