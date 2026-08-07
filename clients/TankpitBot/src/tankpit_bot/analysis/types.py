"""Typed results of scanning the capture archive.

Every archive miner asks the same question in the same order: which
sessions in ``runs/`` can be decoded, and what does each decoded frame
say. Before this package each miner answered it privately — 30 of the
40 scripts re-implemented the load, 26 the XOR bring-up, and 10 forked
the frame walk that has lived in :mod:`tankpit_bot.protocol.framing`
since the beginning.

The shapes here are the vocabulary that replaces those forks. A session
that cannot be analysed is a VALUE, not an exception: a capture whose
XOR magic was never observed decodes to nothing at all, which is an
ordinary and rare fact about the archive (1 of 287 bot captures), not a
failure of the miner. Genuine faults — an unreadable file, malformed
JSON, a body that violates the wire contract — propagate, because a
miner that silently skips those reports a corpus it never actually
read.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_int,
    require_str,
)

#: Why a capture session yielded no decodable frames. The vocabulary is
#: closed: a new skip site means a new documented reason here, not an
#: invented string, so ``skipped`` tallies stay comparable across runs.
#: ``unframed_payload`` (added 2026-08-06 with the direction extension)
#: marks a session whose payloads violate the framing contract —
#: measured archive-wide: 2 of 62,095 sent payloads, both in the one
#: pre-framing capture ``bot-20260331-230406``; 0 of 217,678 received.
SessionSkipReason = Literal["no_magic", "unframed_payload"]

#: Every value :data:`SessionSkipReason` admits, for validation and for
#: exhaustive reporting by callers that tally skips.
SESSION_SKIP_REASONS: tuple[SessionSkipReason, ...] = ("no_magic", "unframed_payload")

#: Which side of the wire a frame travelled. Sent frames carry our own
#: commands (the same XOR cipher covers both directions); received
#: frames are the server's stream.
FrameDirection = Literal["received", "sent"]

#: Every value :data:`FrameDirection` admits.
FRAME_DIRECTIONS: tuple[FrameDirection, ...] = ("received", "sent")


class SkippedSessionDict(TypedDict):
    """A capture session that carries no analysable frames.

    Attributes:
        path: Filesystem path of the capture session file.
        reason: Why the session yielded nothing, from
            :data:`SESSION_SKIP_REASONS`.
    """

    path: str
    reason: SessionSkipReason


class DecodedFrameDict(TypedDict):
    """One wire frame in either direction, decoded, with its timestamp.

    Attributes:
        timestamp_ms: Capture time of the containing message, carried
            through so miners can correlate frames across channels
            without re-reading the session.
        direction: Which side of the wire the frame travelled
            (:data:`FRAME_DIRECTIONS`). Sent frames are our own
            commands; received frames are the server stream.
        msg_type: First byte of the frame, before XOR decoding.
        raw: The frame exactly as captured on the wire, type byte
            included, untouched by the cipher. This is what the
            production receive path hands to its PRE-cipher
            discriminators — ``protocol.try_decode_plaintext_ack``
            and the text-route check — which run before ``xor_decode``
            because plaintext frames are never ciphered.
        body: The frame after the session cipher: type byte stripped,
            remainder XOR-decoded — exactly the data argument
            ``protocol.decode_message(msg_type, body)`` and
            ``sim.commands.decode_client_command(body)`` take.
    """

    timestamp_ms: int
    direction: FrameDirection
    msg_type: int
    raw: bytes
    body: bytes


class ScannedSessionDict(TypedDict):
    """A capture session that decoded, with its frames.

    Attributes:
        path: Filesystem path of the capture session file.
        session_id: Identifier carried by the capture itself.
        frames: Every frame, both directions, in capture order.
    """

    path: str
    session_id: str
    frames: list[DecodedFrameDict]


def encode_skipped_session(skipped: SkippedSessionDict) -> JSONObject:
    """Encode a :class:`SkippedSessionDict` to a JSON object.

    Args:
        skipped: The skip record to encode.

    Returns:
        JSON-serializable object with ``path`` and ``reason``.
    """
    return {"path": skipped["path"], "reason": skipped["reason"]}


def decode_skipped_session(data: JSONObject) -> SkippedSessionDict:
    """Decode a :class:`SkippedSessionDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        The validated skip record.

    Raises:
        JSONTypeError: If a field is missing, of the wrong type, or if
            ``reason`` is outside :data:`SESSION_SKIP_REASONS`.
    """
    reason = require_session_skip_reason(require_str(data, "reason"))
    return SkippedSessionDict(path=require_str(data, "path"), reason=reason)


def require_session_skip_reason(value: str) -> SessionSkipReason:
    """Narrow a string to a :data:`SessionSkipReason`.

    Args:
        value: Candidate reason string.

    Returns:
        The same value, narrowed.

    Raises:
        JSONTypeError: If the value is not a known skip reason. The
            message names both the offending value and the closed
            vocabulary, so a caller can fix the input without reading
            this module.
    """
    for reason in SESSION_SKIP_REASONS:
        if value == reason:
            return reason
    known = ", ".join(SESSION_SKIP_REASONS)
    raise JSONTypeError(f"unknown session skip reason '{value}'; known reasons: {known}")


def encode_decoded_frame(frame: DecodedFrameDict) -> JSONObject:
    """Encode a :class:`DecodedFrameDict` to a JSON object.

    Byte fields are rendered as lowercase hex because JSON has no
    bytes literal and hex round-trips exactly at any byte value.

    Args:
        frame: The decoded frame to encode.

    Returns:
        JSON-serializable object with ``timestamp_ms``, ``direction``,
        ``msg_type`` and hex ``raw`` / ``body``.
    """
    return {
        "timestamp_ms": frame["timestamp_ms"],
        "direction": frame["direction"],
        "msg_type": frame["msg_type"],
        "raw": frame["raw"].hex(),
        "body": frame["body"].hex(),
    }


def decode_decoded_frame(data: JSONObject) -> DecodedFrameDict:
    """Decode a :class:`DecodedFrameDict` from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        The validated frame.

    Raises:
        JSONTypeError: If a field is missing, of the wrong type, or if
            ``raw`` or ``body`` is not valid hex.
    """
    return DecodedFrameDict(
        timestamp_ms=require_int(data, "timestamp_ms"),
        direction=require_frame_direction(require_str(data, "direction")),
        msg_type=require_int(data, "msg_type"),
        raw=require_hex_bytes(require_str(data, "raw")),
        body=require_hex_bytes(require_str(data, "body")),
    )


def require_frame_direction(value: str) -> FrameDirection:
    """Narrow a string to a :data:`FrameDirection`.

    Args:
        value: Candidate direction string.

    Returns:
        The same value, narrowed.

    Raises:
        JSONTypeError: If the value is not a known direction; the
            message names the closed vocabulary.
    """
    for direction in FRAME_DIRECTIONS:
        if value == direction:
            return direction
    known = ", ".join(FRAME_DIRECTIONS)
    raise JSONTypeError(f"unknown frame direction '{value}'; known directions: {known}")


def require_hex_bytes(value: str) -> bytes:
    """Convert a hex string to bytes, rejecting anything malformed.

    ``bytes.fromhex`` raises :class:`ValueError` on bad input; this
    translates that into the JSON error vocabulary so a caller decoding
    a record sees one error type rather than two.

    Args:
        value: Lowercase or uppercase hex, no separators.

    Returns:
        The decoded bytes.

    Raises:
        JSONTypeError: If the string is not valid hex.
    """
    if len(value) % 2 != 0:
        raise JSONTypeError(f"hex body has odd length {len(value)}")
    for index, char in enumerate(value):
        if char not in "0123456789abcdefABCDEF":
            raise JSONTypeError(f"hex body has non-hex character '{char}' at index {index}")
    return bytes.fromhex(value)


__all__ = [
    "FRAME_DIRECTIONS",
    "SESSION_SKIP_REASONS",
    "DecodedFrameDict",
    "FrameDirection",
    "ScannedSessionDict",
    "SessionSkipReason",
    "SkippedSessionDict",
    "decode_decoded_frame",
    "decode_skipped_session",
    "encode_decoded_frame",
    "encode_skipped_session",
    "require_frame_direction",
    "require_hex_bytes",
    "require_session_skip_reason",
]
