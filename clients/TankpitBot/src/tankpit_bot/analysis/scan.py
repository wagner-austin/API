"""Decode a capture session, and walk the archive of them.

This is the load-XOR-split-decode pipeline that thirty of the forty
analysis scripts each wrote for themselves. Every step delegates to the
module that already owned it:

* the session shape and its validation —
  :func:`tankpit_bot.types.session.decode_capture_session`
* the base64 payload —
  :func:`tankpit_bot.capture.xor.decode_base64_safe`
* the frame walk — :func:`tankpit_bot.protocol.framing.split_frames`
* the cipher — :mod:`tankpit_bot.sniffer.xor`

Nothing here re-implements any of them. The value this module adds is
the ORDER, the XOR lifecycle around a session, and a typed answer for
the one session shape that cannot be analysed.

Strictness note. ``split_frames`` raises on a payload that ends
mid-frame, where the scripts it replaces silently returned the frames
they had. That leniency was never exercised: measured over the whole
bot archive on 2026-08-06, 217,678 received payloads across 286
sessions split cleanly and ZERO raised. The strict walk therefore costs
nothing on real captures and turns genuine corruption into a reported
failure instead of a quietly shortened analysis.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONTypeError, load_json_str

from tankpit_bot.analysis import _test_hooks
from tankpit_bot.analysis.types import (
    DecodedFrameDict,
    ScannedSessionDict,
    SkippedSessionDict,
)
from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol.framing import split_frames
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode
from tankpit_bot.types import CaptureSession, decode_capture_session

#: Direction marking a frame the server sent to us. Sent frames carry
#: our own commands and are not part of the received wire stream.
RECEIVED_DIRECTION = "received"


def load_capture_session(path: Path) -> CaptureSession:
    """Read and validate one capture-session file.

    Args:
        path: Capture-session JSON file.

    Returns:
        The validated session.

    Raises:
        OSError: If the file cannot be read.
        InvalidJsonError: If the file is not valid JSON.
        JSONTypeError: If the JSON is not a capture session — a missing
            or mistyped field names itself in the message.
    """
    parsed = load_json_str(_test_hooks.read_text(path))
    if not isinstance(parsed, dict):
        raise JSONTypeError(f"{path.name}: capture session must be a JSON object")
    return decode_capture_session(parsed)


def decode_session_frames(session: CaptureSession) -> list[DecodedFrameDict]:
    """Decode every received frame in a session, in capture order.

    The XOR table is global state owned by :mod:`tankpit_bot.sniffer.xor`,
    so this resets it and rebuilds it from the session's own magic
    before decoding. Sessions must therefore be decoded one at a time,
    which :func:`scan_session` guarantees.

    Args:
        session: A session whose ``magic`` is known to be present.

    Returns:
        Every received frame, decoded.

    Raises:
        FramingError: If a payload ends mid-frame.
        ValueError: If ``session["magic"]`` is None. Callers reach this
            function through :func:`scan_session`, which classifies
            that case as a skip rather than calling here.
    """
    magic = session["magic"]
    if magic is None:
        raise ValueError("session has no XOR magic; nothing in it can be decoded")
    reset_xor_state()
    build_global_xor_table(magic)
    frames: list[DecodedFrameDict] = []
    for message in session["messages"]:
        if message["direction"] != RECEIVED_DIRECTION:
            continue
        payload = decode_base64_safe(message["payload"])
        if not payload:
            continue
        for body in split_frames(payload):
            if not body:
                continue
            frames.append(
                DecodedFrameDict(
                    timestamp_ms=message["timestamp_ms"],
                    msg_type=body[0],
                    body=xor_decode(body),
                )
            )
    return frames


def scan_session(path: Path) -> ScannedSessionDict | SkippedSessionDict:
    """Decode one capture session, or say why it holds nothing.

    A capture whose XOR magic was never observed cannot yield a single
    frame. That is a property of the recording, not an error, so it
    returns a :class:`SkippedSessionDict` the caller can tally. Every
    other fault — unreadable file, malformed JSON, a payload violating
    the framing contract — propagates.

    Args:
        path: Capture-session JSON file.

    Returns:
        The decoded session, or a skip record naming the reason.

    Raises:
        OSError: If the file cannot be read.
        InvalidJsonError: If the file is not valid JSON.
        JSONTypeError: If the JSON is not a capture session.
        FramingError: If a payload ends mid-frame.
    """
    session = load_capture_session(path)
    if session["magic"] is None:
        return SkippedSessionDict(path=str(path), reason="no_magic")
    return ScannedSessionDict(
        path=str(path),
        session_id=session["session_id"],
        frames=decode_session_frames(session),
    )


def scan_archive(directory: Path) -> list[ScannedSessionDict | SkippedSessionDict]:
    """Scan every capture session in a directory, in a stable order.

    Args:
        directory: Directory holding ``*.capture_session.json`` files.

    Returns:
        One result per session, in sorted-by-name order, each either a
        decode or a skip. Sessions are processed sequentially because
        the XOR table is global state.

    Raises:
        OSError: If a file cannot be read.
        InvalidJsonError: If a file is not valid JSON.
        JSONTypeError: If a file is not a capture session.
        FramingError: If a payload ends mid-frame.
    """
    return [scan_session(path) for path in _test_hooks.list_session_paths(directory)]


__all__ = [
    "RECEIVED_DIRECTION",
    "decode_session_frames",
    "load_capture_session",
    "scan_archive",
    "scan_session",
]
