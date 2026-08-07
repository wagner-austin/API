"""Decode a capture session, and walk the archive of them.

This is the load-XOR-split-decode pipeline that thirty of the forty
analysis scripts each wrote for themselves. Every step delegates to the
module that already owned it:

* the session shape and its validation —
  :func:`tankpit_bot.types.session.decode_capture_session`
* the base64 payload —
  :func:`tankpit_bot.capture.xor.decode_base64_safe`
* the frame walk — :func:`tankpit_bot.protocol.framing.split_frames`
* the cipher — :func:`tankpit_bot.capture.xor.build_session_xor_table`

Nothing here re-implements any of them. The value this module adds is
the ORDER, the XOR lifecycle around a session, and a typed answer for
the one session shape that cannot be analysed.

Strictness note. ``split_frames`` raises on a payload that ends
mid-frame, where the scripts it replaces silently returned the frames
they had. Measured over the whole bot archive on 2026-08-06: 217,678
received payloads across 286 sessions split cleanly with ZERO raises,
and - after the direction extension the same night - 62,095 sent
payloads split cleanly with exactly TWO raises, both inside the one
pre-framing capture ``bot-20260331-230406`` (3-byte sent blobs from
before the framing protocol existed). :func:`scan_session` classifies
that archaeology as a typed ``unframed_payload`` skip; the strict walk
still turns any NEW corruption into a reported value instead of a
quietly shortened analysis.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONTypeError, load_json_str
from platform_core.logging import get_logger

from tankpit_bot.analysis import _test_hooks
from tankpit_bot.analysis.types import (
    DecodedFrameDict,
    ScannedSessionDict,
    SkippedSessionDict,
)
from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.capture.xor import (
    build_session_xor_table,
    xor_decode_body,
)
from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.types import CaptureSession, decode_capture_session

log = get_logger(__name__)


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
    """Decode every frame in a session, both directions, in capture order.

    Sent frames carry our own commands and share the session cipher, so
    the direction extension (2026-08-06) tags each frame instead of
    dropping half the traffic - the command-correlating miners
    (displacement semantics, cost pairing) need both sides.

    The XOR table is built from this session's own magic and held as a
    LOCAL, so sessions no longer have to be decoded one at a time. It
    was a module global until 2026-08-06, which is why the archive walk
    was sequential ([[session-state-deglobalisation]] step 1).

    Args:
        session: A session whose ``magic`` is known to be present.

    Returns:
        Every frame, decoded and direction-tagged.

    Raises:
        FramingError: If a payload ends mid-frame (see the module
            strictness note; :func:`scan_session` classifies this).
        ValueError: If ``session["magic"]`` is None. Callers reach this
            function through :func:`scan_session`, which classifies
            that case as a skip rather than calling here.
    """
    magic = session["magic"]
    if magic is None:
        raise ValueError("session has no XOR magic; nothing in it can be decoded")
    xor_table = build_session_xor_table(magic)
    frames: list[DecodedFrameDict] = []
    for message in session["messages"]:
        for body in split_payload_frames(message["payload"]):
            if not body:
                continue
            frames.append(
                DecodedFrameDict(
                    timestamp_ms=message["timestamp_ms"],
                    direction=message["direction"],
                    msg_type=body[0],
                    raw=body,
                    body=xor_decode_body(body, xor_table, offset=1),
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

    A payload that violates the framing contract is likewise a typed
    skip (``unframed_payload``), not a crash: the archive holds exactly
    one such capture (``bot-20260331-230406``, sent blobs predating the
    framing protocol - see the module strictness note), and a skip
    keeps that archaeology visible in every tally while any NEW
    corruption surfaces the same loud way.

    Args:
        path: Capture-session JSON file.

    Returns:
        The decoded session, or a skip record naming the reason.

    Raises:
        OSError: If the file cannot be read.
        InvalidJsonError: If the file is not valid JSON.
        JSONTypeError: If the JSON is not a capture session.
    """
    session = load_capture_session(path)
    if session["magic"] is None:
        return SkippedSessionDict(path=str(path), reason="no_magic")
    try:
        frames = decode_session_frames(session)
    except FramingError as error:
        log.info("Skipping unframed capture %s: %s", path, error)
        return SkippedSessionDict(path=str(path), reason="unframed_payload")
    return ScannedSessionDict(
        path=str(path),
        session_id=session["session_id"],
        frames=frames,
    )


def scan_archive(directory: Path) -> list[ScannedSessionDict | SkippedSessionDict]:
    """Scan every capture session in a directory, in a stable order.

    Args:
        directory: Directory holding ``*.capture_session.json`` files.

    Returns:
        One result per session, in sorted-by-name order, each either a
        decode or a skip. The order is for reproducibility only —
        each session owns its XOR table, so nothing here forces
        sequential processing ([[session-state-deglobalisation]]).

    Raises:
        OSError: If a file cannot be read.
        InvalidJsonError: If a file is not valid JSON.
        JSONTypeError: If a file is not a capture session.
    """
    return [scan_session(path) for path in _test_hooks.list_session_paths(directory)]


__all__ = [
    "decode_session_frames",
    "load_capture_session",
    "scan_archive",
    "scan_session",
]
