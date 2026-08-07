"""Tests for the capture-scan pipeline.

Sessions are built as real JSON on disk and read through the real
production hooks wherever possible; the injected hooks are used only to
prove the seam works and to drive the enumeration order. Frames are
built with the production encoder (``encode_frame``) and the production
cipher, so a change to either shows up here rather than passing a
hand-rolled fixture that agrees with nothing.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str

from tankpit_bot.analysis import _test_hooks
from tankpit_bot.analysis.scan import (
    decode_session_frames,
    load_capture_session,
    scan_archive,
    scan_session,
)
from tankpit_bot.capture.xor import build_session_xor_table, xor_decode_body
from tankpit_bot.protocol.framing import FramingError, encode_frame
from tankpit_bot.types import CaptureSession

MAGIC = "abcdefgh"


def _expected_body(body: bytes) -> bytes:
    """Return the plaintext a frame body decodes to under ``MAGIC``.

    Built through the same production cipher the scanner uses, from
    this fixture's own magic — the table is a value, so the expectation
    no longer depends on global decoder state
    ([[session-state-deglobalisation]]).

    Args:
        body: Raw frame body (type byte + ciphered rest).

    Returns:
        The decoded payload without the leading type byte.
    """
    return xor_decode_body(body, build_session_xor_table(MAGIC), offset=1)


def _payload(*bodies: bytes) -> str:
    """Frame each body with the production encoder and base64 it.

    Args:
        *bodies: Raw frame bodies, each starting with its type byte.

    Returns:
        Base64 payload as a capture stores it.
    """
    return base64.b64encode(b"".join(encode_frame(body) for body in bodies)).decode("ascii")


def _session_json(
    *,
    magic: str | None = MAGIC,
    messages: list[JSONObject] | None = None,
) -> str:
    """Build a capture-session file body.

    Args:
        magic: XOR magic, or None for a session that cannot be decoded.
        messages: Captured messages; defaults to none.

    Returns:
        JSON text of a valid capture session.
    """
    session: JSONObject = {
        "session_id": "s-1",
        "start_timestamp_ms": 1000,
        "end_timestamp_ms": 2000,
        "base_url": "https://tankpit.com",
        "magic": magic,
        "messages": list(messages) if messages is not None else [],
    }
    return dump_json_str(session)


def _received(payload: str, timestamp_ms: int = 1500) -> JSONObject:
    """Build one received message.

    Args:
        payload: Base64 payload.
        timestamp_ms: Capture time.

    Returns:
        A captured-message JSON object.
    """
    return {
        "timestamp_ms": timestamp_ms,
        "direction": "received",
        "payload": payload,
        "ws_url": "wss://tankpit.com/ws",
    }


def _write(tmp_path: Path, name: str, text: str) -> Path:
    """Write a capture-session file.

    Args:
        tmp_path: Directory to write into.
        name: File name.
        text: File contents.

    Returns:
        The written path.
    """
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_capture_session_reads_and_validates(tmp_path: Path) -> None:
    """A well-formed file decodes to the typed session."""
    path = _write(tmp_path, "a.capture_session.json", _session_json())
    session = load_capture_session(path)
    assert session["session_id"] == "s-1"
    assert session["magic"] == MAGIC


def test_load_capture_session_rejects_non_object_json(tmp_path: Path) -> None:
    """A JSON array is not a capture session, and the file is named."""
    path = _write(tmp_path, "b.capture_session.json", "[1, 2]")
    with pytest.raises(JSONTypeError) as excinfo:
        load_capture_session(path)
    assert "b.capture_session.json" in str(excinfo.value)


def test_load_capture_session_rejects_missing_field(tmp_path: Path) -> None:
    """A session without session_id fails decode rather than defaulting."""
    incomplete: JSONObject = {"messages": []}
    path = _write(tmp_path, "c.capture_session.json", dump_json_str(incomplete))
    with pytest.raises(JSONTypeError):
        load_capture_session(path)


def test_decode_session_frames_decodes_body_with_the_real_cipher(tmp_path: Path) -> None:
    """Each frame carries its type byte and the XOR-decoded remainder."""
    body = bytes([0x53, 0x11, 0x22, 0x33])
    path = _write(
        tmp_path,
        "d.capture_session.json",
        _session_json(messages=[_received(_payload(body), timestamp_ms=4242)]),
    )
    frames = decode_session_frames(load_capture_session(path))
    assert frames == [
        {
            "timestamp_ms": 4242,
            "direction": "received",
            "msg_type": 0x53,
            "raw": body,
            "body": _expected_body(body),
        }
    ]


def test_decode_session_frames_keeps_capture_order(tmp_path: Path) -> None:
    """Frames come back in wire order across messages."""
    path = _write(
        tmp_path,
        "e.capture_session.json",
        _session_json(
            messages=[
                _received(_payload(bytes([0x41, 0x01]), bytes([0x42, 0x02])), timestamp_ms=10),
                _received(_payload(bytes([0x43, 0x03])), timestamp_ms=20),
            ]
        ),
    )
    frames = decode_session_frames(load_capture_session(path))
    assert [(f["timestamp_ms"], f["msg_type"]) for f in frames] == [
        (10, 0x41),
        (10, 0x42),
        (20, 0x43),
    ]


def test_decode_session_frames_tags_sent_frames(tmp_path: Path) -> None:
    """Our own commands decode with the same cipher, tagged ``sent``.

    The direction extension (2026-08-06): command-correlating miners
    (displacement semantics, cost pairing) need both sides of the
    wire, so sent frames are decoded and tagged instead of dropped.
    """
    body = bytes([0x70, 0x05])
    sent: JSONObject = {
        "timestamp_ms": 1,
        "direction": "sent",
        "payload": _payload(body),
        "ws_url": "wss://tankpit.com/ws",
    }
    path = _write(tmp_path, "f.capture_session.json", _session_json(messages=[sent]))
    frames = decode_session_frames(load_capture_session(path))
    assert frames == [
        {
            "timestamp_ms": 1,
            "direction": "sent",
            "msg_type": 0x70,
            "raw": body,
            "body": _expected_body(body),
        }
    ]


def test_decode_session_frames_keeps_the_raw_frame_untouched(tmp_path: Path) -> None:
    """``raw`` is the wire frame byte-for-byte, never ciphered.

    The production receive path discriminates plaintext acks and text
    routes on the RAW frame before ``xor_decode`` runs — a consumer
    that applies those discriminators to ``body`` reads garbage. The
    ``raw`` field exists so miners can run the same pre-cipher checks
    the sniffer does (found live: the viewport-probe migration lost
    every autoscroll ack until it switched to ``raw``).
    """
    ack = b"A1"
    path = _write(
        tmp_path, "r.capture_session.json", _session_json(messages=[_received(_payload(ack))])
    )
    frames = decode_session_frames(load_capture_session(path))
    assert [f["raw"] for f in frames] == [ack]
    assert frames[0]["body"] != ack[1:]


def test_decode_session_frames_ignores_empty_payload(tmp_path: Path) -> None:
    """A received message with no payload contributes nothing."""
    path = _write(tmp_path, "g.capture_session.json", _session_json(messages=[_received("")]))
    assert decode_session_frames(load_capture_session(path)) == []


def test_decode_session_frames_skips_a_zero_length_frame(tmp_path: Path) -> None:
    """A zero-length frame has no type byte, so it contributes nothing.

    ``encode_frame(b"")`` is a legal two-byte header with an empty
    body, and ``split_frames`` yields it. Reading ``body[0]`` on it
    would raise IndexError, so the walk skips it and keeps going —
    proven here by the real frame that follows still arriving.
    """
    real = bytes([0x46, 0x07])
    path = _write(
        tmp_path,
        "z.capture_session.json",
        _session_json(messages=[_received(_payload(b"", real), timestamp_ms=55)]),
    )
    frames = decode_session_frames(load_capture_session(path))
    assert [(f["timestamp_ms"], f["msg_type"]) for f in frames] == [(55, 0x46)]


def test_decode_session_frames_raises_without_magic(tmp_path: Path) -> None:
    """Calling the decoder directly on a magicless session is an error."""
    path = _write(tmp_path, "h.capture_session.json", _session_json(magic=None))
    with pytest.raises(ValueError) as excinfo:
        decode_session_frames(load_capture_session(path))
    assert "no XOR magic" in str(excinfo.value)


def test_decode_session_frames_raises_on_truncated_payload(tmp_path: Path) -> None:
    """A payload ending mid-frame is a reported fault, not a short read."""
    truncated = base64.b64encode(encode_frame(bytes([0x53, 0x11, 0x22]))[:-1]).decode("ascii")
    path = _write(
        tmp_path, "i.capture_session.json", _session_json(messages=[_received(truncated)])
    )
    with pytest.raises(FramingError):
        decode_session_frames(load_capture_session(path))


def test_scan_session_returns_frames_for_a_decodable_capture(tmp_path: Path) -> None:
    """The decoded result carries the path, the id, and the frames."""
    body = bytes([0x4C, 0x09])
    path = _write(
        tmp_path,
        "j.capture_session.json",
        _session_json(messages=[_received(_payload(body), timestamp_ms=99)]),
    )
    result = scan_session(path)
    if "frames" not in result:
        raise AssertionError(f"expected a decoded session, got a skip: {result}")
    assert result == {
        "path": str(path),
        "session_id": "s-1",
        "frames": [
            {
                "timestamp_ms": 99,
                "direction": "received",
                "msg_type": 0x4C,
                "raw": body,
                "body": _expected_body(body),
            }
        ],
    }


def test_scan_session_skips_a_magicless_capture(tmp_path: Path) -> None:
    """No magic is a typed value, not an exception and not a crash."""
    path = _write(tmp_path, "k.capture_session.json", _session_json(magic=None))
    assert scan_session(path) == {"path": str(path), "reason": "no_magic"}


def test_scan_session_classifies_an_unframed_payload_as_a_typed_skip(tmp_path: Path) -> None:
    """The pre-framing archaeology class is a value, not a crash.

    The archive holds exactly one capture (``bot-20260331-230406``)
    whose sent blobs predate the framing protocol; a truncated payload
    reproduces the same contract violation here. The direct decoder
    still raises (the strictness pin above); the SESSION-level answer
    is the typed skip that keeps the tally honest.
    """
    truncated = base64.b64encode(encode_frame(bytes([0x53, 0x11, 0x22]))[:-1]).decode("ascii")
    sent: JSONObject = {
        "timestamp_ms": 1,
        "direction": "sent",
        "payload": truncated,
        "ws_url": "wss://tankpit.com/ws",
    }
    path = _write(tmp_path, "u.capture_session.json", _session_json(messages=[sent]))
    assert scan_session(path) == {"path": str(path), "reason": "unframed_payload"}


def test_scan_archive_visits_every_session_in_name_order(tmp_path: Path) -> None:
    """Two scans of the same archive produce the same order."""
    _write(tmp_path, "b.capture_session.json", _session_json())
    _write(tmp_path, "a.capture_session.json", _session_json(magic=None))
    results = scan_archive(tmp_path)
    assert [Path(r["path"]).name for r in results] == [
        "a.capture_session.json",
        "b.capture_session.json",
    ]
    assert results[0] == {
        "path": str(tmp_path / "a.capture_session.json"),
        "reason": "no_magic",
    }


def test_scan_archive_ignores_files_that_are_not_captures(tmp_path: Path) -> None:
    """Only *.capture_session.json is enumerated."""
    _write(tmp_path, "a.capture_session.json", _session_json())
    _write(tmp_path, "notes.txt", "not a capture")
    _write(tmp_path, "run.log", "also not a capture")
    assert len(scan_archive(tmp_path)) == 1


def test_injected_hooks_replace_both_seams(tmp_path: Path) -> None:
    """The DI seam is real: neither production impl runs when replaced."""
    session_text = _session_json(magic=None)
    seen: list[Path] = []

    def fake_list(directory: Path) -> list[Path]:
        return [directory / "injected.capture_session.json"]

    def fake_read(path: Path) -> str:
        seen.append(path)
        return session_text

    _test_hooks.set_analysis_hooks(read_text_fn=fake_read, list_session_paths_fn=fake_list)
    results = scan_archive(tmp_path)
    assert [p.name for p in seen] == ["injected.capture_session.json"]
    assert results == [
        {"path": str(tmp_path / "injected.capture_session.json"), "reason": "no_magic"}
    ]


def test_reset_hooks_restores_the_production_readers(tmp_path: Path) -> None:
    """After reset the real filesystem is used again."""

    def fake_list(directory: Path) -> list[Path]:
        raise AssertionError("production enumeration should have been restored")

    def fake_read(path: Path) -> str:
        raise AssertionError("production reader should have been restored")

    _test_hooks.set_analysis_hooks(read_text_fn=fake_read, list_session_paths_fn=fake_list)
    _test_hooks.reset_analysis_hooks()
    _write(tmp_path, "a.capture_session.json", _session_json(magic=None))
    assert scan_archive(tmp_path) == [
        {"path": str(tmp_path / "a.capture_session.json"), "reason": "no_magic"}
    ]


def test_hook_surface_is_exactly_the_two_seams() -> None:
    """Shape guard: a new seam must be added deliberately, not silently."""
    exported = sorted(name for name in _test_hooks.__all__)
    assert exported == [
        "ListSessionPathsFn",
        "ReadTextFn",
        "list_session_paths",
        "read_text",
        "reset_analysis_hooks",
        "set_analysis_hooks",
    ]


def test_capture_session_type_is_reexported() -> None:
    """The layer speaks the existing session type, not a private copy."""
    session: CaptureSession = {
        "session_id": "x",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": None,
        "base_url": "u",
        "messages": [],
        "magic": None,
        "game_log": [],
        "tank_names": {},
    }
    assert session["magic"] is None
