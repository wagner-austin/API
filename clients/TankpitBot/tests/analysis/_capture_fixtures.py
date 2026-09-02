"""Shared capture-session builders for the analysis tests.

Sessions are built as real JSON on disk and frames go through the
production encoder and the production cipher, so a change to either
surfaces here rather than passing a hand-rolled fixture that agrees
with nothing.

These lived privately in ``test_scan.py`` until the recipient-policy
sweep needed the same builders; one definition serves both suites.
"""

from __future__ import annotations

import base64
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str

from tankpit_bot.capture.xor import build_session_xor_table, xor_decode_body
from tankpit_bot.protocol.framing import encode_frame

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


def _ciphered(plaintext: bytes) -> bytes:
    """Cipher a frame body so the scanner decodes it back to ``plaintext``.

    The XOR table is its own inverse at a fixed offset, so the decode
    helper doubles as the encoder — which keeps the fixture honest: a
    change to the cipher breaks construction and reading together
    rather than leaving them agreeing on the wrong bytes.

    Args:
        plaintext: Frame body as the decoder should see it — type byte
            first, then the payload to be ciphered.

    Returns:
        The wire body: type byte, then the ciphered remainder.
    """
    return plaintext[:1] + _expected_body(plaintext)


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
        "game_log": [],
        "tank_names": {},
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


def _sent(payload: str, timestamp_ms: int = 1500) -> JSONObject:
    """Build one sent message — the client's own command channel.

    Args:
        payload: Base64 payload.
        timestamp_ms: Capture time.

    Returns:
        A captured-message JSON object.
    """
    return {
        "timestamp_ms": timestamp_ms,
        "direction": "sent",
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


__all__ = [
    "MAGIC",
    "_ciphered",
    "_expected_body",
    "_payload",
    "_received",
    "_sent",
    "_session_json",
    "_write",
]
