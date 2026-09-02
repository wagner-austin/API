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
from tankpit_bot.protocol.encoders import (
    encode_build_pickup,
    encode_radar_result,
    encode_tank_info,
)
from tankpit_bot.protocol.framing import encode_frame
from tankpit_bot.protocol.types import BuildPickupDict, RadarResultDict, TankInfoDict

MAGIC = "abcdefgh"

#: The capturing client in fixture sessions, and a tank that is not it.
OWN_TANK = 601
FOREIGN_TANK = 709


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


def _tank_info(tank_id: int) -> bytes:
    """Build a ciphered 0x21 TankInfo frame body.

    Args:
        tank_id: The tank the identity names.

    Returns:
        Wire body: the 0x21 type byte then the ciphered payload.
    """
    payload = encode_tank_info(
        TankInfoDict(
            msg_type=0x21,
            tank_id=tank_id,
            team=1,
            decoration_state=bytes(4),
            persistent_tank_id=7,
            name="red-9",
        )
    )
    return _ciphered(bytes([0x21]) + payload)


def _build_pickup(tank_id: int) -> bytes:
    """Build a ciphered 0x42 BuildPickup frame body.

    Args:
        tank_id: The tank the block action names as its actor.

    Returns:
        Wire body: the 0x42 type byte then the ciphered payload.
    """
    payload = encode_build_pickup(
        BuildPickupDict(
            msg_type=0x42,
            tank_id=tank_id,
            source_x=253,
            source_y=9,
            drop_x=254,
            drop_y=9,
            direction=0,
            obstacle_type=2,
            flag=0,
        )
    )
    return _ciphered(bytes([0x42]) + payload)


def _command(framed: bytes) -> bytes:
    """Cipher one client command into its sent-frame body.

    The production ``build_*_command`` helpers return a COMPLETE framed
    message — a 2-byte LE length prefix, then ``!``, then the command —
    so the prefix is dropped here and :func:`_payload` re-frames. Left
    on, the frame's leading byte reads 0x05 instead of ``!`` and a
    sweep correctly declines to treat it as a command at all.

    Args:
        framed: Bytes from a production ``build_*_command``.

    Returns:
        Wire body the scanner decodes back to that command.
    """
    return _ciphered(framed[2:])


def _radar_result(*, found: bool = True) -> bytes:
    """Build a ciphered 0x46 RadarResult frame body.

    Args:
        found: Whether the scan reports an enemy.

    Returns:
        Wire body: the 0x46 type byte then the ciphered payload.
    """
    payload = encode_radar_result(RadarResultDict(msg_type=0x46, detection_type=0, found=found))
    return _ciphered(bytes([0x46]) + payload)


__all__ = [
    "FOREIGN_TANK",
    "MAGIC",
    "OWN_TANK",
    "_build_pickup",
    "_ciphered",
    "_command",
    "_expected_body",
    "_payload",
    "_radar_result",
    "_received",
    "_sent",
    "_session_json",
    "_tank_info",
    "_write",
]
