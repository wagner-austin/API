"""Wire-layer unit tests: framing, XOR passthrough, and error paths."""

from __future__ import annotations

import base64

import pytest

from tankpit_bot.protocol.commands import CMD_MOVE, COMMAND_PREFIX, TYPE_MOVEMENT
from tankpit_bot.protocol.helpers import DecodeError, pack16
from tankpit_bot.protocol.types import SyncDict
from tankpit_bot.sim.transport import decode_client_payload, encode_tick_payload

_TINY_TABLE = bytes([0x07])


def _client_frame(payload: bytes, table: bytes) -> str:
    """Frame and XOR one client command payload the way the bot does.

    Args:
        payload: Plaintext command payload after the ``!`` prefix.
        table: Session XOR table.

    Returns:
        Base64 wire payload.
    """
    encoded = bytes(byte ^ (table[i] if i < len(table) else 0) for i, byte in enumerate(payload))
    body = bytes([COMMAND_PREFIX]) + encoded
    return base64.b64encode(pack16(len(body)) + body).decode("ascii")


def test_tick_payload_xor_passes_through_beyond_the_table() -> None:
    """Bytes past the table length travel in the clear (xor_decode's rule)."""
    payload = base64.b64decode(encode_tick_payload([SyncDict(msg_type=0x3F)], _TINY_TABLE))
    body_len = payload[0] | (payload[1] << 8)
    body = payload[2 : 2 + body_len]
    assert body[0] == 0x2E
    assert body[1] == 0x3F ^ 0x07
    assert body[2] == 0x01


def test_client_round_trip_with_a_tiny_table() -> None:
    """A framed move command decodes through the transport."""
    frame = _client_frame(bytes([TYPE_MOVEMENT, CMD_MOVE, 42, 161]), _TINY_TABLE)
    commands = decode_client_payload(frame, _TINY_TABLE)
    assert [(c["kind"], c["x"], c["y"]) for c in commands] == [("move", 42, 161)]


def test_invalid_base64_raises() -> None:
    """Garbage payloads fail loudly, never best-effort."""
    with pytest.raises(DecodeError):
        decode_client_payload("not-base64!!!", _TINY_TABLE)


def test_torn_frame_raises() -> None:
    """A length prefix pointing past the payload is a decode failure."""
    torn = base64.b64encode(pack16(50) + bytes([COMMAND_PREFIX, 1, 2])).decode("ascii")
    with pytest.raises(DecodeError):
        decode_client_payload(torn, _TINY_TABLE)


def test_missing_command_prefix_raises() -> None:
    """Frames without the ``!`` prefix are rejected."""
    body = bytes([0x2E, 1, 2, 3])
    payload = base64.b64encode(pack16(len(body)) + body).decode("ascii")
    with pytest.raises(DecodeError):
        decode_client_payload(payload, _TINY_TABLE)
