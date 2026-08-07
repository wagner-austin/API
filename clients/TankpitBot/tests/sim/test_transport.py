"""Wire-layer unit tests: framing, XOR passthrough, and error paths."""

from __future__ import annotations

import base64

import pytest

from tankpit_bot.capture.xor import xor_decode_body
from tankpit_bot.protocol.commands import CMD_MOVE, COMMAND_PREFIX, TYPE_MOVEMENT
from tankpit_bot.protocol.types import SyncDict
from tankpit_bot.sim.commands import SimError
from tankpit_bot.sim.transport import decode_client_payload, encode_tick_payload
from tankpit_bot.wire.helpers import DecodeError, pack16

_TINY_TABLE = bytes([0x07])
"""A one-byte table: anything but the shortest body runs past its end.

Kept to prove the sim REFUSES such a frame. It used to prove the
opposite ([[session-state-deglobalisation]])."""

_TABLE = bytes([0x07, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77])
"""A table long enough to cover these tests' bodies, as the real
1000-byte one covers every frame the server has ever sent (931 max)."""


def _client_frame(payload: bytes, table: bytes) -> str:
    """Frame and XOR one client command payload the way the bot does.

    Args:
        payload: Plaintext command payload after the ``!`` prefix.
        table: Session XOR table.

    Returns:
        Base64 wire payload.
    """
    body = bytes([COMMAND_PREFIX]) + xor_decode_body(payload, table)
    return base64.b64encode(pack16(len(body)) + body).decode("ascii")


def test_tick_payload_refuses_an_envelope_past_the_table() -> None:
    """A body the cipher cannot cover is refused, not passed through.

    The real server's largest observed body is 931 bytes against a
    1000-byte table, measured over 282,783 bodies — the key length is
    the frame bound. A sim frame that exceeds it could not occur on the
    wire and the production decoder cannot read it, so emitting one is
    a sim bug rather than a cipher edge case
    ([[session-state-deglobalisation]]).
    """
    with pytest.raises(SimError, match="past the 1-byte cipher table"):
        encode_tick_payload([SyncDict(msg_type=0x3F)], _TINY_TABLE)


def test_tick_payload_encodes_a_body_the_table_covers() -> None:
    """Within the bound, the envelope ciphers against the real table."""
    payload = base64.b64decode(encode_tick_payload([SyncDict(msg_type=0x3F)], _TABLE))
    body_len = payload[0] | (payload[1] << 8)
    body = payload[2 : 2 + body_len]
    assert body[0] == 0x2E
    assert body[1] == 0x3F ^ _TABLE[0]
    assert body[2] == 0x01 ^ _TABLE[1]


def test_client_round_trip() -> None:
    """A framed move command decodes through the transport."""
    frame = _client_frame(bytes([TYPE_MOVEMENT, CMD_MOVE, 42, 161]), _TABLE)
    commands = decode_client_payload(frame, _TABLE)
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
