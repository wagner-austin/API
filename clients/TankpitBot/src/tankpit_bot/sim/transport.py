"""Step (c) wire layer: sim messages to real bytes and back.

Server -> client: every decoded message becomes a length-prefixed
0x2E envelope frame (the real wire tunnels everything through the
container), XOR'd exactly the way ``capture.xor.xor_decode_body``
inverts it — the production ingestion path consumes the output
unchanged.

Client -> server: the bot's ``!``-prefixed command frames (as built
by ``protocol.commands``) decode back into typed
:class:`~tankpit_bot.sim.commands.ClientCommandDict` values.
"""

from __future__ import annotations

import base64

from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.protocol.commands import COMMAND_PREFIX
from tankpit_bot.protocol.encoders import encode_envelope_body
from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import ClientCommandDict, decode_client_command
from tankpit_bot.wire.helpers import DecodeError, pack16

_ENVELOPE_TYPE = 0x2E


def _xor_with_table(table: bytes, data: bytes) -> bytes:
    """XOR data against the session table, passing through beyond it.

    Mirrors ``capture.xor.xor_decode_body`` (which is its own
    inverse) at ``offset=0``, but bytes past the table length travel
    in the clear here rather than raising. Folding the two is tracked
    as its own step ([[session-state-deglobalisation]]).

    Args:
        table: Session XOR table.
        data: Bytes to transform.

    Returns:
        Transformed bytes, same length.
    """
    out = bytearray(len(data))
    for index in range(len(data)):
        key = table[index] if index < len(table) else 0
        out[index] = data[index] ^ key
    return bytes(out)


def encode_tick_payload(messages: list[BinaryMessage], table: bytes) -> str:
    """Encode one tick's message batch as a wire frame payload.

    Args:
        messages: The tick's decoded messages, in emission order.
        table: Session XOR table.

    Returns:
        Base64 payload holding one length-prefixed 0x2E envelope
        frame per message — exactly what the production
        ``process_received_message`` ingests.
    """
    out = bytearray()
    for message in messages:
        body = bytes([_ENVELOPE_TYPE]) + _xor_with_table(table, encode_envelope_body(message))
        out += pack16(len(body)) + body
    return base64.b64encode(bytes(out)).decode("ascii")


def decode_client_payload(payload: str, table: bytes) -> list[ClientCommandDict]:
    """Decode a client frame payload into typed commands.

    Args:
        payload: Base64 payload as sent by the bot's command sender
            (length-prefixed ``!`` frames, XOR after the prefix).
        table: Session XOR table.

    Returns:
        The decoded commands, in frame order.

    Raises:
        DecodeError: If the payload is not valid base64, a frame is
            torn, or a frame does not carry the ``!`` command prefix.
    """
    # The split is shared; only the translation into this module's
    # DecodeError is local ([[session-state-deglobalisation]]).
    try:
        frames = split_payload_frames(payload)
    except FramingError as error:
        raise DecodeError(f"undecodable client payload: {error}") from error
    commands: list[ClientCommandDict] = []
    for body in frames:
        if body[0] != COMMAND_PREFIX:
            raise DecodeError(f"client frame missing '!' prefix: 0x{body[0]:02X}")
        commands.append(decode_client_command(_xor_with_table(table, body[1:])))
    return commands


__all__ = [
    "decode_client_payload",
    "encode_tick_payload",
]
