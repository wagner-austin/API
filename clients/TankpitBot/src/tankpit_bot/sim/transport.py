"""Step (c) wire layer: sim messages to real bytes and back.

Server -> client: every decoded message becomes a length-prefixed
0x2E envelope frame (the real wire tunnels everything through the
container), XOR'd exactly the way ``capture.xor.xor_decode_body``
inverts it — the production ingestion path consumes the output
unchanged.

Client -> server: the bot's ``!``-prefixed command frames (as built
by ``protocol.commands``) decode back into typed
:class:`~tankpit_bot.sim.commands.ClientCommandDict` values.

Both directions use the production cipher with no local copy, and
:func:`_require_wire_sized` holds the sim to a frame size the real
server has actually been observed to produce. "The simulator is not a
parallel universe" is this module's claim; it is now checked rather
than asserted ([[session-state-deglobalisation]]).
"""

from __future__ import annotations

import base64

from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.capture.xor import xor_decode_body
from tankpit_bot.protocol.commands import COMMAND_PREFIX
from tankpit_bot.protocol.encoders import encode_envelope_body
from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import ClientCommandDict, SimError, decode_client_command
from tankpit_bot.wire.helpers import DecodeError, pack16

_ENVELOPE_TYPE = 0x2E


def encode_tick_payload(messages: list[BinaryMessage], table: bytes) -> str:
    """Encode one tick's message batch as a wire frame payload.

    The cipher is the PRODUCTION one. This module used to carry its own
    copy whose bytes past the table travelled in the clear, which let
    the sim emit envelopes the real server could not produce AND the
    production decoder could not read — 904 such frames sit in
    ``runs/sim`` ([[session-state-deglobalisation]]).

    Args:
        messages: The tick's decoded messages, in emission order.
        table: Session XOR table.

    Returns:
        Base64 payload holding one length-prefixed 0x2E envelope
        frame per message — exactly what the production
        ``process_received_message`` ingests.

    Raises:
        SimError: If a message's envelope would exceed what the cipher
            covers. See :func:`_require_wire_sized`.
    """
    out = bytearray()
    for message in messages:
        plaintext = encode_envelope_body(message)
        _require_wire_sized(message, plaintext, table)
        body = bytes([_ENVELOPE_TYPE]) + xor_decode_body(plaintext, table)
        out += pack16(len(body)) + body
    return base64.b64encode(bytes(out)).decode("ascii")


def _require_wire_sized(message: BinaryMessage, plaintext: bytes, table: bytes) -> None:
    """Refuse to emit an envelope the real server could never send.

    Measured over the whole archive on 2026-08-06: across 282,783
    frame bodies from ``runs/bot``, ``runs/sniff`` and ``runs/probe``
    — every byte the REAL server ever sent us — the largest ciphered
    span is 931 bytes, against a 1000-byte table. The key length is
    the server's frame bound, and nothing on the wire has ever reached
    it.

    The sim was emitting 8,780. A single ``MapData`` carrying the whole
    mined container atlas (8,592 dots, where the real server's largest
    map reveals 656) does not fit any frame the game produces, so a sim
    that sends one is not modelling the wire — it is inventing a shape
    the bot will never meet and cannot decode.

    Args:
        message: The message being encoded, named in the error.
        plaintext: Its encoded envelope body, before the cipher.
        table: Session XOR table; its length is the frame bound.

    Raises:
        SimError: If the body runs past the end of the table.
    """
    if len(plaintext) <= len(table):
        return
    raise SimError(
        f"sim envelope for {message['msg_type']!r} is {len(plaintext)} bytes, "
        f"past the {len(table)}-byte cipher table; the real server's largest "
        f"observed body is 931 bytes, so this frame could not occur on the "
        f"wire and the bot cannot decode it"
    )


def encode_plaintext_payload(frames: list[bytes]) -> str:
    """Encode lobby frames as a wire payload, in the clear.

    Lobby traffic is the one thing on this wire that is neither XOR'd
    nor enveloped: the archive's room lists, join confirms, enter
    responses and toggle acks are all plaintext top-level frames
    ([[session-state-deglobalisation]]).

    Args:
        frames: Frame bodies including their lead byte.

    Returns:
        Base64 payload holding one length-prefixed frame per body.
    """
    out = bytearray()
    for body in frames:
        out += pack16(len(body)) + body
    return base64.b64encode(bytes(out)).decode("ascii")


def split_client_frames(payload: str) -> list[bytes]:
    """Split a client payload into frame bodies without reading them.

    The lobby and the command channel share one socket, and they are
    told apart by the leading byte — ``!`` is a command, anything else
    is lobby. Splitting has to happen before that question can be
    asked, so it is its own step here rather than a branch inside
    :func:`decode_client_payload`.

    Args:
        payload: Base64 payload as sent by the bot.

    Returns:
        The frame bodies, in order.

    Raises:
        DecodeError: If the payload is not valid base64 or a frame is
            torn.
    """
    try:
        return split_payload_frames(payload)
    except FramingError as error:
        raise DecodeError(f"undecodable client payload: {error}") from error


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
    commands: list[ClientCommandDict] = []
    for body in split_client_frames(payload):
        if body[0] != COMMAND_PREFIX:
            raise DecodeError(f"client frame missing '!' prefix: 0x{body[0]:02X}")
        commands.append(decode_client_command(xor_decode_body(body, table, offset=1)))
    return commands


__all__ = [
    "decode_client_payload",
    "encode_plaintext_payload",
    "encode_tick_payload",
    "split_client_frames",
]
