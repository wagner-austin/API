"""Build wire payloads the way the server would, for tests.

Twenty-seven private helpers across twenty test files each framed a
body and ciphered it for themselves — ``make_payload``,
``_make_tracker_payload``, ``_frame``, ``_frame_payload``, ``_payload``,
``_make_text_payload``, ``_xor_encode_bytes`` (four byte-identical
copies), ``_encode_received_frame`` (eight, in three spellings that
differ only in whether they name a ``length`` local), and
``_encode_sent_frame``. None of them called the production framer that
was there the whole time ([[session-state-deglobalisation]]).

Both primitives below are the PRODUCTION ones:

* :func:`~tankpit_bot.protocol.framing.encode_frame` writes the 2-byte
  little-endian length header, so a test can never disagree with the
  splitter about what a frame is;
* :func:`~tankpit_bot.capture.xor.xor_decode_body` ciphers the body —
  XOR is its own inverse, so the decode helper IS the encode helper and
  no separate ``xor_encode`` is needed or wanted.
"""

from __future__ import annotations

import base64

from tankpit_bot.capture.xor import xor_decode_body
from tankpit_bot.protocol.framing import encode_frame


def frame_payload(*bodies: bytes) -> str:
    """Frame one or more bodies into a single base64 wire payload.

    Args:
        *bodies: Message bodies, each framed with its own length
            header, in order. Passing several models the real wire,
            which packs multiple logical messages into one payload.

    Returns:
        The base64 payload a capture would record.
    """
    return base64.b64encode(b"".join(encode_frame(body) for body in bodies)).decode("ascii")


def encode_wire_frame(prefix: int, plaintext: bytes, xor_table: bytes) -> str:
    """Frame a body with its leading byte in the clear and the rest ciphered.

    This is the wire's shape in BOTH directions: a received frame keeps
    its ``msg_type`` byte readable, and a sent command keeps its ``!``
    prefix readable. Direction is not part of the encoding, so it is
    not a parameter — the eight received-side and two sent-side copies
    this replaces were the same function under two names.

    Args:
        prefix: The type or command byte, carried in the clear.
        plaintext: The bytes after it, which the cipher covers.
        xor_table: Session XOR table.

    Returns:
        The base64 payload a capture would record.
    """
    return frame_payload(bytes([prefix]) + xor_decode_body(plaintext, xor_table))


__all__ = [
    "encode_wire_frame",
    "frame_payload",
]
