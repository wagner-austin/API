"""Shared fixtures and helpers for sniffer tests."""

from __future__ import annotations

import base64

from tankpit_bot.capture.xor import build_session_xor_table, xor_decode_body

SNIFFER_TEST_MAGIC = "snifftest"
"""Magic these tests build their session table from.

Any magic works — the decoders take the table as a parameter now, so
the value only has to be stable within a test ([[session-state-deglobalisation]])."""


def sniffer_xor_table() -> bytes:
    """Build the sniffer tests' session XOR table.

    Returns:
        The table for :data:`SNIFFER_TEST_MAGIC`.

    Raises:
        XorStaticKeyUnavailableError: If the repo's static key is
            missing.
    """
    return build_session_xor_table(SNIFFER_TEST_MAGIC)


def make_payload(body: bytes) -> str:
    """Create a base64 payload with 2-byte length header.

    Args:
        body: Raw message body bytes.

    Returns:
        Base64-encoded payload with length header.
    """
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


def make_binary_payload(msg_type: int, plaintext: bytes) -> str:
    """Frame a binary body that decodes to ``plaintext``.

    The type byte travels in the clear and the rest is ciphered under
    :func:`sniffer_xor_table`, so ``process_received_message`` handed
    that same table recovers ``plaintext`` exactly. XOR is its own
    inverse, so encoding runs through the production decode helper.

    Args:
        msg_type: Message type byte, carried in the clear.
        plaintext: The bytes the decoder should recover.

    Returns:
        Base64-encoded payload with length header.

    Raises:
        XorStaticKeyUnavailableError: If the repo's static key is
            missing.
    """
    ciphered = xor_decode_body(bytes(1) + plaintext, sniffer_xor_table(), offset=1)
    return make_payload(bytes([msg_type]) + ciphered)
