"""Shared fixtures and helpers for sniffer tests."""

from __future__ import annotations

from tankpit_bot.capture.xor import build_session_xor_table

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


# ``make_payload`` and ``make_binary_payload`` used to live here. The
# first was one of eleven copies of "length header + base64"; the
# second only bound :func:`sniffer_xor_table` to a call that
# ``tests.wire_builders.encode_wire_frame`` already makes. Callers use
# those two directly ([[session-state-deglobalisation]]).
