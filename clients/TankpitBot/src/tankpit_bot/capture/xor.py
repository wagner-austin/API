"""The capture layer's entry point to the session cipher.

The cipher PRIMITIVES live one layer down in
:mod:`tankpit_bot.protocol.codec` — key loading, table building, and
the key path are that module's, and this one no longer carries copies
of them ([[session-state-deglobalisation]]). What lives here is the
session-scoped concern the capture layer actually needs: build the
table belonging to ONE session's magic, cache the process-wide key
behind it, and the base64 helpers the frame readers use.
"""

from __future__ import annotations

import base64
import re

from tankpit_bot import _test_hooks
from tankpit_bot.protocol.codec import (
    DEFAULT_STATIC_KEY_PATH,
    CodecError,
    build_xor_table,
    load_static_key,
)

#: Process-wide cache of the static key. Unlike a session's table this
#: is NOT session state: the same key builds every session's table, so
#: reading ``xor_static_key.txt`` once is a property of the key itself.
_static_key_cache: str | None = None


class XorStaticKeyUnavailableError(CodecError):
    """Raised when ``xor_static_key.txt`` cannot be read.

    A :class:`~tankpit_bot.protocol.codec.CodecError` rather than a
    parallel error family — the condition belongs to the cipher, and
    the cipher has exactly one owner.

    No session table can be built without the key, and decoding against
    a missing key yields plausible garbage rather than an error, so
    callers must stop rather than continue.
    """


def require_static_key() -> str:
    """Return the process-wide static key, reading it once.

    Args:
        None.

    Returns:
        The 1000-character static key every session's table is built
        from.

    Raises:
        XorStaticKeyUnavailableError: If the key file is missing.
            Decoding against a missing key yields plausible garbage
            rather than an error, so callers must stop rather than
            continue.
    """
    global _static_key_cache
    if _static_key_cache is None:
        if not _test_hooks.path_exists(DEFAULT_STATIC_KEY_PATH):
            raise XorStaticKeyUnavailableError(
                "static XOR key unavailable (xor_static_key.txt missing); "
                "cannot build a session XOR table"
            )
        _static_key_cache = load_static_key(DEFAULT_STATIC_KEY_PATH)
    return _static_key_cache


def build_session_xor_table(magic: str) -> bytes:
    """Build the XOR table belonging to one session.

    The returned table is a VALUE the caller owns. It replaced a module
    global that a second session would silently overwrite, decoding the
    first session's frames against the wrong key. It also replaced
    seventeen hand-rolled copies of load-key-then-build-table, eleven of
    which silently left the table ``None`` when the key was missing
    ([[session-state-deglobalisation]]).

    Args:
        magic: The session magic captured from the client.

    Returns:
        The XOR table for that session.

    Raises:
        XorStaticKeyUnavailableError: If the static key cannot be read.
    """
    return build_xor_table(require_static_key(), magic)


def reset_static_key_cache() -> None:
    """Clear the cached static key.

    Only tests that exercise key loading need this; in production the
    key is a process-wide constant that is never invalidated.
    """
    global _static_key_cache
    _static_key_cache = None


def is_valid_base64(payload: str) -> bool:
    """Check if payload is valid base64.

    Args:
        payload: String to validate.

    Returns:
        True if valid base64, False otherwise.
    """
    if not payload:
        return False
    # Base64 uses A-Z, a-z, 0-9, +, /, and = for padding
    pattern = r"^[A-Za-z0-9+/]*={0,2}$"
    if not re.match(pattern, payload):
        return False
    # Length must be multiple of 4 (with padding)
    return len(payload) % 4 == 0


def decode_base64_safe(payload: str) -> bytes | None:
    """Validate and decode base64 payload.

    Args:
        payload: Base64-encoded string.

    Returns:
        Decoded bytes, or None if invalid.
    """
    if not is_valid_base64(payload):
        return None
    return base64.b64decode(payload)


def xor_decode_body(body: bytes, xor_table: bytes, offset: int = 0) -> bytes:
    """XOR decode a message body, wrapping the table for long spans.

    The wrap is the measured cipher law, not a tolerance: the real
    client's decode is ``l[ja] ^= B[ja % pa]`` ([[xor-cipher]], tpclient.js
    case 46), so the server may send bodies longer than the 1000-byte
    table. A span-length guard here crashed artax live on 2026-08-26
    when a busy practice room grew a 0x5A map frame to 1051 ciphered
    bytes — past the previous 931-byte archive maximum the guard had
    mistaken for a protocol bound.

    Args:
        body: Raw message body bytes.
        xor_table: XOR table to use.
        offset: Starting offset in body (default 0).

    Returns:
        Decoded bytes.
    """
    span = len(body) - offset
    table_len = len(xor_table)
    decoded = bytearray(span)
    for i in range(span):
        decoded[i] = body[i + offset] ^ xor_table[i % table_len]
    return bytes(decoded)


__all__ = [
    "XorStaticKeyUnavailableError",
    "build_session_xor_table",
    "decode_base64_safe",
    "is_valid_base64",
    "require_static_key",
    "reset_static_key_cache",
    "xor_decode_body",
]
