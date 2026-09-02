"""XOR codec for Tankpit game protocol.

The protocol uses per-session XOR encoding with two keys:
1. Static key: 1000-char string embedded in client JS
2. Magic key: Session-specific string set in tankpit.magic after login

The XOR table is built by XORing static key with magic key (cycling):
    table[i] = static_key[i] ^ magic[i % len(magic)]

Game commands are then XOR'd with this table before sending.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot import _test_hooks

# The static XOR key lives in the project root — of a SOURCE CHECKOUT.
# Four parents up from an installed module is site-packages, which is
# why the container names the file by env instead (same law as the
# accounts pool, [[fleet-lifecycle]] container notes).
_CHECKOUT_STATIC_KEY_PATH = Path(__file__).parent.parent.parent.parent / "xor_static_key.txt"


def static_key_file_path() -> Path:
    """Resolve where the static XOR key lives.

    ``TANKPIT_XOR_KEY_FILE`` names the file explicitly — the fleet
    image bakes the tracked key at ``/app/xor_static_key.txt`` and
    sets this in its environment, because the checkout-relative
    default resolves into site-packages once the package is
    pip-installed. Unset, the source checkout's project root is the
    location it always was. Without this file no session can decode
    a single wire byte, so the resolution must never be guessed.

    Returns:
        The static key path.
    """
    override = _test_hooks.get_env("TANKPIT_XOR_KEY_FILE")
    if override is None or override == "":
        return _CHECKOUT_STATIC_KEY_PATH
    return Path(override)


class CodecError(Exception):
    """Base error for codec operations."""


class InvalidKeyError(CodecError):
    """Raised when a key is invalid (empty or wrong format)."""


def load_static_key(path: Path) -> str:
    """Load the static XOR key from a file.

    Args:
        path: Path to the static key file.

    Returns:
        The static key string (stripped of whitespace).

    Raises:
        FileNotFoundError: If the file does not exist.
        InvalidKeyError: If the key is empty after stripping.
    """
    content = _test_hooks.read_text(path)
    key = content.strip()
    if len(key) == 0:
        raise InvalidKeyError("Static key file is empty")
    return key


def build_xor_table(static_key: str, magic: str) -> bytes:
    """Build XOR encoding table from static key and magic.

    The table is built by XORing each character of the static key with
    the magic key (cycling through magic if it's shorter).

    Args:
        static_key: The static XOR key from client JS.
        magic: The session-specific magic key from tankpit.magic.

    Returns:
        XOR table as bytes, same length as static_key.

    Raises:
        InvalidKeyError: If either key is empty.
    """
    if len(static_key) == 0:
        raise InvalidKeyError("Static key is empty")
    if len(magic) == 0:
        raise InvalidKeyError("Magic key is empty")

    magic_len = len(magic)
    table = bytearray(len(static_key))

    for i, char in enumerate(static_key):
        static_byte = ord(char)
        magic_byte = ord(magic[i % magic_len])
        table[i] = static_byte ^ magic_byte

    return bytes(table)


def xor_bytes(table: bytes, data: bytes, offset: int = 0) -> bytes:
    """XOR data with the encoding table.

    Since XOR is symmetric, this function works for both encoding and decoding.

    Args:
        table: The XOR encoding table.
        data: Data bytes to encode/decode.
        offset: Starting position in the table (for commands that skip bytes).

    Returns:
        XOR'd bytes, same length as data.

    Raises:
        InvalidKeyError: If table is empty.
        ValueError: If offset + len(data) exceeds table length.
    """
    if len(table) == 0:
        raise InvalidKeyError("XOR table is empty")
    if len(data) == 0:
        return b""

    end_pos = offset + len(data)
    if end_pos > len(table):
        raise ValueError(
            f"Data extends beyond table: offset={offset}, "
            f"data_len={len(data)}, table_len={len(table)}"
        )

    result = bytearray(len(data))
    for i, byte in enumerate(data):
        result[i] = byte ^ table[offset + i]

    return bytes(result)


class ProtocolCodec:
    """Encoder/decoder for Tankpit game protocol.

    Holds the XOR table and provides encode/decode methods.
    """

    def __init__(self, static_key: str, magic: str) -> None:
        """Initialize codec with keys.

        Args:
            static_key: The static XOR key from client JS.
            magic: The session-specific magic key from tankpit.magic.

        Raises:
            InvalidKeyError: If either key is empty.
        """
        self._table = build_xor_table(static_key, magic)

    @property
    def table(self) -> bytes:
        """Get the XOR encoding table.

        Returns:
            The XOR table as bytes.
        """
        return self._table

    def encode(self, data: bytes, offset: int = 0) -> bytes:
        """Encode data using the XOR table.

        Args:
            data: Data bytes to encode.
            offset: Starting position in the table.

        Returns:
            Encoded bytes.

        Raises:
            ValueError: If offset + len(data) exceeds table length.
        """
        return xor_bytes(self._table, data, offset)

    def decode(self, data: bytes, offset: int = 0) -> bytes:
        """Decode data using the XOR table.

        Since XOR is symmetric, this is identical to encode.

        Args:
            data: Data bytes to decode.
            offset: Starting position in the table.

        Returns:
            Decoded bytes.

        Raises:
            ValueError: If offset + len(data) exceeds table length.
        """
        return xor_bytes(self._table, data, offset)


def extract_magic_from_auth_payload(payload_bytes: bytes) -> str | None:
    """Extract magic key from AUTH message payload.

    AUTH message format: %AUTH !be <session_id>|<hash>|<timestamp> <magic>
    The magic is the last space-separated token.

    Args:
        payload_bytes: Raw AUTH message bytes (including 2-byte length prefix).

    Returns:
        Magic key string, or None if not an AUTH message or extraction fails.
    """
    # Skip 2-byte length prefix
    body = payload_bytes[2:]
    text = body.decode("utf-8", errors="replace")

    if "%AUTH" not in text and "AUTH" not in text:
        return None

    parts = text.split()
    if len(parts) < 3:
        return None

    magic = parts[-1]
    if len(magic) < 10:
        return None

    return magic


def create_codec(static_key_path: Path, magic: str) -> ProtocolCodec:
    """Create a protocol codec by loading static key from file.

    Convenience function that loads the static key and creates a codec.

    Args:
        static_key_path: Path to the static key file.
        magic: The session-specific magic key from tankpit.magic.

    Returns:
        Configured ProtocolCodec instance.

    Raises:
        FileNotFoundError: If static key file does not exist.
        InvalidKeyError: If any key is empty.
    """
    static_key = load_static_key(static_key_path)
    return ProtocolCodec(static_key, magic)


__all__ = [
    "CodecError",
    "InvalidKeyError",
    "ProtocolCodec",
    "build_xor_table",
    "create_codec",
    "extract_magic_from_auth_payload",
    "load_static_key",
    "static_key_file_path",
    "xor_bytes",
]
