"""Tests for tankpit_bot.codec module."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot.protocol.codec import (
    DEFAULT_STATIC_KEY_PATH,
    CodecError,
    InvalidKeyError,
    ProtocolCodec,
    build_xor_table,
    create_codec,
    extract_magic_from_auth_payload,
    load_static_key,
    xor_bytes,
)
from tests.conftest import FakeFileSystem

# =============================================================================
# load_static_key Tests
# =============================================================================


def test_load_static_key_success(fake_fs: FakeFileSystem) -> None:
    """Test loading static key from file."""
    fake_fs.write_text(Path("key.txt"), "ABC123XYZ")

    result = load_static_key(Path("key.txt"))

    assert result == "ABC123XYZ"


def test_load_static_key_strips_whitespace(fake_fs: FakeFileSystem) -> None:
    """Test that whitespace is stripped from static key."""
    fake_fs.write_text(Path("key.txt"), "  ABC123  \n\n")

    result = load_static_key(Path("key.txt"))

    assert result == "ABC123"


def test_load_static_key_empty_raises(fake_fs: FakeFileSystem) -> None:
    """Test that empty key file raises InvalidKeyError."""
    fake_fs.write_text(Path("key.txt"), "   \n\n  ")

    with pytest.raises(InvalidKeyError, match="Static key file is empty"):
        load_static_key(Path("key.txt"))


def test_load_static_key_file_not_found(fake_fs: FakeFileSystem) -> None:
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_static_key(Path("nonexistent.txt"))


# =============================================================================
# build_xor_table Tests
# =============================================================================


def test_build_xor_table_basic() -> None:
    """Test building XOR table with simple keys."""
    static_key = "ABCD"
    magic = "12"

    table = build_xor_table(static_key, magic)

    # A ^ 1 = 0x41 ^ 0x31 = 0x70
    # B ^ 2 = 0x42 ^ 0x32 = 0x70
    # C ^ 1 = 0x43 ^ 0x31 = 0x72
    # D ^ 2 = 0x44 ^ 0x32 = 0x76
    assert table == bytes([0x70, 0x70, 0x72, 0x76])


def test_build_xor_table_magic_cycles() -> None:
    """Test that magic key cycles when shorter than static key."""
    static_key = "AAAAAA"
    magic = "XY"

    table = build_xor_table(static_key, magic)

    # A ^ X, A ^ Y, A ^ X, A ^ Y, A ^ X, A ^ Y
    expected_first = ord("A") ^ ord("X")
    expected_second = ord("A") ^ ord("Y")
    assert table[0] == expected_first
    assert table[1] == expected_second
    assert table[2] == expected_first
    assert table[3] == expected_second
    assert table[4] == expected_first
    assert table[5] == expected_second


def test_build_xor_table_empty_static_raises() -> None:
    """Test that empty static key raises InvalidKeyError."""
    with pytest.raises(InvalidKeyError, match="Static key is empty"):
        build_xor_table("", "magic")


def test_build_xor_table_empty_magic_raises() -> None:
    """Test that empty magic key raises InvalidKeyError."""
    with pytest.raises(InvalidKeyError, match="Magic key is empty"):
        build_xor_table("static", "")


def test_build_xor_table_same_length() -> None:
    """Test that table length matches static key length."""
    static_key = "A" * 100
    magic = "B" * 50

    table = build_xor_table(static_key, magic)

    assert len(table) == 100


# =============================================================================
# xor_bytes Tests
# =============================================================================


def test_xor_bytes_basic() -> None:
    """Test XOR encoding of bytes."""
    table = bytes([0x00, 0xFF, 0xAA, 0x55])
    data = bytes([0x12, 0x34, 0x56, 0x78])

    result = xor_bytes(table, data)

    # 0x12 ^ 0x00 = 0x12
    # 0x34 ^ 0xFF = 0xCB
    # 0x56 ^ 0xAA = 0xFC
    # 0x78 ^ 0x55 = 0x2D
    assert result == bytes([0x12, 0xCB, 0xFC, 0x2D])


def test_xor_bytes_symmetric() -> None:
    """Test that XOR is symmetric (encode then decode = original)."""
    table = bytes([0x12, 0x34, 0x56, 0x78])
    original = b"test"

    encoded = xor_bytes(table, original)
    decoded = xor_bytes(table, encoded)

    assert decoded == original


def test_xor_bytes_with_offset() -> None:
    """Test XOR with offset into table."""
    table = bytes([0x00, 0x00, 0xFF, 0xFF])
    data = bytes([0x12, 0x34])

    result = xor_bytes(table, data, offset=2)

    # 0x12 ^ 0xFF = 0xED
    # 0x34 ^ 0xFF = 0xCB
    assert result == bytes([0xED, 0xCB])


def test_xor_bytes_empty_data() -> None:
    """Test XOR with empty data returns empty bytes."""
    table = bytes([0x12, 0x34])
    data = b""

    result = xor_bytes(table, data)

    assert result == b""


def test_xor_bytes_empty_table_raises() -> None:
    """Test that empty table raises InvalidKeyError."""
    with pytest.raises(InvalidKeyError, match="XOR table is empty"):
        xor_bytes(b"", b"data")


def test_xor_bytes_exceeds_table_raises() -> None:
    """Test that data exceeding table length raises ValueError."""
    table = bytes([0x12, 0x34])
    data = bytes([0x01, 0x02, 0x03, 0x04])

    with pytest.raises(ValueError, match="Data extends beyond table"):
        xor_bytes(table, data)


def test_xor_bytes_offset_exceeds_table_raises() -> None:
    """Test that offset + data exceeding table raises ValueError."""
    table = bytes([0x12, 0x34, 0x56, 0x78])
    data = bytes([0x01, 0x02, 0x03])

    with pytest.raises(ValueError, match="Data extends beyond table"):
        xor_bytes(table, data, offset=2)


# =============================================================================
# ProtocolCodec Tests
# =============================================================================


def test_protocol_codec_init() -> None:
    """Test ProtocolCodec initialization."""
    codec = ProtocolCodec("ABCD", "12")

    assert len(codec.table) == 4


def test_protocol_codec_encode() -> None:
    """Test ProtocolCodec.encode method."""
    codec = ProtocolCodec("ABCD", "12")
    data = b"\x00\x00\x00\x00"

    encoded = codec.encode(data)

    # Encoded should equal the table XOR'd with zeros = table itself
    assert encoded == codec.table


def test_protocol_codec_decode() -> None:
    """Test ProtocolCodec.decode method."""
    codec = ProtocolCodec("ABCD", "12")
    original = b"test"

    encoded = codec.encode(original)
    decoded = codec.decode(encoded)

    assert decoded == original


def test_protocol_codec_encode_with_offset() -> None:
    """Test ProtocolCodec.encode with offset."""
    codec = ProtocolCodec("ABCD", "12")
    data = b"\x00\x00"

    encoded = codec.encode(data, offset=2)

    assert encoded == codec.table[2:4]


def test_protocol_codec_decode_with_offset() -> None:
    """Test ProtocolCodec.decode with offset."""
    codec = ProtocolCodec("ABCD", "12")
    original = b"XY"

    encoded = codec.encode(original, offset=1)
    decoded = codec.decode(encoded, offset=1)

    assert decoded == original


def test_protocol_codec_empty_static_raises() -> None:
    """Test that empty static key raises InvalidKeyError."""
    with pytest.raises(InvalidKeyError, match="Static key is empty"):
        ProtocolCodec("", "magic")


def test_protocol_codec_empty_magic_raises() -> None:
    """Test that empty magic raises InvalidKeyError."""
    with pytest.raises(InvalidKeyError, match="Magic key is empty"):
        ProtocolCodec("static", "")


# =============================================================================
# create_codec Tests
# =============================================================================


def test_create_codec_success(fake_fs: FakeFileSystem) -> None:
    """Test create_codec loads key and creates codec."""
    fake_fs.write_text(Path("static.txt"), "STATICKEY")

    codec = create_codec(Path("static.txt"), "magic123")

    assert len(codec.table) == len("STATICKEY")


def test_create_codec_file_not_found(fake_fs: FakeFileSystem) -> None:
    """Test create_codec raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        create_codec(Path("nonexistent.txt"), "magic")


def test_create_codec_empty_key_raises(fake_fs: FakeFileSystem) -> None:
    """Test create_codec raises InvalidKeyError for empty key file."""
    fake_fs.write_text(Path("empty.txt"), "")

    with pytest.raises(InvalidKeyError, match="Static key file is empty"):
        create_codec(Path("empty.txt"), "magic")


def test_create_codec_empty_magic_raises(fake_fs: FakeFileSystem) -> None:
    """Test create_codec raises InvalidKeyError for empty magic."""
    fake_fs.write_text(Path("static.txt"), "STATICKEY")

    with pytest.raises(InvalidKeyError, match="Magic key is empty"):
        create_codec(Path("static.txt"), "")


# =============================================================================
# Error Class Tests
# =============================================================================


def test_codec_error_is_exception() -> None:
    """Test CodecError is an Exception."""
    assert issubclass(CodecError, Exception)
    err = CodecError("test error")
    assert str(err) == "test error"


def test_invalid_key_error_is_codec_error() -> None:
    """Test InvalidKeyError is a CodecError."""
    assert issubclass(InvalidKeyError, CodecError)
    err = InvalidKeyError("bad key")
    assert str(err) == "bad key"


# =============================================================================
# Default Path Tests
# =============================================================================


def test_default_static_key_path_has_expected_name() -> None:
    """Test DEFAULT_STATIC_KEY_PATH has expected file name."""
    assert DEFAULT_STATIC_KEY_PATH.name == "xor_static_key.txt"


def test_default_static_key_path_is_path_type() -> None:
    """Test DEFAULT_STATIC_KEY_PATH is a Path."""
    # Verify by calling a Path method - this will fail if not a Path
    assert DEFAULT_STATIC_KEY_PATH.suffix == ".txt"


# =============================================================================
# extract_magic_from_auth_payload Tests
# =============================================================================


def test_extract_magic_from_auth_payload_success() -> None:
    """Test extracting magic from valid AUTH message."""
    # AUTH format: 2-byte length prefix + "%AUTH !be <session>|<hash>|<ts> <magic>"
    auth_body = "%AUTH !be abc123|def456|789 test_magic_key_12345"
    # Add 2-byte length prefix (doesn't matter what value for this test)
    payload = bytes([0x00, 0x30]) + auth_body.encode("utf-8")

    result = extract_magic_from_auth_payload(payload)

    assert result == "test_magic_key_12345"


def test_extract_magic_from_auth_payload_short_payload() -> None:
    """Test returns None for payload too short."""
    payload = bytes([0x00, 0x05, 0x41, 0x42])  # Only 4 bytes

    result = extract_magic_from_auth_payload(payload)

    assert result is None


def test_extract_magic_from_auth_payload_not_auth() -> None:
    """Test returns None when payload is not AUTH message."""
    # A message without AUTH keyword
    body = "HELLO !be abc123|def456|789 test_magic_key_12345"
    payload = bytes([0x00, 0x30]) + body.encode("utf-8")

    result = extract_magic_from_auth_payload(payload)

    assert result is None


def test_extract_magic_from_auth_payload_too_few_parts() -> None:
    """Test returns None when AUTH message has too few parts."""
    # Only 2 space-separated parts
    body = "%AUTH something"
    payload = bytes([0x00, 0x10]) + body.encode("utf-8")

    result = extract_magic_from_auth_payload(payload)

    assert result is None


def test_extract_magic_from_auth_payload_two_parts_with_a_long_tail() -> None:
    """A two-part AUTH line yields nothing even when its tail is long enough.

    The sibling ``_too_few_parts`` case above cannot distinguish the
    field-count rule from the magic-length rule: its last token is nine
    characters, so the ``len(magic) < 10`` check rejects it regardless.
    Here the tail is fourteen characters and would be accepted as a
    magic key, so the field-count rule is the only thing refusing it --
    a real AUTH line carries session, hash and timestamp fields before
    the key, and a payload missing them is not one.
    """
    body = "%AUTH abcdefghij1234"
    payload = bytes([0x00, 0x14]) + body.encode("utf-8")

    result = extract_magic_from_auth_payload(payload)

    assert result is None


def test_extract_magic_from_auth_payload_magic_too_short() -> None:
    """Test returns None when magic is too short."""
    # Magic is only 5 chars (needs at least 10)
    body = "%AUTH !be abc123|def456|789 short"
    payload = bytes([0x00, 0x25]) + body.encode("utf-8")

    result = extract_magic_from_auth_payload(payload)

    assert result is None


def test_extract_magic_from_auth_payload_with_auth_variant() -> None:
    """Test extracting magic when AUTH (not %AUTH) is present."""
    # Some messages may have AUTH without % prefix
    body = "AUTH !be abc123|def456|789 another_magic_key"
    payload = bytes([0x00, 0x30]) + body.encode("utf-8")

    result = extract_magic_from_auth_payload(payload)

    assert result == "another_magic_key"
