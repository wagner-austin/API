"""Tests for tankpit_bot.capture.xor module."""

from __future__ import annotations

import pytest

from tankpit_bot.capture.xor import (
    XorStaticKeyUnavailableError,
    build_session_xor_table,
    decode_base64_safe,
    is_valid_base64,
    reset_static_key_cache,
    xor_decode_body,
)
from tankpit_bot.protocol.codec import build_xor_table
from tests.conftest import FakeFileSystem


class TestIsValidBase64:
    """Tests for is_valid_base64 function."""

    def test_valid_base64(self) -> None:
        """Test valid base64 strings return True."""
        assert is_valid_base64("SGVsbG8=") is True
        assert is_valid_base64("QUJDRA==") is True
        assert is_valid_base64("dGVzdA==") is True

    def test_valid_base64_no_padding(self) -> None:
        """Test valid base64 without padding returns True."""
        # Length must be multiple of 4
        assert is_valid_base64("QUJD") is True
        assert is_valid_base64("YWJjZA==") is True

    def test_empty_string_returns_false(self) -> None:
        """Test empty string returns False."""
        assert is_valid_base64("") is False

    def test_invalid_chars_returns_false(self) -> None:
        """Test invalid characters return False."""
        assert is_valid_base64("ABC!") is False
        assert is_valid_base64("AB@D") is False
        assert is_valid_base64("test$") is False

    def test_invalid_length_returns_false(self) -> None:
        """Test strings with length not multiple of 4 return False."""
        assert is_valid_base64("abc") is False
        assert is_valid_base64("abcde") is False
        assert is_valid_base64("ab") is False


class TestDecodeBase64Safe:
    """Tests for decode_base64_safe function."""

    def test_valid_base64_decodes(self) -> None:
        """Test valid base64 decodes correctly."""
        result = decode_base64_safe("SGVsbG8=")
        assert result == b"Hello"

    def test_valid_base64_no_padding(self) -> None:
        """Test valid base64 without padding decodes correctly."""
        result = decode_base64_safe("QUJD")
        assert result == b"ABC"

    def test_invalid_base64_returns_none(self) -> None:
        """Test invalid base64 returns None."""
        result = decode_base64_safe("abc")
        assert result is None

    def test_invalid_chars_returns_none(self) -> None:
        """Test invalid characters return None."""
        result = decode_base64_safe("ABC!")
        assert result is None

    def test_empty_returns_none(self) -> None:
        """Test empty string returns None."""
        result = decode_base64_safe("")
        assert result is None


class TestXorDecodeBody:
    """Tests for xor_decode_body function."""

    def test_basic_decode(self) -> None:
        """Test basic XOR decoding without offset."""
        body = bytes([0x41, 0x42, 0x43, 0x44])  # ABCD
        xor_table = bytes([0x10, 0x20, 0x30, 0x40])
        result = xor_decode_body(body, xor_table)
        assert result == bytes([0x51, 0x62, 0x73, 0x04])

    def test_decode_with_offset(self) -> None:
        """Test XOR decoding with offset skips bytes."""
        body = bytes([0xFF, 0xFF, 0x41, 0x42, 0x43])  # Skip first 2 bytes
        xor_table = bytes([0x10, 0x20, 0x30, 0x40, 0x50])
        result = xor_decode_body(body, xor_table, offset=2)
        # Decodes bytes 2,3,4 using table positions 0,1,2
        assert result == bytes([0x51, 0x62, 0x73])

    def test_decode_empty_body(self) -> None:
        """Test decoding empty body returns empty."""
        body = b""
        xor_table = bytes([0x10, 0x20])
        result = xor_decode_body(body, xor_table)
        assert result == b""

    def test_body_exactly_filling_the_table_decodes(self) -> None:
        """A span the table's exact length uses every key byte once."""
        xor_table = bytes([0x10, 0x20, 0x30])
        assert xor_decode_body(bytes([0x41, 0x42, 0x43]), xor_table) == bytes([0x51, 0x62, 0x73])

    def test_body_past_the_table_wraps_the_key(self) -> None:
        """The cipher wraps: ``body[i] ^ table[i % len]``, the JS law.

        The real client's inbound decode is ``l[ja] ^= B[ja % pa]``
        ([[xor-cipher]], tpclient.js case 46) — the table length was
        never a frame bound. The guard that treated it as one crashed
        artax live on 2026-08-26 when a busy practice room grew a 0x5A
        map frame to 1051 ciphered bytes.
        """
        body = bytes([0x41, 0x42, 0x43, 0x44])
        xor_table = bytes([0x10, 0x20, 0x30])
        assert xor_decode_body(body, xor_table) == bytes([0x51, 0x62, 0x73, 0x54])

    def test_offset_span_wraps_from_table_start(self) -> None:
        """The wrap indexes the CIPHERED span, not the whole body."""
        body = bytes([0x2E, 0x41, 0x42, 0x43, 0x44])
        xor_table = bytes([0x10, 0x20, 0x30])
        assert xor_decode_body(body, xor_table, offset=1) == bytes([0x51, 0x62, 0x73, 0x54])

    def test_decode_body_shorter_than_table(self) -> None:
        """Test decoding body shorter than XOR table."""
        body = bytes([0x41, 0x42])
        xor_table = bytes([0x10, 0x20, 0x30, 0x40])
        result = xor_decode_body(body, xor_table)
        assert result == bytes([0x51, 0x62])


class TestBuildSessionXorTable:
    """Tests for the per-session table builder.

    The table is a VALUE the caller owns; it replaced a module global
    that a second session would silently overwrite
    ([[session-state-deglobalisation]] step 1).
    """

    def test_returns_the_table_for_the_given_magic(self, fake_fs: FakeFileSystem) -> None:
        """The result equals the static key combined with that magic."""
        from tankpit_bot.protocol.codec import static_key_file_path

        static_key = "ABCDEF"
        fake_fs.write_text(static_key_file_path(), static_key)
        reset_static_key_cache()

        assert build_session_xor_table("testmagic") == build_xor_table(static_key, "testmagic")

    def test_two_magics_yield_two_independent_tables(self, fake_fs: FakeFileSystem) -> None:
        """A second session's table does not disturb the first's.

        This is the property the module global could not provide: the
        second build overwrote the first, so the first session's frames
        decoded against the wrong key.
        """
        from tankpit_bot.protocol.codec import static_key_file_path

        fake_fs.write_text(static_key_file_path(), "ABCDEF")
        reset_static_key_cache()

        first = build_session_xor_table("alpha1")
        second = build_session_xor_table("bravo2")

        assert first != second
        assert first == build_session_xor_table("alpha1")

    def test_raises_when_the_static_key_is_missing(self, fake_fs: FakeFileSystem) -> None:
        """A missing key raises rather than yielding a silent identity decode."""
        from tankpit_bot.protocol.codec import static_key_file_path

        fake_fs.remove(static_key_file_path())
        reset_static_key_cache()

        with pytest.raises(XorStaticKeyUnavailableError, match=r"xor_static_key\.txt missing"):
            build_session_xor_table("testmagic")

    def test_the_static_key_is_read_once_and_cached(self, fake_fs: FakeFileSystem) -> None:
        """Deleting the key file after a build still serves later builds.

        The KEY is process-wide (the same key builds every session's
        table); only the TABLE is session state.
        """
        from tankpit_bot.protocol.codec import static_key_file_path

        fake_fs.write_text(static_key_file_path(), "ABCDEF")
        reset_static_key_cache()
        first = build_session_xor_table("testmagic")

        fake_fs.remove(static_key_file_path())

        assert build_session_xor_table("testmagic") == first

    def test_resetting_the_cache_re_reads_the_key(self, fake_fs: FakeFileSystem) -> None:
        """After a reset the next build sees the file's current contents."""
        from tankpit_bot.protocol.codec import static_key_file_path

        fake_fs.write_text(static_key_file_path(), "ABCDEF")
        reset_static_key_cache()
        first = build_session_xor_table("testmagic")

        fake_fs.write_text(static_key_file_path(), "UVWXYZ")
        reset_static_key_cache()

        assert build_session_xor_table("testmagic") == build_xor_table("UVWXYZ", "testmagic")
        assert build_session_xor_table("testmagic") != first
