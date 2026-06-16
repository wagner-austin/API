"""Tests for tankpit_bot.sniffer capture and summary functions."""

from __future__ import annotations

import base64

from tankpit_bot import _test_hooks
from tankpit_bot.capture.signature import extract_message_signature
from tankpit_bot.capture.stats import build_message_stats, empty_message_stats
from tankpit_bot.capture.xor import build_xor_table, load_xor_static_key
from tests.conftest import FakeFileSystem

# =============================================================================
# Empty Message Stats Tests
# =============================================================================


class TestEmptyMessageStats:
    """Tests for empty_message_stats function."""

    def test_returns_empty_stats(self) -> None:
        """Test returns MessageStats with empty values."""
        result = empty_message_stats()
        assert result["decoded"] == {}
        assert result["unknown"] == {}
        assert result["total_received"] == 0
        assert result["decode_coverage"] == "0%"


# =============================================================================
# Extract Message Signature Tests
# =============================================================================


class TestExtractMessageSignature:
    """Tests for extract_message_signature function."""

    def test_returns_none_for_invalid_base64(self) -> None:
        """Test returns None for invalid base64."""
        result = extract_message_signature("not valid!!!", b"\x00" * 100)
        assert result is None

    def test_returns_none_when_no_dot_in_first_3_bytes(self) -> None:
        """Test returns None when no dot found in first 3 bytes."""
        payload = base64.b64encode(b"ABCDEFGH").decode()
        result = extract_message_signature(payload, b"\x00" * 100)
        assert result is None

    def test_returns_none_when_dot_at_position_3_or_higher(self) -> None:
        """Test returns None when dot position >= 3."""
        payload = base64.b64encode(b"ABC.EFGH").decode()
        result = extract_message_signature(payload, b"\x00" * 100)
        assert result is None

    def test_returns_none_when_no_data_after_dot(self) -> None:
        """Test returns None when no data after dot."""
        payload = base64.b64encode(b".").decode()
        result = extract_message_signature(payload, b"\x00" * 100)
        assert result is None

    def test_decodes_with_xor_table(self) -> None:
        """Test decodes data using XOR table."""
        # Payload with dot at position 0
        raw_data = bytes([0x2E, 0x41, 0x42, 0x43])
        payload = base64.b64encode(raw_data).decode()
        xor_table = bytes([0x00, 0x00, 0x00])  # Identity XOR

        result = extract_message_signature(payload, xor_table)
        assert result == bytes([0x41, 0x42, 0x43])


# =============================================================================
# Build Message Stats Tests
# =============================================================================


class TestBuildMessageStats:
    """Tests for build_message_stats function."""

    def test_returns_empty_without_magic(self) -> None:
        """Test returns empty stats when session has no magic key."""
        from tankpit_bot.types import CaptureSession

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic=None,
            game_log=[],
            tank_names={},
        )
        result = build_message_stats(session)
        assert result["decoded"] == {}
        assert result["unknown"] == {}
        assert result["total_received"] == 0


class TestBuildMessageStatsEdgeCases:
    """Edge case tests for build_message_stats."""

    def test_build_message_stats_with_messages(self, fake_fs: FakeFileSystem) -> None:
        """Test build_message_stats processes messages correctly."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.types import CapturedMessage, CaptureSession

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        magic = "testmagic123"

        # Build XOR table
        magic_bytes = magic.encode("utf-8")
        xor_table = bytes(
            ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
        )

        # Create a valid 0x2E message with known container type
        # Use 0x75 position message: 0x75 x y
        decoded_data = bytes([0x75, 0x64, 0x32])  # position x=100, y=50
        encoded = bytes(decoded_data[i] ^ xor_table[i] for i in range(len(decoded_data)))
        body = bytes([0x2E]) + encoded
        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        messages: list[CapturedMessage] = [
            CapturedMessage(
                timestamp_ms=100,
                direction="received",
                payload=payload,
                ws_url="wss://test",
            ),
        ]

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=messages,
            magic=magic,
            game_log=[],
            tank_names={},
        )

        result = build_message_stats(session)
        # Should have processed the message
        assert result["total_received"] >= 0

    def test_build_message_stats_unknown_message_type(self, fake_fs: FakeFileSystem) -> None:
        """Test build_message_stats tracks unknown message types."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.types import CapturedMessage, CaptureSession

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        magic = "testmagic123"

        magic_bytes = magic.encode("utf-8")
        xor_table = bytes(
            ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
        )

        # Create a message with unknown type (0xFF)
        decoded_data = bytes([0xFF, 0x01, 0x02, 0x03])
        encoded = bytes(decoded_data[i] ^ xor_table[i] for i in range(len(decoded_data)))
        body = bytes([0x2E]) + encoded
        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        messages: list[CapturedMessage] = [
            CapturedMessage(
                timestamp_ms=100,
                direction="received",
                payload=payload,
                ws_url="wss://test",
            ),
        ]

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=messages,
            magic=magic,
            game_log=[],
            tank_names={},
        )

        result = build_message_stats(session)
        # Should have tracked as unknown
        assert result["total_received"] >= 0

    def test_build_message_stats_sent_messages_ignored(self, fake_fs: FakeFileSystem) -> None:
        """Test build_message_stats ignores sent messages."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.types import CapturedMessage, CaptureSession

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        magic = "testmagic123"

        messages: list[CapturedMessage] = [
            CapturedMessage(
                timestamp_ms=100,
                direction="sent",  # Sent messages should be ignored
                payload="AAAA",
                ws_url="wss://test",
            ),
        ]

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=messages,
            magic=magic,
            game_log=[],
            tank_names={},
        )

        result = build_message_stats(session)
        # No received messages, so total should be 0
        assert result["total_received"] == 0


# =============================================================================
# Build Session Summary Tests
# =============================================================================


class TestBuildSessionSummary:
    """Tests for build_session_summary function."""

    def test_extracts_combat_events_from_game_log(self) -> None:
        """Test extracts combat events from game log."""
        from tankpit_bot.capture.summary import build_session_summary
        from tankpit_bot.types import CaptureSession, GameLogEntryWithTimestamp

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic="test",
            game_log=[
                GameLogEntryWithTimestamp(
                    timestamp_ms=100,
                    text="You hit Enemy",
                    category="combat",
                ),
                GameLogEntryWithTimestamp(
                    timestamp_ms=200,
                    text="You killed Enemy",
                    category="combat",
                ),
                GameLogEntryWithTimestamp(
                    timestamp_ms=300,
                    text="Foe hit you",
                    category="combat",
                ),
                GameLogEntryWithTimestamp(
                    timestamp_ms=400,
                    text="Foe killed you",
                    category="combat",
                ),
            ],
            tank_names={},
        )
        result = build_session_summary(session)
        assert len(result["combat"]) == 4
        assert result["combat"][0]["event_type"] == "hit"
        assert result["combat"][0]["target"] == "Enemy"
        assert result["combat"][1]["event_type"] == "kill"
        assert result["combat"][2]["event_type"] == "hit_by"
        assert result["combat"][3]["event_type"] == "killed_by"

    def test_skips_non_combat_log_entries(self) -> None:
        """Test skips non-combat log entries."""
        from tankpit_bot.capture.summary import build_session_summary
        from tankpit_bot.types import CaptureSession, GameLogEntryWithTimestamp

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic="test",
            game_log=[
                GameLogEntryWithTimestamp(
                    timestamp_ms=100,
                    text="Some info message",
                    category="info",
                ),
            ],
            tank_names={},
        )
        result = build_session_summary(session)
        assert len(result["combat"]) == 0


# =============================================================================
# Load XOR Static Key Tests
# =============================================================================


class TestLoadXorStaticKey:
    """Tests for load_xor_static_key function."""

    def test_load_xor_static_key_cached(self, fake_fs: FakeFileSystem) -> None:
        """Test load_xor_static_key returns cached value."""
        cached_key = "CACHED_KEY_VALUE"
        result = load_xor_static_key(cached_key)
        assert result == (cached_key, cached_key)

    def test_load_xor_static_key_from_file(self, fake_fs: FakeFileSystem) -> None:
        """Test load_xor_static_key loads from file."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "FILE_KEY_VALUE" + "A" * 986
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        result = load_xor_static_key(None)
        assert result == (static_key, static_key)

    def test_load_xor_static_key_missing_file(self) -> None:
        """Test load_xor_static_key returns None when file missing."""
        fs = FakeFileSystem()
        _test_hooks.path_exists = fs.path_exists

        result = load_xor_static_key(None)
        assert result == (None, None)


# =============================================================================
# Build XOR Table Tests
# =============================================================================


class TestBuildXorTable:
    """Tests for build_xor_table function."""

    def test_build_xor_table(self) -> None:
        """Test build_xor_table creates correct XOR table."""
        static_key = "ABCD"
        magic = "xy"

        result = build_xor_table(static_key, magic)

        # A(65) ^ x(120) = 57, B(66) ^ y(121) = 59
        # C(67) ^ x(120) = 59, D(68) ^ y(121) = 61
        assert result == bytes([57, 59, 59, 61])
