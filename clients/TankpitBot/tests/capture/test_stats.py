"""Tests for tankpit_bot.capture.stats module."""

from __future__ import annotations

import base64

from tankpit_bot import _test_hooks
from tankpit_bot.capture.stats import build_message_stats
from tankpit_bot.types import CapturedMessage, CaptureSession
from tests.conftest import FakeFileSystem


class TestBuildMessageStats:
    """Tests for build_message_stats function."""

    def test_returns_empty_when_no_static_key_file(self) -> None:
        """Test returns empty stats when static key file doesn't exist."""
        fs = FakeFileSystem()
        _test_hooks.path_exists = fs.path_exists
        _test_hooks.read_text = fs.read_text

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic="testmagic",
            game_log=[],
            tank_names={},
        )

        result = build_message_stats(session)
        assert result["decoded"] == {}
        assert result["unknown"] == {}
        assert result["total_received"] == 0

    def test_skips_messages_with_invalid_signature(self, fake_fs: FakeFileSystem) -> None:
        """Test skips messages that can't be decoded."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        # Create an invalid payload (no dot in first 3 bytes)
        payload = base64.b64encode(b"ABCDEFGH").decode()

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
            magic="testmagic",
            game_log=[],
            tank_names={},
        )

        result = build_message_stats(session)
        # Message should be skipped, total remains 0
        assert result["total_received"] == 0

    def test_tracks_unknown_message_types(self, fake_fs: FakeFileSystem) -> None:
        """Test tracks unknown message types in unknown dict."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        magic = "testmagic123"

        # Build XOR table for encoding
        magic_bytes = magic.encode("utf-8")
        xor_table = bytes(
            ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
        )

        # Create a message with 8 bytes - no known structure for 8 bytes
        decoded_data = bytes([0xFF] * 8)
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
        # Should have tracked as unknown (8-byte messages have no known structure)
        assert result["total_received"] == 1
        assert "len=08" in result["unknown"]

    def test_tracks_decoded_known_structure(self, fake_fs: FakeFileSystem) -> None:
        """Known container structures land in the decoded counter.

        Use a 4-byte player_list_short body (subtype 0x79) -- the
        identifier returns its name + level, exercising the decoded
        branch in build_message_stats.
        """
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        magic = "testmagic123"
        magic_bytes = magic.encode("utf-8")
        xor_table = bytes(
            ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
        )

        # 4-byte body starting with 0x79 -> player_list_short
        decoded_data = bytes([0x79, 0x99, 0x05, 0x07])
        encoded = bytes(decoded_data[i] ^ xor_table[i] for i in range(len(decoded_data)))
        body = bytes([0x2E]) + encoded
        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[
                CapturedMessage(
                    timestamp_ms=100, direction="received", payload=payload, ws_url="wss://test"
                ),
            ],
            magic=magic,
            game_log=[],
            tank_names={},
        )

        result = build_message_stats(session)
        assert result["total_received"] == 1
        assert "len=04 player_list_short" in result["decoded"]

    def test_unknown_samples_limited_to_3(self, fake_fs: FakeFileSystem) -> None:
        """Test unknown samples are limited to 3 per length key."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        magic = "testmagic123"

        magic_bytes = magic.encode("utf-8")
        xor_table = bytes(
            ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
        )

        # Create 5 messages with 8 bytes - no known structure
        messages: list[CapturedMessage] = []
        for i in range(5):
            # 8 bytes with varying first byte for different samples
            decoded_data = bytes([i, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
            encoded = bytes(decoded_data[j] ^ xor_table[j] for j in range(len(decoded_data)))
            body = bytes([0x2E]) + encoded
            header = len(body).to_bytes(2, "little")
            payload = base64.b64encode(header + body).decode()
            messages.append(
                CapturedMessage(
                    timestamp_ms=100 + i,
                    direction="received",
                    payload=payload,
                    ws_url="wss://test",
                )
            )

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
        # Should have 5 unknown messages
        assert result["total_received"] == 5
        # But only 3 samples stored per length key
        assert "len=08" in result["unknown"]
        entry = result["unknown"]["len=08"]
        assert entry["count"] == 5
        # Samples is a list[str] of 3 hex strings
        samples = entry["samples"]
        assert samples[0].startswith("00")  # First sample starts with 0x00
        assert samples[1].startswith("01")  # Second sample starts with 0x01
        assert samples[2].startswith("02")  # Third sample starts with 0x02
