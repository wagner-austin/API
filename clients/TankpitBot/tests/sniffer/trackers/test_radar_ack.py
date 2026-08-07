"""Tests for RadarAckTracker class."""

from __future__ import annotations

import base64

from tankpit_bot.capture.trackers import RadarAckTracker
from tankpit_bot.protocol.codec import build_xor_table
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import assert_set_magic_requires_static_key
from tests.wire_builders import encode_wire_frame, frame_payload


class TestRadarAckTracker:
    """Tests for RadarAckTracker class."""

    def test_init(self) -> None:
        """Test RadarAckTracker initialization."""
        tracker = RadarAckTracker()
        assert tracker._xor_table is None
        assert tracker._count == 0

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = RadarAckTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_count_property(self) -> None:
        """Test count property returns acknowledgement count."""
        tracker = RadarAckTracker()
        assert tracker.count == 0
        tracker._count = 5
        assert tracker.count == 5

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = RadarAckTracker()
        payload = frame_payload(b"\x2e\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = RadarAckTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


class TestRadarAckTrackerProcessMessage:
    """Tests for RadarAckTracker.process_message with XOR decoding."""

    def test_process_message_returns_ack_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes radar ack correctly."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = RadarAckTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        # Radar ack decoded: 0x46 byte1 byte2
        decoded_data = bytes([0x46, 0x01, 0x00])
        payload = encode_wire_frame(0x2E, decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "RADAR" in result

    def test_radar_count_tracks(self, fake_fs: FakeFileSystem) -> None:
        """Test radar ack count is tracked."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = RadarAckTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        payload = encode_wire_frame(0x2E, bytes([0x46, 0x01, 0x00]), xor_table)
        tracker.process_message(payload)
        tracker.process_message(payload)

        assert tracker.count == 2


class TestRadarAckTrackerEdgeCases:
    """Tests for RadarAckTracker edge cases and uncovered branches."""

    def test_set_magic_raises_when_no_static_key(self) -> None:
        """A missing static key is fatal, not a silent no-op."""
        assert_set_magic_requires_static_key(RadarAckTracker())

    def test_process_message_returns_none_for_short_data(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for data < 4 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = RadarAckTracker()
        tracker.set_magic("testmagic123")

        # Payload that decodes to < 4 bytes
        payload = base64.b64encode(b"\x01\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_body_length(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for body != 4 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = RadarAckTracker()
        tracker.set_magic("testmagic123")

        # Body length 5 (should be 4)
        payload = frame_payload(b"\x2e\x01\x02\x03\x04")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_prefix(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for body not starting with 0x2E."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = RadarAckTracker()
        tracker.set_magic("testmagic123")

        # Body starts with 0x30 instead of 0x2E
        payload = frame_payload(b"\x30\x01\x02\x03")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_decoded_type(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None when decoded[0] != 0x46."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = RadarAckTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_xor_table(static_key, "testmagic123")

        # Use 0x99 instead of 0x46
        decoded_data = bytes([0x99, 0x01, 0x00])
        payload = encode_wire_frame(0x2E, decoded_data, xor_table)
        result = tracker.process_message(payload)
        assert result is None
