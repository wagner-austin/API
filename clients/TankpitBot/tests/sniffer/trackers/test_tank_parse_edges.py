"""Tests for tank-tracker parse edge cases.

Short payloads, unknown tanks, and the name-extraction branches.
"""

from __future__ import annotations

from tankpit_bot.capture.trackers import TankTracker
from tests.conftest import FakeFileSystem
from tests.wire_builders import frame_payload


class TestTankTrackerParseEdges:
    """Tests for tank-tracker parse edge cases."""

    def test_parse_tank_status_exactly_13_bytes_no_name(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_tank_status with exactly 13 bytes skips name extraction."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()

        info_byte = 0x01 | (0x02 << 4)
        decoded = bytearray([info_byte, 0x64, 0x00])
        decoded.extend([0] * 10)

        assert len(decoded) == 13

        result = tracker._parse_tank_status(decoded)
        assert result, "Expected non-None result"
        assert "id=100" in result
        assert "purple" in result
        assert "corporal" in result
        assert "'" not in result

    def test_parse_tank_info_name_extraction_with_break(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_tank_info extracts name and breaks on non-printable char."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()

        decoded = bytearray([0, 0x64, 0x00])
        decoded.extend([0] * 7)
        decoded.extend(b"ABC\x01DEF")

        result = tracker._parse_tank_info(decoded)
        assert result, "Expected non-None result"
        assert "ABC" in result
        assert "DEF" not in result

    def test_parse_movement_returns_none_for_short_decoded(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_movement returns None when decoded < 5 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        result = tracker._parse_movement(bytearray(b"\x01\x02\x03\x04"))
        assert result is None

    def test_parse_shooting_returns_none_for_short_decoded(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_shooting returns None when decoded < 4 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        result = tracker._parse_shooting(bytearray(b"\x01\x02\x03"))
        assert result is None

    def test_parse_shooting_with_known_tank(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_shooting includes tank name when known."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker.register_name(100, "Shooter")

        decoded = bytearray([2, 0x64, 0x00, 50, 60])
        result = tracker._parse_shooting(decoded)
        assert result, "Expected non-None result"
        assert "Shooter" in result

    def test_parse_shooting_unknown_team(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_shooting handles unknown team index."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()

        decoded = bytearray([10, 0x01, 0x00, 50, 60])
        result = tracker._parse_shooting(decoded)
        assert result, "Expected non-None result"
        assert "team10" in result

    def test_parse_tank_info_returns_none_for_short_decoded(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_tank_info returns None when decoded < 11 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        result = tracker._parse_tank_info(bytearray(bytes(8)))
        assert result is None

    def test_parse_tank_info_returns_none_for_empty_name(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_tank_info returns None when name is empty."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        decoded = bytearray(bytes(12))
        result = tracker._parse_tank_info(decoded)
        assert result is None

    def test_process_message_status_sync_short(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for 0x2E with short decoded."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = TankTracker()
        tracker.set_magic(magic)

        body = b"\x2e\x01\x02\x03"
        payload = frame_payload(body)
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_status_sync_valid(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message calls _parse_status_sync for 0x2E with 8+ decoded bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = TankTracker()
        tracker.set_magic(magic)
        tracker.register_name(100, "StatusTank")

        # Body: 0x2E + 9 bytes (decoded will be 8+ bytes)
        body = b"\x2e\x50\x64\x00\x01\x02\x03\x04\x05"
        payload = frame_payload(body)
        result = tracker.process_message(payload)

        # Should call _parse_status_sync and return a result
        if result is None:
            raise AssertionError("Expected non-None result from process_message")
        assert "STATUS" in result

    def test_process_message_handler_too_short(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None when decoded < handler's min_len."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = TankTracker()
        tracker.set_magic(magic)

        # Message type 0x28 (tank join) requires min_len=4
        # Body: 0x28 + only 2 bytes (decoded will be < 4)
        body = b"\x28\x01\x02"
        payload = frame_payload(body)
        result = tracker.process_message(payload)
        assert result is None

    def test_decode_payload_body_exceeds_xor_table(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_payload handles body longer than xor_table."""
        import base64

        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        # Create a short static key to make xor_table shorter
        static_key = "ABCD"  # Only 4 chars
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "test"

        tracker = TankTracker()
        tracker.set_magic(magic)

        # Create body longer than xor_table (4 bytes)
        # Body = msg_type + data (10 bytes total)
        body = b"\x28" + bytes(10)  # 11 bytes in body
        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        result = tracker._decode_payload(payload)
        # Should still work - bytes beyond xor_table are copied as-is
        if result is None:
            raise AssertionError("Expected non-None result from _decode_payload")
        msg_type, decoded, _raw_body = result
        assert msg_type == 0x28
        assert len(decoded) == 10

    def test_process_message_calls_handler(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message calls handler and returns result (line 131)."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = TankTracker()
        tracker.set_magic(magic)

        # Message type 0x28 (tank join) with min_len=4
        # Body: 0x28 + 5 bytes (decoded will be 4+ bytes, meeting min_len)
        body = b"\x28\x00\x64\x00\xab"  # tank_id=100
        payload = frame_payload(body)
        result = tracker.process_message(payload)

        # Should call _parse_tank_join handler and return result
        if result is None:
            raise AssertionError("Expected non-None result from process_message")
        assert "JOIN" in result
