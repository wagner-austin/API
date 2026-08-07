"""Tests for RadarTracker class."""

from __future__ import annotations

import base64

from tankpit_bot.capture.trackers import RadarTracker
from tankpit_bot.capture.xor import xor_decode_body
from tankpit_bot.protocol.codec import build_xor_table
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import assert_set_magic_requires_static_key
from tests.wire_builders import frame_payload


class TestRadarTracker:
    """Tests for RadarTracker class."""

    def test_init(self) -> None:
        """Test RadarTracker initialization."""
        tracker = RadarTracker()
        assert tracker._xor_table is None

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = RadarTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_classify_entity_tank(self) -> None:
        """Test _classify_entity identifies tanks (0xFFFF)."""
        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(10, 20, 0xFFFF)
        assert category == "tanks"
        assert formatted == "(10,20)"

    def test_classify_entity_equipment(self) -> None:
        """Test _classify_entity identifies equipment (>= 0x8000)."""
        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(15, 25, 0x8005)
        assert category == "equip"
        assert "(15,25)" in formatted

    def test_classify_entity_fuel(self) -> None:
        """Test _classify_entity identifies fuel (< 0x8000)."""
        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(5, 10, 500)
        assert category == "fuel"
        assert formatted == "(5,10)=500"

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = RadarTracker()
        payload = frame_payload(b"\x2e\x70\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = RadarTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


class TestRadarTrackerProcessMessage:
    """Tests for RadarTracker.process_message with XOR decoding."""

    def test_classify_entity_fuel(self) -> None:
        """Test _classify_entity for fuel containers."""
        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(50, 60, 100)
        assert category == "fuel"
        assert formatted == "(50,60)=100"

    def test_classify_entity_tank(self) -> None:
        """Test _classify_entity for tanks (0xFFFF)."""
        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(70, 80, 0xFFFF)
        assert category == "tanks"
        assert formatted == "(70,80)"

    def test_classify_entity_equipment(self) -> None:
        """Test _classify_entity for equipment (negative values)."""
        tracker = RadarTracker()
        # Values >= 0x8000 are treated as negative (equipment)
        category, _formatted = tracker._classify_entity(30, 40, 0x8001)
        assert category == "equip"


class TestRadarTrackerEdgeCases:
    """Tests for RadarTracker edge cases and uncovered branches."""

    def test_set_magic_raises_when_no_static_key(self) -> None:
        """A missing static key is fatal, not a silent no-op."""
        assert_set_magic_requires_static_key(RadarTracker())

    def test_decode_radar_returns_none_for_short_data(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_radar returns None for data < 4 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = RadarTracker()
        tracker.set_magic("testmagic")

        payload = base64.b64encode(b"\x01\x00\x2e").decode()
        result = tracker._decode_radar(payload)
        assert result is None

    def test_decode_radar_returns_none_for_wrong_prefix(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_radar returns None when body doesn't match radar format."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = RadarTracker()
        tracker.set_magic("testmagic")

        body = b"\x2e\x00\x00\x00\x00"
        payload = frame_payload(body)
        result = tracker._decode_radar(payload)
        assert result is None

    def test_process_message_returns_empty_radar(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns no entities message when count is 0."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        # RadarTracker expects: body[1]=0x70 AND decoded[0]=0x4F after XOR
        # This requires table[0] = 0x70 XOR 0x4F = 0x3F = 63
        # With static_key[0]='A'(65), we need magic[0] = 65 XOR 63 = 126 = '~'
        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "~estmagic123"

        tracker = RadarTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        # Build radar body: 0x2E + 0x70 + XOR-encoded(count, padding, entities...)
        rest_decoded = bytes([0x00, 0x00])  # count=0, padding
        rest_encoded = xor_decode_body(rest_decoded, xor_table[1:])
        body = bytes([0x2E, 0x70]) + rest_encoded

        payload = frame_payload(body)
        result = tracker.process_message(payload)
        assert result == "[RADAR] No entities found"

    def test_process_message_with_entities(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message formats entities correctly."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "~estmagic123"

        tracker = RadarTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        # Radar with 2 entities:
        # Entity 1: x=10, y=20, value=100 (fuel)
        # Entity 2: x=30, y=40, value=0xFFFF (tank)
        rest_decoded = bytes([0x02, 0x00, 10, 20, 0x64, 0x00, 30, 40, 0xFF, 0xFF])
        rest_encoded = xor_decode_body(rest_decoded, xor_table[1:])
        body = bytes([0x2E, 0x70]) + rest_encoded

        payload = frame_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result"
        assert "RADAR" in result
        assert "2 found" in result

    def test_classify_entity_equip(self, fake_fs: FakeFileSystem) -> None:
        """Test _classify_entity returns equip for values >= 0x8000."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(5, 10, 0x8001)
        assert category == "equip"
        assert "(5,10)" in formatted

    def test_decode_radar_returns_none_wrong_decoded_type(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_radar returns None when decoded[0] != 0x4F."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        # Use a magic that when XOR'd with the static key produces decoded[0] != 0x4F
        # body[1] = 0x70 is required, and decoded[0] = body[1] ^ xor_table[0]
        # We want decoded[0] != 0x4F, so xor_table[0] != 0x70 ^ 0x4F = 0x3F
        # With static_key[0]='A'(65), magic[0] should not be 65 ^ 63 = 126 = '~'
        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"  # xor_table[0] = ord('A') ^ ord('t') = 65 ^ 116 = 53

        tracker = RadarTracker()
        tracker.set_magic(magic)

        # Build body with 0x2E + 0x70 format but XOR table yields decoded[0] != 0x4F
        # decoded[0] = 0x70 ^ xor_table[0] = 0x70 ^ 53 = 112 ^ 53 = 69 != 0x4F (79)
        rest = bytes([0x00, 0x00, 0x00])  # some padding
        body = bytes([0x2E, 0x70]) + rest

        payload = frame_payload(body)
        result = tracker._decode_radar(payload)
        assert result is None
