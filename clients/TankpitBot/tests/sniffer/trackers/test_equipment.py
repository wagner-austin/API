"""Tests for EquipmentToggleTracker and EquipmentGainTracker classes."""

from __future__ import annotations

import base64

from tankpit_bot.capture import EquipmentGainTracker, EquipmentToggleTracker
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import build_test_xor_table, make_payload


def _make_xor_payload(decoded_data: bytes, xor_table: bytes) -> str:
    """Create XOR-encoded base64 payload for testing."""
    encoded = bytes(decoded_data[i] ^ xor_table[i] for i in range(len(decoded_data)))
    body = bytes([0x2E]) + encoded
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


class TestEquipmentToggleTracker:
    """Tests for EquipmentToggleTracker class."""

    def test_init(self) -> None:
        """Test EquipmentToggleTracker initialization."""
        tracker = EquipmentToggleTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._state == [False, False, False, False, False]
        assert tracker._prev_state is None

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = EquipmentToggleTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_state_property(self) -> None:
        """Test state property returns equipment state dict."""
        tracker = EquipmentToggleTracker()
        state = tracker.state
        assert state == {
            "armor": False,
            "dual": False,
            "missile": False,
            "homing": False,
            "radar": False,
        }

    def test_detect_changes_no_previous(self) -> None:
        """Test _detect_changes returns empty list without previous state."""
        tracker = EquipmentToggleTracker()
        changes = tracker._detect_changes([True, False, True, False, True])
        assert changes == []

    def test_detect_changes_with_changes(self) -> None:
        """Test _detect_changes detects state changes."""
        tracker = EquipmentToggleTracker()
        tracker._prev_state = [False, False, False, False, False]
        changes = tracker._detect_changes([True, False, True, False, False])
        assert "armor=ON" in changes
        assert "missile=ON" in changes

    def test_detect_changes_off_transitions(self) -> None:
        """Test _detect_changes detects OFF transitions."""
        tracker = EquipmentToggleTracker()
        tracker._prev_state = [True, True, False, False, False]
        changes = tracker._detect_changes([False, True, False, False, False])
        assert "armor=OFF" in changes
        assert len(changes) == 1

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = EquipmentToggleTracker()
        payload = make_payload(b"\x2e\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = EquipmentToggleTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


class TestEquipmentToggleTrackerParseMethods:
    """Tests for EquipmentToggleTracker methods."""

    def test_decode_toggle_with_xor_table(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_toggle works with proper setup."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = EquipmentToggleTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Toggle format decoded: 0x74 armor dual missile homing radar
        # armor=ON, dual=OFF, missile=ON, homing=OFF, radar=ON
        decoded_data = bytes([0x74, 0x01, 0x00, 0x01, 0x00, 0x01])
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "EQUIP" in result

    def test_state_property_returns_current_state(self, fake_fs: FakeFileSystem) -> None:
        """Test state property returns current equipment state."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = EquipmentToggleTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        decoded_data = bytes([0x74, 0x01, 0x00, 0x01, 0x00, 0x01])
        payload = _make_xor_payload(decoded_data, xor_table)
        tracker.process_message(payload)

        state = tracker.state
        expected = {
            "armor": True,
            "dual": False,
            "missile": True,
            "homing": False,
            "radar": True,
        }
        assert state == expected

    def test_detect_changes_returns_changes(self, fake_fs: FakeFileSystem) -> None:
        """Test _detect_changes identifies equipment state changes.

        The tracker compares new state with _prev_state, which is set to
        the state from two messages ago. So changes are detected relative
        to that baseline, not the immediately previous message.
        """
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = EquipmentToggleTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # First message: all OFF (sets baseline for next comparison)
        payload1 = _make_xor_payload(bytes([0x74, 0x00, 0x00, 0x00, 0x00, 0x00]), xor_table)
        tracker.process_message(payload1)

        # Second message: dual ON (compared to initial all-OFF state)
        payload2 = _make_xor_payload(bytes([0x74, 0x00, 0x01, 0x00, 0x00, 0x00]), xor_table)
        result = tracker.process_message(payload2)

        assert result, "Expected non-None result from process_message"
        assert "TOGGLE" in result
        assert "dual=ON" in result


class TestEquipmentGainTracker:
    """Tests for EquipmentGainTracker class."""

    def test_init(self) -> None:
        """Test EquipmentGainTracker initialization."""
        tracker = EquipmentGainTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = EquipmentGainTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = EquipmentGainTracker()
        payload = make_payload(b"\x2e\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = EquipmentGainTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


class TestEquipmentGainTrackerProcessMessage:
    """Tests for EquipmentGainTracker.process_message with XOR decoding."""

    def test_process_message_returns_gain_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes equipment gain correctly."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = EquipmentGainTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Equipment gain decoded: 0x67 type zeros... equipment_flags
        # 7 bytes total for the decoded data
        decoded_data = bytes([0x67, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00])
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "EQUIP" in result or "GAIN" in result


class TestEquipmentToggleTrackerEdgeCases:
    """Tests for EquipmentToggleTracker edge cases and uncovered branches."""

    def test_set_magic_returns_early_when_no_static_key(self) -> None:
        """Test set_magic does nothing when static key missing."""
        from tankpit_bot import _test_hooks
        from tests.conftest import FakeFileSystem

        fs = FakeFileSystem()
        _test_hooks.path_exists = fs.path_exists
        _test_hooks.read_text = fs.read_text

        tracker = EquipmentToggleTracker()
        tracker.set_magic("testmagic")
        assert tracker._xor_table is None

    def test_decode_toggle_returns_none_for_short_data(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_toggle returns None for data < 4 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = EquipmentToggleTracker()
        tracker.set_magic("testmagic123")

        # Payload that decodes to < 4 bytes
        payload = base64.b64encode(b"\x01\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None

    def test_decode_toggle_wrong_body_length(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_toggle returns None for body != 7 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = EquipmentToggleTracker()
        tracker.set_magic("testmagic123")

        # Body length 6 (should be 7)
        payload = make_payload(b"\x2e\x01\x02\x03\x04\x05")
        result = tracker.process_message(payload)
        assert result is None

    def test_decode_toggle_wrong_prefix(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_toggle returns None for body not starting with 0x2E."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = EquipmentToggleTracker()
        tracker.set_magic("testmagic123")

        # Body starts with 0x30 instead of 0x2E
        payload = make_payload(b"\x30\x01\x02\x03\x04\x05\x06")
        result = tracker.process_message(payload)
        assert result is None

    def test_decode_toggle_wrong_decoded_type(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_toggle returns None when decoded[0] != 0x74."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = EquipmentToggleTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_test_xor_table(static_key, "testmagic123")

        # Use 0x99 instead of 0x74
        decoded_data = bytes([0x99, 0x01, 0x00, 0x01, 0x00, 0x01])
        payload = _make_xor_payload(decoded_data, xor_table)
        result = tracker.process_message(payload)
        assert result is None


class TestEquipmentGainTrackerEdgeCases:
    """Tests for EquipmentGainTracker edge cases and uncovered branches."""

    def test_set_magic_returns_early_when_no_static_key(self) -> None:
        """Test set_magic does nothing when static key missing."""
        from tankpit_bot import _test_hooks
        from tests.conftest import FakeFileSystem

        fs = FakeFileSystem()
        _test_hooks.path_exists = fs.path_exists
        _test_hooks.read_text = fs.read_text

        tracker = EquipmentGainTracker()
        tracker.set_magic("testmagic")
        assert tracker._xor_table is None

    def test_process_message_returns_none_for_short_data(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for data < 4 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = EquipmentGainTracker()
        tracker.set_magic("testmagic123")

        # Payload that decodes to < 4 bytes
        payload = base64.b64encode(b"\x01\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_body_length(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for body != 8 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = EquipmentGainTracker()
        tracker.set_magic("testmagic123")

        # Body length 7 (should be 8)
        payload = make_payload(b"\x2e\x01\x02\x03\x04\x05\x06")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_prefix(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for body not starting with 0x2E."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = EquipmentGainTracker()
        tracker.set_magic("testmagic123")

        # Body starts with 0x30 instead of 0x2E
        payload = make_payload(b"\x30\x01\x02\x03\x04\x05\x06\x07")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_decoded_type(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None when decoded[0] != 0x67."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = EquipmentGainTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_test_xor_table(static_key, "testmagic123")

        # Use 0x99 instead of 0x67
        decoded_data = bytes([0x99, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00])
        payload = _make_xor_payload(decoded_data, xor_table)
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_no_equipment_gained(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns flags when no equipment matches."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = EquipmentGainTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_test_xor_table(static_key, "testmagic123")

        # Equipment gain with no matching flags (bits outside 0-4)
        # flags5=0x00, flags6=0x00 -> no equipment, but decoded[0]=0x67
        decoded_data = bytes([0x67, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        payload = _make_xor_payload(decoded_data, xor_table)
        result = tracker.process_message(payload)

        if result is None:
            raise AssertionError("Expected non-None result from process_message")
        assert "EQUIP:GAIN" in result
        assert "flags=0,0" in result
