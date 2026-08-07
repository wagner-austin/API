"""Tests for TankExitTracker class."""

from __future__ import annotations

import base64

from tankpit_bot.capture.trackers import TankExitTracker
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import (
    assert_set_magic_requires_static_key,
    build_test_xor_table,
    make_payload,
)


def _make_xor_payload(decoded_data: bytes, xor_table: bytes) -> str:
    """Create XOR-encoded base64 payload for testing."""
    encoded = bytes(decoded_data[i] ^ xor_table[i] for i in range(len(decoded_data)))
    body = bytes([0x2E]) + encoded
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


class TestTankExitTracker:
    """Tests for TankExitTracker class."""

    def test_init(self) -> None:
        """Test TankExitTracker initialization."""
        tracker = TankExitTracker()
        assert tracker._xor_table is None
        assert tracker._exited == set()

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = TankExitTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_exited_tanks_property(self) -> None:
        """Test exited_tanks property returns copy of set."""
        tracker = TankExitTracker()
        tracker._exited = {1, 2, 3}
        result = tracker.exited_tanks
        assert result == {1, 2, 3}
        # Verify it's a copy
        result.add(4)
        assert 4 not in tracker._exited

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = TankExitTracker()
        payload = make_payload(b"\x2e\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = TankExitTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


class TestTankExitTrackerProcessMessage:
    """Tests for TankExitTracker.process_message with XOR decoding."""

    def test_process_message_returns_exit_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes tank exit correctly."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = TankExitTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Tank exit format decoded: 0x58 tank_id_lo tank_id_hi
        decoded_data = bytes([0x58, 0x64, 0x00])  # tank_id=100
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "EXIT" in result
        assert "100" in result

    def test_exited_tanks_property_tracks(self, fake_fs: FakeFileSystem) -> None:
        """Test exited_tanks property tracks exited tanks."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = TankExitTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        payload = _make_xor_payload(bytes([0x58, 0x64, 0x00]), xor_table)
        tracker.process_message(payload)

        assert 100 in tracker.exited_tanks


class TestTankExitTrackerEdgeCases:
    """Tests for TankExitTracker edge cases and uncovered branches."""

    def test_set_magic_raises_when_no_static_key(self) -> None:
        """A missing static key is fatal, not a silent no-op."""
        assert_set_magic_requires_static_key(TankExitTracker())

    def test_process_message_returns_none_for_short_data(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for data < 4 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankExitTracker()
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

        tracker = TankExitTracker()
        tracker.set_magic("testmagic123")

        # Body length 5 (should be 4)
        payload = make_payload(b"\x2e\x01\x02\x03\x04")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_prefix(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for body not starting with 0x2E."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankExitTracker()
        tracker.set_magic("testmagic123")

        # Body starts with 0x30 instead of 0x2E
        payload = make_payload(b"\x30\x01\x02\x03")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_wrong_decoded_type(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None when decoded[0] != 0x58."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankExitTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_test_xor_table(static_key, "testmagic123")

        # Use 0x99 instead of 0x58
        decoded_data = bytes([0x99, 0x64, 0x00])
        payload = _make_xor_payload(decoded_data, xor_table)
        result = tracker.process_message(payload)
        assert result is None
