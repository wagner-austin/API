"""Tests for ItemPickupTracker class."""

from __future__ import annotations

import base64

from tankpit_bot.capture.trackers import ItemPickupTracker
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import build_test_xor_table, make_payload


def _xor_encode_bytes(data: bytes, xor_table: bytes) -> bytes:
    """XOR encode bytes with table."""
    result = bytearray(len(data))
    for i in range(len(data)):
        if i < len(xor_table):
            result[i] = data[i] ^ xor_table[i]
        else:
            result[i] = data[i]
    return bytes(result)


def _make_xor_payload(decoded_data: bytes, xor_table: bytes) -> str:
    """Create XOR-encoded base64 payload for testing."""
    encoded = bytes(decoded_data[i] ^ xor_table[i] for i in range(len(decoded_data)))
    body = bytes([0x2E]) + encoded
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


class TestItemPickupTracker:
    """Tests for ItemPickupTracker class."""

    def test_init(self) -> None:
        """Test ItemPickupTracker initialization."""
        tracker = ItemPickupTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._total_armor == 0
        assert tracker._total_missile == 0
        assert tracker._total_homing == 0

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = ItemPickupTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = ItemPickupTracker()
        payload = make_payload(b"\x2e\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = ItemPickupTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_short_data(self) -> None:
        """Test process_message returns None for short data."""
        tracker = ItemPickupTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x02\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_non_0x2e(self) -> None:
        """Test process_message returns None for non-0x2E messages."""
        tracker = ItemPickupTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = make_payload(b"\x30\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None


class TestItemPickupTrackerProcessMessage:
    """Tests for ItemPickupTracker.process_message with XOR decoding."""

    def test_process_message_returns_pickup_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes item pickup correctly."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ItemPickupTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Pickup format: body = [0x2E, XOR(0x67, 0x01, armor, dual, missile, homing, radar)]
        # Body must be 8 bytes. Pick up 1 armor, 0 dual, 2 missiles, 1 homing, 0 radar
        decoded_data = bytes([0x67, 0x01, 0x01, 0x00, 0x02, 0x01, 0x00])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data  # 8 bytes

        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "PICKUP" in result
        assert "armor" in result
        assert "missile" in result

    def test_process_message_returns_none_for_all_zeros(self, fake_fs: FakeFileSystem) -> None:
        """Test returns None when all quantities are zero."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ItemPickupTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # All zeros - still 8 byte body with 0x2E prefix
        decoded_data = bytes([0x67, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result is None


class TestItemPickupTrackerEdgeCases:
    """Tests for ItemPickupTracker edge cases and uncovered branches."""

    def test_set_magic_returns_early_when_no_static_key(self) -> None:
        """Test set_magic does nothing when static key missing."""
        from tankpit_bot import _test_hooks
        from tests.conftest import FakeFileSystem

        fs = FakeFileSystem()
        _test_hooks.path_exists = fs.path_exists
        _test_hooks.read_text = fs.read_text

        tracker = ItemPickupTracker()
        tracker.set_magic("testmagic")
        assert tracker._xor_table is None

    def test_decode_pickup_wrong_decoded_prefix(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_pickup returns None when decoded[0] != 0x67."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = ItemPickupTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_test_xor_table(static_key, "testmagic123")

        # Use 0x99 instead of 0x67
        decoded_data = bytes([0x99, 0x01, 0x01, 0x00, 0x02, 0x01, 0x00])
        payload = _make_xor_payload(decoded_data, xor_table)
        result = tracker.process_message(payload)
        assert result is None

    def test_decode_pickup_wrong_second_byte(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_pickup returns None when decoded[1] != 0x01."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = ItemPickupTracker()
        tracker.set_magic("testmagic123")

        xor_table = build_test_xor_table(static_key, "testmagic123")

        # decoded[0]=0x67 but decoded[1]=0x02 instead of 0x01
        decoded_data = bytes([0x67, 0x02, 0x01, 0x00, 0x02, 0x01, 0x00])
        payload = _make_xor_payload(decoded_data, xor_table)
        result = tracker.process_message(payload)
        assert result is None
