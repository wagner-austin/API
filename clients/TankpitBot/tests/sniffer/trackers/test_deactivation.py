"""Tests for DeactivationTracker class."""

from __future__ import annotations

import base64

from tankpit_bot import _test_hooks
from tankpit_bot.capture import DeactivationTracker
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


def _make_tracker_payload(body: bytes) -> str:
    """Wrap body in length header and base64 encode."""
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


class TestDeactivationTracker:
    """Tests for DeactivationTracker class."""

    def test_init(self) -> None:
        """Test DeactivationTracker initialization."""
        tracker = DeactivationTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._my_tank_id is None
        assert tracker._kills == 0
        assert tracker._deaths == 0

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_set_my_tank_id(self) -> None:
        """Test set_my_tank_id stores tank ID."""
        tracker = DeactivationTracker()
        tracker.set_my_tank_id(123)
        assert tracker._my_tank_id == 123

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = DeactivationTracker()
        payload = make_payload(b"\x2e\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_short_data(self) -> None:
        """Test process_message returns None for short data."""
        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x02\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_non_0x2e(self) -> None:
        """Test process_message returns None for non-0x2E messages."""
        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = make_payload(b"\x30\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_wrong_length(self) -> None:
        """Test process_message returns None for wrong length messages."""
        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        # 7 bytes instead of 8
        payload = make_payload(b"\x2e\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_kills_property(self) -> None:
        """Test kills property returns kill count."""
        tracker = DeactivationTracker()
        assert tracker.kills == 0

    def test_deaths_property(self) -> None:
        """Test deaths property returns death count."""
        tracker = DeactivationTracker()
        assert tracker.deaths == 0


class TestDeactivationTrackerProcessMessage:
    """Tests for DeactivationTracker.process_message with XOR decoding."""

    def test_process_message_returns_kill_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes kill events correctly."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        # Set up static key
        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = DeactivationTracker()
        tracker.set_magic(magic)

        # Build XOR table
        xor_table = build_test_xor_table(static_key, magic)

        # Deactivation format: body = [0x2E, XOR(0x41, victim_lo, victim_hi,
        #                                        killer_lo, killer_hi, extra, extra)]
        # Body must be 8 bytes. Victim ID = 100 (0x0064), Killer ID = 200 (0x00C8)
        decoded_data = bytes([0x41, 0x64, 0x00, 0xC8, 0x00, 0x00, 0x00])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data  # 8 bytes total

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "KILL" in result
        assert "100" in result or "Tank" in result

    def test_process_message_returns_death_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message detects own death."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = DeactivationTracker()
        tracker.set_magic(magic)
        tracker.set_my_tank_id(100)  # Set our tank ID

        xor_table = build_test_xor_table(static_key, magic)

        # Victim = 100 (our tank), Killer = 200
        decoded_data = bytes([0x41, 0x64, 0x00, 0xC8, 0x00, 0x00, 0x00])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "DEATH" in result

    def test_process_message_returns_none_for_wrong_signature(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """Test returns None when decoded signature is not 0x41."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = DeactivationTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Use wrong signature 0x99 instead of 0x41
        decoded_data = bytes([0x99, 0x64, 0x00, 0xC8, 0x00, 0x00, 0x00])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result is None


class TestDeactivationTrackerEdgeCases:
    """Edge case tests for DeactivationTracker."""

    def test_set_magic_returns_early_when_no_static_key(self) -> None:
        """Test set_magic returns early when static key file is missing."""
        fs = FakeFileSystem()
        _test_hooks.path_exists = fs.path_exists
        _test_hooks.read_text = fs.read_text

        tracker = DeactivationTracker()
        tracker.set_magic("testmagic")
        # Should not have set xor_table since static key is missing
        assert tracker._xor_table is None
