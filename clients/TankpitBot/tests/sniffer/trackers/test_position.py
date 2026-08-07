"""Tests for PositionTracker class."""

from __future__ import annotations

import base64

from tankpit_bot.capture.trackers import PositionTracker
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import assert_set_magic_requires_static_key, make_payload


def test_position_tracker_set_magic_builds_xor_table() -> None:
    """Test PositionTracker.set_magic builds XOR table from static key."""
    tracker = PositionTracker()
    assert tracker._xor_table is None

    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After set_magic, _xor_table must be populated with 1000 bytes
    xor_table = tracker._xor_table
    if xor_table is None:
        raise AssertionError("_xor_table was not populated after set_magic")
    assert len(xor_table) == 1000


def test_position_tracker_decode_position_from_0x75() -> None:
    """Test PositionTracker.decode_position extracts x,y from movement response."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After MOVE TO (93, 113): 0x75 shows FROM position (93, 118)
    body = bytes.fromhex("2e757d7e584a0932765910304b1f690d473d20")
    pos = tracker.decode_position(body)
    assert pos == (93, 118)


def test_position_tracker_decode_position_after_move() -> None:
    """Test position decoding shows previous position after movement."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After MOVE (93,113)->(93,118): 0x75 shows (93, 113) = FROM position
    body = bytes.fromhex("2e757d7e584d0132765910304b1f74105a203d")
    pos = tracker.decode_position(body)
    assert pos == (93, 113)


def test_position_tracker_decode_position_x_changes() -> None:
    """Test position decoding when x coordinate changes."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After MOVE (90,118)->(93,118): 0x75 shows (90, 118) = FROM position
    body = bytes.fromhex("2e757d7e5f4a0d32765910304b1f62064c")
    pos = tracker.decode_position(body)
    assert pos == (90, 118)


def test_position_tracker_decode_position_diagonal() -> None:
    """Test position decoding after diagonal movement."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After MOVE (91,115)->(93,118): 0x75 shows (91, 115) = FROM position
    body = bytes.fromhex("2e757d7e5e4f0f32765910304b1f74105a362b")
    pos = tracker.decode_position(body)
    assert pos == (91, 115)


def test_position_tracker_decode_position_wrong_type_returns_none() -> None:
    """Test decode_position returns None for non-0x75 messages."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # 0x1c message (fuel), not 0x75
    body = bytes.fromhex("2e1c4240073f033137191232741b")
    pos = tracker.decode_position(body)
    assert pos is None


def test_position_tracker_decode_position_no_magic_returns_none() -> None:
    """Test decode_position returns None without magic key."""
    tracker = PositionTracker()
    # No magic set

    body = bytes.fromhex("2e757d7e584a0932765910304b1f690d473d20")
    pos = tracker.decode_position(body)
    assert pos is None


def test_position_tracker_process_message_returns_status() -> None:
    """Test process_message returns formatted position status."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # Build payload with 2-byte length header
    body = bytes.fromhex("2e757d7e584a0932765910304b1f690d473d20")
    length = len(body)
    payload_bytes = length.to_bytes(2, "little") + body
    payload = base64.b64encode(payload_bytes).decode()

    result = tracker.process_message(payload)
    if result is None:
        raise AssertionError("process_message returned None for valid payload")
    assert "[POS:FROM]" in result
    assert "(93, 118)" in result


def test_position_tracker_update_from_move() -> None:
    """Test update_from_move sets current position."""
    tracker = PositionTracker()
    assert tracker.current_position is None

    tracker.update_from_move(93, 118)
    assert tracker.current_position == (93, 118)

    tracker.update_from_move(90, 115)
    assert tracker.current_position == (90, 115)


def test_position_tracker_is_blocked_response() -> None:
    """Test is_blocked_response detects 5-byte blocking messages."""
    tracker = PositionTracker()

    # Blocked movement response (5 bytes)
    blocked = bytes.fromhex("2e6347320d")
    assert tracker.is_blocked_response(blocked) is True

    # Normal movement response (not blocked)
    normal = bytes.fromhex("2e757d7e584a0932765910304b1f690d473d20")
    assert tracker.is_blocked_response(normal) is False

    # Too short
    short = bytes.fromhex("2e63")
    assert tracker.is_blocked_response(short) is False


def test_position_tracker_process_message_blocked() -> None:
    """Test process_message returns BLOCKED status for 5-byte messages."""
    tracker = PositionTracker()
    tracker.set_magic("hwvoiew1x26uiv6zlvas")

    # Build payload with 2-byte length header
    body = bytes.fromhex("2e6347320d")
    length = len(body)
    payload_bytes = length.to_bytes(2, "little") + body
    payload = base64.b64encode(payload_bytes).decode()

    result = tracker.process_message(payload)
    assert result == "[POS:BLOCKED]"


def test_position_tracker_variable_subtype() -> None:
    """Test position decoding works with different subtypes per session."""
    tracker = PositionTracker()
    tracker.set_magic("hwvoiew1x26uiv6zlvas")

    # Session with 0x76 subtype (different from 0x75)
    body = bytes.fromhex("2e767a306a4e1c3d704c576d084575004f3b3f6e")
    pos = tracker.decode_position(body)
    # Should decode position regardless of subtype
    assert pos == (102, 125)
    # Verify subtype was tracked
    assert tracker._move_subtype == 0x76


class TestPositionTrackerEdgeCases:
    """Tests for PositionTracker edge cases and uncovered branches."""

    def test_set_magic_raises_when_no_static_key(self) -> None:
        """A missing static key is fatal, not a silent no-op."""
        assert_set_magic_requires_static_key(PositionTracker())

    def test_decode_position_returns_none_for_short_body(self, fake_fs: FakeFileSystem) -> None:
        """Test decode_position returns None when body < 6 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = PositionTracker()
        tracker.set_magic("testmagic")

        # Body too short (< 6 bytes)
        body = bytes.fromhex("2e7501")
        pos = tracker.decode_position(body)
        assert pos is None

    def test_decode_position_with_minimum_length(self, fake_fs: FakeFileSystem) -> None:
        """Test decode_position with exactly 17 bytes (minimum)."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = PositionTracker()
        tracker.set_magic("testmagic")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("xor_table should be set")

        # Build a 17-byte body (minimum valid length for position decoding)
        # Position is at bytes 4-5, XOR'd with xor_table[3-4]
        encoded_x = 10 ^ xor_table[3]
        encoded_y = 20 ^ xor_table[4]
        # [0x2E, sig, byte2, byte3, x, y, ...padding to 17 bytes...]
        body = bytes([0x2E, 0x75, 0x00, 0x00, encoded_x, encoded_y]) + bytes(11)

        pos = tracker.decode_position(body)
        assert pos == (10, 20)

    def test_process_message_returns_none_for_short_payload(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for payloads < 4 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = PositionTracker()
        tracker.set_magic("testmagic")

        # Payload that decodes to < 4 bytes
        payload = base64.b64encode(b"\x01\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = PositionTracker()
        tracker.set_magic("testmagic")

        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_non_0x2e(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for non-0x2E messages."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = PositionTracker()
        tracker.set_magic("testmagic")

        # Not starting with 0x2E
        payload = make_payload(b"\x30\x75\x01\x02\x03\x04")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_non_position_type_returns_none(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for non-position message types."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = PositionTracker()
        tracker.set_magic("testmagic")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("xor_table should be set")

        # Build a non-position message (sig != 0x75/0x76)
        # Use sig=0x41 (random)
        encoded_sig = 0x41 ^ xor_table[0]
        body = bytes([0x2E, encoded_sig, 0x00, 0x00, 0x10, 0x20])

        payload = make_payload(body)
        result = tracker.process_message(payload)
        assert result is None


class TestPositionTrackerMoveSubtype:
    """Tests for PositionTracker move subtype handling."""

    def test_first_position_sets_move_subtype(self, fake_fs: FakeFileSystem) -> None:
        """Test first position message sets the move subtype."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = PositionTracker()
        tracker.set_magic("testmagic")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("xor_table should be set")

        # Build 17-byte position message
        # Position at bytes 4-5, XOR'd with xor_table[3-4]
        encoded_x = 16 ^ xor_table[3]
        encoded_y = 32 ^ xor_table[4]
        body = bytes([0x2E, 0x76, 0x00, 0x00, encoded_x, encoded_y]) + bytes(11)

        pos = tracker.decode_position(body)
        assert pos == (16, 32)
        # Subtype is stored as the raw byte from position 1
        assert tracker._move_subtype == 0x76

    def test_subsequent_wrong_subtype_decodes_anyway(self, fake_fs: FakeFileSystem) -> None:
        """Test subsequent messages with different subtype still decode."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = PositionTracker()
        tracker.set_magic("testmagic")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("xor_table should be set")

        # Build 17-byte bodies (minimum valid length)
        # Position at bytes 4-5, XOR'd with xor_table[3-4]

        # First message sets subtype to 0x75
        encoded_x1 = 16 ^ xor_table[3]
        encoded_y1 = 32 ^ xor_table[4]
        body1 = bytes([0x2E, 0x75, 0x00, 0x00, encoded_x1, encoded_y1]) + bytes(11)
        pos1 = tracker.decode_position(body1)
        assert pos1 == (16, 32)
        assert tracker._move_subtype == 0x75

        # Second message with different subtype should be decoded
        # (both 0x75 and 0x76 are valid position subtypes)
        encoded_x2 = 48 ^ xor_table[3]
        encoded_y2 = 64 ^ xor_table[4]
        body2 = bytes([0x2E, 0x76, 0x00, 0x00, encoded_x2, encoded_y2]) + bytes(11)
        pos2 = tracker.decode_position(body2)
        # Both are valid, so it should decode
        assert pos2 == (48, 64)

    def test_process_message_returns_none_when_decode_fails(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None when decode_position returns None.

        This tests line 121 - when body passes length checks but decode_position
        returns None due to xor_table being too short.
        """
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        # Create a short static key (< 5 chars) so xor_table will be < 5 bytes
        static_key = "ABCD"  # Only 4 chars
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = PositionTracker()
        tracker.set_magic("test")  # xor_table will be 4 bytes

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("xor_table should be set")
        assert len(xor_table) == 4  # Verify short xor_table

        # Build 17-byte body that passes process_message checks but fails in decode_position
        # because xor_table has < 5 bytes
        body = bytes([0x2E, 0x75, 0x00, 0x00, 0x10, 0x20]) + bytes(11)

        payload = make_payload(body)
        result = tracker.process_message(payload)
        # decode_position returns None due to len(xor_table) < 5
        assert result is None
