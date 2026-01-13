"""Tests for MineTracker class."""

from __future__ import annotations

import base64

from tankpit_bot import _test_hooks
from tankpit_bot.capture import MineTracker
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import build_test_xor_table, make_payload


class TestMineTracker:
    """Tests for MineTracker class."""

    def test_init(self) -> None:
        """Test MineTracker initialization."""
        tracker = MineTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._mines_placed == 0
        assert tracker._mines_detonated == 0

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = MineTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_mines_placed_property(self) -> None:
        """Test mines_placed property returns count."""
        tracker = MineTracker()
        assert tracker.mines_placed == 0

    def test_mines_detonated_property(self) -> None:
        """Test mines_detonated property returns count."""
        tracker = MineTracker()
        assert tracker.mines_detonated == 0

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = MineTracker()
        payload = make_payload(b"\x2e\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = MineTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_short_data(self) -> None:
        """Test process_message returns None for short data."""
        tracker = MineTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x02\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None


class TestMineTrackerParseMethods:
    """Tests for MineTracker._parse_* methods."""

    def test_parse_mine_placed(self) -> None:
        """Test _parse_mine_placed parses mine placement."""
        tracker = MineTracker()
        # decoded: [0x4B, owner_id_lo, owner_id_hi, x, y, ...]
        decoded = bytearray([0x4B, 0x64, 0x00, 0x32, 0x3C, 0x00, 0x00])
        result = tracker._parse_mine_placed(decoded)
        assert "PLACED" in result
        assert tracker.mines_placed == 1

    def test_parse_mine_detonation(self) -> None:
        """Test _parse_mine_detonation parses mine explosions."""
        tracker = MineTracker()
        # decoded: [0x45, count, x1, y1, x2, y2]
        decoded = bytearray([0x45, 0x02, 0x32, 0x3C, 0x33, 0x3D])
        result = tracker._parse_mine_detonation(decoded)
        assert "EXPLODE" in result
        assert "2 mines" in result
        assert tracker.mines_detonated == 2


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


class TestMineTrackerEdgeCases:
    """Tests for MineTracker edge cases and uncovered branches."""

    def test_set_magic_returns_early_when_no_static_key(self) -> None:
        """Test set_magic does nothing when static key missing."""
        fs = FakeFileSystem()
        _test_hooks.path_exists = fs.path_exists
        _test_hooks.read_text = fs.read_text

        tracker = MineTracker()
        tracker.set_magic("testmagic")
        assert tracker._xor_table is None

    def test_process_message_mine_command_sent(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message handles sent mine drop command."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Mine drop: ! type=4 id=98 x=50 y=60
        decrypted = bytes([0x21, 4, 98, 50, 60])
        # Encrypt back (skip first byte)
        encrypted = bytearray(len(decrypted))
        encrypted[0] = decrypted[0]
        for i in range(1, len(decrypted)):
            encrypted[i] = decrypted[i] ^ xor_table[i - 1]

        body = bytes(encrypted)
        payload = make_payload(body)
        result = tracker.process_message(payload, direction="sent")
        assert result, "Expected non-None result"
        assert "MINE:DROP" in result

    def test_process_message_mine_placed(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes mine placed message."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Mine placed: 0x4B owner_id(2) x y
        decoded_data = bytes([0x4B, 0x64, 0x00, 50, 60])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result"
        assert "MINE:PLACED" in result
        assert tracker.mines_placed == 1

    def test_process_message_mine_detonation(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes mine detonation message."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Mine detonation: 0x45 count positions...
        decoded_data = bytes([0x45, 2, 10, 20, 30, 40])  # 2 mines at (10,20), (30,40)
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result"
        assert "MINE:EXPLODE" in result
        assert "CHAIN" in result

    def test_parse_mine_placed_short(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_mine_placed handles short decoded."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = MineTracker()
        result = tracker._parse_mine_placed(bytearray([0x4B, 0x01, 0x02]))
        assert "total:" in result

    def test_parse_mine_detonation_short(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_mine_detonation handles short decoded."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = MineTracker()
        result = tracker._parse_mine_detonation(bytearray([0x45]))
        assert result == "[MINE:EXPLODE]"

    def test_process_message_returns_none_for_wrong_body(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for body not starting with 0x2E."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        # Body not starting with 0x2E and not a command
        body = b"\x99" + bytes(10)
        payload = make_payload(body)
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_unknown_decoded_sig(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for unknown decoded signature."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Build body with 0x2E prefix and decoded sig that is neither 0x4B nor 0x45
        decoded_data = bytes([0x99, 0x01, 0x02, 0x03])  # sig=0x99
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result is None

    def test_process_mine_command_wrong_type_or_id(self, fake_fs: FakeFileSystem) -> None:
        """Test _process_mine_command returns None for wrong cmd_type or cmd_id."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Command with type=5 instead of type=4
        decrypted = bytes([0x21, 5, 98, 50, 60])  # type=5
        encrypted = bytearray(len(decrypted))
        encrypted[0] = decrypted[0]
        for i in range(1, len(decrypted)):
            encrypted[i] = decrypted[i] ^ xor_table[i - 1]

        body = bytes(encrypted)
        payload = make_payload(body)
        result = tracker.process_message(payload, direction="sent")
        assert result is None

    def test_parse_mine_detonation_no_readable_positions(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_mine_detonation when positions can't be read."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_test_xor_table(static_key, magic)

        # Mine detonation with count=2 but not enough data for positions
        # decoded: 0x45 count=2 (only 2 bytes, no position data)
        decoded_data = bytes([0x45, 0x02])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)

        # Should return count without positions
        if result is None:
            raise AssertionError("Expected non-None result from process_message")
        assert "MINE:EXPLODE" in result
        assert "2 mines" in result
