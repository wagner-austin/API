"""Tests for MineTracker class."""

from __future__ import annotations

import base64

from tankpit_bot.capture.trackers.mine import MineTracker
from tankpit_bot.capture.xor import xor_decode_body
from tankpit_bot.protocol.codec import build_xor_table
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import assert_set_magic_requires_static_key
from tests.wire_builders import frame_payload


class TestMineTracker:
    """Tests for MineTracker class."""

    def test_init(self) -> None:
        """Test MineTracker initialization."""
        tracker = MineTracker()
        assert tracker._xor_table is None
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

    def test_process_message_before_the_magic_arrives_returns_none(self) -> None:
        """Messages seen before the session magic are skipped, not decoded.

        A capture starts before the AUTH frame that carries the magic,
        so the tracker sees traffic while ``_xor_table`` is still None.
        Every decode path below needs that table --
        ``_process_mine_command`` asserts on it outright -- so reaching
        them turns an ordinary ordering into an ``AssertionError`` on a
        payload the tracker simply cannot read yet.
        """
        tracker = MineTracker()
        assert tracker._xor_table is None
        payload = frame_payload(bytes([0x21, 0x6B, 10, 20, 0]))

        assert tracker.process_message(payload, direction="sent") is None

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
        payload = frame_payload(b"\x2e\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = MineTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_short_data(self) -> None:
        """Test process_message returns None for short data.

        The third byte is 0x45 -- a mine-detonation type -- on purpose.
        With ``\\x2e`` there the length guard was unfalsifiable: the later
        ``msg_type not in (0x45, 0x4B)`` check returned None too, so the
        test passed whether the guard existed or not. Mutating the guard
        to a no-op left the suite green (2026-08-08 mutation sample).
        A tracked type reaches the decode path when the guard is absent,
        so now only the guard can produce None here.
        """
        tracker = MineTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x02\x00\x45").decode()
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_short_body(self) -> None:
        """Test process_message returns None when framed body is too short."""
        tracker = MineTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x03\x00\x2e\x00\x00").decode()
        result = tracker.process_message(payload)
        assert result is None


class TestMineTrackerParseMethods:
    """Tests for MineTracker._parse_* methods."""

    def test_parse_mine_placed(self) -> None:
        """Test _parse_mine_placed parses mine placement."""
        tracker = MineTracker()
        # decoded: [0x4B, mine_type, owner_id_lo, owner_id_hi, count, x1, y1]
        decoded = bytearray([0x4B, 0x00, 0x64, 0x00, 0x01, 0x32, 0x3C])
        result = tracker._parse_mine_placed(decoded)
        assert "PLACED" in result
        assert "owner=100" in result
        assert "count=1" in result
        assert tracker.mines_placed == 1

    def test_parse_mine_detonation(self) -> None:
        """Test _parse_mine_detonation parses mine explosions."""
        tracker = MineTracker()
        # decoded: [0x45, x1, y1, x2, y2]
        decoded = bytearray([0x45, 0x32, 0x3C, 0x33, 0x3D])
        result = tracker._parse_mine_detonation(decoded)
        assert "EXPLODE" in result
        assert "2 mines" in result
        assert tracker.mines_detonated == 2


class TestMineTrackerEdgeCases:
    """Tests for MineTracker edge cases and uncovered branches."""

    def test_set_magic_raises_when_no_static_key(self) -> None:
        """A missing static key is fatal, not a silent no-op."""
        assert_set_magic_requires_static_key(MineTracker())

    def test_process_message_mine_command_sent(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message handles sent mine drop command."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(static_key_file_path(), static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        # Mine drop: ! type=4 id=98 x=50 y=60
        decrypted = bytes([0x21, 4, 98, 50, 60])
        # Encrypt back (skip first byte)
        encrypted = bytearray(len(decrypted))
        encrypted[0] = decrypted[0]
        for i in range(1, len(decrypted)):
            encrypted[i] = decrypted[i] ^ xor_table[i - 1]

        body = bytes(encrypted)
        payload = frame_payload(body)
        result = tracker.process_message(payload, direction="sent")
        assert result, "Expected non-None result"
        assert "MINE:DROP" in result

    def test_process_message_mine_placed(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes mine placed message."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(static_key_file_path(), static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        # Mine placed: top-level 0x4B, decoded body is mine_type owner_id(2) count x y
        decoded_data = bytes([0x00, 0x64, 0x00, 1, 50, 60])
        encoded_data = xor_decode_body(decoded_data, xor_table)
        body = bytes([0x4B]) + encoded_data

        payload = frame_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result"
        assert "MINE:PLACED" in result
        assert tracker.mines_placed == 1

    def test_process_message_mine_detonation(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes mine detonation message."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(static_key_file_path(), static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        # Mine detonation: top-level 0x45 with positions...
        decoded_data = bytes([10, 20, 30, 40])  # 2 mines at (10,20), (30,40)
        encoded_data = xor_decode_body(decoded_data, xor_table)
        body = bytes([0x45]) + encoded_data

        payload = frame_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result"
        assert "MINE:EXPLODE" in result
        assert "CHAIN" in result

    def test_parse_mine_placed_short(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_mine_placed handles short decoded."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(static_key_file_path(), static_key)

        tracker = MineTracker()
        result = tracker._parse_mine_placed(bytearray([0x4B, 0x01, 0x02]))
        assert "total:" in result

    def test_parse_mine_placed_without_readable_positions(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_mine_placed reports count when positions are truncated."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(static_key_file_path(), static_key)

        tracker = MineTracker()
        result = tracker._parse_mine_placed(bytearray([0x4B, 0x00, 0x34, 0x12, 0x02]))
        assert result == "[MINE:PLACED] owner=4660 count=2"

    def test_parse_mine_detonation_short(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_mine_detonation handles short decoded."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(static_key_file_path(), static_key)

        tracker = MineTracker()
        result = tracker._parse_mine_detonation(bytearray([0x45]))
        assert result == "[MINE:EXPLODE]"

    def test_process_message_returns_none_for_wrong_body(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for body not starting with 0x2E."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(static_key_file_path(), static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        # Body not starting with a mine type and not a command
        body = b"\x99" + bytes(10)
        payload = frame_payload(body)
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_unknown_decoded_sig(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for unknown decoded signature."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(static_key_file_path(), static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        # Build body with unsupported top-level type
        decoded_data = bytes([0x01, 0x02, 0x03])
        encoded_data = xor_decode_body(decoded_data, xor_table)
        body = bytes([0x99]) + encoded_data

        payload = frame_payload(body)
        result = tracker.process_message(payload)
        assert result is None

    def test_process_mine_command_wrong_type_or_id(self, fake_fs: FakeFileSystem) -> None:
        """Test _process_mine_command returns None for wrong cmd_type or cmd_id."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(static_key_file_path(), static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        # Command with type=5 instead of type=4
        decrypted = bytes([0x21, 5, 98, 50, 60])  # type=5
        encrypted = bytearray(len(decrypted))
        encrypted[0] = decrypted[0]
        for i in range(1, len(decrypted)):
            encrypted[i] = decrypted[i] ^ xor_table[i - 1]

        body = bytes(encrypted)
        payload = frame_payload(body)
        result = tracker.process_message(payload, direction="sent")
        assert result is None

    def test_parse_mine_detonation_no_readable_positions(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_mine_detonation when positions can't be read."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(static_key_file_path(), static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        # Mine detonation with odd position data length: one complete pair and one dangling byte
        decoded_data = bytes([10, 20, 30])
        encoded_data = xor_decode_body(decoded_data, xor_table)
        body = bytes([0x45]) + encoded_data

        payload = frame_payload(body)
        result = tracker.process_message(payload)

        if result is None:
            raise AssertionError("Expected non-None result from process_message")
        assert "MINE:EXPLODE" in result
        assert "1 mines" in result

    def test_process_message_ignores_tunneled_tank_status_sync(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """Test tunneled 0x2E subtype 0x45 is not misclassified as mine detonation."""
        from tankpit_bot.resources import static_key_file_path

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(static_key_file_path(), static_key)
        magic = "testmagic123"

        tracker = MineTracker()
        tracker.set_magic(magic)

        xor_table = build_xor_table(static_key, magic)

        decoded_data = bytes([0x45, 0x37, 0xDC])
        encoded_data = xor_decode_body(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = frame_payload(body)
        result = tracker.process_message(payload)
        assert result is None
