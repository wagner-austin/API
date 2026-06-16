"""Tests for TankTracker class."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.capture.trackers import TankTracker
from tests.conftest import FakeFileSystem
from tests.sniffer.trackers.conftest import make_payload


class TestTankTracker:
    """Tests for TankTracker class."""

    def test_init(self) -> None:
        """Test TankTracker initialization."""
        tracker = TankTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._tanks == {}

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        tracker = TankTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_register_name(self) -> None:
        """Test register_name stores tank name."""
        tracker = TankTracker()
        tracker.register_name(123, "TestTank")

        assert tracker._tanks[123]["name"] == "TestTank"

    def test_register_name_updates_existing(self) -> None:
        """Test register_name updates existing tank entry."""
        tracker = TankTracker()
        tracker._tanks[123] = {"team": "red", "rank": "private"}
        tracker.register_name(123, "NewName")

        assert tracker._tanks[123]["name"] == "NewName"
        assert tracker._tanks[123]["team"] == "red"

    def test_get_name_returns_name(self) -> None:
        """Test get_name returns stored name."""
        tracker = TankTracker()
        tracker.register_name(123, "TestTank")

        result = tracker.get_name(123)
        assert result == "TestTank"

    def test_get_name_returns_none_for_unknown(self) -> None:
        """Test get_name returns None for unknown tank."""
        tracker = TankTracker()
        result = tracker.get_name(999)
        assert result is None

    def test_get_name_returns_none_if_name_not_set(self) -> None:
        """Test get_name returns None if name not set."""
        tracker = TankTracker()
        tracker._tanks[123] = {"team": "red"}

        result = tracker.get_name(123)
        assert result is None

    def test_get_all_names_returns_dict(self) -> None:
        """Test get_all_names returns all name mappings."""
        tracker = TankTracker()
        tracker.register_name(1, "Tank1")
        tracker.register_name(2, "Tank2")
        tracker._tanks[3] = {"team": "blue"}  # No name

        result = tracker.get_all_names()
        assert result == {1: "Tank1", 2: "Tank2"}

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        tracker = TankTracker()
        payload = make_payload(b"\x2e\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        tracker = TankTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


class TestTankTrackerParseMethods:
    """Tests for TankTracker._parse_* methods."""

    def test_parse_tank_join(self) -> None:
        """Test _parse_tank_join parses tank join message."""
        tracker = TankTracker()
        # decoded: [subtype, tank_id_lo, tank_id_hi, extra...]
        decoded = bytearray([0x01, 0x64, 0x00, 0xAB, 0xCD])
        result = tracker._parse_tank_join(decoded)
        assert result, "Expected non-None result from _parse_tank_join"
        assert "JOIN" in result
        assert "100" in result or "id=100" in result

    def test_parse_tank_leave(self) -> None:
        """Test _parse_tank_leave parses tank leave message."""
        tracker = TankTracker()
        decoded = bytearray([0x01, 0x64, 0x00, 0xAB, 0xCD])
        result = tracker._parse_tank_leave(decoded)
        assert result, "Expected non-None result from _parse_tank_leave"
        assert "LEAVE" in result

    def test_parse_tank_status(self) -> None:
        """Test _parse_tank_status parses and stores tank info."""
        tracker = TankTracker()
        # decoded: [info_byte, tank_id_lo, tank_id_hi, 10 bytes, name...]
        # info_byte: team in bits 0-1, rank in bits 4-6
        # Team=1 (purple), Rank=3 (sergeant) -> 0x31
        decoded = bytearray([0x31, 0x64, 0x00]) + bytearray(10) + bytearray(b"TestPlayer")
        result = tracker._parse_tank_status(decoded)
        assert result, "Expected non-None result from _parse_tank_status"
        assert "STATUS" in result
        assert "sergeant" in result or "TestPlayer" in result

    def test_parse_movement(self) -> None:
        """Test _parse_movement parses tank movement."""
        tracker = TankTracker()
        # decoded: [tank_id_lo, tank_id_hi, x, y, direction]
        decoded = bytearray([0x64, 0x00, 0x32, 0x3C, 0x02])
        result = tracker._parse_movement(decoded)
        assert result, "Expected non-None result from _parse_movement"
        assert "MOVE" in result
        assert "(50,60)" in result

    def test_parse_shooting(self) -> None:
        """Test _parse_shooting parses shot events."""
        tracker = TankTracker()
        # decoded: [team, shooter_id_lo, shooter_id_hi, x, y]
        decoded = bytearray([0x01, 0x64, 0x00, 0x32, 0x3C])
        result = tracker._parse_shooting(decoded)
        assert result, "Expected non-None result from _parse_shooting"
        assert "SHOT" in result

    def test_parse_tank_info(self) -> None:
        """Test _parse_tank_info registers tank name."""
        tracker = TankTracker()
        # decoded: [team, tank_id_lo, tank_id_hi, 7 bytes, name...]
        decoded = bytearray([0x01, 0x64, 0x00]) + bytearray(7) + bytearray(b"PlayerName")
        result = tracker._parse_tank_info(decoded)
        assert result, "Expected non-None result from _parse_tank_info"
        assert "INFO" in result
        assert "PlayerName" in result
        # Verify name was registered
        assert tracker.get_name(100) == "PlayerName"

    def test_parse_player_list(self) -> None:
        """Test _parse_player_list parses player list."""
        tracker = TankTracker()
        # decoded: [tank_id_lo, tank_id_hi, b2, b3, b4]
        decoded = bytearray([0x64, 0x00, 0x01, 0x02, 0x03])
        result = tracker._parse_player_list(decoded)
        assert "PLAYERS" in result

    def test_parse_player_update(self) -> None:
        """Test _parse_player_update parses player updates."""
        tracker = TankTracker()
        # decoded: repeating [tank_id_lo, tank_id_hi, data]
        decoded = bytearray([0x64, 0x00, 0x01, 0xC8, 0x00, 0x02])
        result = tracker._parse_player_update(decoded)
        assert "PLAYERS" in result

    def test_parse_statistics(self) -> None:
        """Test _parse_statistics parses stats message."""
        tracker = TankTracker()
        # decoded: [hours_lo, hours_hi, mins, secs, pad(3), destroyed, deactivated, pad(3), promo]
        decoded = bytearray(
            [
                0x05,
                0x00,  # 5 hours
                0x1E,
                0x0A,  # 30 mins, 10 secs
                0x00,
                0x00,
                0x00,  # padding
                0x10,
                0x08,  # destroyed=16, deactivated=8
                0x00,
                0x00,
                0x00,  # padding
                0x00,
                0x64,  # promo_pts=100
            ]
        )
        result = tracker._parse_statistics(decoded)
        assert "STATS" in result
        assert "5h" in result

    def test_parse_promotion(self) -> None:
        """Test _parse_promotion parses promotion message."""
        tracker = TankTracker()
        # decoded: [rank, promoted_flag]
        decoded = bytearray([0x04, 0x01])  # Promoted to lieutenant
        result = tracker._parse_promotion(decoded)
        assert "PROMOTED" in result
        assert "lieutenant" in result

    def test_parse_promotion_demoted(self) -> None:
        """Test _parse_promotion handles demotion."""
        tracker = TankTracker()
        decoded = bytearray([0x03, 0x00])  # Demoted to sergeant
        result = tracker._parse_promotion(decoded)
        assert "DEMOTED" in result

    def test_parse_supervisor_msg(self) -> None:
        """Test _parse_supervisor_msg parses supervisor message."""
        tracker = TankTracker()
        # decoded: [0x01, 0x00, status]
        decoded = bytearray([0x01, 0x00, 0x04])
        result = tracker._parse_supervisor_msg(decoded)
        assert "SUPERVISOR" in result

    def test_get_all_names_returns_registered_names(self) -> None:
        """Test get_all_names returns all registered tank names."""
        tracker = TankTracker()
        tracker.register_name(100, "Player1")
        tracker.register_name(200, "Player2")

        names = tracker.get_all_names()
        assert names == {100: "Player1", 200: "Player2"}


class TestTankTrackerEdgeCases:
    """Tests for TankTracker edge cases and uncovered branches."""

    def test_set_magic_returns_early_when_no_static_key(self) -> None:
        """Test set_magic does nothing when static key missing."""
        fs = FakeFileSystem()
        _test_hooks.path_exists = fs.path_exists
        _test_hooks.read_text = fs.read_text

        tracker = TankTracker()
        tracker.set_magic("testmagic")
        assert tracker._xor_table is None

    def test_decode_payload_returns_none_for_short_data(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_payload returns None for data < 4 bytes."""
        import base64

        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker.set_magic("testmagic")

        payload = base64.b64encode(b"\x01\x00\x2e").decode()
        result = tracker._decode_payload(payload)
        assert result is None

    def test_decode_payload_returns_none_for_short_body(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_payload returns None for body < 2 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker.set_magic("testmagic")

        body = b"\x2e"
        payload = make_payload(body)
        result = tracker._decode_payload(payload)
        assert result is None

    def test_process_message_returns_none_for_unknown_type(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message returns None for unhandled message types."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = TankTracker()
        tracker.set_magic(magic)

        body = b"\x99" + bytes(10)
        payload = make_payload(body)
        result = tracker.process_message(payload)
        assert result is None

    def test_parse_tank_join_returns_none_for_short_decoded(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_tank_join returns None when decoded < 3 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker.set_magic("testmagic")

        result = tracker._parse_tank_join(bytearray(b"\x01\x02"))
        assert result is None

    def test_parse_tank_leave_returns_none_for_short_decoded(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_tank_leave returns None when decoded < 3 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        result = tracker._parse_tank_leave(bytearray(b"\x01\x02"))
        assert result is None

    def test_parse_tank_status_returns_none_for_short_decoded(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """Test _parse_tank_status returns None when decoded < 13 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        result = tracker._parse_tank_status(bytearray(bytes(10)))
        assert result is None

    def test_parse_movement_response_returns_none_for_short_decoded(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """Test _parse_movement_response returns None when decoded < 11 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        result = tracker._parse_movement_response(bytearray(bytes(8)))
        assert result is None

    def test_parse_movement_response_valid_data(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_movement_response with valid 11+ byte data."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker.register_name(100, "TestTank")

        # team=2(blue), tank_id=100, x=50, y=60, dir=1, unk=0, rank=3, lb_pos=12345
        decoded = bytearray([2, 100, 0, 50, 60, 1, 0, 3, 0, 0x30, 0x39])
        result = tracker._parse_movement_response(decoded)
        if result is None:
            raise AssertionError("expected non-None result")
        assert "TestTank" in result
        assert "blue" in result
        assert "sergeant" in result

    def test_parse_movement_response_unknown_tank(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_movement_response creates tank entry for unknown tank."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()

        decoded = bytearray([0, 50, 0, 30, 40, 1, 0, 2, 0, 0, 10])
        result = tracker._parse_movement_response(decoded)
        if result is None:
            raise AssertionError("expected non-None result")
        assert "id=50" in result
        assert "red" in result

    def test_parse_status_sync_valid(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_status_sync with valid data."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker.register_name(100, "SyncTank")

        decoded = bytearray([0x50, 100, 0, 0x01, 0x02])
        result = tracker._parse_status_sync(decoded, b"\x2e\x50\x64\x00")
        assert "STATUS:0x50 'P'" in result
        assert "SyncTank" in result

    def test_parse_status_sync_empty(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_status_sync with empty data."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        result = tracker._parse_status_sync(bytearray(), b"")
        assert "STATUS:0x00" in result

    def test_parse_movement_with_known_tank(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_movement includes tank name when known."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker.register_name(100, "TestTank")

        decoded = bytearray([0x64, 0x00, 50, 60, 1])
        result = tracker._parse_movement(decoded)
        assert result, "Expected non-None result"
        assert "TestTank" in result

    def test_parse_movement_with_tracked_tank_no_name(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_movement with tank in _tanks but no name (empty string)."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker._tanks[100] = {"team": "red", "rank": "private", "name": ""}

        decoded = bytearray([0x64, 0x00, 50, 60, 1])
        result = tracker._parse_movement(decoded)
        assert result, "Expected non-None result"
        assert "tank=100" in result

    def test_parse_movement_response_with_tracked_tank_no_name(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """Test _parse_movement_response with tank in _tanks but no name."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker._tanks[100] = {"team": "blue", "rank": "sergeant", "name": ""}

        decoded = bytearray([2, 100, 0, 50, 60, 1, 0, 3, 0, 0x30, 0x39])
        result = tracker._parse_movement_response(decoded)
        assert result, "Expected non-None result"
        assert "id=100" in result

    def test_parse_shooting_with_tracked_tank_no_name(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_shooting with tank in _tanks but no name."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()
        tracker._tanks[100] = {"team": "red", "rank": "private", "name": ""}

        decoded = bytearray([0, 0x64, 0x00, 50, 60])
        result = tracker._parse_shooting(decoded)
        assert result, "Expected non-None result"
        assert "id=100" in result

    def test_parse_tank_status_with_name(self, fake_fs: FakeFileSystem) -> None:
        """Test _parse_tank_status extracts name when decoded > 13 bytes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = TankTracker()

        info_byte = 0x01 | (0x02 << 4)  # team=1, rank=2
        decoded = bytearray([info_byte, 0x64, 0x00])
        decoded.extend([0] * 10)
        decoded.extend(b"Tank\x00")

        result = tracker._parse_tank_status(decoded)
        assert result, "Expected non-None result"
        assert "Tank" in result

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
        payload = make_payload(body)
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
        payload = make_payload(body)
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
        payload = make_payload(body)
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
        payload = make_payload(body)
        result = tracker.process_message(payload)

        # Should call _parse_tank_join handler and return result
        if result is None:
            raise AssertionError("Expected non-None result from process_message")
        assert "JOIN" in result
