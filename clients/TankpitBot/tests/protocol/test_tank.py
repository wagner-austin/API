"""Tests for tank message decoders.

Tests for tank info, entry, exit, status, and supervisor decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    DecodeError,
    SupervisorDict,
    decode_action_done,
    decode_supervisor,
    decode_tank_entry,
    decode_tank_exit,
    decode_tank_info,
    decode_tank_status,
    decode_tank_status_sync,
    supervisor_error_code,
    supervisor_is_cant_go,
    supervisor_is_insufficient_fuel,
    x24,
)


class TestDecodeTankInfo:
    """Tests for decode_tank_info function."""

    def test_decodes_tank_info_with_name(self) -> None:
        """Decodes tank info with name."""
        # team=2, tank_id=0x0102, decoration=4 bytes, score=0x030405, name="Test"
        data = bytes([2, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF, 0x03, 0x04, 0x05]) + b"Test"
        result = decode_tank_info(data)
        assert result["msg_type"] == 0x21
        assert result["team"] == 2
        assert result["tank_id"] == 0x0102
        assert result["decoration_state"] == bytes([0xDE, 0xAD, 0xBE, 0xEF])
        assert result["persistent_tank_id"] == x24(0x03, 0x04, 0x05)
        assert result["name"] == "Test"

    def test_decodes_tank_info_without_name(self) -> None:
        """Decodes tank info without name."""
        data = bytes([2, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF, 0x03, 0x04, 0x05])
        result = decode_tank_info(data)
        assert result["name"] == ""

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_info(bytes([1, 2, 3]))


class TestDecodeTankEntry:
    """Tests for decode_tank_entry function."""

    def test_decodes_tank_entry(self) -> None:
        """Decodes tank entry per JS Uf.h layout.

        Wire: [flags, tank_id_lo, tank_id_hi, packed, score_hi, score_mid, score_lo, x, y]
        packed: team=bits0-1, damage_state=bits2-3, rank=bits4-7
        """
        packed = 0b0010_01_10  # rank=2, damage_state=1, team=2
        data = bytes([255, 5, 0, packed, 0, 0, 100, 80, 90])
        result = decode_tank_entry(data)
        assert result["msg_type"] == 0x28
        assert result["tank_id"] == 5
        assert result["team"] == 2
        assert result["rank"] == 2
        assert result["damage_state"] == 1
        assert result["score"] == 100
        assert result["x"] == 80
        assert result["y"] == 90

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_entry(bytes([1, 2, 3]))


class TestDecodeTankExit:
    """Tests for decode_tank_exit function."""

    def test_decodes_tank_exit(self) -> None:
        """Decodes tank exit message."""
        data = bytes([0x02, 0x01])  # tank_id=0x0102
        result = decode_tank_exit(data)
        assert result["msg_type"] == 0x58
        assert result["tank_id"] == 0x0102

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_exit(bytes([1]))


class TestDecodeTankStatusSync:
    """Tests for decode_tank_status_sync function."""

    def test_decodes_short_format(self) -> None:
        """Decodes 8-byte tank status sync."""
        # subtype=1, tank_id=0x0102, damage=2, rank=4, flags, lb_pos
        data = bytes([1, 0x02, 0x01, 2, 4, 0, 0x10, 0x00])
        result = decode_tank_status_sync(data)
        assert result["msg_type"] == 0x2E
        assert result["subtype"] == 1
        assert result["tank_id"] == 0x0102
        assert result["damage_state"] == 2
        assert result["rank"] == 4
        assert result["fuel"] is None

    def test_decodes_long_format(self) -> None:
        """Decodes 12+ byte tank status sync with fuel."""
        data = bytes([3, 0x02, 0x01, 0, 5, 0, 0x10, 0x00, 0, 0, 0xE8, 0x03])  # fuel=1000
        result = decode_tank_status_sync(data)
        assert result["subtype"] == 3
        assert result["fuel"] == 1000

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_status_sync(bytes([1, 2, 3]))


class TestDecodeTankStatus:
    """Tests for decode_tank_status function."""

    def test_decodes_tank_status_with_name(self) -> None:
        """Decodes full tank status with name."""
        # info_byte: team=2, rank=4 -> (4<<4)|2 = 0x42
        # tank_id, decoration(4), lb_score(3), lb_pos(3), name
        header = bytes([0x42, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF])
        lb_bytes = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        data = header + lb_bytes + b"Tank"
        result = decode_tank_status(data)
        assert result["msg_type"] == 0x3E
        assert result["team"] == 2
        assert result["rank"] == 4
        assert result["tank_id"] == 0x0102
        assert result["name"] == "Tank"

    def test_decodes_tank_status_without_name(self) -> None:
        """Decodes tank status without name."""
        data = bytes([0x42, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        result = decode_tank_status(data)
        assert result["name"] == ""

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_status(bytes([1, 2, 3, 4, 5]))


class TestDecodeSupervisor:
    """Tests for decode_supervisor function."""

    def test_decodes_supervisor(self) -> None:
        """Decodes supervisor message."""
        data = bytes([1, 0, 3])  # reset_action=1, close_map=0, error_code=3
        result = decode_supervisor(data)
        assert result["msg_type"] == 0x52
        assert result["reset_action"] == 1
        assert result["close_map"] == 0
        assert result["error_code"] == 3

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_supervisor(bytes([1, 2]))


class TestSupervisorHelpers:
    """Tests for supervisor error code helpers."""

    def test_supervisor_error_code(self) -> None:
        """Returns the data field as error code."""
        msg: SupervisorDict = {"msg_type": 0x52, "reset_action": 1, "close_map": 0, "error_code": 8}
        assert supervisor_error_code(msg) == 8

    def test_supervisor_is_cant_go(self) -> None:
        """Detects 'You can't go there!' error (code 1)."""
        cant_go: SupervisorDict = {
            "msg_type": 0x52,
            "reset_action": 1,
            "close_map": 0,
            "error_code": 1,
        }
        other: SupervisorDict = {
            "msg_type": 0x52,
            "reset_action": 1,
            "close_map": 0,
            "error_code": 5,
        }
        assert supervisor_is_cant_go(cant_go) is True
        assert supervisor_is_cant_go(other) is False

    def test_supervisor_is_insufficient_fuel(self) -> None:
        """Detects 'Insufficient fuel' error (code 8)."""
        low_fuel: SupervisorDict = {
            "msg_type": 0x52,
            "reset_action": 1,
            "close_map": 1,
            "error_code": 8,
        }
        other: SupervisorDict = {
            "msg_type": 0x52,
            "reset_action": 0,
            "close_map": 0,
            "error_code": 4,
        }
        assert supervisor_is_insufficient_fuel(low_fuel) is True
        assert supervisor_is_insufficient_fuel(other) is False


class TestDecodeActionDone:
    """Tests for decode_action_done function."""

    def test_decodes_action_done(self) -> None:
        """Decodes action done message (always succeeds)."""
        result = decode_action_done(b"")
        assert result["msg_type"] == 0x54
