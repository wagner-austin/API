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
    supervisor_has_promo_kill,
    supervisor_is_promo_eligible,
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
        assert result["score"] == x24(0x03, 0x04, 0x05)
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

    def test_decodes_tank_entry_with_name(self) -> None:
        """Decodes tank entry with name."""
        # tank_id=5, x=0x0102, y=60, padding to 10 bytes, then name
        data = bytes([5, 0x02, 0x01, 60, 0, 0, 0, 0, 0, 0]) + b"Tank"
        result = decode_tank_entry(data)
        assert result["msg_type"] == 0x28
        assert result["tank_id"] == 5
        assert result["x"] == 0x0102
        assert result["y"] == 60
        assert result["name"] == "Tank"

    def test_decodes_tank_entry_without_name(self) -> None:
        """Decodes tank entry without name."""
        data = bytes([5, 0x02, 0x01, 60, 0, 0, 0, 0, 0, 0])
        result = decode_tank_entry(data)
        assert result["name"] == ""

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
        data = bytes([1, 0, 3])  # status=1, reserved=0, data=3
        result = decode_supervisor(data)
        assert result["msg_type"] == 0x52
        assert result["status"] == 1
        assert result["reserved"] == 0
        assert result["data"] == 3

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_supervisor(bytes([1, 2]))


class TestSupervisorHelpers:
    """Tests for supervisor helper functions."""

    def test_supervisor_is_promo_eligible(self) -> None:
        """Checks promo eligibility correctly."""
        eligible: SupervisorDict = {"msg_type": 0x52, "status": 1, "reserved": 0, "data": 0}
        not_eligible: SupervisorDict = {"msg_type": 0x52, "status": 8, "reserved": 0, "data": 0}
        assert supervisor_is_promo_eligible(eligible) is True
        assert supervisor_is_promo_eligible(not_eligible) is False

    def test_supervisor_has_promo_kill(self) -> None:
        """Checks promo kill correctly."""
        has_kill: SupervisorDict = {"msg_type": 0x52, "status": 8, "reserved": 0, "data": 0}
        no_kill: SupervisorDict = {"msg_type": 0x52, "status": 1, "reserved": 0, "data": 0}
        assert supervisor_has_promo_kill(has_kill) is True
        assert supervisor_has_promo_kill(no_kill) is False


class TestDecodeActionDone:
    """Tests for decode_action_done function."""

    def test_decodes_action_done(self) -> None:
        """Decodes action done message (always succeeds)."""
        result = decode_action_done(b"")
        assert result["msg_type"] == 0x54
