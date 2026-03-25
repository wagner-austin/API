"""Tests for resource message decoders.

Tests for fuel, inventory, and equipment decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    DecodeError,
    decode_equipment_gain,
    decode_equipment_toggle,
    decode_fuel_deposit,
    decode_fuel_gain,
    decode_inventory,
)


class TestDecodeFuelGain:
    """Tests for decode_fuel_gain function."""

    def test_decodes_paid_fuel(self) -> None:
        """Decodes paid fuel gain."""
        # fuel_total=0x1234, is_free=False (data[2] != 0)
        data = bytes([0x34, 0x12, 1])
        result = decode_fuel_gain(data)
        assert result["msg_type"] == 0x44
        assert result["fuel_total"] == 0x1234
        assert result["is_free"] is False

    def test_decodes_free_fuel(self) -> None:
        """Decodes free fuel gain."""
        data = bytes([0x34, 0x12, 0])
        result = decode_fuel_gain(data)
        assert result["is_free"] is True

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_fuel_gain(bytes([1, 2]))


class TestDecodeFuelDeposit:
    """Tests for decode_fuel_deposit function."""

    def test_decodes_fuel_deposit(self) -> None:
        """Decodes fuel deposit fuel_total."""
        data = bytes([0x64, 0x00])  # fuel_total=100
        result = decode_fuel_deposit(data)
        assert result["msg_type"] == 0x64
        assert result["fuel_total"] == 100

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_fuel_deposit(bytes([1]))


class TestDecodeInventory:
    """Tests for decode_inventory function."""

    def test_decodes_inventory_show(self) -> None:
        """Decodes inventory with show flag."""
        # show=1, counts with enabled flags
        data = bytes([1, 5, 10 | 128, 3, 7, 0])  # armor enabled, dual disabled, others enabled
        result = decode_inventory(data)
        assert result["msg_type"] == 0x49
        assert result["show"] is True
        assert result["alternate"] is False
        assert result["counts"] == [5, 10, 3, 7, 0]
        assert result["enabled"] == [True, False, True, True, True]

    def test_decodes_inventory_alternate(self) -> None:
        """Decodes inventory with alternate flag."""
        data = bytes([2, 0, 0, 0, 0, 0])
        result = decode_inventory(data)
        assert result["alternate"] is True

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_inventory(bytes([1, 2, 3]))


class TestDecodeEquipmentGain:
    """Tests for decode_equipment_gain function."""

    def test_decodes_equipment_gain(self) -> None:
        """Decodes equipment gain message."""
        # show_message=1, gained=[1,2,0,1,0]
        data = bytes([1, 1, 2, 0, 1, 0])
        result = decode_equipment_gain(data)
        assert result["msg_type"] == 0x67
        assert result["show_message"] is True
        assert result["gained"] == [1, 2, 0, 1, 0]

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_equipment_gain(bytes([1, 2, 3]))


class TestDecodeEquipmentToggle:
    """Tests for decode_equipment_toggle function."""

    def test_decodes_equipment_toggle(self) -> None:
        """Decodes equipment toggle message."""
        data = bytes([1, 0, 1, 1, 0])  # armor=on, dual=off, missile=on, homing=on, radar=off
        result = decode_equipment_toggle(data)
        assert result["msg_type"] == 0x74
        assert result["enabled"] == [True, False, True, True, False]

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_equipment_toggle(bytes([1, 2]))
