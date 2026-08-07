"""Resource message decoders.

This module handles decoding of resource-related messages:
fuel gain/deposit, inventory, equipment gain/toggle.
"""

from __future__ import annotations

from tankpit_bot.protocol.types import (
    EquipmentGainDict,
    EquipmentToggleDict,
    FuelDepositDict,
    FuelGainDict,
    InventoryDict,
)
from tankpit_bot.wire.helpers import require_min_length, x16


def decode_fuel_gain(data: bytes) -> FuelGainDict:
    """Decode fuel gain from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded fuel gain.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 3, "FuelGain")
    return FuelGainDict(
        msg_type=0x44,
        fuel_total=x16(data[0], data[1]),
        is_free=data[2] == 0,
        flag=data[2],
    )


def decode_fuel_deposit(data: bytes) -> FuelDepositDict:
    """Decode fuel deposit from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded fuel deposit.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 2, "FuelDeposit")
    return FuelDepositDict(msg_type=0x64, fuel_total=x16(data[0], data[1]))


def decode_inventory(data: bytes) -> InventoryDict:
    """Decode inventory from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded inventory.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 6, "Inventory")
    counts: list[int] = []
    enabled: list[bool] = []
    for i in range(5):
        byte = data[i + 1]
        counts.append(byte & 127)
        enabled.append((byte & 128) == 0)
    return InventoryDict(
        msg_type=0x49,
        show=data[0] == 1,
        alternate=data[0] == 2,
        counts=counts,
        enabled=enabled,
    )


def decode_equipment_gain(data: bytes) -> EquipmentGainDict:
    """Decode equipment gain from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded equipment gain.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 6, "EquipmentGain")
    return EquipmentGainDict(
        msg_type=0x67,
        show_message=data[0] == 1,
        gained=[data[i + 1] for i in range(5)],
    )


def decode_equipment_toggle(data: bytes) -> EquipmentToggleDict:
    """Decode equipment toggle from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded equipment toggle.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 5, "EquipmentToggle")
    return EquipmentToggleDict(msg_type=0x74, enabled=[data[i] == 1 for i in range(5)])


__all__ = [
    "decode_equipment_gain",
    "decode_equipment_toggle",
    "decode_fuel_deposit",
    "decode_fuel_gain",
    "decode_inventory",
]
