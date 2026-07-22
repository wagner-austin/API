"""Resource message encoders — exact byte inverses of ``decoders.resources``."""

from __future__ import annotations

from tankpit_bot.protocol.helpers import pack16
from tankpit_bot.protocol.types import (
    EquipmentGainDict,
    EquipmentToggleDict,
    FuelDepositDict,
    FuelGainDict,
    InventoryDict,
)


def encode_fuel_gain(message: FuelGainDict) -> bytes:
    """Encode a 0x44 FuelGain payload (inverse of ``decode_fuel_gain``).

    Args:
        message: Decoded fuel gain.

    Returns:
        Payload bytes without the 0x44 prefix. The raw ``flag`` byte is
        emitted verbatim (``is_free`` is derived from it).
    """
    return pack16(message["fuel_total"]) + bytes([message["flag"]])


def encode_fuel_deposit(message: FuelDepositDict) -> bytes:
    """Encode a 0x64 FuelDeposit payload (inverse of ``decode_fuel_deposit``).

    Args:
        message: Decoded fuel deposit.

    Returns:
        Payload bytes without the 0x64 prefix.
    """
    return pack16(message["fuel_total"])


def encode_inventory(message: InventoryDict) -> bytes:
    """Encode a 0x49 Inventory payload (inverse of ``decode_inventory``).

    Args:
        message: Decoded inventory snapshot.

    Returns:
        Payload bytes without the 0x49 prefix: a display byte (1=show,
        2=alternate, 0=neither) then five count bytes with bit 7 set
        for DISABLED slots.
    """
    head = 1 if message["show"] else (2 if message["alternate"] else 0)
    slots = bytes(
        (message["counts"][i] & 127) | (0 if message["enabled"][i] else 128) for i in range(5)
    )
    return bytes([head]) + slots


def encode_equipment_gain(message: EquipmentGainDict) -> bytes:
    """Encode a 0x67 EquipmentGain payload (inverse of ``decode_equipment_gain``).

    Args:
        message: Decoded equipment gain.

    Returns:
        Payload bytes without the 0x67 prefix.
    """
    return bytes([1 if message["show_message"] else 0]) + bytes(message["gained"])


def encode_equipment_toggle(message: EquipmentToggleDict) -> bytes:
    """Encode a 0x74 EquipmentToggle payload (inverse of ``decode_equipment_toggle``).

    Args:
        message: Decoded equipment toggle.

    Returns:
        Payload bytes without the 0x74 prefix.
    """
    return bytes(1 if enabled else 0 for enabled in message["enabled"])


__all__ = [
    "encode_equipment_gain",
    "encode_equipment_toggle",
    "encode_fuel_deposit",
    "encode_fuel_gain",
    "encode_inventory",
]
