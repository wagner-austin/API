"""Container message encoders — exact byte inverses of ``container.decoders``.

Container bodies include their own lead byte (the 0x2E envelope's
subtype), so these encoders return the FULL envelope body — unlike the
``protocol.encoders`` family, whose payloads get the subtype prepended
by the envelope encoder.
"""

from __future__ import annotations

from tankpit_bot.container.types import (
    ContainerPickupDict,
    MineDetonationDict,
    MinePlacementDict,
    TeleportLandedDict,
    UnknownContainerDict,
)
from tankpit_bot.protocol.helpers import pack16


def encode_container_pickup(message: ContainerPickupDict) -> bytes:
    """Encode a ContainerPickup body (inverse of ``decode_container_pickup``).

    Args:
        message: Decoded pickup with 1+ records.

    Returns:
        Full 0x2E body: 0x43 subtype + one 4-byte record per pickup.
    """
    out = bytearray([0x43])
    for record in message["pickups"]:
        out += bytes([record["x"], record["y"]]) + pack16(record["remaining_volume"])
    return bytes(out)


def encode_teleport_landed(message: TeleportLandedDict) -> bytes:
    """Encode a TeleportLanded body (inverse of ``decode_teleport_landed``).

    Args:
        message: Decoded teleport confirmation.

    Returns:
        The 1-byte body.
    """
    return bytes([message["subtype"]])


def encode_mine_detonation(message: MineDetonationDict) -> bytes:
    """Encode a MineDetonation body (inverse of ``decode_mine_detonation``).

    Args:
        message: Decoded detonation with destroyed-mine positions.

    Returns:
        Full 0x2E body: 0x45 subtype + one (x, y) pair per position.
    """
    out = bytearray([0x45])
    for x, y in message["positions"]:
        out += bytes([x, y])
    return bytes(out)


def encode_mine_placement(message: MinePlacementDict) -> bytes:
    """Encode a MinePlacement body (inverse of ``decode_mine_placement``).

    Args:
        message: Decoded placement with landed-mine positions.

    Returns:
        Full 0x2E body: 0x4B subtype, mine type, tank id, count, then
        one (x, y) pair per position.
    """
    out = bytearray([0x4B, message["mine_type"]])
    out += pack16(message["tank_id"])
    out.append(len(message["positions"]))
    for x, y in message["positions"]:
        out += bytes([x, y])
    return bytes(out)


def encode_unknown_container(message: UnknownContainerDict) -> bytes:
    """Encode an UnknownContainer body (inverse of ``decode_unknown_container``).

    Args:
        message: Preserved raw container body.

    Returns:
        The raw bytes verbatim.
    """
    return message["data"]


__all__ = [
    "encode_container_pickup",
    "encode_mine_detonation",
    "encode_mine_placement",
    "encode_teleport_landed",
    "encode_unknown_container",
]
