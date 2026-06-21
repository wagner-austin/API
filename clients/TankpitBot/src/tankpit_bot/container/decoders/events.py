"""Container event decoders.

Three container-only message bodies:
- 1-byte TeleportLanded (server confirmation, 0x0C subtype in production)
- (1 + 4N)-byte ContainerPickup (subtype 0x43, fuel/equipment pickup with N records)
- Unknown fallback that preserves the raw body for diagnostics
"""

from __future__ import annotations

from tankpit_bot.container.helpers import (
    ContainerDecodeError,
    require_exact_length,
    require_min_length,
)
from tankpit_bot.container.types import (
    ContainerPickupDict,
    ContainerPickupRecordDict,
    TeleportLandedDict,
    UnknownContainerDict,
)


def is_teleport_landed_structure(data: bytes) -> bool:
    """Check if data matches teleport landed structure.

    Teleport landed criteria:
    - Exactly 1 byte

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches teleport landed pattern.
    """
    return len(data) == 1


def decode_teleport_landed(data: bytes) -> TeleportLandedDict:
    """Decode teleport landed container message.

    Args:
        data: Decoded container body bytes (must be 1 byte).

    Returns:
        Decoded teleport landed data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 1, "TeleportLanded")

    return TeleportLandedDict(
        msg_type="teleport_landed",
        subtype=data[0],
    )


def is_container_pickup_structure(data: bytes) -> bool:
    """Check if data matches container pickup structure.

    Criteria:
    - Subtype byte (first byte) is 0x43
    - Remaining bytes (after subtype) form 1+ complete 4-byte records
      (each record is ``[x, y, remaining_lo, remaining_hi]``)

    Equivalent to: ``data[0] == 0x43 and len(data) >= 5 and
    (len(data) - 1) % 4 == 0``. Bodies of length 5, 9, 13, 17... match.

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches container pickup pattern.
    """
    if len(data) < 5 or data[0] != 0x43:
        return False
    return (len(data) - 1) % 4 == 0


def decode_container_pickup(data: bytes) -> ContainerPickupDict:
    """Decode container pickup message (one or more 4-byte records).

    Each record is ``[x, y, remaining_lo, remaining_hi]``;
    ``remaining_volume`` is the fuel that REMAINS in the container after
    this pickup, not the fuel transferred. See
    :class:`ContainerPickupDict` for the empirical evidence and the
    discriminator for fuel vs equipment (the paired 0x67
    EquipmentGain or the 0x2E TankStatusSync fuel delta).

    Args:
        data: Decoded container body bytes. Must start with 0x43 and
            have at least one 4-byte record after the subtype byte.

    Returns:
        Decoded ``ContainerPickupDict`` with a tuple of pickup records.

    Raises:
        ContainerDecodeError: If structure validation fails (wrong
            subtype, body too short, or trailing partial record).
    """
    require_min_length(data, 5, "ContainerPickup")
    if data[0] != 0x43:
        raise ContainerDecodeError(f"ContainerPickup: expected subtype 0x43, got 0x{data[0]:02X}")
    if (len(data) - 1) % 4 != 0:
        raise ContainerDecodeError(
            "ContainerPickup: body after subtype must be a multiple of 4 bytes "
            f"(one 4-byte record per pickup), got {len(data) - 1} bytes"
        )

    pickups = tuple(
        ContainerPickupRecordDict(
            x=data[i],
            y=data[i + 1],
            remaining_volume=data[i + 2] | (data[i + 3] << 8),
        )
        for i in range(1, len(data), 4)
    )

    return ContainerPickupDict(
        msg_type="container_pickup",
        pickups=pickups,
    )


# PlayerListShort (4-byte 0x79) and PlayerListExtended (7-byte 0x79) were
# deleted 2026-06-20. Corpus sweep of 150 sessions / 48,304 0x2E bodies:
# 0 fires for either. The bot never sends the '/' query and 4-byte 0x2E
# bodies are 0x44 FuelGain or 0x52 SupervisorText; 7-byte bodies route to
# other subtypes. The protocol path is the single source of truth.


# is_tip_notification_structure / decode_tip_notification,
# is_chunk_data_structure / decode_chunk_data, and
# is_world_state_structure / decode_world_state were all deleted
# 2026-06-19 -- 0 production samples after 0x4C MapData was tunneled
# correctly inside 0x2E (2933 samples). The freshness-regression
# fixture was regenerated under its run's real magic so the bodies
# now decode as genuine 0x4C MapData via the protocol path.


def decode_unknown_container(data: bytes) -> UnknownContainerDict:
    """Create unknown container result for unrecognized structures.

    Args:
        data: Decoded container body bytes.

    Returns:
        Unknown container data for debugging.

    Raises:
        ContainerDecodeError: If data is empty.
    """
    require_min_length(data, 1, "UnknownContainer")

    return UnknownContainerDict(
        msg_type="unknown_container",
        subtype=data[0],
        length=len(data),
        data=bytes(data),
    )


__all__ = [
    "decode_container_pickup",
    "decode_teleport_landed",
    "decode_unknown_container",
    "is_container_pickup_structure",
    "is_teleport_landed_structure",
]
