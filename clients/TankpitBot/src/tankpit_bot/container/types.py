"""Container message types and enumerations.

This module contains all TypedDict definitions for container messages and
enumerations for message types and decode levels.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import Literal, TypedDict


class ContainerMessageType(IntEnum):
    """Types of messages found inside 0x2E containers.

    Identified by structure, not subtype byte value.
    """

    UNKNOWN = 0
    MINE_DETONATION = auto()
    MINE_PLACEMENT = auto()
    TELEPORT_LANDED = auto()
    CONTAINER_PICKUP = auto()


class DecodeLevel(IntEnum):
    """Decode understanding level for message types.

    Used to calculate decode coverage percentage in stats.
    The integer value represents the weight for coverage calculation.

    Levels:
        UNKNOWN: Message type not recognized (0 points).
        IDENTIFIED: Type known but fields not decoded (25 points).
        PARTIAL: Key fields decoded but some unknown (50 points).
        FULL: All fields fully decoded and understood (100 points).
    """

    UNKNOWN = 0
    IDENTIFIED = 25
    PARTIAL = 50
    FULL = 100


# =============================================================================
# Decode Level Registry
# =============================================================================
# Maps each message type to its decode understanding level.
# This is the single source of truth for stats coverage calculation.

MESSAGE_TYPE_LEVELS: dict[ContainerMessageType, DecodeLevel] = {
    ContainerMessageType.UNKNOWN: DecodeLevel.UNKNOWN,
    # Fully decoded message types (all fields understood)
    ContainerMessageType.MINE_DETONATION: DecodeLevel.FULL,
    ContainerMessageType.MINE_PLACEMENT: DecodeLevel.FULL,
    ContainerMessageType.TELEPORT_LANDED: DecodeLevel.FULL,
    ContainerMessageType.CONTAINER_PICKUP: DecodeLevel.FULL,
}


def get_decode_level(msg_type: ContainerMessageType) -> DecodeLevel:
    """Get the decode understanding level for a message type.

    Returns the level from MESSAGE_TYPE_LEVELS registry.
    This is used by stats calculation to determine coverage percentage.

    Args:
        msg_type: The container message type to look up.

    Returns:
        The decode level for the message type.
        Returns DecodeLevel.UNKNOWN if type not in registry.
    """
    return MESSAGE_TYPE_LEVELS.get(msg_type, DecodeLevel.UNKNOWN)


# =============================================================================
# Combat Messages
# =============================================================================
# 0x53 ShootEvent lives in tankpit_bot.protocol (ShootEventDict). The
# container path is intentionally not duplicated -- the protocol path
# is the single source of truth (re-verified 2026-06-19 against JS
# Gg.h and capture bot-20260619-050303).


class MineDetonationDict(TypedDict):
    """Mine detonation decoded from 0x2E container.

    Structure (proven from captures):
      [subtype:1] [positions: repeated (x, y) bytes]
    """

    msg_type: Literal[0x45]
    positions: list[tuple[int, int]]


# 0x41 Deactivation lives in tankpit_bot.protocol (DeactivationDict).
# The container path was deleted 2026-06-19; routing min_len fix in
# protocol/decoders/routing.py (7 -> 6) ensures the protocol path now
# fires for the wire 6-byte body.

# Container DeactivationDeath (7-byte 0x43) was deleted 2026-06-20 after
# empirical proof of zero production fires: 7-byte 0x2E bodies all route
# to 0x49 Inventory / 0x67 EquipmentGain / 0x4A TerrainUpdate / 0x4F
# RadarScanResult via the tunneled protocol dispatch.


class MinePlacementDict(TypedDict):
    """Mine placement decoded from 0x2E container.

    Structure (15 bytes, proven from capture):
      [subtype:1] [mine_type:1] [tank_id:2 LE] [count:1] [positions: count*2]

    This is a tunneled mine placement payload carried inside a 0x2E frame.
    """

    msg_type: Literal[0x4B]
    mine_type: int
    tank_id: int
    positions: list[tuple[int, int]]


# =============================================================================
# Tank Messages
# =============================================================================
# Container TankRegistry / TankLeave / TankStatusShort / TankUpdateCompact
# / TankUpdateExtended / TankUpdateFull / TankStatusSync all deleted.
# Single source of truth lives in tankpit_bot.protocol:
#
# * 0x21 TankInfo replaces TankRegistry (5143 corpus samples cover the
#   16/17/19-byte 0x2E body slots that TankRegistry previously ate).
# * 0x47 Movement (Lg.h) replaces TankUpdate / TankLeave.
# * 0x3D MovementResponse (Lg.h) replaces 13-byte TankStatusSync / the
#   long-form TankStatusFull.
# * Og.h TankStatusSync (short/full) handles 9-byte and 13-byte self
#   status with fuel.
#
# Empirical proof: corpus sweep of 150 sessions / 48,304 0x2E bodies,
# 0 production fires for any of the deleted types (2026-06-20).


# =============================================================================
# Position/Movement Messages
# =============================================================================
# Container PositionUpdate (13-byte 0x24) deleted 2026-06-20. 13-byte
# 0x2E slots are all 0x3D MovementResponse (4197 corpus samples).

# 0x47 Movement lives in tankpit_bot.protocol (MovementDict). The
# container path was deleted 2026-06-19: it had misinterpreted bytes
# 8-11 as a "player_id" but tpclient.js Lg.h reads those bytes as
# lb_score (24-bit BE at 6-8) + rank (a[9]). PlayerIdMapper was also
# deleted -- the protocol decoder has tank_id directly.


# =============================================================================
# Radar Messages
# =============================================================================


# RadarContainerDict, RadarMineDict, and the 0x4F RadarResponse decoder
# all live in tankpit_bot.protocol (single source of truth). Their
# container duplicates were deleted 2026-06-19.


# =============================================================================
# Player List Messages
# =============================================================================
# Container PlayerListShort (4-byte 0x79) and PlayerListExtended
# (7-byte 0x79) deleted 2026-06-20. The bot never sends the '/' query
# and 4/7-byte 0x2E bodies route to 0x44 FuelGain / 0x52 SupervisorText
# / other tunneled subtypes via the protocol path.


# =============================================================================
# Miscellaneous Messages
# =============================================================================


class TeleportLandedDict(TypedDict):
    """Teleport landed confirmation container message.

    Structure (1 byte):
      [subtype:1] (0x0C = 12)

    Sent by server after teleport completes and tank has landed at new location.
    Arrives 150-2000ms after teleport initiated, just before UI updates position.
    """

    msg_type: Literal["teleport_landed"]
    subtype: int


class ContainerPickupRecordDict(TypedDict):
    """One pickup record inside a 0x43 ContainerPickup body.

    Each record is 4 wire bytes: ``[x:1] [y:1] [remaining_volume:2 LE]``.
    """

    x: int
    y: int
    remaining_volume: int


class ContainerPickupDict(TypedDict):
    """Container pickup event (one OR more pickups in one wire message).

    Wire format (per JS V.C = $g handler, tpclient.pretty.js:4743):
      [subtype:1=0x43] [record_1: 4 bytes] [record_2: 4 bytes] ...
      Each record = ``[x:1] [y:1] [remaining_volume:2 LE]``.

    Corpus distribution (156 sessions, 2026-06-20 sweep):
      1 record (5-byte body):  2653 samples (regular single pickup)
      2 records (9-byte body):   80 samples (two pickups same tick)
      3 records (13-byte body):   2 samples (three pickups same tick)

    Multi-record bodies fire when a tank's movement causes multiple
    container tiles to update in one server tick -- for example a
    tank walking from one container tile into another, or a deposit
    happening simultaneously with a pickup on an adjacent tile.

    ``remaining_volume`` is the fuel that remains in the container
    **after** this pickup -- it is NOT the fuel transferred to the
    picker. Empirically verified 2026-06-20 against an annotated
    multi-pickup capture (runs/sniff/sniff-20260620-155103) where the
    user walked over fuel containers of known volume (300, 400, 100)
    and reported the server's remaining-vol read-out after each
    pickup; the wire ``remaining_volume`` matched exactly
    (283, 397, 96).

    Discriminator:
    - ``remaining_volume == 0`` -- container is empty after pickup.
      This covers both equipment containers (no fuel attribute) AND
      fuel containers fully consumed by the pickup.
    - ``remaining_volume > 0`` -- the picker took only part of a fuel
      container (typically because their tank is near the 1100 cap).

    Equipment pickups also fire a separate ``0x67 EquipmentGain``
    wire message in the same tick that carries the actual items
    received -- consumers that need to distinguish equipment vs fuel
    pickups should check for the paired ``0x67``. The fuel transferred
    on a fuel pickup is observable from the ``0x2E TankStatusSync``
    fuel-delta in the same tick.

    Server broadcasts this event TWICE per real pickup (one to the
    picker, one to the world view). Both arrive within ~200 ms as
    separate WS frames -- measured 43.9% duplicate rate across 13
    sniff sessions, 2026-06-20. The dispatcher de-duplicates by
    (subtype, body-bytes) within ``WIRE_DUP_WINDOW_MS``; downstream
    handlers can assume each pickup fires exactly once.
    """

    msg_type: Literal["container_pickup"]
    pickups: tuple[ContainerPickupRecordDict, ...]


# TipNotificationDict / ChunkDataDict / WorldStateDict all deleted
# 2026-06-19 -- 0 corpus samples landed on any of them after 0x4C
# MapData was tunneled inside 0x2E (2933 samples) and the regression
# fixture was regenerated under its run's real magic.


class UnknownContainerDict(TypedDict):
    """Unknown container message that didn't match any known structure.

    Preserved for debugging and future analysis.
    """

    msg_type: Literal["unknown_container"]
    subtype: int
    length: int
    data: bytes


# =============================================================================
# Status Messages (0x3D position+status, 0x2E self-status)
# =============================================================================


# 0x3D TankPositionStatus moved to tankpit_bot.protocol.MovementResponseDict
# 2026-06-19 (single source of truth, with the carrying byte at offset 11
# restored after being dropped from the prior protocol decoder).


# SelfStatusDict was deleted 2026-06-19. The 13-byte 0x2E-nested
# self-status form (with fuel) is decoded by the protocol path's
# TankStatusSync (decode_tank_status_sync), which handles both the short
# 9-byte form and the 13-byte form with fuel at the tail.


# =============================================================================
# Union Types
# =============================================================================

ContainerMessage = (
    MineDetonationDict
    | MinePlacementDict
    | TeleportLandedDict
    | ContainerPickupDict
    | UnknownContainerDict
)


__all__ = [
    "MESSAGE_TYPE_LEVELS",
    "ContainerMessage",
    "ContainerMessageType",
    "ContainerPickupDict",
    "ContainerPickupRecordDict",
    "DecodeLevel",
    "MineDetonationDict",
    "MinePlacementDict",
    "TeleportLandedDict",
    "UnknownContainerDict",
    "get_decode_level",
]
