"""Sniffer configuration constants and message signatures.

This module contains configuration defaults and protocol signature
documentation for the WebSocket sniffer.
"""

from __future__ import annotations

# Default configuration constants
DEFAULT_TARGET_URL = "https://tankpit.com"
DEFAULT_OUTPUT_PATH = "capture_session.json"
DEFAULT_CAPTURE_DURATION_MS = 0  # 0 = indefinite (wait until browser closed)

# Message type byte -> display name mapping
MSG_TYPE_NAMES: dict[int, str] = {
    0x21: "TankInfo",
    0x28: "TankJoin",
    0x29: "TankLeave",
    0x2E: "TankStatus",
    0x3D: "MoveResponse",
    0x3E: "TankStatus",
    0x41: "Deactivation",
    0x43: "Container",
    0x44: "FuelDeposit",
    0x45: "MineDetonate",
    0x46: "RadarAck",
    0x47: "Movement",
    0x48: "MovementShort",
    0x49: "ItemPickup",
    0x4A: "TerrainUpdate",
    0x4B: "MinePlace",
    0x4C: "WorldEntry",
    0x4D: "PlayerList",
    0x4F: "RadarResult",
    0x52: "Supervisor",
    0x53: "Shooting",
    0x54: "ActionDone",
    0x56: "Statistics",
    0x58: "TankExit",
    0x5A: "ViewportUpdate",
    0x64: "FuelDeposit",
    0x67: "EquipGain",
    0x74: "EquipToggle",
}

# Message type categories for formatting dispatch
COMBAT_MSG_TYPES: frozenset[int] = frozenset({0x53, 0x41})
TANK_MSG_TYPES: frozenset[int] = frozenset({0x28, 0x58, 0x2E, 0x3E, 0x21, 0x47, 0x3D, 0x48})
RESOURCE_MSG_TYPES: frozenset[int] = frozenset({0x44, 0x64, 0x49, 0x43})
POSITION_MSG_TYPES: frozenset[int] = frozenset({0x4B, 0x45})
RADAR_MSG_TYPES: frozenset[int] = frozenset({0x46, 0x4F, 0x5A})
MISC_MSG_TYPES: frozenset[int] = frozenset({0x67, 0x74, 0x56, 0x52, 0x4D})

# Text message type bytes (ASCII chars that indicate text, not binary)
TEXT_MESSAGE_TYPES: frozenset[int] = frozenset({0x3D, 0x2B, 0x24, 0x2A, 0x25, 0x2D})

# Message type -> minimum data length (from protocol.py _require_* calls)
MSG_MIN_LENGTHS: dict[int, int] = {
    ord("S"): 12,  # ShootEvent
    ord("A"): 5,  # Deactivation
    ord("K"): 4,  # MinePlacement
    ord("E"): 0,  # MineDetonation (no minimum)
    ord("D"): 3,  # FuelGain
    ord("d"): 2,  # FuelDeposit
    ord("I"): 6,  # Inventory
    ord("g"): 6,  # EquipmentGain
    ord("t"): 5,  # EquipmentToggle
    ord("F"): 2,  # RadarResult
    ord("H"): 6,  # EnemyDetection
    ord("O"): 2,  # RadarScanResult
    ord("("): 10,  # TankEntry
    ord("X"): 2,  # TankExit
    ord("."): 1,  # 0x2E container (TankStatusSync or tunneled message)
    ord(">"): 13,  # TankStatus
    ord("!"): 10,  # TankInfo
    ord("G"): 9,  # Movement
    ord("="): 11,  # MovementResponse
    ord("Z"): 2,  # ViewportUpdate
    ord("J"): 0,  # TerrainUpdate (no minimum)
    ord("?"): 0,  # Sync (no minimum)
    ord("C"): 4,  # Container
    ord("M"): 3,  # ChatMessage
    ord("V"): 16,  # Statistics
    ord("*"): 4,  # ActiveForces
    ord("R"): 3,  # Supervisor
    ord("T"): 0,  # ActionDone (no minimum)
}

# Known decoded signatures with understanding level
# FULL = complete decoder, PARTIAL = key fields known, IDENTIFIED = type known
DECODED_SIGS: dict[int, tuple[str, str]] = {
    # Binary control messages (0x00-0x1F)
    0x00: ("sync_state", "IDENTIFIED"),
    0x01: ("heartbeat", "IDENTIFIED"),
    0x04: ("position_update", "IDENTIFIED"),
    0x08: ("entity_state", "IDENTIFIED"),
    0x0E: ("tick_update", "IDENTIFIED"),
    0x14: ("world_state", "IDENTIFIED"),
    0x15: ("spawn_state", "IDENTIFIED"),
    0x1D: ("combat_state", "IDENTIFIED"),
    # ASCII message types (0x20+)
    0x21: ("tank_info", "FULL"),
    0x22: ("entity_position", "IDENTIFIED"),  # '"' 13-byte position update
    0x28: ("tank_join", "IDENTIFIED"),
    0x29: ("tank_leave", "IDENTIFIED"),
    0x2B: ("promotion", "FULL"),
    0x2D: ("world_entity", "IDENTIFIED"),  # '-' 16-byte entity state
    0x2E: ("tank_status_sync", "PARTIAL"),
    0x2F: ("player_update", "IDENTIFIED"),
    0x31: ("top10_list", "IDENTIFIED"),  # '1' MSG_TOP10
    0x32: ("top10_extended", "IDENTIFIED"),  # '2' similar to top10
    0x33: ("score_update", "IDENTIFIED"),  # '3'
    0x3D: ("movement", "FULL"),
    0x3E: ("tank_status", "PARTIAL"),
    0x3F: ("position", "FULL"),
    0x40: ("mine_status", "IDENTIFIED"),  # '@' MSG_MINE_STATUS
    0x41: ("kill", "FULL"),
    0x43: ("container", "FULL"),
    0x45: ("mine_detonate", "FULL"),
    0x46: ("radar_ack", "FULL"),
    0x47: ("shooting", "FULL"),
    0x49: ("item_pickup", "FULL"),
    0x4A: ("terrain_update", "IDENTIFIED"),  # 'J' MSG_TERRAIN_UPDATE
    0x4B: ("mine_place", "FULL"),
    0x4C: ("tank_entry", "PARTIAL"),
    0x4D: ("player_list", "IDENTIFIED"),
    0x4F: ("deactivation", "FULL"),
    0x52: ("supervisor", "PARTIAL"),
    0x53: ("tank_move", "FULL"),
    0x54: ("tank_shoot", "FULL"),
    0x56: ("statistics", "FULL"),
    0x58: ("tank_exit", "FULL"),
    0x5A: ("viewport_update", "PARTIAL"),
    0x5F: ("action_event", "IDENTIFIED"),  # '_' 11-byte action/event
    0x64: ("fuel_deposit", "FULL"),
    0x66: ("fuel_state", "IDENTIFIED"),  # 'f' - lowercase variant
    0x67: ("equip_gain", "FULL"),
    0x69: ("inventory_state", "IDENTIFIED"),  # 'i' - lowercase variant
    0x74: ("equip_toggle", "FULL"),
    0x78: ("tank_disconnect", "IDENTIFIED"),  # 'x' - lowercase variant
    0x79: ("entity_spawn", "IDENTIFIED"),  # 'y'
    0x7A: ("zone_update", "IDENTIFIED"),  # 'z' - lowercase variant
}

# Rank number -> name mapping
RANK_NAMES: tuple[str, ...] = (
    "recruit",
    "private",
    "corporal",
    "sergeant",
    "lieutenant",
    "captain",
    "major",
    "general",
)

# Damage state -> description
DAMAGE_NAMES: tuple[str, ...] = ("full", "light", "medium", "critical")

# Team number -> name
TEAM_NAMES: tuple[str, ...] = ("red", "blue", "green", "purple")


__all__ = [
    "COMBAT_MSG_TYPES",
    "DAMAGE_NAMES",
    "DECODED_SIGS",
    "DEFAULT_CAPTURE_DURATION_MS",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TARGET_URL",
    "MISC_MSG_TYPES",
    "MSG_MIN_LENGTHS",
    "MSG_TYPE_NAMES",
    "POSITION_MSG_TYPES",
    "RADAR_MSG_TYPES",
    "RANK_NAMES",
    "RESOURCE_MSG_TYPES",
    "TANK_MSG_TYPES",
    "TEAM_NAMES",
    "TEXT_MESSAGE_TYPES",
]
