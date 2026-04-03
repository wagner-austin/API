"""Protocol message TypedDicts.

This module contains all TypedDict definitions for protocol messages,
organized by category for easy reference.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# =============================================================================
# Text Messages (no XOR encoding)
# =============================================================================


class JoinConfirmDict(TypedDict):
    """Join confirmation (= message) - TEXT format.

    Format: =<team>|<join_date>|<name>|<rank>|<eq1>|<eq2>|<eq3>|<eq4>
    """

    msg_type: Literal[0x3D]
    team: int
    join_date: str
    name: str
    rank: int
    equipment: list[int]


class WorldInfoDict(TypedDict):
    """World/map info (+ message) - TEXT format.

    Format: +<id>|<name>|<field>|<flags>|<team>|<mode>|<image>|<year>
    """

    msg_type: Literal[0x2B]
    world_id: int
    name: str
    field_id: int
    flags: list[int]
    team: int
    mode: str
    image: str
    year: int


# =============================================================================
# Combat Messages
# =============================================================================


class ShootEventDict(TypedDict):
    """Shooting/hit event (S message)."""

    msg_type: Literal[0x53]
    shooter_id: int
    target_x: int
    target_y: int
    projectile_x: int
    projectile_y: int
    fuel: int
    weapon: int
    ammo: int
    friendly_fire: bool


class HitConfirmationDict(TypedDict):
    """Fire confirmation (0x2E len=12, subtype 0x7E)."""

    msg_type: Literal[0x2E]
    target_y: int
    target_x: int


class DeactivationDict(TypedDict):
    """Kill/deactivation event (0x41 'A' message)."""

    msg_type: Literal[0x41]
    victim_id: int
    killer_id: int
    rank: int
    points: int


class MinePlacementDict(TypedDict):
    """Mine placement (K message)."""

    msg_type: Literal[0x4B]
    mine_type: int
    tank_id: int
    positions: list[tuple[int, int]]


class MineDetonationDict(TypedDict):
    """Mine detonation (E message)."""

    msg_type: Literal[0x45]
    positions: list[tuple[int, int]]


# =============================================================================
# Movement Messages
# =============================================================================


class MovementDict(TypedDict):
    """Movement path (0x47 'G' message)."""

    msg_type: Literal[0x47]
    tank_id: int
    start_x: int
    start_y: int
    direction: int
    flag: int
    leaderboard_position: int
    waypoints: list[tuple[int, int]]


class MovementResponseDict(TypedDict):
    """Movement response (0x3D '=' binary message)."""

    msg_type: Literal[0x3D]
    team: int
    tank_id: int
    x: int
    y: int
    direction: int
    rank: int
    leaderboard_position: int


# =============================================================================
# Resource Messages (Fuel, Equipment)
# =============================================================================


class FuelGainDict(TypedDict):
    """Fuel gain event (D message). fuel_total is the new absolute fuel level."""

    msg_type: Literal[0x44]
    fuel_total: int
    is_free: bool


class FuelDepositDict(TypedDict):
    """Fuel deposit event (d message). fuel_total is the new absolute fuel level."""

    msg_type: Literal[0x64]
    fuel_total: int


class InventoryDict(TypedDict):
    """Inventory display (I message)."""

    msg_type: Literal[0x49]
    show: bool
    alternate: bool
    counts: list[int]
    enabled: list[bool]


class EquipmentGainDict(TypedDict):
    """Equipment gain (g message)."""

    msg_type: Literal[0x67]
    show_message: bool
    gained: list[int]


class EquipmentToggleDict(TypedDict):
    """Equipment toggle (t message)."""

    msg_type: Literal[0x74]
    enabled: list[bool]


# =============================================================================
# Radar Messages
# =============================================================================


class RadarResultDict(TypedDict):
    """Radar scan result (F message)."""

    msg_type: Literal[0x46]
    detection_type: int
    found: bool


class EnemyDetectionDict(TypedDict):
    """Enemy detection (H message)."""

    msg_type: Literal[0x48]
    tank_id: int
    x: int
    y: int
    rank: int
    team: int


class RadarContainerDict(TypedDict):
    """Container entry in radar scan result.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        volume: Fuel volume (0-32767), or -1 for equipment.
    """

    x: int
    y: int
    volume: int


class RadarMineDict(TypedDict):
    """Mine entry in radar scan result.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team that placed the mine (0=red, 1=purple, 2=blue, 3=orange).
    """

    x: int
    y: int
    team: int


class RadarScanResultDict(TypedDict):
    """Radar scan result (tunneled 0x2E -> 0x4F).

    Contains containers (fuel/equipment) and mines discovered by radar.

    Attributes:
        msg_type: Message type (0x4F).
        containers: List of container entries.
        mines: List of mine entries.
    """

    msg_type: Literal[0x4F]
    containers: list[RadarContainerDict]
    mines: list[RadarMineDict]


# =============================================================================
# Tank Messages
# =============================================================================


class TankInfoDict(TypedDict):
    """Tank info (0x21 '!' message).

    NOTE: This message does NOT contain the tank's current rank!
    Use 0x3E TankStatus for own rank, or 0x2E short for other tanks.
    """

    msg_type: Literal[0x21]
    tank_id: int
    team: int
    decoration_state: bytes
    score: int
    name: str


class TankEntryDict(TypedDict):
    """Tank entry (( message)."""

    msg_type: Literal[0x28]
    tank_id: int
    x: int
    y: int
    name: str


class TankExitDict(TypedDict):
    """Tank exit (0x58 'X' message)."""

    msg_type: Literal[0x58]
    tank_id: int


class TankStatusSyncDict(TypedDict):
    """Tank status sync (0x2E message).

    Long format (13 bytes, subtype 0x03) for self.
    Short format (8 bytes, subtype 0x01) for other tanks.
    The damage_state controls how dark the enemy tank name appears.
    """

    msg_type: Literal[0x2E]
    subtype: int
    tank_id: int
    damage_state: int
    rank: int
    flags: bytes
    leaderboard_position: int
    fuel: int | None


class TankStatusDict(TypedDict):
    """Full tank status (0x3E '>' message)."""

    msg_type: Literal[0x3E]
    team: int
    rank: int
    tank_id: int
    decoration_state: bytes
    leaderboard_score: int
    leaderboard_position: int
    name: str


# =============================================================================
# World Messages (Viewport, Terrain, Sync)
# =============================================================================


class SyncDict(TypedDict):
    """Sync/heartbeat (0x3F '?' message)."""

    msg_type: Literal[0x3F]


class CacheUpdateDict(TypedDict):
    """Cache-only tile patch (0x43 'C' message)."""

    msg_type: Literal[0x43]
    updates: list[tuple[int, int, int]]


class OverlayUpdateDict(TypedDict):
    """Overlay-only tile patch (0x40 '@' message)."""

    msg_type: Literal[0x40]
    updates: list[tuple[int, int, int]]


class CombinedTileUpdateDict(TypedDict):
    """Combined cache+overlay tile patch (0x4F 'O' message)."""

    msg_type: Literal[0x4F]
    cache_updates: list[tuple[int, int, int]]
    overlay_updates: list[tuple[int, int, int]]


class ViewportEntityDict(TypedDict):
    """Single tile row in viewport update."""

    col: int
    row: int
    cache_value: int
    overlay_value: int
    terrain_type: int


class ViewportUpdateDict(TypedDict):
    """Viewport/map update (0x5A 'Z' message)."""

    msg_type: Literal[0x5A]
    viewport_left: int
    viewport_top: int
    entities: list[ViewportEntityDict]


class TerrainUpdateDict(TypedDict):
    """Terrain type update (0x4A 'J' message)."""

    msg_type: Literal[0x4A]
    updates: list[tuple[int, int, int]]


class SupervisorDict(TypedDict):
    """Supervisor/promotion eligibility message (0x52 'R' message)."""

    msg_type: Literal[0x52]
    status: int
    reserved: int
    data: int


# =============================================================================
# Misc Messages
# =============================================================================


class ActionDoneDict(TypedDict):
    """Action completion marker (0x54 'T' message)."""

    msg_type: Literal[0x54]


class ChatMessageDict(TypedDict):
    """Chat message (M message)."""

    msg_type: Literal[0x4D]
    sender_id: int
    message_type: int
    x: int | None
    y: int | None


class StatisticsDict(TypedDict):
    """Statistics display (V message)."""

    msg_type: Literal[0x56]
    playtime_hours: int
    playtime_minutes: int
    playtime_seconds: int
    destroyed: int
    deactivated: int
    score: int


class ActiveForcesDict(TypedDict):
    """Active forces count (* message)."""

    msg_type: Literal[0x2A]
    team_counts: list[int]


__all__ = [
    "ActionDoneDict",
    "ActiveForcesDict",
    "CacheUpdateDict",
    "ChatMessageDict",
    "CombinedTileUpdateDict",
    "DeactivationDict",
    "EnemyDetectionDict",
    "EquipmentGainDict",
    "EquipmentToggleDict",
    "FuelDepositDict",
    "FuelGainDict",
    "HitConfirmationDict",
    "InventoryDict",
    "JoinConfirmDict",
    "MineDetonationDict",
    "MinePlacementDict",
    "MovementDict",
    "MovementResponseDict",
    "OverlayUpdateDict",
    "RadarContainerDict",
    "RadarMineDict",
    "RadarResultDict",
    "RadarScanResultDict",
    "ShootEventDict",
    "StatisticsDict",
    "SupervisorDict",
    "SyncDict",
    "TankEntryDict",
    "TankExitDict",
    "TankInfoDict",
    "TankStatusDict",
    "TankStatusSyncDict",
    "TerrainUpdateDict",
    "ViewportEntityDict",
    "ViewportUpdateDict",
    "WorldInfoDict",
]
