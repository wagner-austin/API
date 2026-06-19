"""Protocol message TypedDicts.

This module contains all TypedDict definitions for protocol messages,
organized by category for easy reference.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot.container.types import ContainerMessage

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
    """Shooting / hit event (0x53 'S' message).

    Layout from tpclient.js Gg.h (V.S), re-verified 2026-06-19 against
    real wire bytes from runs/bot/bot-20260619-050303 msg t+25.47s
    `53 02 15 05 b2 7d b2 7e b2 7e 01`:
      a[0]    = team byte (red=0, purple=1, blue=2, orange=3)
      a[1:3]  = shooter_id (LE u16)  -- the tank that fired
      a[3]    = source_x  -- shooter's tile X (live position)
      a[4]    = source_y  -- shooter's tile Y
      a[5]    = target_x  -- shot's landing tile X (homing's final tile)
      a[6]    = target_y  -- shot's landing tile Y
      a[7]    = unk1 (often duplicates target tile -- semantics TBD)
      a[8]    = unk2
      a[9]    = weapon (0=single, 1=dual, 2=missile, 3=homing)

    Prior decoder named a[3,4] as target and a[5,6] as projectile_start
    -- reversed. Prior names for a[7..9] as fuel/weapon/ammo were also
    wrong. Three-way validation (enemy src tracking, homing tgt tile,
    wire damage events) confirmed the corrected layout.

    Hit detection per JS Gg.prototype.h case 18: shot landed on a named
    tank tile -> hit. That tile lookup uses (target_x, target_y).
    """

    msg_type: Literal[0x53]
    team: int
    shooter_id: int
    source_x: int
    source_y: int
    target_x: int
    target_y: int
    unk1: int
    unk2: int
    weapon: int


class DeactivationDict(TypedDict):
    """Kill/deactivation event (0x41 'A' message).

    Layout from tpclient.js Pg.h (V.A), verified 2026-06-19:
      a[0]  = status byte
      a[1:3] = victim_id (LE u16)
      a[3]  = promo_eligible (1=earned extra points)
      a[4:6] = killer_id (LE u16)
      If killer_id >= 65530: mine kill (team = killer_id - 65530)
    """

    msg_type: Literal[0x41]
    status: int
    victim_id: int
    promo_eligible: bool
    killer_id: int
    is_mine_kill: bool


# 0x4B MinePlacement and 0x45 MineDetonation TypedDicts live in
# tankpit_bot.container.types -- both are container-subtype messages
# that never arrive standalone. Protocol versions deleted 2026-06-19.


# =============================================================================
# Movement Messages
# =============================================================================


class MovementDict(TypedDict):
    """Movement path (0x47 'G' message).

    Layout from tpclient.js Lg.h (V.G), verified 2026-06-19:
      a[0:2]  = tank_id (LE u16)
      a[2]    = start_x
      a[3]    = start_y
      a[4]    = direction
      a[5]    = damage_state (assigned to b.u in Lg.prototype.h; NOT damage_state)
      a[6:9]  = lb_score (24-bit BE)
      a[9]    = rank (assigned to b.l in Lg.prototype.h)
      a[10]   = animation flag (passed to Re constructor, not tank state)
      a[11]   = is_carrying (1=true)
      a[12:]  = waypoints (direction chars)
    """

    msg_type: Literal[0x47]
    tank_id: int
    start_x: int
    start_y: int
    direction: int
    damage_state: int
    lb_score: int
    rank: int
    flag: int
    is_carrying: bool
    waypoints: list[tuple[int, int]]


class MovementResponseDict(TypedDict):
    """Movement response (0x3D '=' binary message).

    Layout from tpclient.js Mg.h (V["="]):
      a[0]    = team
      a[1:3]  = tank_id (LE u16)
      a[3]    = x
      a[4]    = y
      a[5]    = direction
      a[6]    = damage_state (assigned to b.u; NOT damage_state)
      a[7]    = rank
      a[8:11] = lb_score (24-bit BE)
      a[11]   = carrying flag
    """

    msg_type: Literal[0x3D]
    team: int
    tank_id: int
    x: int
    y: int
    direction: int
    damage_state: int
    rank: int
    lb_score: int
    carrying: int


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

    Trace-verified from tpclient.js Tf.h (line 3896-3901):
      a[0]    = team (a[0] & 255)
      a[1:3]  = tank_id (LE u16)
      a[3:7]  = decoration_state (4 bytes, decoded by yg() into 9 x 2-bit slots)
      a[7:10] = persistent_tank_id (24-bit BE, sets a.aa for profile links)
      a[10:]  = name (UTF-8 string)

    NOTE: This message does NOT contain the tank's current rank.
    """

    msg_type: Literal[0x21]
    tank_id: int
    team: int
    decoration_state: bytes
    persistent_tank_id: int
    name: str


class TankEntryDict(TypedDict):
    """Tank entry (( message).

    Layout from tpclient.js Uf.h (V["("]), verified 2026-06-19:
      a[0]   = flags (255=known tank)
      a[1:3] = tank_id (LE u16)
      a[3]   = packed byte: team(bits 0-1), damage_state(bits 2-3), rank(bits 4-7)
      a[4:7] = score (24-bit BE)
      a[7]   = x position
      a[8]   = y position
    """

    msg_type: Literal[0x28]
    team: int
    tank_id: int
    rank: int
    damage_state: int
    score: int
    x: int
    y: int


class TankExitDict(TypedDict):
    """Tank exit (0x58 'X' message)."""

    msg_type: Literal[0x58]
    tank_id: int


class TankStatusSyncDict(TypedDict):
    """Tank status sync (0x2E message).

    Layout from tpclient.js Og.h (V["."]), verified 2026-06-19:
      a[0]    = team (subtype)
      a[1:3]  = tank_id (LE u16)
      a[3]    = damage_state (b.u; dual-purpose: rank_category on init, damage during gameplay)
      a[4]    = rank (b.l)
      a[5:8]  = lb_score (24-bit BE)
      a[8]    = promo_state (if long form)
      a[9]    = has_fuel_bar (if long form)
      a[10:12] = fuel (LE u16, if long form)
    """

    msg_type: Literal[0x2E]
    subtype: int
    tank_id: int
    damage_state: int
    rank: int
    lb_score: int
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
    """Command failure response (0x52 'R' message).

    Trace-verified from tpclient.js xg.h (line 4317-4322):
      a[0] = reset_action (1=reset to idle)
      a[1] = close_map (1=close map view)
      a[2] = error_code (index into Gb[] error strings; 128+=custom text)
    """

    msg_type: Literal[0x52]
    reset_action: int
    close_map: int
    error_code: int


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
    "BinaryMessage",
    "CacheUpdateDict",
    "ChatMessageDict",
    "CombinedTileUpdateDict",
    "DeactivationDict",
    "DecodedMessage",
    "EnemyDetectionDict",
    "EquipmentGainDict",
    "EquipmentToggleDict",
    "FuelDepositDict",
    "FuelGainDict",
    "InventoryDict",
    "JoinConfirmDict",
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
    "TextMessage",
    "ViewportEntityDict",
    "ViewportUpdateDict",
    "WorldInfoDict",
]


# =============================================================================
# Message union types
# =============================================================================

TextMessage = JoinConfirmDict | WorldInfoDict

BinaryMessage = (
    ShootEventDict
    | DeactivationDict
    | FuelGainDict
    | FuelDepositDict
    | RadarResultDict
    | EnemyDetectionDict
    | InventoryDict
    | EquipmentGainDict
    | EquipmentToggleDict
    | RadarScanResultDict
    | MovementDict
    | TankInfoDict
    | MovementResponseDict
    | SyncDict
    | CacheUpdateDict
    | OverlayUpdateDict
    | CombinedTileUpdateDict
    | TankEntryDict
    | TankExitDict
    | ActionDoneDict
    | ChatMessageDict
    | StatisticsDict
    | ActiveForcesDict
    | TankStatusSyncDict
    | TankStatusDict
    | SupervisorDict
    | TerrainUpdateDict
    | ViewportUpdateDict
    | ContainerMessage
)

DecodedMessage = TextMessage | BinaryMessage
