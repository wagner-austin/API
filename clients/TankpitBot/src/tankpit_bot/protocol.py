"""Tankpit protocol decoder - decodes all server→client messages.

Based on analysis of tpclient JS handlers.
All messages use XOR encoding with session-specific table.

Container messages (0x2E) use structure-based decoding via container_decoder.py
since XOR subtype bytes vary per session.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal, TypedDict

from tankpit_bot.container_decoder import (
    ContainerMessage,
    decode_container_message,
)


class Rank(IntEnum):
    """Tank rank levels (0-7)."""

    RECRUIT = 0
    PRIVATE = 1
    CORPORAL = 2
    SERGEANT = 3
    LIEUTENANT = 4
    CAPTAIN = 5
    MAJOR = 6
    GENERAL = 7


# Starting fuel by rank (resets on death/respawn)
RANK_FUEL: dict[Rank, int] = {
    Rank.RECRUIT: 1000,
    Rank.PRIVATE: 1100,
    Rank.CORPORAL: 1200,
    Rank.SERGEANT: 1300,
    Rank.LIEUTENANT: 1400,
    Rank.CAPTAIN: 1500,
    Rank.MAJOR: 1600,
    Rank.GENERAL: 1700,
}


class Team(IntEnum):
    """Team colors (0-3)."""

    RED = 0
    PURPLE = 1
    BLUE = 2
    ORANGE = 3


class Equipment(IntEnum):
    """Equipment types (0-4)."""

    ARMOR_SHIELD = 0
    DUAL_SHOT = 1
    MISSILE_SHOT = 2
    HOMING_SHOT = 3
    EXTRA_RADAR = 4


class TerrainType(IntEnum):
    """Terrain types from JS rendering code."""

    GROUND = 0
    ROCK_A = 1
    ROCK_B = 2
    ROCK_AB = 3
    FERRY = 5
    FERRY_ROCK = 7


# Message type characters
MSG_TANK_STATS = ord(".")
MSG_TANK_INFO = ord("!")
MSG_TANK_POS = ord("=")
MSG_MOVEMENT = ord("G")
MSG_SHOOT = ord("S")
MSG_DEACTIVATE = ord("A")
MSG_FUEL_GAIN = ord("D")
MSG_FUEL_DEPOSIT = ord("d")
MSG_RADAR_RESULT = ord("F")
MSG_ENEMY_DETECT = ord("H")
MSG_INVENTORY = ord("I")
MSG_EQUIP_GAIN = ord("g")
MSG_EQUIP_TOGGLE = ord("t")
MSG_MINE_PLACE = ord("K")
MSG_MINE_DETONATE = ord("E")
MSG_CHAT = ord("M")
MSG_TANK_REMOVE = ord("X")
MSG_MAP_UPDATE = ord("Z")
MSG_TANK_ENTRY = ord("(")
MSG_TANK_EXIT = ord(")")
MSG_TANK_STATUS = ord(">")
MSG_PROMOTION = ord("+")
MSG_DECORATION = ord("N")
MSG_STATISTICS = ord("V")
MSG_ACTIVE_FORCES = ord("*")
MSG_ACTIVE_PLAYERS = ord("/")
MSG_TOP10 = ord("1")
MSG_TILE_UPDATE = ord("O")
MSG_BUILD_PICKUP = ord("B")
MSG_ACTION_DONE = ord("T")
MSG_MINE_STATUS = ord("@")
MSG_TERRAIN_UPDATE = ord("J")
MSG_PING = ord("`")
MSG_DISCONNECT = ord("~")
MSG_SUPERVISOR = ord("R")
MSG_TANK_STATUS_FULL = ord(">")
MSG_VIEWPORT = ord("Z")
MSG_SYNC = ord("?")
MSG_CONTAINER = ord("C")
MSG_MOVE_RESPONSE = ord("=")


def _x16(low: int, high: int) -> int:
    """Combine two bytes into 16-bit value (JS X function).

    Args:
        low: Low byte (0-255).
        high: High byte (0-255).

    Returns:
        Combined 16-bit unsigned value.
    """
    return (low & 255) + 256 * (high & 255)


def _x24(a: int, b: int, c: int) -> int:
    """Combine three bytes into 24-bit value (big-endian).

    Args:
        a: High byte.
        b: Middle byte.
        c: Low byte.

    Returns:
        Combined 24-bit unsigned value.
    """
    return 256 * (256 * a + b) + c


class DecodeError(Exception):
    """Raised when message decoding fails."""


def _require_min_length(data: bytes, min_len: int, msg_name: str) -> None:
    """Validate minimum data length.

    Args:
        data: Raw bytes to validate.
        min_len: Minimum required length.
        msg_name: Message type name for error messages.

    Raises:
        DecodeError: If data is too short.
    """
    if len(data) < min_len:
        raise DecodeError(f"{msg_name}: expected >= {min_len} bytes, got {len(data)}")


def _require_exact_length(data: bytes, exact_len: int, msg_name: str) -> None:
    """Validate exact data length.

    Args:
        data: Raw bytes to validate.
        exact_len: Required exact length.
        msg_name: Message type name for error messages.

    Raises:
        DecodeError: If data length doesn't match.
    """
    if len(data) != exact_len:
        raise DecodeError(f"{msg_name}: expected {exact_len} bytes, got {len(data)}")


def _require_prefix(text: str, prefix: str, msg_name: str) -> None:
    """Validate text starts with expected prefix.

    Args:
        text: Text to validate.
        prefix: Required prefix.
        msg_name: Message type name for error messages.

    Raises:
        DecodeError: If prefix is missing.
    """
    if not text.startswith(prefix):
        raise DecodeError(f"{msg_name}: expected prefix '{prefix}'")


def _require_parts(parts: list[str], min_parts: int, msg_name: str) -> None:
    """Validate minimum number of pipe-separated parts.

    Args:
        parts: Split parts to validate.
        min_parts: Minimum required parts.
        msg_name: Message type name for error messages.

    Raises:
        DecodeError: If not enough parts.
    """
    if len(parts) < min_parts:
        raise DecodeError(f"{msg_name}: expected >= {min_parts} parts, got {len(parts)}")


# --- TypedDicts for each message type ---


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


def decode_join_confirm(data: bytes) -> JoinConfirmDict:
    """Decode join confirmation from raw message body.

    Args:
        data: Raw message body (including = prefix).

    Returns:
        Decoded join confirmation.

    Raises:
        DecodeError: If decoding fails.
    """
    text = data.decode("utf-8", errors="replace")
    _require_prefix(text, "=", "JoinConfirm")
    parts = text[1:].split("|")
    _require_parts(parts, 4, "JoinConfirm")
    return JoinConfirmDict(
        msg_type=0x3D,
        team=int(parts[0]),
        join_date=parts[1],
        name=parts[2],
        rank=int(parts[3]),
        equipment=[int(p) for p in parts[4:8] if p.isdigit()],
    )


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


def decode_world_info(data: bytes) -> WorldInfoDict:
    """Decode world info from raw message body.

    Args:
        data: Raw message body (including + prefix).

    Returns:
        Decoded world info.

    Raises:
        DecodeError: If decoding fails.
    """
    text = data.decode("utf-8", errors="replace")
    _require_prefix(text, "+", "WorldInfo")
    parts = text[1:].split("|")
    _require_parts(parts, 8, "WorldInfo")
    flags_str = parts[3].split(",")
    return WorldInfoDict(
        msg_type=0x2B,
        world_id=int(parts[0]),
        name=parts[1],
        field_id=int(parts[2]),
        flags=[int(f) for f in flags_str if f.isdigit()],
        team=int(parts[4]),
        mode=parts[5],
        image=parts[6],
        year=int(parts[7]) if parts[7].isdigit() else 0,
    )


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


def decode_shoot_event(data: bytes) -> ShootEventDict:
    """Decode shooting event from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x53 prefix).

    Returns:
        Decoded shoot event.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 12, "ShootEvent")
    return ShootEventDict(
        msg_type=0x53,
        shooter_id=_x16(data[0], data[1]),
        target_x=data[2],
        target_y=data[3],
        projectile_x=data[4],
        projectile_y=data[5],
        fuel=_x24(data[6], data[7], data[8]),
        weapon=data[9],
        ammo=data[10],
        friendly_fire=data[11] == 1,
    )


class HitConfirmationDict(TypedDict):
    """Fire confirmation (0x2E len=12, subtype 0x7E)."""

    msg_type: Literal[0x2E]
    target_y: int
    target_x: int


def decode_hit_confirmation(data: bytes, xor_table: bytes) -> HitConfirmationDict:
    """Decode HIT message from raw body.

    Args:
        data: Raw message body (12 bytes, starts with 0x2E).
        xor_table: XOR table for decoding.

    Returns:
        Decoded hit confirmation.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_exact_length(data, 12, "HitConfirmation")
    if data[0] != 0x2E:
        raise DecodeError("HitConfirmation: expected 0x2E prefix")

    decoded = bytearray(len(data) - 1)
    for i in range(len(decoded)):
        decoded[i] = data[i + 1] ^ xor_table[i]

    return HitConfirmationDict(
        msg_type=0x2E,
        target_y=decoded[5],
        target_x=decoded[6],
    )


class DeactivationDict(TypedDict):
    """Kill/deactivation event (0x41 'A' message)."""

    msg_type: Literal[0x41]
    victim_id: int
    killer_id: int
    rank: int
    points: int


def decode_deactivation(data: bytes) -> DeactivationDict:
    """Decode deactivation event from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x41 prefix).

    Returns:
        Decoded deactivation.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 5, "Deactivation")
    return DeactivationDict(
        msg_type=0x41,
        victim_id=_x16(data[0], data[1]),
        killer_id=_x16(data[2], data[3]),
        rank=data[4],
        points=_x16(data[5], data[6]) if len(data) >= 7 else 0,
    )


class FuelGainDict(TypedDict):
    """Fuel gain event (D message)."""

    msg_type: Literal[0x44]
    amount: int
    is_free: bool


def decode_fuel_gain(data: bytes) -> FuelGainDict:
    """Decode fuel gain from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded fuel gain.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 3, "FuelGain")
    return FuelGainDict(
        msg_type=0x44,
        amount=_x16(data[0], data[1]),
        is_free=data[2] == 0,
    )


class FuelDepositDict(TypedDict):
    """Fuel deposit event (d message)."""

    msg_type: Literal[0x64]
    amount: int


def decode_fuel_deposit(data: bytes) -> FuelDepositDict:
    """Decode fuel deposit from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded fuel deposit.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 2, "FuelDeposit")
    return FuelDepositDict(msg_type=0x64, amount=_x16(data[0], data[1]))


class RadarResultDict(TypedDict):
    """Radar scan result (F message)."""

    msg_type: Literal[0x46]
    detection_type: int
    found: bool


def decode_radar_result(data: bytes) -> RadarResultDict:
    """Decode radar result from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded radar result.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 2, "RadarResult")
    return RadarResultDict(
        msg_type=0x46,
        detection_type=data[0],
        found=data[1] == 1,
    )


class EnemyDetectionDict(TypedDict):
    """Enemy detection (H message)."""

    msg_type: Literal[0x48]
    tank_id: int
    x: int
    y: int
    rank: int
    team: int


def decode_enemy_detection(data: bytes) -> EnemyDetectionDict:
    """Decode enemy detection from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded enemy detection.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 6, "EnemyDetection")
    return EnemyDetectionDict(
        msg_type=0x48,
        tank_id=_x16(data[0], data[1]),
        x=data[2],
        y=data[3],
        rank=data[4],
        team=data[5],
    )


class InventoryDict(TypedDict):
    """Inventory display (I message)."""

    msg_type: Literal[0x49]
    show: bool
    alternate: bool
    counts: list[int]
    enabled: list[bool]


def decode_inventory(data: bytes) -> InventoryDict:
    """Decode inventory from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded inventory.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 6, "Inventory")
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


class EquipmentGainDict(TypedDict):
    """Equipment gain (g message)."""

    msg_type: Literal[0x67]
    show_message: bool
    gained: list[int]


def decode_equipment_gain(data: bytes) -> EquipmentGainDict:
    """Decode equipment gain from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded equipment gain.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 6, "EquipmentGain")
    return EquipmentGainDict(
        msg_type=0x67,
        show_message=data[0] == 1,
        gained=[data[i + 1] for i in range(5)],
    )


class EquipmentToggleDict(TypedDict):
    """Equipment toggle (t message)."""

    msg_type: Literal[0x74]
    enabled: list[bool]


def decode_equipment_toggle(data: bytes) -> EquipmentToggleDict:
    """Decode equipment toggle from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded equipment toggle.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 5, "EquipmentToggle")
    return EquipmentToggleDict(msg_type=0x74, enabled=[data[i] == 1 for i in range(5)])


class MinePlacementDict(TypedDict):
    """Mine placement (K message)."""

    msg_type: Literal[0x4B]
    mine_type: int
    tank_id: int
    positions: list[tuple[int, int]]


def decode_mine_placement(data: bytes) -> MinePlacementDict:
    """Decode mine placement from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded mine placement.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 4, "MinePlacement")
    mine_type = data[0]
    tank_id = _x16(data[1], data[2])
    count = data[3]
    positions: list[tuple[int, int]] = []
    idx = 4
    for _ in range(count):
        if idx + 1 >= len(data):
            break
        positions.append((data[idx], data[idx + 1]))
        idx += 2
    return MinePlacementDict(
        msg_type=0x4B, mine_type=mine_type, tank_id=tank_id, positions=positions
    )


class MineDetonationDict(TypedDict):
    """Mine detonation (E message)."""

    msg_type: Literal[0x45]
    positions: list[tuple[int, int]]


def decode_mine_detonation(data: bytes) -> MineDetonationDict:
    """Decode mine detonation from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded mine detonation.
    """
    positions: list[tuple[int, int]] = []
    for i in range(0, len(data) - 1, 2):
        positions.append((data[i], data[i + 1]))
    return MineDetonationDict(msg_type=0x45, positions=positions)


class RadarScanResultDict(TypedDict):
    """Radar scan result (0x4F 'O' message)."""

    msg_type: Literal[0x4F]
    entities: list[tuple[int, int, int]]


def decode_radar_scan_result(data: bytes) -> RadarScanResultDict:
    """Decode radar scan result from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded radar scan result with (x, y, value) tuples.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 2, "RadarScanResult")
    count = data[0]
    entities: list[tuple[int, int, int]] = []
    idx = 2
    for _ in range(count):
        if idx + 4 > len(data):
            break
        x = data[idx]
        y = data[idx + 1]
        val = data[idx + 2] | (data[idx + 3] << 8)
        if val >= 0x8000:
            val -= 0x10000
        entities.append((x, y, val))
        idx += 4
    return RadarScanResultDict(msg_type=0x4F, entities=entities)


class MovementDict(TypedDict):
    """Movement path (0x47 'G' message)."""

    msg_type: Literal[0x47]
    tank_id: int
    start_x: int
    start_y: int
    direction: int
    flag: int
    fuel: int
    waypoints: list[tuple[int, int]]


def decode_movement(data: bytes) -> MovementDict:
    """Decode movement from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded movement.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 9, "Movement")
    return MovementDict(
        msg_type=0x47,
        tank_id=_x16(data[0], data[1]),
        start_x=data[2],
        start_y=data[3],
        direction=data[4],
        flag=data[5],
        fuel=_x24(data[6], data[7], data[8]),
        waypoints=[],
    )


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


def decode_tank_info(data: bytes) -> TankInfoDict:
    """Decode tank info from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x21 prefix).

    Returns:
        Decoded tank info.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 10, "TankInfo")
    team = data[0]
    tank_id = _x16(data[1], data[2])
    decoration_state = bytes(data[3:7])
    score = 256 * (256 * data[7] + data[8]) + data[9] if len(data) >= 10 else 0
    name = data[10:].decode("utf-8", errors="replace") if len(data) > 10 else ""
    return TankInfoDict(
        msg_type=0x21,
        tank_id=tank_id,
        team=team,
        decoration_state=decoration_state,
        score=score,
        name=name,
    )


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


def decode_movement_response(data: bytes) -> MovementResponseDict:
    """Decode movement response from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x3D prefix).

    Returns:
        Decoded movement response.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 11, "MovementResponse")
    return MovementResponseDict(
        msg_type=0x3D,
        team=data[0],
        tank_id=_x16(data[1], data[2]),
        x=data[3],
        y=data[4],
        direction=data[5],
        rank=data[7],
        leaderboard_position=_x24(data[8], data[9], data[10]),
    )


class SyncDict(TypedDict):
    """Sync/heartbeat (0x3F '?' message)."""

    msg_type: Literal[0x3F]


def decode_sync(data: bytes) -> SyncDict:
    """Decode sync message.

    Args:
        data: XOR-decoded message body.

    Returns:
        Empty sync dict.
    """
    return SyncDict(msg_type=0x3F)


class ContainerDict(TypedDict):
    """Container fuel update (0x43 'C' message)."""

    msg_type: Literal[0x43]
    container_id: int
    fuel: int


def decode_container(data: bytes) -> ContainerDict:
    """Decode container from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x43 prefix).

    Returns:
        Decoded container.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 4, "Container")
    return ContainerDict(
        msg_type=0x43,
        container_id=_x16(data[0], data[1]),
        fuel=_x16(data[2], data[3]),
    )


class TankEntryDict(TypedDict):
    """Tank entry (( message)."""

    msg_type: Literal[0x28]
    tank_id: int
    x: int
    y: int
    name: str


def decode_tank_entry(data: bytes) -> TankEntryDict:
    """Decode tank entry from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded tank entry.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 10, "TankEntry")
    tank_id = data[0]
    x = _x16(data[1], data[2])
    y = data[3]
    name = data[10:].decode("utf-8", errors="replace") if len(data) > 10 else ""
    return TankEntryDict(msg_type=0x28, tank_id=tank_id, x=x, y=y, name=name)


class TankExitDict(TypedDict):
    """Tank exit (0x58 'X' message)."""

    msg_type: Literal[0x58]
    tank_id: int


def decode_tank_exit(data: bytes) -> TankExitDict:
    """Decode tank exit from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x58 prefix).

    Returns:
        Decoded tank exit.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 2, "TankExit")
    return TankExitDict(msg_type=0x58, tank_id=_x16(data[0], data[1]))


class ActionDoneDict(TypedDict):
    """Action completion marker (0x54 'T' message)."""

    msg_type: Literal[0x54]


def decode_action_done(data: bytes) -> ActionDoneDict:
    """Decode action done message.

    Args:
        data: XOR-decoded message body.

    Returns:
        Empty action done dict.
    """
    return ActionDoneDict(msg_type=0x54)


class ChatMessageDict(TypedDict):
    """Chat message (M message)."""

    msg_type: Literal[0x4D]
    sender_id: int
    message_type: int
    x: int | None
    y: int | None


def decode_chat_message(data: bytes) -> ChatMessageDict:
    """Decode chat message from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded chat message.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 3, "ChatMessage")
    return ChatMessageDict(
        msg_type=0x4D,
        sender_id=_x16(data[0], data[1]),
        message_type=data[2],
        x=data[3] if len(data) > 3 else None,
        y=data[4] if len(data) > 4 else None,
    )


class StatisticsDict(TypedDict):
    """Statistics display (V message)."""

    msg_type: Literal[0x56]
    playtime_hours: int
    playtime_minutes: int
    playtime_seconds: int
    destroyed: int
    deactivated: int
    score: int


def decode_statistics(data: bytes) -> StatisticsDict:
    """Decode statistics from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded statistics.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 16, "Statistics")
    return StatisticsDict(
        msg_type=0x56,
        playtime_hours=_x16(data[0], data[1]),
        playtime_minutes=data[2],
        playtime_seconds=data[3],
        destroyed=int.from_bytes(data[4:8], "little"),
        deactivated=int.from_bytes(data[8:12], "little"),
        score=int.from_bytes(data[12:16], "little"),
    )


class ActiveForcesDict(TypedDict):
    """Active forces count (* message)."""

    msg_type: Literal[0x2A]
    team_counts: list[int]


def decode_active_forces(data: bytes) -> ActiveForcesDict:
    """Decode active forces from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded active forces.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 4, "ActiveForces")
    return ActiveForcesDict(msg_type=0x2A, team_counts=[data[i] for i in range(4)])


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


def decode_tank_status_sync(data: bytes) -> TankStatusSyncDict:
    """Decode tank status sync from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x2E prefix).

    Returns:
        Decoded tank status sync.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 8, "TankStatusSync")
    subtype = data[0]
    tank_id = _x16(data[1], data[2])
    damage_state = data[3]
    rank = data[4]
    flags = bytes(data[5:8]) if len(data) > 7 else b""

    if len(data) >= 12:
        lb_pos = _x16(data[6], data[7])
        fuel: int | None = _x16(data[10], data[11])
    else:
        lb_pos = _x16(data[6], data[7]) if len(data) > 7 else 0
        fuel = None

    return TankStatusSyncDict(
        msg_type=0x2E,
        subtype=subtype,
        tank_id=tank_id,
        damage_state=damage_state,
        rank=rank,
        flags=flags,
        leaderboard_position=lb_pos,
        fuel=fuel,
    )


def decode_0x2e_message(data: bytes) -> ContainerMessage:
    """Decode 0x2E container message using structure-based matching.

    Uses container_decoder module which identifies messages by STRUCTURE
    (length, field positions) rather than subtype bytes, since XOR encoding
    with session-specific magic keys causes subtype values to vary.

    Args:
        data: XOR-decoded message body (without 0x2E prefix).

    Returns:
        Decoded container message as appropriate TypedDict.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    return decode_container_message(data)


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


def decode_tank_status(data: bytes) -> TankStatusDict:
    """Decode full tank status from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x3E prefix).

    Returns:
        Decoded tank status.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 13, "TankStatus")
    info_byte = data[0]
    team = info_byte & 0x03
    rank = (info_byte >> 4) & 0x07
    tank_id = _x16(data[1], data[2])
    decoration_state = bytes(data[3:7])
    lb_score = 256 * (256 * data[7] + data[8]) + data[9] if len(data) >= 10 else 0
    lb_pos = 256 * (256 * data[10] + data[11]) + data[12] if len(data) >= 13 else 0
    name = data[13:].decode("utf-8", errors="replace") if len(data) > 13 else ""
    return TankStatusDict(
        msg_type=0x3E,
        team=team,
        rank=rank,
        tank_id=tank_id,
        decoration_state=decoration_state,
        leaderboard_score=lb_score,
        leaderboard_position=lb_pos,
        name=name,
    )


# Supervisor status constants
SUPERVISOR_STATUS_PROMO_ELIGIBLE = 1
SUPERVISOR_STATUS_PROMO_KILL = 8
SUPERVISOR_STATUS_TEXT_FOLLOWS = 128


class SupervisorDict(TypedDict):
    """Supervisor/promotion eligibility message (0x52 'R' message)."""

    msg_type: Literal[0x52]
    status: int
    reserved: int
    data: int


def decode_supervisor(data: bytes) -> SupervisorDict:
    """Decode supervisor message from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x52 prefix).

    Returns:
        Decoded supervisor message.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 3, "Supervisor")
    return SupervisorDict(
        msg_type=0x52,
        status=data[0],
        reserved=data[1],
        data=data[2],
    )


def supervisor_is_promo_eligible(supervisor: SupervisorDict) -> bool:
    """Check if player is eligible for promotion.

    Args:
        supervisor: Decoded supervisor message.

    Returns:
        True if eligible for promotion.
    """
    return supervisor["status"] == SUPERVISOR_STATUS_PROMO_ELIGIBLE


def supervisor_has_promo_kill(supervisor: SupervisorDict) -> bool:
    """Check if player got a promotion kill.

    Args:
        supervisor: Decoded supervisor message.

    Returns:
        True if got promotion kill.
    """
    return supervisor["status"] == SUPERVISOR_STATUS_PROMO_KILL


class TerrainUpdateDict(TypedDict):
    """Terrain type update (0x4A 'J' message)."""

    msg_type: Literal[0x4A]
    updates: list[tuple[int, int, int]]


def decode_terrain_update(data: bytes) -> TerrainUpdateDict:
    """Decode terrain update from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x4A prefix).

    Returns:
        Decoded terrain update with (x, y, terrain_type) tuples.
    """
    updates: list[tuple[int, int, int]] = []
    for i in range(0, len(data) - 2, 3):
        x = data[i]
        y = data[i + 1]
        terrain_type = data[i + 2]
        updates.append((x, y, terrain_type))
    return TerrainUpdateDict(msg_type=0x4A, updates=updates)


class ViewportEntityDict(TypedDict):
    """Single entity in viewport update."""

    col: int
    row: int
    entity_id: int
    value: int
    terrain_type: int


def viewport_entity_is_tank(entity: ViewportEntityDict) -> bool:
    """Check if entity is a tank.

    Args:
        entity: Viewport entity.

    Returns:
        True if entity is a tank.
    """
    return entity["entity_id"] == -1


def viewport_entity_is_container(entity: ViewportEntityDict) -> bool:
    """Check if entity is a fuel container.

    Args:
        entity: Viewport entity.

    Returns:
        True if entity is a container.
    """
    return entity["entity_id"] > 0


def viewport_entity_is_empty(entity: ViewportEntityDict) -> bool:
    """Check if tile is empty.

    Args:
        entity: Viewport entity.

    Returns:
        True if tile is empty.
    """
    return entity["entity_id"] == 0


class ViewportUpdateDict(TypedDict):
    """Viewport/map update (0x5A 'Z' message)."""

    msg_type: Literal[0x5A]
    direction: int
    flags: int
    entities: list[ViewportEntityDict]


def decode_viewport_update(data: bytes) -> ViewportUpdateDict:
    """Decode viewport update from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x5A prefix).

    Returns:
        Decoded viewport update.

    Raises:
        DecodeError: If decoding fails.
    """
    _require_min_length(data, 2, "ViewportUpdate")

    direction = data[0]
    flags = data[1]
    entities: list[ViewportEntityDict] = []
    col, row, t = 0, 0, 2

    while t < len(data):
        v = data[t]
        t += 1

        col += v % 18
        row += v // 18
        while col >= 18:
            row += 1
            col -= 18

        if v != 255:
            if t + 3 > len(data):
                break

            b1, b2, b3 = data[t], data[t + 1], data[t + 2]
            t += 3

            z = 256 * (256 * b1 + b2) + b3
            z &= 0xFFFFFF

            terrain_type = z & 0xF
            z >>= 4
            value = z & 0xF
            if value >= 8:
                value = 255
            z >>= 4
            entity_id = z if z != 65535 else -1

            entities.append(
                ViewportEntityDict(
                    col=col,
                    row=row,
                    entity_id=entity_id,
                    value=value,
                    terrain_type=terrain_type,
                )
            )

    return ViewportUpdateDict(msg_type=0x5A, direction=direction, flags=flags, entities=entities)


# Text message types (no XOR encoding)
TEXT_MSG_TYPES: frozenset[int] = frozenset(
    {
        MSG_TANK_POS,
        MSG_PROMOTION,
        ord("%"),
        ord("*"),
        ord("$"),
        ord("-"),
        ord("~"),
        ord("`"),
        ord("R"),
    }
)


def is_text_message(msg_type: int) -> bool:
    """Check if a message type uses text format (not XOR encoded).

    Args:
        msg_type: Message type byte.

    Returns:
        True if message uses text format.
    """
    return msg_type in TEXT_MSG_TYPES


# Text message types (no XOR decoding, ASCII format)
TextMessage = JoinConfirmDict | WorldInfoDict

# Binary message types (XOR decoded)
# Note: Container types (CombatHitDict, etc.) are imported from container_decoder
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
    | MinePlacementDict
    | MineDetonationDict
    | RadarScanResultDict
    | MovementDict
    | TankInfoDict
    | MovementResponseDict
    | SyncDict
    | ContainerDict
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
    | ContainerMessage  # All container types from container_decoder
)

# Union type for all decoded messages
DecodedMessage = TextMessage | BinaryMessage


def decode_text_message(raw_body: bytes) -> JoinConfirmDict | WorldInfoDict:
    """Decode a text-format message (no XOR decoding needed).

    Args:
        raw_body: Raw message body including type byte.

    Returns:
        Decoded message object.

    Raises:
        DecodeError: If message type is unknown or decoding fails.
    """
    if len(raw_body) < 1:
        raise DecodeError("decode_text_message: empty body")

    msg_type = raw_body[0]

    if msg_type == MSG_TANK_POS:
        return decode_join_confirm(raw_body)
    if msg_type == MSG_PROMOTION:
        return decode_world_info(raw_body)

    raise DecodeError(f"decode_text_message: unknown type 0x{msg_type:02X}")


def _decode_combat_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode combat-related messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a combat message.
    """
    if msg_type == MSG_SHOOT:
        return decode_shoot_event(data)
    if msg_type == MSG_DEACTIVATE:
        return decode_deactivation(data)
    if msg_type == MSG_MINE_PLACE:
        return decode_mine_placement(data)
    if msg_type == MSG_MINE_DETONATE:
        return decode_mine_detonation(data)
    return None


def _decode_resource_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode resource-related messages (fuel, equipment).

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a resource message.
    """
    if msg_type == MSG_FUEL_GAIN:
        return decode_fuel_gain(data)
    if msg_type == MSG_FUEL_DEPOSIT:
        return decode_fuel_deposit(data)
    if msg_type == MSG_INVENTORY:
        return decode_inventory(data)
    if msg_type == MSG_EQUIP_GAIN:
        return decode_equipment_gain(data)
    if msg_type == MSG_EQUIP_TOGGLE:
        return decode_equipment_toggle(data)
    return None


def _decode_radar_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode radar and detection messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a radar message.
    """
    if msg_type == MSG_RADAR_RESULT:
        return decode_radar_result(data)
    if msg_type == MSG_ENEMY_DETECT:
        return decode_enemy_detection(data)
    if msg_type == MSG_TILE_UPDATE:
        return decode_radar_scan_result(data)
    return None


def _decode_tank_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode tank status and info messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a tank message.
    """
    if msg_type == MSG_TANK_ENTRY:
        return decode_tank_entry(data)
    if msg_type == MSG_TANK_EXIT:
        return decode_tank_exit(data)
    if msg_type == MSG_TANK_STATS:
        # Container decoder handles all structures including unknown
        return decode_0x2e_message(data)
    if msg_type == MSG_TANK_STATUS_FULL:
        return decode_tank_status(data)
    if msg_type == MSG_TANK_INFO:
        return decode_tank_info(data)
    return None


def _decode_movement_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode movement-related messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a movement message.
    """
    if msg_type == MSG_MOVEMENT:
        return decode_movement(data)
    if msg_type == MSG_MOVE_RESPONSE:
        return decode_movement_response(data)
    return None


def _decode_world_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode world/environment messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a world message.
    """
    if msg_type == MSG_VIEWPORT:
        return decode_viewport_update(data)
    if msg_type == MSG_TERRAIN_UPDATE:
        return decode_terrain_update(data)
    if msg_type == MSG_SYNC:
        return decode_sync(data)
    if msg_type == MSG_CONTAINER:
        return decode_container(data)
    return None


def _decode_misc_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Decode miscellaneous messages.

    Args:
        msg_type: Message type byte.
        data: XOR decoded message bytes.

    Returns:
        Decoded message, or None if not a misc message.
    """
    if msg_type == MSG_CHAT:
        return decode_chat_message(data)
    if msg_type == MSG_STATISTICS:
        return decode_statistics(data)
    if msg_type == MSG_ACTIVE_FORCES:
        return decode_active_forces(data)
    if msg_type == MSG_SUPERVISOR:
        return decode_supervisor(data)
    if msg_type == MSG_ACTION_DONE:
        return decode_action_done(data)
    return None


def decode_message(msg_type: int, data: bytes) -> BinaryMessage:
    """Decode a BINARY message based on its type.

    NOTE: For text messages, use decode_text_message() instead.

    Args:
        msg_type: First byte of message (NOT XOR encoded).
        data: Remaining message bytes (XOR decoded).

    Returns:
        Decoded message object.

    Raises:
        DecodeError: If message type is unknown or decoding fails.
    """
    result = _decode_combat_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_resource_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_radar_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_tank_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_movement_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_world_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_misc_message(msg_type, data)
    if result is not None:
        return result

    raise DecodeError(f"decode_message: unknown type 0x{msg_type:02X}")


def try_decode_message(msg_type: int, data: bytes) -> DecodedMessage | None:
    """Try to decode a message, returning None if unsupported.

    Unlike decode_message(), this does not raise DecodeError for unknown types.
    Use this when you want to handle unknown types gracefully without exceptions.

    Args:
        msg_type: First byte of message (NOT XOR encoded).
        data: Remaining message bytes (XOR decoded).

    Returns:
        Decoded message object, or None if type is unknown/unsupported.
    """
    return try_decode_binary_message(msg_type, data)


def try_decode_binary_message(msg_type: int, data: bytes) -> BinaryMessage | None:
    """Try to decode a BINARY message, returning None if unsupported.

    Same as try_decode_message but with narrower return type for binary-only contexts.

    Args:
        msg_type: First byte of message (NOT XOR encoded).
        data: Remaining message bytes (XOR decoded).

    Returns:
        Decoded binary message, or None if type is unknown/unsupported.
    """
    result = _decode_combat_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_resource_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_radar_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_tank_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_movement_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_world_message(msg_type, data)
    if result is not None:
        return result

    result = _decode_misc_message(msg_type, data)
    if result is not None:
        return result

    return None
