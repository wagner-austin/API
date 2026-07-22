"""Simulator world state — the fake server's single source of truth.

Phase 4 step (b), wiki [[physics-module-roadmap]]. The world dict holds
everything the wiki laws mutate: tanks, fuel containers, mines, and
the tick counter. Every dict has an encode/decode codec so worlds can
be seeded from files and snapshots can be asserted in tests.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_bool,
    require_int,
)

from tankpit_bot.physics.capacity import fuel_capacity

EQUIPMENT_SLOTS = 5


class SimTankDict(TypedDict):
    """One tank's server-side state.

    ``counts``/``enabled`` mirror the five 0x49 equipment slots
    (armor, dual, missile, homing, radar). ``fuel`` is the tank's
    absolute fuel; ``damage_state`` follows the wire convention
    (0 = full, then 3 -> 2 -> 1 counting down toward deactivation).
    """

    tank_id: int
    team: int
    rank: int
    x: int
    y: int
    fuel: int
    damage_state: int
    counts: list[int]
    enabled: list[bool]
    alive: bool


class SimContainerDict(TypedDict):
    """One fuel container tile: position and remaining volume."""

    x: int
    y: int
    volume: int


class SimMineDict(TypedDict):
    """One mine tile: position and owning team."""

    x: int
    y: int
    team: int


class SimEquipmentDict(TypedDict):
    """One equipment container tile: position only.

    Contents are not determined until pickup (wiki
    [[equipment-system]] — the archive-mined law grants one slot per
    pickup); the container is consumed by a successful grant.
    """

    x: int
    y: int


class SimWorldDict(TypedDict):
    """The whole simulated world at one tick.

    ``field`` names the terrain GIF the router loads
    (e.g. ``field01_r.gif``); ``tick`` counts processed server ticks.
    """

    field: str
    tick: int
    tanks: dict[int, SimTankDict]
    containers: list[SimContainerDict]
    mines: list[SimMineDict]
    equipment: list[SimEquipmentDict]


def make_sim_tank(
    tank_id: int,
    team: int,
    rank: int,
    x: int,
    y: int,
    fuel: int,
) -> SimTankDict:
    """Build a live tank with empty equipment slots.

    Args:
        tank_id: Wire tank id.
        team: Team index (0-3).
        rank: Rank (0-8) — determines fuel capacity.
        x: Tile X.
        y: Tile Y.
        fuel: Starting fuel (clamped to capacity).

    Returns:
        A live tank at full damage state.
    """
    return SimTankDict(
        tank_id=tank_id,
        team=team,
        rank=rank,
        x=x,
        y=y,
        fuel=min(fuel, fuel_capacity(rank)),
        damage_state=0,
        counts=[0] * EQUIPMENT_SLOTS,
        enabled=[True] * EQUIPMENT_SLOTS,
        alive=True,
    )


def make_sim_world(field: str) -> SimWorldDict:
    """Build an empty world on the given terrain field.

    Args:
        field: Terrain GIF file name (e.g. ``field01_r.gif``).

    Returns:
        A world at tick 0 with no tanks, containers, mines, or
        equipment.
    """
    return SimWorldDict(field=field, tick=0, tanks={}, containers=[], mines=[], equipment=[])


def encode_sim_tank(tank: SimTankDict) -> JSONObject:
    """Encode one tank to a JSON-serializable dict.

    Args:
        tank: Tank to encode.

    Returns:
        JSON object with all tank fields.
    """
    return {
        "tank_id": tank["tank_id"],
        "team": tank["team"],
        "rank": tank["rank"],
        "x": tank["x"],
        "y": tank["y"],
        "fuel": tank["fuel"],
        "damage_state": tank["damage_state"],
        "counts": list(tank["counts"]),
        "enabled": list(tank["enabled"]),
        "alive": tank["alive"],
    }


def _require_int_list(data: JSONObject, key: str, length: int) -> list[int]:
    """Validate a fixed-length list of ints.

    Args:
        data: Enclosing JSON object.
        key: List field name.
        length: Required element count.

    Returns:
        The validated list.

    Raises:
        ValueError: If the field is not a list of ``length`` ints.
    """
    raw = data.get(key)
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError(f"SimTank.{key}: expected a list of {length} entries")
    values: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"SimTank.{key}: entries must be ints")
        values.append(item)
    return values


def _require_bool_list(data: JSONObject, key: str, length: int) -> list[bool]:
    """Validate a fixed-length list of bools.

    Args:
        data: Enclosing JSON object.
        key: List field name.
        length: Required element count.

    Returns:
        The validated list.

    Raises:
        ValueError: If the field is not a list of ``length`` bools.
    """
    raw = data.get(key)
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError(f"SimTank.{key}: expected a list of {length} entries")
    values: list[bool] = []
    for item in raw:
        if not isinstance(item, bool):
            raise ValueError(f"SimTank.{key}: entries must be bools")
        values.append(item)
    return values


def decode_sim_tank(data: JSONObject) -> SimTankDict:
    """Decode one tank from a JSON object with validation.

    Args:
        data: JSON object carrying the tank fields.

    Returns:
        Validated tank.

    Raises:
        JSONTypeError: If a scalar field has the wrong type.
        ValueError: If an equipment list is malformed.
        KeyError: If a field is missing.
    """
    return SimTankDict(
        tank_id=require_int(data, "tank_id"),
        team=require_int(data, "team"),
        rank=require_int(data, "rank"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        fuel=require_int(data, "fuel"),
        damage_state=require_int(data, "damage_state"),
        counts=_require_int_list(data, "counts", EQUIPMENT_SLOTS),
        enabled=_require_bool_list(data, "enabled", EQUIPMENT_SLOTS),
        alive=require_bool(data, "alive"),
    )


def encode_sim_world(world: SimWorldDict) -> JSONObject:
    """Encode the world to a JSON-serializable dict.

    Args:
        world: World to encode.

    Returns:
        JSON object with all world fields.
    """
    tanks: list[JSONValue] = [encode_sim_tank(tank) for tank in world["tanks"].values()]
    containers: list[JSONValue] = [
        {"x": c["x"], "y": c["y"], "volume": c["volume"]} for c in world["containers"]
    ]
    mines: list[JSONValue] = [{"x": m["x"], "y": m["y"], "team": m["team"]} for m in world["mines"]]
    equipment: list[JSONValue] = [{"x": e["x"], "y": e["y"]} for e in world["equipment"]]
    return {
        "field": world["field"],
        "tick": world["tick"],
        "tanks": tanks,
        "containers": containers,
        "mines": mines,
        "equipment": equipment,
    }


def decode_sim_world(data: JSONObject) -> SimWorldDict:
    """Decode a world from a JSON object with validation.

    Args:
        data: JSON object carrying the world fields.

    Returns:
        Validated world, tanks keyed by id.

    Raises:
        JSONTypeError: If a scalar field has the wrong type.
        ValueError: If a nested record is malformed.
        KeyError: If a field is missing.
    """
    field = data.get("field")
    if not isinstance(field, str):
        raise ValueError("SimWorld.field: expected a string")
    raw_tanks = data.get("tanks")
    if not isinstance(raw_tanks, list):
        raise ValueError("SimWorld.tanks: expected a list")
    tanks: dict[int, SimTankDict] = {}
    for item in raw_tanks:
        tank = decode_sim_tank(narrow_json_to_dict(item))
        tanks[tank["tank_id"]] = tank
    raw_containers = data.get("containers")
    if not isinstance(raw_containers, list):
        raise ValueError("SimWorld.containers: expected a list")
    containers: list[SimContainerDict] = []
    for item in raw_containers:
        record = narrow_json_to_dict(item)
        containers.append(
            SimContainerDict(
                x=require_int(record, "x"),
                y=require_int(record, "y"),
                volume=require_int(record, "volume"),
            )
        )
    raw_mines = data.get("mines")
    if not isinstance(raw_mines, list):
        raise ValueError("SimWorld.mines: expected a list")
    mines: list[SimMineDict] = []
    for item in raw_mines:
        record = narrow_json_to_dict(item)
        mines.append(
            SimMineDict(
                x=require_int(record, "x"),
                y=require_int(record, "y"),
                team=require_int(record, "team"),
            )
        )
    raw_equipment = data.get("equipment")
    if not isinstance(raw_equipment, list):
        raise ValueError("SimWorld.equipment: expected a list")
    equipment: list[SimEquipmentDict] = []
    for item in raw_equipment:
        record = narrow_json_to_dict(item)
        equipment.append(SimEquipmentDict(x=require_int(record, "x"), y=require_int(record, "y")))
    return SimWorldDict(
        field=field,
        tick=require_int(data, "tick"),
        tanks=tanks,
        containers=containers,
        mines=mines,
        equipment=equipment,
    )


__all__ = [
    "EQUIPMENT_SLOTS",
    "SimContainerDict",
    "SimEquipmentDict",
    "SimMineDict",
    "SimTankDict",
    "SimWorldDict",
    "decode_sim_tank",
    "decode_sim_world",
    "encode_sim_tank",
    "encode_sim_world",
    "make_sim_tank",
    "make_sim_world",
]
