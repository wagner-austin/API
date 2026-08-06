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
    require_str,
)

from tankpit_bot.physics.capacity import fuel_capacity

EQUIPMENT_SLOTS = 5


class SimTankDict(TypedDict):
    """One tank's server-side state.

    ``counts``/``enabled`` mirror the five 0x49 equipment slots
    (armor, dual, missile, homing, radar). ``fuel`` is the tank's
    absolute fuel — it IS the health pool; the wire damage tier is
    not stored state but the fuel quartile, derived at emission via
    ``physics.damage_tier`` (corpus-fitted 2026-07-23,
    [[deactivation-format]]).

    ``name`` is the wire display name the 0x21 identity broadcast
    carries. The bot classifies humans by NAME SHAPE
    (``bot/ai/humans.py``): the practice shape ``red-<id>`` (the
    :func:`make_sim_tank` default) keeps a tank on the farmable
    practice-bot tier, while any other non-empty name puts it behind
    the human-consent gate and the fair-fight contracts (2026-07-31)
    — seed one to exercise the human-fight paths in sim.
    """

    tank_id: int
    team: int
    rank: int
    name: str
    x: int
    y: int
    fuel: int
    counts: list[int]
    enabled: list[bool]
    alive: bool
    carrying: bool


class SimContainerDict(TypedDict):
    """One fuel container tile: position, remaining volume, dot state.

    ``dotted`` is the container's 0x4C map-atlas membership — set the
    first time the container is EXPOSED (radar reveal) while holding
    ``MAP_DOT_MIN_VOLUME`` (500) or more, and persistent afterwards
    even as the volume drains (measured law 2026-07-25: dots are
    exposure memory, ~60% of live dots no longer hold usable fuel).
    Containers persist at volume 0 (the server answers pickup
    attempts with 0x52 code 4 "Empty container").
    """

    x: int
    y: int
    volume: int
    dotted: bool


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


class SimFerryDict(TypedDict):
    """One ferry: a single dynamic water tile that moves with its rider.

    Ferries are wire terrain (``TERRAIN_FERRY`` in 0x5A viewport
    patches), not static map content — the client composes them over
    the minimap ([[ferry-mechanics]]).
    """

    x: int
    y: int


class SimBlockDict(TypedDict):
    """One resting movable concrete block ([[movable-blocks]]).

    A carried block has NO tile — it leaves this list on pickup and
    returns on drop. The wire value derives from context: one block
    over static water is a walkable bridge (1), a block on land is an
    obstacle (2), two blocks on a water tile are stacked terrain (3).
    """

    x: int
    y: int


class SimWorldDict(TypedDict):
    """The whole simulated world at one tick.

    ``field`` names the terrain GIF the router loads
    (e.g. ``field01_r.gif``); ``tick`` counts processed server ticks.

    ``revealed_mine_keys_by_team`` holds, per team (string key for
    JSON round-tripping), the "x,y" keys of mines that team has been
    SHOWN. Mine visibility is TEAM-scoped (user contract 2026-08-04):
    any teammate's scan reveals for the whole color, a fresh plant is
    invisible even to a tank sharing the planter's viewport until
    someone on its team radars, and own-team mines are always visible
    (and walkable, so they never enter routing). The server routes a
    walk around VISIBLE enemy mines only ([[walk-mechanics]]); a
    hidden one is walked into and detonates. Detonated mines need no
    key cleanup — routing intersects these keys with the live mine
    list.
    """

    field: str
    tick: int
    tanks: dict[int, SimTankDict]
    containers: list[SimContainerDict]
    mines: list[SimMineDict]
    equipment: list[SimEquipmentDict]
    ferries: list[SimFerryDict]
    blocks: list[SimBlockDict]
    revealed_mine_keys_by_team: dict[str, list[str]]


def make_sim_tank(
    tank_id: int,
    team: int,
    rank: int,
    x: int,
    y: int,
    fuel: int,
    name: str = "",
) -> SimTankDict:
    """Build a live tank with empty equipment slots.

    Args:
        tank_id: Wire tank id.
        team: Team index (0-3).
        rank: Rank (0-8) — determines fuel capacity.
        x: Tile X.
        y: Tile Y.
        fuel: Starting fuel (clamped to capacity).
        name: Wire display name. Empty resolves to the practice-bot
            shape ``red-<id>``; a human-shaped name (e.g. ``guest``)
            puts the tank behind the human-consent gate and the
            fair-fight contracts.

    Returns:
        A live tank.
    """
    return SimTankDict(
        tank_id=tank_id,
        team=team,
        rank=rank,
        name=name if name else f"red-{tank_id}",
        x=x,
        y=y,
        fuel=min(fuel, fuel_capacity(rank)),
        counts=[0] * EQUIPMENT_SLOTS,
        enabled=[True] * EQUIPMENT_SLOTS,
        alive=True,
        carrying=False,
    )


def make_sim_world(field: str) -> SimWorldDict:
    """Build an empty world on the given terrain field.

    Args:
        field: Terrain GIF file name (e.g. ``field01_r.gif``).

    Returns:
        A world at tick 0 with no tanks, containers, mines,
        equipment, ferries, or blocks.
    """
    return SimWorldDict(
        field=field,
        tick=0,
        tanks={},
        containers=[],
        mines=[],
        equipment=[],
        ferries=[],
        blocks=[],
        revealed_mine_keys_by_team={},
    )


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
        "name": tank["name"],
        "x": tank["x"],
        "y": tank["y"],
        "fuel": tank["fuel"],
        "counts": list(tank["counts"]),
        "enabled": list(tank["enabled"]),
        "alive": tank["alive"],
        "carrying": tank["carrying"],
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
        name=require_str(data, "name"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        fuel=require_int(data, "fuel"),
        counts=_require_int_list(data, "counts", EQUIPMENT_SLOTS),
        enabled=_require_bool_list(data, "enabled", EQUIPMENT_SLOTS),
        alive=require_bool(data, "alive"),
        carrying=require_bool(data, "carrying"),
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
        {"x": c["x"], "y": c["y"], "volume": c["volume"], "dotted": c["dotted"]}
        for c in world["containers"]
    ]
    mines: list[JSONValue] = [{"x": m["x"], "y": m["y"], "team": m["team"]} for m in world["mines"]]
    equipment: list[JSONValue] = [{"x": e["x"], "y": e["y"]} for e in world["equipment"]]
    ferries: list[JSONValue] = [{"x": f["x"], "y": f["y"]} for f in world["ferries"]]
    blocks: list[JSONValue] = [{"x": b["x"], "y": b["y"]} for b in world["blocks"]]
    revealed: JSONValue = {
        team: list(keys) for team, keys in world["revealed_mine_keys_by_team"].items()
    }
    return {
        "field": world["field"],
        "tick": world["tick"],
        "tanks": tanks,
        "containers": containers,
        "mines": mines,
        "equipment": equipment,
        "ferries": ferries,
        "blocks": blocks,
        "revealed_mine_keys_by_team": revealed,
    }


def _require_record_list(data: JSONObject, key: str) -> list[JSONObject]:
    """Validate a world section as a list of JSON objects.

    Args:
        data: Enclosing JSON object.
        key: Section field name.

    Returns:
        The section's records.

    Raises:
        ValueError: If the section is not a list.
    """
    raw = data.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"SimWorld.{key}: expected a list")
    return [narrow_json_to_dict(item) for item in raw]


def _decode_xy_list(data: JSONObject, key: str) -> list[tuple[int, int]]:
    """Decode a section of bare (x, y) records.

    Args:
        data: Enclosing JSON object.
        key: Section field name.

    Returns:
        The validated positions.

    Raises:
        ValueError: If the section is malformed.
    """
    return [
        (require_int(record, "x"), require_int(record, "y"))
        for record in _require_record_list(data, key)
    ]


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
    containers = [
        SimContainerDict(
            x=require_int(record, "x"),
            y=require_int(record, "y"),
            volume=require_int(record, "volume"),
            dotted=require_bool(record, "dotted"),
        )
        for record in _require_record_list(data, "containers")
    ]
    mines = [
        SimMineDict(
            x=require_int(record, "x"),
            y=require_int(record, "y"),
            team=require_int(record, "team"),
        )
        for record in _require_record_list(data, "mines")
    ]
    raw_revealed = data.get("revealed_mine_keys_by_team")
    if not isinstance(raw_revealed, dict):
        raise ValueError("SimWorld.revealed_mine_keys_by_team: expected an object")
    revealed: dict[str, list[str]] = {}
    for team_key, raw_keys in raw_revealed.items():
        if not isinstance(raw_keys, list):
            raise ValueError("SimWorld.revealed_mine_keys_by_team: entries must be lists")
        keys: list[str] = []
        for raw_key in raw_keys:
            if not isinstance(raw_key, str):
                raise ValueError("SimWorld.revealed_mine_keys_by_team: keys must be strings")
            keys.append(raw_key)
        revealed[team_key] = keys
    return SimWorldDict(
        field=field,
        tick=require_int(data, "tick"),
        tanks=tanks,
        containers=containers,
        mines=mines,
        equipment=[SimEquipmentDict(x=x, y=y) for x, y in _decode_xy_list(data, "equipment")],
        ferries=[SimFerryDict(x=x, y=y) for x, y in _decode_xy_list(data, "ferries")],
        blocks=[SimBlockDict(x=x, y=y) for x, y in _decode_xy_list(data, "blocks")],
        revealed_mine_keys_by_team=revealed,
    )


__all__ = [
    "EQUIPMENT_SLOTS",
    "SimBlockDict",
    "SimContainerDict",
    "SimEquipmentDict",
    "SimFerryDict",
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
