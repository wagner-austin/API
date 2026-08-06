"""Laws 5-8 (step d): teleport, radar, map, and mine placement.

Each processor mutates the world under a measured wiki law and
returns typed outcomes; the tick processor turns them into wire
messages. Documented sim assumptions (wiki [[physics-module-roadmap]]
Phase 4): displacement south is tried LAST (only E->N->W are
measured), beyond-ring-1 displacement does not exist (blocked ring-1
rejects the hop), and mine placement clips to map bounds (the
viewport-edge clip needs a per-client viewport model).
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.physics.capacity import damage_tier, free_radar_radius
from tankpit_bot.physics.costs import MINE_PRESS_COST, RADAR_COST, teleport_cost
from tankpit_bot.physics.map import MAP_DOT_MIN_VOLUME
from tankpit_bot.physics.supervisor import teleport_refusal
from tankpit_bot.protocol.types import MapDataDict, MapTankEntry, RadarContainerDict, RadarMineDict
from tankpit_bot.sim.blocks import blocks_at
from tankpit_bot.sim.combat import SLOT_RADAR
from tankpit_bot.sim.movement import PickupRecordDict, ferry_at, resolve_pickup
from tankpit_bot.sim.world import SimMineDict, SimWorldDict

# Ring-1 displacement preference when the teleport target is blocked
# (wiki [[teleport-mechanics]], 2026-07-21: E, then N, then W measured;
# S is the documented last-resort assumption).
_DISPLACEMENT_ORDER: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, -1), (-1, 0), (0, 1))

# The client's viewport as a Chebyshev radius around the tank: the
# extra-radar scan covers exactly the viewport (wiki
# [[radar-mechanics]]: extra = whole viewport), and tank-position
# visibility is viewport-scoped — leaving it triggers the 0x58
# TankRemove the law-4 reroute clock starts from
# ([[shoot-event-format]]).
VIEWPORT_RADIUS = 8


class TeleportOutcomeDict(TypedDict):
    """One processed teleport: landing, cost, and arrival pickups."""

    kind: Literal["landed", "blocked", "insufficient_fuel"]
    tank_id: int
    landed_x: int
    landed_y: int
    cost: int
    pickups: list[PickupRecordDict]


class RadarOutcomeDict(TypedDict):
    """One processed radar scan: what it revealed and what it consumed."""

    tank_id: int
    containers: list[RadarContainerDict]
    mines: list[RadarMineDict]
    enemy_found: bool
    consumed_extra: bool


class MinePressOutcomeDict(TypedDict):
    """One processed mine press: placed mines and 1:1 detonations."""

    tank_id: int
    mine_type: int
    placed: list[tuple[int, int]]
    detonated: list[tuple[int, int]]


def _tile_blocked_for_landing(
    world: SimWorldDict, terrain: TerrainMapProtocol, tank_id: int, team: int, x: int, y: int
) -> bool:
    """Report whether a teleport may land on a tile.

    Rock/water, any OTHER living tank (self-occupancy of the target
    tile counts as blocked only for other tanks — the mover vacates
    its own tile), and enemy mines all block; own-color mines do not
    (wiki [[teleport-mechanics]]). A FERRY tile is a legal landing
    even though its water is not: boarding by teleport is the F5
    doctrine's core move (user, [[ferry-mechanics]]: "you generally
    will need to teleport to the ferry since many times it will be on
    its own area in the water") and what ``ferry_landing.py``'s
    boarding-tile hops dispatch.

    Args:
        world: Simulated world.
        terrain: Static terrain.
        tank_id: The teleporting tank.
        team: The teleporting tank's team.
        x: Candidate landing X.
        y: Candidate landing Y.

    Returns:
        True when the tile cannot be landed on.
    """
    if not terrain.is_passable(x, y) and ferry_at(world, x, y) is None:
        return True
    if blocks_at(world, x, y) > 0:
        return True
    for tank in world["tanks"].values():
        if tank["alive"] and tank["tank_id"] != tank_id and (tank["x"], tank["y"]) == (x, y):
            return True
    return any(mine["team"] != team and (mine["x"], mine["y"]) == (x, y) for mine in world["mines"])


def process_teleport(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    tank_id: int,
    target_x: int,
    target_y: int,
) -> TeleportOutcomeDict:
    """Process one map-teleport command (law 5).

    Args:
        world: Simulated world (mutated on success).
        terrain: Static terrain.
        tank_id: The teleporting tank.
        target_x: Clicked map tile X.
        target_y: Clicked map tile Y.

    Returns:
        The typed outcome: cost is ``floor(6 x euclid)`` to the ACTUAL
        landing tile; a blocked target displaces E -> N -> W (-> S)
        within ring 1 or rejects the hop.
    """
    tank = world["tanks"][tank_id]
    outcome = TeleportOutcomeDict(
        kind="blocked",
        tank_id=tank_id,
        landed_x=tank["x"],
        landed_y=tank["y"],
        cost=0,
        pickups=[],
    )
    for dx, dy in _DISPLACEMENT_ORDER:
        x, y = target_x + dx, target_y + dy
        if _tile_blocked_for_landing(world, terrain, tank_id, tank["team"], x, y):
            continue
        cost = teleport_cost(tank["x"], tank["y"], x, y)
        if teleport_refusal(tank["fuel"], cost) is not None:
            outcome["kind"] = "insufficient_fuel"
            return outcome
        tank["x"] = x
        tank["y"] = y
        tank["fuel"] -= cost
        outcome["kind"] = "landed"
        outcome["landed_x"] = x
        outcome["landed_y"] = y
        outcome["cost"] = cost
        resolve_pickup(world, tank_id, outcome["pickups"])
        return outcome
    return outcome


def process_radar(
    world: SimWorldDict, tank_id: int, window: tuple[int, int] | None = None
) -> RadarOutcomeDict:
    """Process one radar command (law 8, scan side).

    An available extra radar is consumed and covers the full viewport;
    otherwise the rank-scaled built-in radius applies, intersected
    with the viewport (production ``scan_coverage``: free radar =
    5x5 around the tank clipped to the viewport). The scan reveals
    containers (fuel by volume, equipment as the wire's
    ``0xFFFF -> -1`` cache marker) and mines in the covered tiles and
    reports whether any living enemy sits inside them. The 10-fuel
    cost is billed by the caller.

    Args:
        world: Simulated world (mutated: extra-radar consumption).
        tank_id: The scanning tank.
        window: The tank's current 0x5A window origin ``(left, top)``,
            or None for a window centered on the tank (server-driven
            roster bots act from rest, where the two coincide; the
            client's window can drift from its position by walking —
            autoscroll OFF never recenters, [[viewport-shift-protocol]]).

    Returns:
        The typed scan result.
    """
    tank = world["tanks"][tank_id]
    consumed = False
    if tank["enabled"][SLOT_RADAR] and tank["counts"][SLOT_RADAR] > 0:
        tank["counts"][SLOT_RADAR] -= 1
        consumed = True
        radius = VIEWPORT_RADIUS
    else:
        radius = free_radar_radius(tank["rank"])
    cx, cy = tank["x"], tank["y"]
    span = 2 * VIEWPORT_RADIUS
    left, top = window if window is not None else (cx - VIEWPORT_RADIUS, cy - VIEWPORT_RADIUS)

    def inside(x: int, y: int) -> bool:
        """Report whether a tile lies inside the scan coverage."""
        if not (left <= x < left + span and top <= y < top + span):
            return False
        if consumed:
            return True
        return abs(x - cx) <= radius and abs(y - cy) <= radius

    containers = []
    for c in world["containers"]:
        if not inside(c["x"], c["y"]):
            continue
        # Exposure is the dot law (2026-07-25): a container revealed
        # while holding MAP_DOT_MIN_VOLUME or more joins the 0x4C
        # atlas permanently. Volume-0 containers ARE sent — the wire's
        # cache value 0 is the client's "tile is empty" removal signal
        # (323 zero-volume reveals in the archive).
        if not c["dotted"] and c["volume"] >= MAP_DOT_MIN_VOLUME:
            c["dotted"] = True
        containers.append(RadarContainerDict(x=c["x"], y=c["y"], volume=c["volume"]))
    containers.extend(
        RadarContainerDict(x=e["x"], y=e["y"], volume=-1)
        for e in world["equipment"]
        if inside(e["x"], e["y"])
    )
    mines = [
        RadarMineDict(x=m["x"], y=m["y"], team=m["team"])
        for m in world["mines"]
        if inside(m["x"], m["y"])
    ]
    # A scan is the REVEAL event, and reveals are TEAM-scoped (user
    # contract 2026-08-04): any teammate's scan makes the mines
    # visible to the whole color, while a fresh plant stays hidden
    # from enemies — even ones sharing the planter's viewport — until
    # someone on their team radars. The movement law consults this to
    # route around VISIBLE enemy mines only ([[walk-mechanics]]).
    team_key = str(tank["team"])
    revealed = set(world["revealed_mine_keys_by_team"].get(team_key, []))
    revealed.update(f"{m['x']},{m['y']}" for m in mines)
    world["revealed_mine_keys_by_team"][team_key] = sorted(revealed)
    enemy_found = any(
        other["alive"] and other["team"] != tank["team"] and inside(other["x"], other["y"])
        for other in world["tanks"].values()
    )
    return RadarOutcomeDict(
        tank_id=tank_id,
        containers=containers,
        mines=mines,
        enemy_found=enemy_found,
        consumed_extra=consumed,
    )


def _atlas_order(dot: tuple[int, int]) -> int:
    """Linear atlas position of one fuel dot (row-major over 256 columns).

    Args:
        dot: The (x, y) dot position.

    Returns:
        The dot's linear stream position.
    """
    return dot[1] * 256 + dot[0]


def build_map_data(world: SimWorldDict) -> MapDataDict:
    """Build the 0x4C strategic-map snapshot (law 8, map side).

    Fuel dots are the DOTTED containers — exposure memory, the
    measured 2026-07-25 law ([[map-data-decode]]): a dot appears when
    a container is revealed holding >= 500 volume and persists as the
    container drains (even to 0), so the atlas over-promises exactly
    the way the live one does (~40% of live dots still hold fuel).
    Dots are emitted in atlas stream order (ascending linear position
    — the skip-RLE encoder's requirement); tank blips cover every
    living tank. Mines are NOT on the map ([[map-data-decode]],
    user-confirmed 2026-07-21).

    Args:
        world: Simulated world.

    Returns:
        The map snapshot.
    """
    dots = sorted(
        ((c["x"], c["y"]) for c in world["containers"] if c["dotted"]),
        key=_atlas_order,
    )
    tanks = [
        MapTankEntry(
            x=tank["x"],
            y=tank["y"],
            tank_id=tank["tank_id"],
            rank=tank["rank"],
            damage=damage_tier(tank["fuel"], tank["rank"]),
            team=tank["team"],
        )
        for tank_id, tank in sorted(world["tanks"].items())
        if tank["alive"]
    ]
    return MapDataDict(msg_type=0x4C, fuel_dots=dots, tanks=tanks)


def process_mine_press(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    tank_id: int,
) -> MinePressOutcomeDict:
    """Process one mine press (law 6).

    A 3x3 placement centered on the placer: rock/water/tank tiles are
    skipped, tiles holding an enemy mine trade 1:1 (the enemy mine
    detonates, nothing is placed there), and clear tiles receive the
    placer's mine. The flat 10-fuel press cost is billed by the
    caller. Mines are not inventory — nothing is consumed.

    Args:
        world: Simulated world (mutated).
        terrain: Static terrain.
        tank_id: The placing tank.

    Returns:
        The typed outcome with placed and detonated positions.
    """
    tank = world["tanks"][tank_id]
    outcome = MinePressOutcomeDict(tank_id=tank_id, mine_type=tank["team"], placed=[], detonated=[])
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = tank["x"] + dx, tank["y"] + dy
            if not terrain.is_passable(x, y):
                continue
            if blocks_at(world, x, y) > 0:
                continue
            if any(
                other["alive"]
                and other["tank_id"] != tank_id
                and (other["x"], other["y"]) == (x, y)
                for other in world["tanks"].values()
            ):
                continue
            enemy_mines = [
                mine
                for mine in world["mines"]
                if mine["team"] != tank["team"] and (mine["x"], mine["y"]) == (x, y)
            ]
            if enemy_mines:
                for mine in enemy_mines:
                    world["mines"].remove(mine)
                outcome["detonated"].append((x, y))
                continue
            if any((mine["x"], mine["y"]) == (x, y) for mine in world["mines"]):
                continue
            world["mines"].append(SimMineDict(x=x, y=y, team=tank["team"]))
            outcome["placed"].append((x, y))
    return outcome


RADAR_FUEL_COST = RADAR_COST
MINE_PRESS_FUEL_COST = MINE_PRESS_COST

__all__ = [
    "MINE_PRESS_FUEL_COST",
    "RADAR_FUEL_COST",
    "VIEWPORT_RADIUS",
    "MinePressOutcomeDict",
    "RadarOutcomeDict",
    "TeleportOutcomeDict",
    "build_map_data",
    "process_mine_press",
    "process_radar",
    "process_teleport",
]
