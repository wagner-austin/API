"""Laws 2 and 2b — instant server movement and ferry surface routing
(wiki [[walk-mechanics]], [[ferry-mechanics]]).

One move command resolves entirely at its processing tick: the router
finds the path, the tank relocates, the per-tile cost is billed, and
destination-tile effects (container pickup, enemy-mine detonation)
happen in the same tick. There is no walking over time — the
on-screen walk is client animation.

Ferry law (user contract 2026-07-19, single-command routing): one
command NEVER chains surfaces. On land, water is unreachable
(cant_go) but a ferry tile boards; while riding, water is open sea
and the ferry moves with the tank. The FIRST queue-consuming surface
transition — stepping onto a ferry (boarding) or from water/ferry
onto land (disembarking) — STOPS the move at that tile; the billed
cost and the echoed path cover only the tiles actually walked.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import WALK_COST_PER_TILE
from tankpit_bot.physics.damage import SINGLE_HIT_VICTIM_COST
from tankpit_bot.sim.blocks import BLOCK_BRIDGE, block_tile_value
from tankpit_bot.sim.pathfind import route
from tankpit_bot.sim.world import SimFerryDict, SimTankDict, SimWorldDict
from tankpit_bot.state.scan_coverage import tile_key

_STEP_DELTAS: dict[str, tuple[int, int]] = {
    "n": (0, -1),
    "s": (0, 1),
    "e": (1, 0),
    "w": (-1, 0),
}

Surface = Literal["land", "water", "ferry"]

StopReason = Literal["exhausted", "transition", "mine", "contact"]


def ferry_at(world: SimWorldDict, x: int, y: int) -> SimFerryDict | None:
    """Return the ferry occupying a tile, if any.

    Args:
        world: Simulated world.
        x: Tile X.
        y: Tile Y.

    Returns:
        The ferry at (x, y), or None.
    """
    for ferry in world["ferries"]:
        if (ferry["x"], ferry["y"]) == (x, y):
            return ferry
    return None


def tile_surface(
    world: SimWorldDict, terrain: TerrainMapProtocol, x: int, y: int
) -> Surface | None:
    """Classify one tile's movement surface.

    Args:
        world: Simulated world (ferries override the static map).
        terrain: Static terrain of the world's field.
        x: Tile X.
        y: Tile Y.

    Returns:
        ``ferry`` for a live ferry tile, ``land`` for passable static
        ground (including a walkable water-block bridge —
        [[movable-blocks]]: walking on a bridge is ordinary movement),
        ``water`` for static water, and None for rock, land-block
        obstacles, and stacked blocks (impassable on every surface).
    """
    if ferry_at(world, x, y) is not None:
        return "ferry"
    block_value = block_tile_value(world, terrain, x, y)
    if block_value != 0:
        return "land" if block_value == BLOCK_BRIDGE else None
    if terrain.is_passable(x, y):
        return "land"
    if terrain.get_terrain(x, y) == terrain.WATER:
        return "water"
    return None


class PickupRecordDict(TypedDict):
    """One container drained by an arrival: tile and remaining volume."""

    x: int
    y: int
    remaining_volume: int


class MoveOutcomeDict(TypedDict):
    """Everything one processed move command changed.

    ``kind`` is ``moved`` on success; ``cant_go`` when the route to
    the destination is severed. A cant_go is NOT a refusal: the
    server accepts the walk, executes the longest realizable prefix
    toward the destination, stops at the first blocker, and reports
    code 1 — exact-window measured 2026-08-04 across 12 live code-1s
    (11 carried a 0x47 prefix echo in the receipt window, two stopping
    cardinally adjacent to a known tank body; one was a zero-tile pure
    refusal with no echo). ``path`` therefore may be non-empty on a
    ``cant_go``. Fuel never rejects a walk: the debit clamps to
    remaining fuel, and fuel-0 walks execute in full (density runs
    2-3, 2026-07-25 — wiki [[game-economy]]: "walking and scanning
    are both free at fuel 0"). ``mine_positions`` lists enemy mines
    detonated by the walk (at most one — the tile the walk arrests
    on, whether mid-path walk-over or the destination).
    """

    kind: Literal["moved", "cant_go"]
    tank_id: int
    start_x: int
    start_y: int
    path: str
    pickups: list[PickupRecordDict]
    mine_positions: list[tuple[int, int]]
    # What ended the walk, and whether the clicked destination was
    # reached. The emission layer needs both: a surface-transition
    # stop short of the click closes with code 1 (live 2026-08-03:
    # four ferry-disembark collects and one riding move, every one
    # echo + 0x52 in-window), while a mine walk-over arrest does NOT
    # (2026-07-29/30 archive: 18 detonations, zero paired code-1s).
    stop_reason: StopReason
    dest_reached: bool


def _team_revealed_keys(world: SimWorldDict, team: int) -> list[str]:
    """Return the "x,y" mine keys revealed to one team.

    Args:
        world: Simulated world.
        team: Team index.

    Returns:
        The team's revealed-mine keys (empty when it never scanned).
    """
    return world["revealed_mine_keys_by_team"].get(str(team), [])


def _blocked_by_world(world: SimWorldDict, mover: SimTankDict, x: int, y: int) -> bool:
    """Report whether world entities make the router avoid a tile.

    The router avoids other living tanks and VISIBLE enemy mines
    only: own-color mines are walkable (user contract 2026-07-21),
    and mine visibility is TEAM-scoped (user contract 2026-08-04) —
    a mine no teammate has scanned does not exist to the route
    planner; the server auto-paths around visible mines only
    ([[walk-mechanics]]) and a hidden one is walked into and
    detonates.

    Args:
        world: Simulated world.
        mover: The moving tank (never blocks itself; its team carries
            the revealed-mine knowledge).
        x: Tile X.
        y: Tile Y.

    Returns:
        True when another living tank or a revealed enemy mine
        occupies the tile.
    """
    for tank in world["tanks"].values():
        if (
            tank["alive"]
            and tank["tank_id"] != mover["tank_id"]
            and (tank["x"], tank["y"]) == (x, y)
        ):
            return True
    key = tile_key(x, y)
    mine = world["mines"].get(key)
    if mine is not None and mine["team"] != mover["team"]:
        return key in _team_revealed_keys(world, mover["team"])
    return False


def _unrevealed_enemy_mine_at(world: SimWorldDict, mover: SimTankDict, x: int, y: int) -> bool:
    """Report whether a hidden enemy mine sits on a tile.

    Args:
        world: Simulated world.
        mover: The moving tank.
        x: Tile X.
        y: Tile Y.

    Returns:
        True when an enemy mine the mover's team has never been shown
        occupies the tile — the walk-over detonation case.
    """
    key = tile_key(x, y)
    mine = world["mines"].get(key)
    if mine is not None and mine["team"] != mover["team"]:
        return key not in _team_revealed_keys(world, mover["team"])
    return False


def resolve_pickup(world: SimWorldDict, tank_id: int, pickups: list[PickupRecordDict]) -> None:
    """Drain the container under a tank into it, capacity-clamped.

    Shared by walk arrivals and teleport landings (both auto-pick-up
    on the destination tile).

    Args:
        world: Simulated world (mutated).
        tank_id: The arriving tank.
        pickups: Pickup records accumulator (appended).
    """
    tank = world["tanks"][tank_id]
    x, y = tank["x"], tank["y"]
    capacity = fuel_capacity(tank["rank"])
    for container in world["containers"]:
        if (container["x"], container["y"]) == (x, y) and container["volume"] > 0:
            transfer = min(container["volume"], capacity - tank["fuel"])
            container["volume"] -= transfer
            tank["fuel"] += transfer
            pickups.append(PickupRecordDict(x=x, y=y, remaining_volume=container["volume"]))


def _resolve_arrival(world: SimWorldDict, tank_id: int, outcome: MoveOutcomeDict) -> None:
    """Apply destination-tile effects: mine detonation and pickup.

    Args:
        world: Simulated world (mutated).
        tank_id: The arriving tank.
        outcome: Outcome being built (mutated).
    """
    tank = world["tanks"][tank_id]
    x, y = tank["x"], tank["y"]
    key = tile_key(x, y)
    mine = world["mines"].get(key)
    if mine is not None and mine["team"] != tank["team"]:
        del world["mines"][key]
        # Walking into an enemy mine costs the mine's 45 (wiki
        # [[game-economy]]) — the same number as a single hit takes.
        tank["fuel"] = max(0, tank["fuel"] - SINGLE_HIT_VICTIM_COST)
        outcome["mine_positions"].append((x, y))
    resolve_pickup(world, tank_id, outcome["pickups"])


def _execute_walk(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    mover: SimTankDict,
    start_surface: Surface,
    path: str,
    *,
    stop_on_contact: bool,
) -> tuple[str, int, int, StopReason]:
    """Execute a planned path step by step, stopping at the first event.

    Three stop classes, in the order the walk meets them:

    * ``transition`` — boarding a ferry from land or stepping from
      water/ferry onto land consumes the whole action: the tank stops
      ON the transition tile (ferry law, user contract 2026-07-19).
    * ``mine`` — an UNREVEALED enemy mine on the next tile is stepped
      onto and arrests the walk there (walk-over law,
      [[mine-mechanics]]; the detonation itself is applied by
      ``_resolve_arrival`` at the stop tile).
    * ``contact`` — only when ``stop_on_contact`` is set (the severed
      fallback walk): a living tank body, a REVEALED enemy mine, or a
      block obstacle on the next tile stops the tank BEFORE it —
      live-measured 2026-08-04: the 18:12:35 echo stopped at (16,24)
      with Belton's body on (16,23).

    Args:
        world: Simulated world.
        terrain: Static terrain of the world's field.
        mover: The walking tank (revealed-mine knowledge).
        start_surface: Surface the tank stands on at route start.
        path: The planned step string.
        stop_on_contact: Whether dynamic blockers stop the walk (the
            fallback walk of a severed route). The primary walk never
            meets them — its plan avoided them.

    Returns:
        ``(walked_path, final_x, final_y, stop_reason)``.
    """
    x, y = mover["x"], mover["y"]
    previous = start_surface
    walked = 0
    for step in path:
        dx, dy = _STEP_DELTAS[step]
        nx, ny = x + dx, y + dy
        if stop_on_contact and (
            _blocked_by_world(world, mover, nx, ny)
            or block_tile_value(world, terrain, nx, ny) not in (0, BLOCK_BRIDGE)
        ):
            return path[:walked], x, y, "contact"
        x, y = nx, ny
        walked += 1
        if _unrevealed_enemy_mine_at(world, mover, x, y):
            return path[:walked], x, y, "mine"
        surface = tile_surface(world, terrain, x, y)
        if previous == "land" and surface == "ferry":
            return path[:walked], x, y, "transition"
        if previous in ("water", "ferry") and surface == "land":
            return path[:walked], x, y, "transition"
        if surface is not None:
            previous = surface
    return path[:walked], x, y, "exhausted"


def _plan_walk(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    tank: SimTankDict,
    riding: SimFerryDict | None,
    dest_x: int,
    dest_y: int,
) -> tuple[str | None, bool]:
    """Plan the route for one move command, primary then as-if-clear.

    The primary plan avoids every blocker the router knows (tanks,
    revealed enemy mines, block obstacles, cross-surface water). When
    it is severed, the fallback plans AS IF the dynamic blockers were
    absent — the measured cant_go choreography (2026-08-04, 12 live
    code-1s): the 18:12:35 echo is a 14-step direct route toward the
    target truncated one tile before a tank body. Static terrain is
    never planned through: rock and cross-surface water stay
    unroutable, and a corridor THEY sever has nothing to walk (the
    zero-tile pure refusal, live 20:58:45).

    Args:
        world: Simulated world.
        terrain: Static terrain of the world's field.
        tank: The commanding tank.
        riding: The ferry under the tank, if any (surface rules).
        dest_x: Clicked destination X.
        dest_y: Clicked destination Y.

    Returns:
        ``(path, severed)`` — the planned step string (None when even
        the as-if-clear plan finds no route) and whether the primary
        plan was severed (the walk must then stop on contact).
    """

    def surface_open(x: int, y: int) -> bool:
        """Report whether the tile's surface admits this walk at all."""
        surface = tile_surface(world, terrain, x, y)
        if surface is None:
            return False
        return riding is not None or surface != "water"

    def passable(x: int, y: int) -> bool:
        """Primary routing; the destination tile may hold a mine."""
        if not surface_open(x, y):
            return False
        if (x, y) == (dest_x, dest_y):
            return not any(
                other["alive"]
                and other["tank_id"] != tank["tank_id"]
                and (other["x"], other["y"]) == (x, y)
                for other in world["tanks"].values()
            )
        return not _blocked_by_world(world, tank, x, y)

    def passable_as_if_clear(x: int, y: int) -> bool:
        """Fallback routing for a severed corridor: terrain only."""
        surface = tile_surface(world, terrain, x, y)
        if surface is None and block_tile_value(world, terrain, x, y) not in (0, BLOCK_BRIDGE):
            return True
        return surface_open(x, y)

    path = route(passable, tank["x"], tank["y"], dest_x, dest_y)
    if path is not None:
        return path, False
    return route(passable_as_if_clear, tank["x"], tank["y"], dest_x, dest_y), True


def _update_ridden_ferry(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    riding: SimFerryDict,
    start_x: int,
    start_y: int,
    walked: str,
    final_x: int,
    final_y: int,
) -> None:
    """Carry the ferry with its rider, or dock it on a disembark.

    Args:
        world: Simulated world.
        terrain: Static terrain of the world's field.
        riding: The ferry the tank started the move on (mutated).
        start_x: Walk start X.
        start_y: Walk start Y.
        walked: The executed step string.
        final_x: The tile the tank stopped on.
        final_y: The tile the tank stopped on.
    """
    if tile_surface(world, terrain, final_x, final_y) == "land":
        last_x, last_y = start_x, start_y
        for step in walked[:-1]:
            dx, dy = _STEP_DELTAS[step]
            last_x, last_y = last_x + dx, last_y + dy
        riding["x"], riding["y"] = last_x, last_y
    else:
        riding["x"], riding["y"] = final_x, final_y


def process_move(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    tank_id: int,
    dest_x: int,
    dest_y: int,
) -> MoveOutcomeDict:
    """Process one destination-click move command at the current tick.

    Routing is single-surface (the ferry law): on land the router
    opens ground and ferry tiles only — a water click is cant_go;
    while riding it opens water, ferry, and land tiles, with the
    disembark stop truncating at the first land step. The ferry
    carries its rider: while the rider stays afloat the ferry ends
    the move under the tank; a disembark leaves it on the last water
    tile.

    Args:
        world: Simulated world (mutated on success).
        terrain: Static terrain of the world's field.
        tank_id: The commanding tank (must exist and be alive).
        dest_x: Clicked destination X.
        dest_y: Clicked destination Y.

    Returns:
        The typed outcome; the world reflects it on ``moved`` and on
        a partial-walk ``cant_go`` (the walked prefix is real).
    """
    tank = world["tanks"][tank_id]
    start_x, start_y = tank["x"], tank["y"]
    riding = ferry_at(world, start_x, start_y)
    start_surface: Surface = "ferry" if riding is not None else "land"
    outcome = MoveOutcomeDict(
        kind="moved",
        tank_id=tank_id,
        start_x=start_x,
        start_y=start_y,
        path="",
        pickups=[],
        mine_positions=[],
        stop_reason="exhausted",
        dest_reached=False,
    )
    path, severed = _plan_walk(world, terrain, tank, riding, dest_x, dest_y)
    if path is None:
        outcome["kind"] = "cant_go"
        return outcome
    walked, final_x, final_y, stop_reason = _execute_walk(
        world, terrain, tank, start_surface, path, stop_on_contact=severed
    )
    outcome["stop_reason"] = stop_reason
    outcome["dest_reached"] = (final_x, final_y) == (dest_x, dest_y)
    if severed:
        # A severed walk that arrests on a hidden mine IS the
        # walk-over law (detonation, not refusal); any other stop is
        # the partial-walk cant_go receipt. Zero tiles walked is the
        # pure refusal: the first step was already blocked.
        outcome["kind"] = "moved" if stop_reason == "mine" else "cant_go"
        if walked == "":
            return outcome
    # The debit clamps to remaining fuel (radar-analog law): the walk
    # itself never rejects for fuel — fuel-0 walks executed live.
    cost = min(WALK_COST_PER_TILE * len(walked), tank["fuel"])
    tank["x"] = final_x
    tank["y"] = final_y
    tank["fuel"] -= cost
    outcome["path"] = walked
    if riding is not None:
        _update_ridden_ferry(world, terrain, riding, start_x, start_y, walked, final_x, final_y)
    _resolve_arrival(world, tank_id, outcome)
    return outcome


__all__ = [
    "MoveOutcomeDict",
    "PickupRecordDict",
    "StopReason",
    "Surface",
    "ferry_at",
    "process_move",
    "resolve_pickup",
    "tile_surface",
]
