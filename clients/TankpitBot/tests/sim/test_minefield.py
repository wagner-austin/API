"""The room's standing minefield, and its archived density."""

from __future__ import annotations

from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tankpit_bot.sim.world_seed_mines import (
    MINE_DENSITY,
    MINE_TEAM_CYCLE,
    seed_minefield,
)
from tankpit_bot.state.viewport_geometry import VIEWPORT_PATCH_HEIGHT, VIEWPORT_PATCH_WIDTH
from tests.in_memory_terrain_map import InMemoryTerrainMap

#: The archive's own component split, from 2,236 components lifted out
#: of 27 real captures replayed through the production dispatcher.
_ARCHIVED_SINGLE_SHARE = 0.393
_ARCHIVED_SMALL_SHARE = 0.525


def _components(world: SimWorldDict) -> list[set[tuple[int, int]]]:
    """Split a seeded minefield into 8-connected components.

    The same connectivity the archive probe used, so the two censuses
    are comparable.
    """
    remaining = {(mine["x"], mine["y"]) for mine in world["mines"].values()}
    blobs: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        blob = {seed}
        frontier = [seed]
        while frontier:
            x, y = frontier.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbour = (x + dx, y + dy)
                    if neighbour in remaining:
                        remaining.discard(neighbour)
                        blob.add(neighbour)
                        frontier.append(neighbour)
        blobs.append(blob)
    return blobs


def test_the_minefield_lands_at_the_measured_density() -> None:
    """The travelled-area density, not the spawn carpet."""
    world = make_sim_world("field01_r.gif")
    terrain = InMemoryTerrainMap()

    laid = seed_minefield(world, terrain)

    passable = sum(1 for x in range(256) for y in range(256) if terrain.is_passable(x, y))
    assert abs(laid / passable - MINE_DENSITY) < 0.01
    assert len(world["mines"]) == laid


def test_the_field_splits_into_the_archives_component_shapes() -> None:
    """Mostly single mines and press-sized blobs, as the archive shows.

    Shape is the load-bearing property: 2,236 components lifted from 27
    real captures are 39.3% single mines and 52.5% blobs of 12 tiles or
    fewer, and the gaps between them are what a route needs
    ([[session-state-deglobalisation]]).
    """
    world = make_sim_world("field01_r.gif")
    seed_minefield(world, InMemoryTerrainMap())

    blobs = _components(world)
    singles = sum(1 for blob in blobs if len(blob) == 1)
    small = sum(1 for blob in blobs if 1 < len(blob) <= 12)
    assert abs(singles / len(blobs) - _ARCHIVED_SINGLE_SHARE) < 0.05
    assert small / len(blobs) >= _ARCHIVED_SMALL_SHARE
    assert singles + small == len(blobs)


def test_components_are_solid_like_the_archives() -> None:
    """The archive's blobs fill their bounding box at a median 1.00.

    Not every one: two neighbours that touch at a corner make a
    sparser merged blob, which the archive shows too (fill runs down
    to 0.27). The MEDIAN is the law.
    """
    world = make_sim_world("field01_r.gif")
    seed_minefield(world, InMemoryTerrainMap())

    fills: list[float] = []
    for blob in _components(world):
        xs = [tile[0] for tile in blob]
        ys = [tile[1] for tile in blob]
        box = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        fills.append(len(blob) / box)
    fills.sort()
    assert fills[len(fills) // 2] == 1.0
    assert fills[0] >= 0.27


def test_the_client_window_meets_mines_without_being_carpeted() -> None:
    """An 18x18 window holds enough mines to route around, not a wall.

    The archive's OPENING patch is 88-159 — but that is spawn ground.
    What this pins is that the bot's window is neither empty (the old
    sim) nor solid (the first cut, which starved the router).
    """
    world = make_sim_world("field01_r.gif")
    seed_minefield(world, InMemoryTerrainMap())

    in_patch = sum(
        1
        for key in world["mines"]
        for x, y in [tuple(int(part) for part in key.split(","))]
        if 100 <= x < 100 + VIEWPORT_PATCH_WIDTH and 100 <= y < 100 + VIEWPORT_PATCH_HEIGHT
    )
    assert 20 <= in_patch <= 88


def test_mines_never_land_under_a_living_tank() -> None:
    """A placement never resolves onto an occupied tile."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 100, 100, 1100)
    world["tanks"][11] = make_sim_tank(11, 1, 1, 101, 100, 500)
    world["tanks"][12] = make_sim_tank(12, 3, 1, 102, 100, 500)
    world["tanks"][12]["alive"] = False

    seed_minefield(world, InMemoryTerrainMap())

    assert "100,100" not in world["mines"]
    assert "101,100" not in world["mines"]


def test_mines_share_tiles_with_containers() -> None:
    """Containers and mines coexist, which is where clearance matters.

    [[mine-mechanics]]: "Containers can coexist with mines on the same
    tile." Seeding mines around the containers would have left the
    bot's clearance and landing-displacement paths unexercised on
    exactly the tiles it cares about.
    """
    from tankpit_bot.sim.world import SimContainerDict

    world = make_sim_world("field01_r.gif")
    for x in range(100, 140):
        world["containers"].append(SimContainerDict(x=x, y=100, volume=500, dotted=True))

    seed_minefield(world, InMemoryTerrainMap())

    shared = [c for c in world["containers"] if f"{c['x']},{c['y']}" in world["mines"]]
    assert shared != []


def test_the_minefield_is_deterministic() -> None:
    """The same field yields the same minefield, so soaks replay."""
    first = make_sim_world("field01_r.gif")
    second = make_sim_world("field01_r.gif")
    terrain = InMemoryTerrainMap()

    seed_minefield(first, terrain)
    seed_minefield(second, terrain)

    assert first["mines"] == second["mines"]


def test_owning_teams_follow_the_archived_skew() -> None:
    """Teams 3 and 0 own the field; 1 and 2 are too rare to place.

    The patch census counted 13,672 tiles for team 3 and 7,519 for
    team 0 — 64.5% / 35.5% — against 35 and 4 for teams 2 and 1, a
    combined 0.18% that no sample size justifies inventing a rate for.
    """
    world = make_sim_world("field01_r.gif")
    seed_minefield(world, InMemoryTerrainMap())

    teams = {mine["team"] for mine in world["mines"].values()}
    assert teams == {0, 3}
    assert teams == set(MINE_TEAM_CYCLE)
    threes = sum(1 for mine in world["mines"].values() if mine["team"] == 3)
    assert abs(threes / len(world["mines"]) - 0.645) < 0.02


def test_rock_never_carries_a_mine() -> None:
    """Impassable terrain is skipped, as a real placement skips it."""
    world = make_sim_world("field01_r.gif")
    rocks = {(x, 50): "#" for x in range(256)}

    seed_minefield(world, InMemoryTerrainMap(terrain_data=rocks))

    assert [key for key in world["mines"] if key.endswith(",50")] == []


def test_the_real_field_gets_a_minefield_the_walk_can_afford() -> None:
    """On the real GIF the field seeds, and a walk stays fast.

    The list form cost 1,902 ms per walk at this size, which is why
    there was no minefield at all before the mapping
    ([[session-state-deglobalisation]]).
    """
    from tankpit_bot.sim.movement import process_move

    terrain = _test_hooks.load_terrain_map(Path("field01_r.gif"))
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 100, 100, 1100)
    laid = seed_minefield(world, terrain)

    # Slightly UNDER the open-terrain figure: field01's passable ground
    # is fragmented, so components lose tiles to rock at their edges.
    passable = sum(1 for x in range(256) for y in range(256) if terrain.is_passable(x, y))
    assert MINE_DENSITY - 0.03 < laid / passable <= MINE_DENSITY
    outcome = process_move(world, terrain, 9, 104, 100)
    assert outcome["kind"] in ("moved", "cant_go")
