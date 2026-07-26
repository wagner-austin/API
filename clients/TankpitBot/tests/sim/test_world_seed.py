"""World seeding: mined layouts, static population, exposure atlas."""

from __future__ import annotations

import pytest

from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.map import MAP_DOT_MIN_VOLUME
from tankpit_bot.protocol.types import ShootEventDict
from tankpit_bot.sim.practice_room import PracticeRoomDriver
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tankpit_bot.sim.world_seed import (
    DOTTED_FUEL_COUNT,
    DOTTED_STOCKED_PERIOD,
    HIDDEN_DRAINED_PERIOD,
    HIDDEN_EQUIPMENT_COUNT,
    HIDDEN_FUEL_COUNT,
    PRACTICE_LAYOUTS,
    SMALL_PERIOD,
    seed_field_population,
    seed_practice_client,
    select_practice_layout,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _world() -> SimWorldDict:
    """An empty field01 world."""
    return make_sim_world("field01_r.gif")


def test_layouts_are_full_practice_rooms() -> None:
    """Every mined layout carries the real 36-bot roster shape.

    Ids 500-535, exactly 9 bots per team, ranks 0-1 only — the shape
    every one of the 223 archived sessions shows.
    """
    for layout in PRACTICE_LAYOUTS:
        roster = layout["roster"]
        assert len(roster) == 36
        assert [row[0] for row in roster] == list(range(500, 536))
        teams = [row[1] for row in roster]
        assert all(teams.count(team) == 9 for team in range(4))
        assert all(row[2] in (0, 1) for row in roster)


def test_layout_selection_is_stamp_deterministic() -> None:
    """Same stamp, same layout; the three layouts are all reachable."""
    assert select_practice_layout("stamp-a") is select_practice_layout("stamp-a")
    chosen = {select_practice_layout(f"stamp-{n}")["provenance"] for n in range(24)}
    assert chosen == {layout["provenance"] for layout in PRACTICE_LAYOUTS}


def test_population_counts_mix_and_exposure_threshold() -> None:
    """The seeded field carries the documented static population.

    Dotted containers hold fuel at the measured ~40% rate (2 of every
    5); hidden fuel is never dotted, half drained, sub-500 at the
    measured 2-in-5 of stocked;
    every stocked dotted container satisfies the ≥500 dot law; no two
    entities share a tile; nothing sits in the unencodable pre-(1,1)
    atlas region.
    """
    world = _world()
    seed_field_population(world, InMemoryTerrainMap(), seed=7)
    dotted = [c for c in world["containers"] if c["dotted"]]
    hidden = [c for c in world["containers"] if not c["dotted"]]
    assert len(dotted) == DOTTED_FUEL_COUNT
    assert len(hidden) == HIDDEN_FUEL_COUNT
    assert len(world["equipment"]) == HIDDEN_EQUIPMENT_COUNT
    stocked = [c for c in dotted if c["volume"] > 0]
    assert len(stocked) == DOTTED_FUEL_COUNT // DOTTED_STOCKED_PERIOD * 2
    assert all(c["volume"] >= MAP_DOT_MIN_VOLUME for c in stocked)
    drained = [c for c in hidden if c["volume"] == 0]
    assert len(drained) == HIDDEN_FUEL_COUNT // HIDDEN_DRAINED_PERIOD
    stocked_hidden = len(hidden) - len(drained)
    small = [c for c in hidden if 0 < c["volume"] < MAP_DOT_MIN_VOLUME]
    assert len(small) == stocked_hidden // SMALL_PERIOD * 2
    tiles = [(c["x"], c["y"]) for c in world["containers"]]
    tiles.extend((e["x"], e["y"]) for e in world["equipment"])
    assert len(tiles) == len(set(tiles))
    assert all(y * 256 + x >= 257 for x, y in tiles)


def test_population_seeding_is_deterministic_and_skips_walls() -> None:
    """Same seed, same world; a wall on a probe tile is walked past."""
    plain_a = _world()
    seed_field_population(plain_a, InMemoryTerrainMap(), seed=3)
    plain_b = _world()
    seed_field_population(plain_b, InMemoryTerrainMap(), seed=3)
    assert plain_a["containers"] == plain_b["containers"]
    first = (plain_a["containers"][0]["x"], plain_a["containers"][0]["y"])
    walled = _world()
    seed_field_population(walled, InMemoryTerrainMap(terrain_data={first: "#"}), seed=3)
    walled_tiles = {(c["x"], c["y"]) for c in walled["containers"]}
    assert first not in walled_tiles


def test_sealed_map_fails_population_seeding_loudly() -> None:
    """A map with no open tile raises instead of under-seeding."""
    sealed = InMemoryTerrainMap(terrain_data={(x, y): "#" for x in range(256) for y in range(256)})
    with pytest.raises(RuntimeError, match="no open tile"):
        seed_field_population(_world(), sealed, seed=1)


def test_practice_client_boots_full_at_the_mined_spawn() -> None:
    """The client lands at the layout's real join spawn, hunt-ready."""
    world = _world()
    layout = PRACTICE_LAYOUTS[0]
    seed_practice_client(world, InMemoryTerrainMap(), layout, 9)
    client = world["tanks"][9]
    assert (client["x"], client["y"]) == layout["client_spawn"]
    assert client["fuel"] == fuel_capacity(client["rank"])
    assert client["counts"] == [25, 25, 25, 25, 25]


def test_practice_client_seeding_fails_loudly_on_sealed_ground() -> None:
    """No open tile near the mined spawn is a boot error, not a guess."""
    sealed = InMemoryTerrainMap(terrain_data={(x, y): "#" for x in range(256) for y in range(256)})
    with pytest.raises(RuntimeError, match="no open tile near layout spawn"):
        seed_practice_client(_world(), sealed, PRACTICE_LAYOUTS[0], 9)


def test_practice_room_notes_hits_and_returns_fire() -> None:
    """A landed 0x53 on a roster bot queues its shot-for-shot return
    and the next ``decide_all`` emits it (plus sighted team aggro)."""
    world = _world()
    world["tanks"][9] = make_sim_tank(9, 2, 1, 100, 100, 1100)
    driver = PracticeRoomDriver(
        world,
        InMemoryTerrainMap(),
        9,
        ((510, 1, 0, 101, 100), (511, 1, 0, 103, 100)),
    )
    shot = ShootEventDict(
        msg_type=0x53,
        team=2,
        shooter_id=9,
        source_x=100,
        source_y=100,
        target_x=101,
        target_y=100,
        aim_x=101,
        aim_y=100,
        weapon=1,
    )
    driver.note_batch(world, [shot])
    assert driver.states[510]["has_pending_return"] is True
    assert driver.states[511]["has_pending_return"] is True
    decisions = driver.decide_all(world, InMemoryTerrainMap())
    kinds = {bot_id: command["kind"] for bot_id, command in decisions}
    assert kinds[510] == "shoot"
    assert kinds[511] == "shoot"
