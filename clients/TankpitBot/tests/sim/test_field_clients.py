"""Who the rival bots of a multi-bot field are: their seats and accounts."""

from __future__ import annotations

import pytest

from tankpit_bot.sim.field_clients import (
    MAX_FIELD_CLIENTS,
    RIVAL_RING_MAX,
    RIVAL_RING_MIN,
    RIVAL_TEAM,
    FieldSeatError,
    field_account,
    seed_field_rivals,
)
from tankpit_bot.sim.lobby import SIM_ACCOUNT
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

PRIMARY = 9


def _world() -> SimWorldDict:
    """The primary client, rank 3 with partial stocks and armor off, at (100, 100)."""
    world = make_sim_world("field01_r.gif")
    primary = make_sim_tank(PRIMARY, 2, 3, 100, 100, 900)
    primary["counts"] = [3, 7, 0, 12, 4]
    primary["enabled"] = [False, True, True, True, True]
    world["tanks"][PRIMARY] = primary
    return world


def test_rivals_take_the_next_ids_and_alternate_sides() -> None:
    """The first rival plays RIVAL_TEAM, the next the primary's own team."""
    world = _world()

    seated = seed_field_rivals(world, InMemoryTerrainMap(), PRIMARY, 3)

    assert seated == (10, 11, 12)
    assert [world["tanks"][tank_id]["team"] for tank_id in seated] == [RIVAL_TEAM, 2, RIVAL_TEAM]


def test_rivals_mirror_the_primary_so_a_duel_has_no_handicap() -> None:
    """Rank, fuel, stocks and enabled slots are the primary's; names are bot-shaped."""
    world = _world()

    (rival,) = seed_field_rivals(world, InMemoryTerrainMap(), PRIMARY, 1)

    tank = world["tanks"][rival]
    assert (tank["rank"], tank["fuel"]) == (3, 900)
    assert tank["counts"] == [3, 7, 0, 12, 4]
    assert tank["enabled"] == [False, True, True, True, True]
    assert tank["name"] == "red-10"
    # A copy, not the primary's own list: the rival's stocks spend alone.
    tank["counts"][0] = 0
    assert world["tanks"][PRIMARY]["counts"][0] == 3


def test_rivals_stand_in_the_ring_band_on_distinct_tiles() -> None:
    """Each rival is 8-14 tiles (Chebyshev) from the primary, none sharing a tile."""
    world = _world()

    seated = seed_field_rivals(world, InMemoryTerrainMap(), PRIMARY, MAX_FIELD_CLIENTS - 1)

    tiles = {(world["tanks"][t]["x"], world["tanks"][t]["y"]) for t in seated}
    assert len(tiles) == len(seated)
    for x, y in tiles:
        assert RIVAL_RING_MIN <= max(abs(x - 100), abs(y - 100)) <= RIVAL_RING_MAX


def test_seating_none_changes_nothing() -> None:
    """Zero rivals is the one-bot session."""
    world = _world()

    assert seed_field_rivals(world, InMemoryTerrainMap(), PRIMARY, 0) == ()
    assert list(world["tanks"]) == [PRIMARY]


def test_more_bots_than_a_field_seats_is_refused() -> None:
    """The bound counts the primary too."""
    with pytest.raises(FieldSeatError, match="SIM_FIELD_SEATS: 9 bots asked for"):
        seed_field_rivals(_world(), InMemoryTerrainMap(), PRIMARY, MAX_FIELD_CLIENTS)


def test_a_taken_id_is_refused() -> None:
    """A rival never overwrites a tank already on the field."""
    world = _world()
    world["tanks"][10] = make_sim_tank(10, 1, 1, 50, 50, 500)

    with pytest.raises(FieldSeatError, match="SIM_FIELD_SEAT_TAKEN: tank id 10"):
        seed_field_rivals(world, InMemoryTerrainMap(), PRIMARY, 1)


def test_a_closed_ring_band_is_refused() -> None:
    """With nothing passable around the primary there is nowhere to seat."""
    terrain = InMemoryTerrainMap.from_passable_set({(100, 100)})

    with pytest.raises(FieldSeatError, match=r"SIM_FIELD_NO_TILE: no open tile 8-14 .*\(100,100\)"):
        seed_field_rivals(_world(), terrain, PRIMARY, 1)


def test_a_rivals_account_names_it_and_keeps_the_rest_of_the_record() -> None:
    """The lobby and the room agree about who joined."""
    world = _world()
    (rival,) = seed_field_rivals(world, InMemoryTerrainMap(), PRIMARY, 1)

    account = field_account(world, rival)

    assert account == {
        "game_start": SIM_ACCOUNT["game_start"],
        "name": "red-10",
        "rank": 3,
        "active_forces": SIM_ACCOUNT["active_forces"],
    }
