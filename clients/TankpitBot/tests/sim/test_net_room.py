"""One networked room: its lobby row, its world, and the players it seats."""

from __future__ import annotations

import pytest

from tankpit_bot.sim.commands import SimError
from tankpit_bot.sim.field_choice import FieldChoiceError
from tankpit_bot.sim.lobby import SimAccountDict
from tankpit_bot.sim.net_room import (
    MAX_ROOM_PLAYERS,
    NET_PLAYER_ID_BASE,
    NET_PLAYER_IDS,
    NetError,
    NetRoom,
    field_room_info,
    open_field_room,
)
from tankpit_bot.sim.scenarios import SIM_CLIENT_ID, SIM_FIELD
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim._net_client import TOKEN, account_book

_LAYOUT = "bot-20260706-223721"


def _player() -> SimAccountDict:
    """The first test account, as the book admits it."""
    return account_book().verify("1001", TOKEN)


def _arena() -> NetRoom:
    """An open room on field01's real terrain with no roster."""
    return open_field_room(
        field_room_info("1", "Arena", "field01.gif", practice=False),
        layout=_LAYOUT,
        population_seed=7,
    )


def test_a_rooms_lobby_row_follows_its_field_and_mode() -> None:
    """Practice rows carry mode p and no game modes; open rows mode n and the archive's modes."""
    assert field_room_info("1", "Practice", "field01.gif", practice=True) == {
        "room_id": "1",
        "name": "Practice",
        "field_id": 1,
        "game_modes": "0,0,0,0,0,0,0",
        "default_troop": 2,
        "mode_code": "p",
        "image": "field01.gif",
        "year": "2026",
    }
    desert = field_room_info("5", "World (Desert)", "field05.gif", practice=False)
    assert (desert["field_id"], desert["mode_code"], desert["game_modes"]) == (
        5,
        "n",
        "1,1,1,0,1,0,0",
    )
    with pytest.raises(FieldChoiceError, match="SIM_FIELD_UNKNOWN"):
        field_room_info("9", "Nowhere", "field99.gif", practice=False)


def test_a_practice_room_holds_the_roster_and_no_layout_client() -> None:
    """The layout's client spawn is not a player; the roster plays on with nobody seated."""
    room = open_field_room(
        field_room_info("1", "Practice", "field01.gif", practice=True),
        layout=_LAYOUT,
        population_seed=7,
    )
    tanks = room.server.world["tanks"]
    assert SIM_CLIENT_ID not in tanks
    assert len(tanks) == 36
    assert room.server.world["field"] == SIM_FIELD

    assert room.advance() == {}
    assert room.server.world["tick"] == 1


def test_a_room_on_another_field_plays_that_field() -> None:
    """The row's image names the terrain the world is built on."""
    room = open_field_room(
        field_room_info("5", "Desert", "field05.gif", practice=False),
        layout=_LAYOUT,
        population_seed=7,
    )
    assert room.server.world["field"] == "field05_r.gif"


def test_a_seated_player_stands_on_open_ground_under_its_account() -> None:
    """Name and rank from the account, team from the troop, full fuel for the rank."""
    room = _arena()

    tank_id = room.seat(_player(), 1)

    tank = room.server.world["tanks"][tank_id]
    assert tank_id == NET_PLAYER_ID_BASE
    assert (tank["name"], tank["team"], tank["rank"], tank["alive"]) == ("austin", 1, 3, True)
    assert room.server.terrain.is_passable(tank["x"], tank["y"])
    assert list(room.advance()) == [tank_id]


@pytest.mark.parametrize("troop", [-1, 4])
def test_a_troop_that_is_no_team_is_refused(troop: int) -> None:
    """Teams are 0 to 3, the wire's own color ids."""
    with pytest.raises(NetError, match=f"SIM_NET_TROOP: troop {troop} is not a team"):
        _arena().seat(_player(), troop)


def test_a_full_room_refuses_the_next_player() -> None:
    """At most MAX_ROOM_PLAYERS seated at once."""
    room = _arena()
    for _ in range(MAX_ROOM_PLAYERS):
        room.seat(_player(), 0)
    with pytest.raises(NetError, match=f"SIM_NET_ROOM_FULL: room 1 seats {MAX_ROOM_PLAYERS}"):
        room.seat(_player(), 0)


def test_ids_are_never_reused_and_run_out() -> None:
    """A player who leaves frees a seat but not an id."""
    room = _arena()
    for _ in range(NET_PLAYER_IDS):
        room.leave(room.seat(_player(), 0))
    with pytest.raises(NetError, match="SIM_NET_IDS_EXHAUSTED: room 1 gave every id"):
        room.seat(_player(), 0)


def test_a_field_with_no_open_tile_seats_nobody() -> None:
    """Rock everywhere is a full room, said by name."""
    info = field_room_info("1", "Rock", "field01.gif", practice=False)
    room = NetRoom(
        info, SimServer(make_sim_world(SIM_FIELD), InMemoryTerrainMap(default="#")), None
    )
    with pytest.raises(NetError, match="SIM_NET_ROOM_FULL: room 1 has no open tile"):
        room.seat(_player(), 0)


def test_leaving_without_a_seat_is_a_harness_error() -> None:
    """Only a connected tank can leave."""
    with pytest.raises(SimError, match="has no connection to close"):
        _arena().leave(NET_PLAYER_ID_BASE)
