"""The networked server's connections: AUTH, lobby, seat, play, quit, close.

The host is driven with the frames a page client sends, built by the
production builders, and what it answers is read through the production
capture decoder (:mod:`tests.sim._net_client`).
"""

from __future__ import annotations

import pytest

from tankpit_bot.sim.lobby import LobbyError
from tankpit_bot.sim.net_accounts import MemoryAccountBook, SeatResult
from tankpit_bot.sim.net_host import NetHost
from tankpit_bot.sim.net_room import NET_PLAYER_ID_BASE, NetError, field_room_info, open_field_room
from tankpit_bot.wire.helpers import DecodeError
from tests.sim._net_client import (
    ENTER,
    MAGIC,
    OTHER_ACCOUNT,
    OTHER_MAGIC,
    OTHER_TOKEN,
    QUIT,
    SELECT,
    account_book,
    auth,
    enter_game,
    move,
    payload,
    plaintext,
    received_kinds,
)

_JOIN_BURST = [0x21, 0x3E, 0x5A, 0x3D, 0x2E, 0x49, 0x49, 0x74, 0x3F]
"""The archived join burst for a room with no other tank: self block, then the tail."""


def _host(book: MemoryAccountBook | None = None) -> NetHost:
    """One open room on field01, no roster, over the two test accounts or a given book."""
    room = open_field_room(
        field_room_info("1", "Arena", "field01.gif", practice=False),
        layout="bot-20260706-223721",
        population_seed=7,
    )
    return NetHost((room,), account_book() if book is None else book)


def _seated(host: NetHost) -> int:
    """Open a connection and walk it through AUTH, select and enter."""
    connection = host.open()
    host.receive(connection, payload(auth()))
    host.receive(connection, payload(SELECT, ENTER))
    return connection


def test_a_client_walks_the_lobby_and_is_seated_on_the_team_it_entered_with() -> None:
    """AUTH draws the room list, select the confirm, enter the response and a tank."""
    host = _host()
    connection = host.open()

    assert plaintext(host.receive(connection, payload(auth()))) == [
        "+1|Arena|1|1,1,1,0,1,0,0|2|n|field01.gif|2026"
    ]
    assert plaintext(host.receive(connection, payload(SELECT))) == [
        "=1|Oct. 05, 2026|austin|3|9|9|9|9"
    ]
    assert plaintext(host.receive(connection, payload(ENTER))) == ["$1|0"]
    tank = host.rooms[0].server.world["tanks"][NET_PLAYER_ID_BASE]
    assert (tank["name"], tank["team"], tank["rank"]) == ("austin", 2, 3)
    assert [session.client_id for session in host.rooms[0].server.sessions] == [NET_PLAYER_ID_BASE]


def test_enter_game_is_answered_with_the_join_burst_on_the_next_tick() -> None:
    """The burst is an answer the client asks for, read back by the production decoder."""
    host = _host()
    connection = _seated(host)

    assert host.receive(connection, payload(enter_game())) == []
    burst = host.tick()[connection]

    assert received_kinds([burst])[: len(_JOIN_BURST)] == _JOIN_BURST


def test_a_move_command_moves_the_players_tank() -> None:
    """A command queues for the tank its connection speaks for."""
    host = _host()
    connection = _seated(host)
    tank = host.rooms[0].server.world["tanks"][NET_PLAYER_ID_BASE]
    start = (tank["x"], tank["y"])

    host.receive(connection, payload(move(tank["x"] + 1, tank["y"])))
    host.tick()

    assert (tank["x"], tank["y"]) != start


def test_a_second_player_is_announced_to_the_first_and_its_quit_too() -> None:
    """Two connections share the field: arrival is a 0x28, departure a 0x29."""
    host = _host()
    first = _seated(host)
    host.receive(first, payload(enter_game()))
    host.tick()
    second = host.open()
    host.receive(second, payload(auth(OTHER_ACCOUNT, OTHER_TOKEN, OTHER_MAGIC)))
    host.receive(second, payload(SELECT, ENTER))

    arrival = host.tick()
    assert received_kinds([arrival[first]])[0] == 0x28
    assert sorted(host.rooms[0].server.world["tanks"]) == [
        NET_PLAYER_ID_BASE,
        NET_PLAYER_ID_BASE + 1,
    ]

    assert plaintext(host.receive(second, payload(QUIT))) == ["-"]
    departure = host.tick()
    assert sorted(departure) == [first]
    assert received_kinds([departure[first]])[0] == 0x29
    with pytest.raises(NetError, match="SIM_NET_NOT_SEATED"):
        host.receive(second, payload(move(1, 1, OTHER_MAGIC)))


def test_closing_a_connection_takes_its_tank_off_the_field() -> None:
    """A socket that closes leaves the room the way a quit does."""
    host = _host()
    connection = _seated(host)
    idle = host.open()

    host.close(connection)
    host.close(idle)

    assert host.rooms[0].server.world["tanks"] == {}
    assert host.connections == 0
    with pytest.raises(NetError, match=f"SIM_NET_UNKNOWN: no open connection {connection}"):
        host.receive(connection, payload(SELECT))


def test_a_seat_that_leaves_is_recorded_and_the_account_rejoins_as_it_left() -> None:
    """Quit and close both record; a later join seats the recorded rank and levels."""
    book = account_book()
    host = _host(book)
    quitter = _seated(host)
    host.tick()
    host.tick()
    host.receive(quitter, payload(QUIT))
    host.close(quitter)

    assert book.results == [
        ("1001", SeatResult("1", "field01_r.gif", 2, 3, 0, 0, (1, 2, 0, 0, 0, 0, 0, 0, 0)))
    ]
    book.record("1001", SeatResult("1", "field01_r.gif", 9, 5, 4, 1, (1, 2, 1, 0, 0, 0, 0, 0, 0)))
    _seated(host)
    tank = host.rooms[0].server.world["tanks"][NET_PLAYER_ID_BASE + 1]
    session = host.rooms[0].server.require_session(NET_PLAYER_ID_BASE + 1)
    assert (tank["rank"], session.awards.levels) == (5, [1, 2, 1, 0, 0, 0, 0, 0, 0])


def test_a_connection_that_has_not_entered_is_sent_nothing() -> None:
    """Only seated connections have a batch."""
    host = _host()
    connection = host.open()
    host.receive(connection, payload(auth()))

    assert host.tick() == {}


def test_the_first_frame_must_be_an_auth() -> None:
    """A select before AUTH names no account."""
    host = _host()
    with pytest.raises(LobbyError, match="SIM_LOBBY_AUTH"):
        host.receive(host.open(), payload(SELECT))


def test_a_wrong_token_is_denied() -> None:
    """The book decides who joins."""
    host = _host()
    with pytest.raises(LobbyError, match="SIM_LOBBY_DENIED: account '1001' not admitted"):
        host.receive(host.open(), payload(auth(token="guessed")))


def test_an_account_joins_once_at_a_time() -> None:
    """A second connection for an account already connected is refused."""
    host = _host()
    host.receive(host.open(), payload(auth()))
    with pytest.raises(NetError, match="SIM_NET_IN_USE: account '1001' is connected"):
        host.receive(host.open(), payload(auth()))


def test_commands_need_a_seat_and_a_cipher() -> None:
    """Before AUTH a command cannot be read; after AUTH and before entry it has no tank."""
    host = _host()
    connection = host.open()
    with pytest.raises(DecodeError, match="SIM_COMMAND_BEFORE_AUTH"):
        host.receive(connection, payload(move(1, 1)))
    host.receive(connection, payload(auth()))
    with pytest.raises(NetError, match=f"SIM_NET_NOT_SEATED: connection {connection} has no tank"):
        host.receive(connection, payload(move(1, 1, MAGIC)))


def test_a_host_needs_rooms_with_distinct_ids() -> None:
    """No rooms, or two rooms under one id, is no host."""
    with pytest.raises(NetError, match=r"SIM_NET_ROOMS: .* not \[\]"):
        NetHost((), account_book())
    room = _host().rooms[0]
    with pytest.raises(NetError, match=r"not \['1', '1'\]"):
        NetHost((room, room), account_book())
