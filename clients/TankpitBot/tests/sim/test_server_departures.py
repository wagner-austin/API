"""A connection quitting a running field: the tank leaves, the room is told.

``SimServer.disconnect`` is the inverse of ``connect``. Tanks 9 and 20
are connected and in the room, standing in each other's window; tank 30
is on the field with no connection. When 20's player quits, 9 is told
with the 0x29 a departing churn visitor draws, and every trace the field
kept of 20 goes with it.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import ClientCommandDict, ClientCommandKind, SimError
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.wire_statements import exit_statement
from tankpit_bot.sim.world import make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _room() -> SimServer:
    """Tanks 9 and 20 connected and joined side by side; 30 unconnected."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 100, 100, 1000)
    world["tanks"][20] = make_sim_tank(20, 1, 1, 104, 100, 1000)
    world["tanks"][30] = make_sim_tank(30, 2, 1, 200, 200, 1000)
    server = SimServer(world, InMemoryTerrainMap())
    for tank_id in (9, 20):
        server.connect(tank_id)
        server.handshake(tank_id)
    server.advance_tick()
    return server


def _command(kind: ClientCommandKind, *, x: int = 0, y: int = 0) -> ClientCommandDict:
    """A decoded client command of one kind."""
    return ClientCommandDict(
        kind=kind,
        command=0,
        x=x,
        y=y,
        target_id=0,
        slot=0,
        message_id=0,
        direction=0,
        amount=0,
    )


def _types(batch: list[BinaryMessage]) -> list[int | str]:
    """The msg_type of every message in a batch, in order."""
    return [message["msg_type"] for message in batch]


def test_the_exit_statement_is_a_plain_announced_departure() -> None:
    """Every archived exit with a matching entry: not silent, not eliminated."""
    assert exit_statement(1, 20) == {
        "msg_type": 0x29,
        "team": 1,
        "tank_id": 20,
        "was_silent": False,
        "was_eliminated": False,
    }


def test_a_quitting_player_leaves_and_the_room_is_told() -> None:
    """The tank is gone, its batch is gone, and 9 reads the 0x29."""
    server = _room()

    server.disconnect(20)
    batches = server.advance_tick()

    assert list(batches) == [9]
    assert 20 not in server.world["tanks"]
    assert server.session_for(20) is None
    assert [m for m in batches[9] if m["msg_type"] == 0x29] == [exit_statement(1, 20)]
    # 20 was in 9's view; it left by quitting, so no 0x58 follows the 0x29.
    assert 0x58 not in _types(batches[9])
    assert 20 not in server.require_session(9).viewport.visible


def test_a_quitting_players_queued_commands_go_with_it() -> None:
    """A chat queued before the quit is never processed."""
    server = _room()
    server.queue_command(20, _command(ClientCommandKind.CHAT, x=104, y=100))
    server.queue_command(9, _command(ClientCommandKind.CHAT, x=100, y=100))

    server.disconnect(20)
    batch = server.advance_tick()[9]

    assert [m["sender_id"] for m in batch if m["msg_type"] == 0x4D] == [9]


def test_a_quitting_shooters_unbilled_cost_is_dropped() -> None:
    """The shot's cost was due next tick; with the shooter gone it is not billed.

    The tick after the quit runs cleanly, which it could not if the
    deferred debit still named a tank the field no longer holds.
    """
    server = _room()
    server.queue_command(20, _command(ClientCommandKind.SHOOT, x=100, y=100))
    fired = server.advance_tick()[9]
    assert [m["shooter_id"] for m in fired if m["msg_type"] == 0x53] == [20]

    server.disconnect(20)
    batch = server.advance_tick()[9]

    assert exit_statement(1, 20) in batch


def test_closing_a_tank_with_no_connection_is_refused() -> None:
    """The unconnected tank 30 has no connection to close."""
    server = _room()

    with pytest.raises(SimError, match="tank 30 has no connection to close"):
        server.disconnect(30)
    assert 30 in server.world["tanks"]
