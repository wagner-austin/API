"""Two connections on one field: who receives what.

The Phase 2 watch, written down as tests before the bot harness ever
drives two tanks: a per-recipient receipt must reach exactly one
connection, and a broadcast must reach all of them
([[recipient-policy]]). Tank 9 and tank 20 are connected and stand in
each other's window; tank 30 is on the field with no connection, the
way a practice bot or the scripted opponent is.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol.commands import SCOPE_EAST
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import ClientCommandDict, ClientCommandKind, SimError
from tankpit_bot.sim.progression import DEMOTED_RANK, PROMOTED_RANK, PROMOTION_RECOVERY_TICKS
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _field() -> SimServer:
    """Tanks 9 and 20 connected side by side; tank 30 unconnected, far off."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 100, 100, 1000)
    world["tanks"][20] = make_sim_tank(20, 1, 1, 104, 100, 1000)
    world["tanks"][30] = make_sim_tank(30, 2, 1, 200, 200, 1000)
    server = SimServer(world, InMemoryTerrainMap())
    server.connect(9)
    server.connect(20)
    return server


def _command(
    kind: ClientCommandKind,
    *,
    x: int = 0,
    y: int = 0,
    message_id: int = 0,
    direction: int = 0,
) -> ClientCommandDict:
    """A decoded client command of one kind."""
    return ClientCommandDict(
        kind=kind,
        command=0,
        x=x,
        y=y,
        target_id=0,
        slot=0,
        message_id=message_id,
        direction=direction,
        amount=0,
    )


def _types(batch: list[BinaryMessage]) -> list[int | str]:
    """The msg_type of every message in a batch, in order."""
    return [message["msg_type"] for message in batch]


def test_batches_are_keyed_by_connection_in_join_order() -> None:
    """One batch per connected tank, and none for the unconnected one."""
    batches = _field().advance_tick()

    assert list(batches) == [9, 20]


def _joined_host() -> SimServer:
    """Tank 9 connected and sent its join burst; 20 and 30 on the field, unconnected."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 100, 100, 1000)
    world["tanks"][20] = make_sim_tank(20, 1, 1, 104, 100, 1000)
    world["tanks"][30] = make_sim_tank(30, 2, 1, 200, 200, 1000)
    server = SimServer(world, InMemoryTerrainMap())
    server.connect(9)
    server.handshake(9)
    return server


def test_a_joiner_is_announced_to_the_room_and_not_to_itself() -> None:
    """The 0x28 a churn visitor draws announces a connection joining.

    Tank 9 was already in the room, so it learns of 20; tank 20 learns
    the room from its own join burst instead.
    """
    server = _joined_host()
    server.connect(20)
    batches = server.advance_tick()

    entries = [m for m in batches[9] if m["msg_type"] == 0x28]
    assert entries == [
        {
            "msg_type": 0x28,
            "team": 1,
            "tank_id": 20,
            "rank": 1,
            "damage_state": 3,
            "score": 0,
            "x": 0,
            "y": 0,
        }
    ]
    assert 0x28 not in _types(batches[20])


def test_a_connection_still_joining_is_not_told_of_an_arrival() -> None:
    """Its own join burst lists the room, so no 0x28 is owed to it.

    Neither tank in the plain field has been sent its burst, so neither
    is told of the other's arrival.
    """
    batches = _field().advance_tick()

    assert 0x28 not in _types(batches[9])
    assert 0x28 not in _types(batches[20])


def test_a_joiner_in_view_is_placed_again_after_its_positionless_entry() -> None:
    """An entry reports (0, 0), so a joiner already in view is re-placed.

    Tank 20 stands in 9's window, so 9's view already holds it; when
    20's player connects, 9 gets the 0x28 and then, from the membership
    pass, the 0x3D that says where the tank really is.
    """
    server = _joined_host()
    session = server.require_session(9)
    assert 20 in session.viewport.visible

    server.connect(20)
    batch = server.advance_tick()[9]

    entry = _types(batch).index(0x28)
    placements = [
        index
        for index, message in enumerate(batch)
        if message["msg_type"] == 0x3D and message["tank_id"] == 20
    ]
    assert len(placements) == 1
    assert placements[0] > entry


def test_connecting_an_absent_dead_or_connected_tank_is_refused() -> None:
    """One living tank, one connection."""
    server = _field()
    server.world["tanks"][30]["alive"] = False

    with pytest.raises(SimError, match="no living tank 41 to connect"):
        server.connect(41)
    with pytest.raises(SimError, match="no living tank 30 to connect"):
        server.connect(30)
    with pytest.raises(SimError, match="tank 20 is already connected"):
        server.connect(20)


def test_a_join_burst_needs_a_connection() -> None:
    """The burst is a connection's own answer; an unconnected tank has none."""
    server = _field()

    with pytest.raises(SimError, match="tank 30 has no connection"):
        server.handshake(30)
    assert server.require_session(20).client_id == 20
    assert [session.client_id for session in server.sessions] == [9, 20]


def test_a_refusal_reaches_the_refused_connection_only() -> None:
    """A move outside tank 20's window draws a 0x52 to tank 20 alone."""
    server = _field()
    server.queue_command(20, _command(ClientCommandKind.MOVE, x=150, y=100))

    batches = server.advance_tick()

    assert 0x52 in _types(batches[20])
    assert 0x52 not in _types(batches[9])


@pytest.mark.parametrize(
    ("kind", "answer"),
    [
        (ClientCommandKind.MAP_OPEN, 0x4C),
        (ClientCommandKind.INVENTORY, 0x49),
        (ClientCommandKind.STATISTICS, 0x56),
        (ClientCommandKind.ENTER_GAME, 0x3E),
    ],
)
def test_a_connection_query_is_answered_to_the_asker_only(
    kind: ClientCommandKind, answer: int
) -> None:
    """Each answer goes to the connection that asked, never the other."""
    server = _field()
    server.queue_command(20, _command(kind))

    batches = server.advance_tick()

    assert answer in _types(batches[20])
    assert answer not in _types(batches[9])


def test_a_query_from_an_unconnected_tank_is_answered_to_nobody() -> None:
    """Nobody asked on a socket, so no socket hears an answer."""
    expected = _field().advance_tick()
    server = _field()
    for kind in (ClientCommandKind.MAP_OPEN, ClientCommandKind.ENTER_GAME):
        server.queue_command(30, _command(kind))

    assert server.advance_tick() == expected


def test_a_chat_is_broadcast_identically_including_the_sender() -> None:
    """The 0x4D echo is the sender's receipt AND everyone else's message."""
    server = _field()
    server.queue_command(9, _command(ClientCommandKind.CHAT, x=1, y=2, message_id=7))

    batches = server.advance_tick()

    chats = {tank: [m for m in batch if m["msg_type"] == 0x4D] for tank, batch in batches.items()}
    assert chats[9] == chats[20]
    assert chats[9] == [{"msg_type": 0x4D, "sender_id": 9, "message_type": 7, "x": 1, "y": 2}]


def test_one_outcome_is_narrated_to_each_observer_through_its_own_eyes() -> None:
    """An equipment toggle is resolved once and shown only to its actor."""
    server = _field()
    server.queue_command(20, _command(ClientCommandKind.TOGGLE_EQUIPMENT))

    batches = server.advance_tick()

    assert _types(batches[20]).count(0x74) == 1
    assert 0x74 not in _types(batches[9])


def test_each_connection_syncs_fuel_for_its_own_tank_only() -> None:
    """The long-form 0x2E carries fuel to the tank's own connection."""
    batches = _field().advance_tick()

    for tank, batch in batches.items():
        fueled = [m["tank_id"] for m in batch if m["msg_type"] == 0x2E and m["fuel"] is not None]
        assert fueled == [tank]


def test_a_scope_pan_moves_only_the_panning_connection_s_window() -> None:
    """Every connection keeps its own stored 0x5A window."""
    server = _field()
    before_9 = server.require_session(9).viewport.window
    before_20 = server.require_session(20).viewport.window
    server.queue_command(20, _command(ClientCommandKind.SCOPE, direction=SCOPE_EAST))

    batches = server.advance_tick()

    assert server.require_session(9).viewport.window == before_9
    assert server.require_session(20).viewport.window != before_20
    assert 0x5A in _types(batches[20])
    assert 0x5A not in _types(batches[9])


def test_a_dead_connected_tank_s_clicks_drop_silently() -> None:
    """A connection outlives its tank; the corpse's clicks are ignored."""
    server = _field()
    server.world["tanks"][20]["alive"] = False

    server.queue_command(20, _command(ClientCommandKind.MOVE, x=104, y=101))

    assert 0x52 not in _types(server.advance_tick()[20])


def test_a_relocation_is_restated_to_every_connection_that_sees_it() -> None:
    """A recorded relocation inside both windows draws a 0x3D in both."""
    server = _field()
    server.advance_tick()
    server.relocate_tank(20, 103, 100)

    batches = server.advance_tick()

    restated = [m for m in batches[9] if m["msg_type"] == 0x3D and m["tank_id"] == 20]
    assert len(restated) == 1
    assert not [m for m in batches[20] if m["msg_type"] == 0x3D and m["tank_id"] == 20]


def test_an_activation_is_announced_to_every_connection() -> None:
    """A 0x21 identity broadcast heads every batch of the next tick."""
    server = _field()
    server.advance_tick()
    server.announce_tank(30)

    batches = server.advance_tick()

    assert batches[9][0] == batches[20][0]
    assert batches[9][0]["msg_type"] == 0x21


def test_a_promotion_closed_for_one_connection_reaches_every_sync() -> None:
    """Every connection's rank settles before any connection's syncs.

    Tank 20 finishes its recovery window this tick. Tank 9's batch is
    built FIRST, so if syncs were emitted inside the same pass as the
    promotion, tank 9 would read tank 20 at the demoted rank.
    """
    server = _field()
    server.world["tanks"][20]["rank"] = DEMOTED_RANK
    server.require_session(20).progression.demoted_tick = (
        server.world["tick"] + 1 - PROMOTION_RECOVERY_TICKS
    )

    batches = server.advance_tick()

    seen_by_9 = [m["rank"] for m in batches[9] if m["msg_type"] == 0x2E and m["tank_id"] == 20]
    assert seen_by_9 == [PROMOTED_RANK]
