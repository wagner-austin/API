"""Tests for sim-server actions: teleport, radar, equipment, map, chat,
pickup, and the statistics key.
"""

from __future__ import annotations

from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_GO,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
)
from tankpit_bot.sim.commands import (
    ClientCommandDict,
)
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim._server_harness import (
    _command,
    _kinds,
    _server,
    _statistics_key,
    _supervisors,
)


def test_teleport_tick_emits_landing_position_and_sync() -> None:
    """A landed hop: teleport_landed, 0x3D, pickup, viewport exit, syncs.

    The hop to (30, 30) leaves enemy 11 at (15, 10) outside the
    viewport, so the batch carries its 0x58 TankRemove — the law-4
    reroute clock starts here.
    """
    server = _server()
    server.world["containers"].append(SimContainerDict(x=30, y=30, volume=40, dotted=True))
    server.queue_command(9, _command(("teleport", 116), 30, 30))
    messages = server.advance_tick()
    # Wire order law (archive-measured 2026-08-01, 7,176 live
    # teleports): the recentered 0x5A LEADS the landing batch, then
    # the position statement, then the landed confirm, then the
    # pickup — ``5A -> 3D -> landed -> pickup``. The 0x3D still
    # precedes the confirm (the displacement receipt reads position
    # at confirm time).
    assert _kinds(messages) == [
        0x5A,
        0x3D,
        "teleport_landed",
        "container_pickup",
        "container_pickup",
        0x58,
        0x2E,
        0x2E,
    ]
    assert (server.world["tanks"][9]["x"], server.world["tanks"][9]["y"]) == (30, 30)


def test_enemy_teleport_emits_no_client_viewport_update() -> None:
    """0x5A is the CLIENT's window — another tank's hop never moves it."""
    server = _server()
    server.queue_command(11, _command(("teleport", 116), 18, 12))
    messages = server.advance_tick()
    assert 0x5A not in _kinds(messages)
    assert (server.world["tanks"][11]["x"], server.world["tanks"][11]["y"]) == (18, 12)


def test_teleport_rejections_emit_supervisor_with_map_close() -> None:
    """An unaffordable hop surfaces as 0x52 with the map-close flag."""
    server = _server()
    server.world["tanks"][9]["fuel"] = 3
    server.queue_command(9, _command(("teleport", 116), 30, 30))
    poor = _supervisors(server.advance_tick())
    assert [(r["error_code"], r["close_map"]) for r in poor] == [
        (SUPERVISOR_ERROR_INSUFFICIENT_FUEL, 1)
    ]


def test_teleport_onto_sealed_tile_is_cant_go() -> None:
    """A fully sealed ring rejects the hop with cant_go."""
    world: SimWorldDict = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    walls = {(30, 30): "#", (31, 30): "#", (30, 29): "#", (29, 30): "#", (30, 31): "#"}
    server = SimServer(world, InMemoryTerrainMap(terrain_data=walls), client_id=9)
    server.queue_command(9, _command(("teleport", 116), 30, 30))
    rejected = _supervisors(server.advance_tick())
    assert [r["error_code"] for r in rejected] == [SUPERVISOR_ERROR_CANT_GO]


def test_teleport_to_empty_ground_has_no_pickup_message() -> None:
    """A landing on bare ground emits no container message."""
    server = _server()
    server.queue_command(9, _command(("teleport", 116), 30, 30))
    assert _kinds(server.advance_tick()) == [0x5A, 0x3D, "teleport_landed", 0x58, 0x2E, 0x2E]


def test_equipment_toggle_flips_the_slot_and_answers_0x74() -> None:
    """A toggle press flips the slot server-side and reports all five."""
    server = _server()
    toggle = ClientCommandDict(
        kind="toggle_equipment",
        command=114,
        x=0,
        y=0,
        target_id=0,
        slot=2,
        message_id=0,
        direction=0,
    )
    server.queue_command(9, toggle)
    messages = server.advance_tick()
    assert _kinds(messages) == [0x74, 0x2E, 0x2E]
    toggled = messages[0]
    assert toggled["msg_type"] == 0x74
    assert toggled["enabled"] == [True, False, True, True, True]
    assert server.world["tanks"][9]["enabled"][1] is False
    out_of_range = ClientCommandDict(
        kind="toggle_equipment",
        command=114,
        x=0,
        y=0,
        target_id=0,
        slot=9,
        message_id=0,
        direction=0,
    )
    server.queue_command(9, out_of_range)
    ignored = server.advance_tick()
    assert _kinds(ignored) == [0x74, 0x2E, 0x2E]
    assert server.world["tanks"][9]["enabled"] == [True, False, True, True, True]


def test_map_open_tick_emits_map_data() -> None:
    """A map open costs nothing and returns the 0x4C snapshot."""
    server = _server()
    server.queue_command(9, _command(("map_open", 108)))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x4C, 0x2E, 0x2E]
    assert server.world["tanks"][9]["fuel"] == 1000


def test_pickup_click_routes_through_the_move_law() -> None:
    """A pickup-fuel click walks, drains, and closes with the code-4 shape.

    The byte-mined drain choreography (2026-08-01): the walk echo,
    the DUPLICATE remaining-0 records, then the 0x52 code-4 close
    with ``reset_action=1`` — the walked-drain close the archive
    shows in 274+ windows.
    """
    server = _server()
    server.world["containers"].append(SimContainerDict(x=12, y=10, volume=30, dotted=True))
    server.queue_command(9, _command(("pickup_fuel", 100), 12, 10))
    messages = server.advance_tick()
    assert _kinds(messages) == [
        0x47,
        "container_pickup",
        "container_pickup",
        0x52,
        0x3F,
        0x2E,
        0x2E,
    ]
    assert messages[1] == messages[2]
    assert [(c["error_code"], c["reset_action"]) for c in _supervisors(messages)] == [(4, 1)]


def test_chat_tick_echoes_the_0x4d_broadcast() -> None:
    """A queued chat comes back as the 0x4D echo, then the cadence syncs.

    Mirrors the un-muted real server (sniff-20260729-214411): the echo
    carries sender id, preset message id, and the send frame's tile —
    and reaches the sender too (the delivery receipt the live bot's
    ``chat_received`` diagnostic latches on).
    """
    server = _server()
    chat = ClientCommandDict(
        kind="chat", command=0x6D, x=10, y=10, target_id=0, slot=0, message_id=41, direction=0
    )
    server.queue_command(9, chat)
    messages = server.advance_tick()
    assert _kinds(messages) == [0x4D, 0x2E, 0x2E]
    echo = messages[0]
    assert echo["msg_type"] == 0x4D
    assert echo["sender_id"] == 9
    assert echo["message_type"] == 41
    assert echo["x"] == 10
    assert echo["y"] == 10


def test_drained_pickup_answers_empty_container_only_to_the_client() -> None:
    """A pickup click on a drained container: 0x52 code 4, client-only.

    The real server validates the destination before any movement
    (the belief-removal signal the production code=4 handler
    consumes); another tank's identical click stays silent on our
    wire — rejections are per-connection.
    """
    from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_EMPTY_CONTAINER

    server = _server()
    server.world["containers"].append(SimContainerDict(x=12, y=10, volume=0, dotted=True))
    server.queue_command(
        9,
        ClientCommandDict(
            kind="pickup_fuel",
            command=100,
            x=12,
            y=10,
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
        ),
    )
    own = _supervisors(server.advance_tick())
    assert [record["error_code"] for record in own] == [SUPERVISOR_ERROR_EMPTY_CONTAINER]
    server.queue_command(
        11,
        ClientCommandDict(
            kind="pickup_fuel",
            command=100,
            x=12,
            y=10,
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
        ),
    )
    assert _supervisors(server.advance_tick()) == []


def test_the_statistics_key_is_answered_with_the_session_counters() -> None:
    """0x56 answers the statistics key, and counts this session.

    279 of the 386 archived 0x56 frames follow ``CMD_STATISTICS`` as
    the client's most recent sent command, which is what makes the
    frame a response rather than a broadcast
    ([[session-state-deglobalisation]]).
    """
    server = _server()
    server.queue_command(9, _statistics_key())
    batch = server.advance_tick()

    reports = [m for m in batch if m["msg_type"] == 0x56]
    assert reports == [
        {
            "msg_type": 0x56,
            "playtime_hours": 0,
            "playtime_minutes": 0,
            "playtime_seconds": 2,
            "destroyed": 0,
            "deactivated": 0,
            "score": 0,
        }
    ]


def test_statistics_playtime_counts_the_session_from_tick_zero() -> None:
    """Playtime is ticks x 2 s, carried into hours/minutes/seconds."""
    server = _server()
    server.world["tick"] = 1849  # 3700 s -> 1 h 01 m 40 s on the next tick
    server.queue_command(9, _statistics_key())

    reports = [m for m in server.advance_tick() if m["msg_type"] == 0x56]
    assert [
        (r["playtime_hours"], r["playtime_minutes"], r["playtime_seconds"]) for r in reports
    ] == [(1, 1, 40)]


def test_another_tanks_statistics_key_is_not_answered_to_the_client() -> None:
    """Every server answer is per-connection, this one included."""
    server = _server()
    server.queue_command(11, _statistics_key())
    assert [m for m in server.advance_tick() if m["msg_type"] == 0x56] == []


def test_a_rejected_teleport_emits_no_landing_confirm() -> None:
    """An unaffordable hop emits the refusal and nothing else.

    The sibling rejection test filters the batch to supervisors, so it
    cannot see what else the tick sent -- and that is the whole
    question. Without the early return the rejected hop falls straight
    into the landed path and the client receives a recentered 0x5A, a
    0x3D position statement and a ``teleport_landed`` confirm for a hop
    the server refused: ``[0x5A, 0x52, 0x3D, teleport_landed, 0x2E,
    0x2E]`` against a tank still standing on its origin tile.

    A bot told it landed while the server kept it in place believes it
    is somewhere it is not, which is the belief every later decision is
    built on.
    """
    server = _server()
    server.world["tanks"][9]["fuel"] = 3
    origin = (server.world["tanks"][9]["x"], server.world["tanks"][9]["y"])
    server.queue_command(9, _command(("teleport", 116), 30, 30))

    messages = server.advance_tick()

    assert _kinds(messages) == [0x52, 0x2E, 0x2E]
    assert (server.world["tanks"][9]["x"], server.world["tanks"][9]["y"]) == origin


def test_a_pickup_outside_the_client_window_does_nothing_but_refuse() -> None:
    """An out-of-window pickup is refused, and the world does not move.

    The window check is the server's own reachability rule, and it has
    to stop the command rather than merely report on it. The target here
    is deliberately STOCKED so the empty-container check cannot be what
    ends the tick: without the early return the tank is relocated to
    (200, 200), the container is drained to zero and a pickup pair is
    broadcast -- while the refusal goes out in the same batch. The
    client is told no and the world says yes.
    """
    server = _server()
    server.world["containers"].append(SimContainerDict(x=200, y=200, volume=90, dotted=True))
    origin = (server.world["tanks"][9]["x"], server.world["tanks"][9]["y"])
    server.queue_command(9, _command(("pickup_fuel", 112), 200, 200))

    messages = server.advance_tick()

    assert _kinds(messages) == [0x52, 0x2E, 0x2E]
    assert (server.world["tanks"][9]["x"], server.world["tanks"][9]["y"]) == origin
    assert [c["volume"] for c in server.world["containers"] if c["x"] == 200] == [90]


def test_only_map_open_is_answered_with_a_map_dump() -> None:
    """No other command kind gets a full 0x4C map in its batch.

    ``_process_other_command`` is a cascade whose FALL-THROUGH is the
    map_open answer, so each earlier arm's return is the only thing
    keeping the whole world out of that command's reply. Removing them
    appends a 0x4C to every one: a block press, a scope shift and a
    statistics key each start broadcasting the entire map, which is the
    largest message the server sends and one the client answers by
    rebuilding its terrain belief.

    ``test_map_open_tick_emits_map_data`` is the standing control for
    the fall-through itself -- it must keep emitting the map, or these
    assertions would pass against a server that never dumps at all.
    """
    scope_shift = ClientCommandDict(
        kind="scope", command=115, x=0, y=0, target_id=0, slot=0, message_id=0, direction=2
    )
    press = ClientCommandDict(
        kind="block", command=98, x=10, y=11, target_id=0, slot=0, message_id=0, direction=0
    )

    for label, command, expected in (
        ("block", press, [0x52, 0x2E, 0x2E]),
        ("scope", scope_shift, [0x5A, 0x3D, 0x2E, 0x2E]),
        ("statistics", _statistics_key(), [0x56, 0x2E, 0x2E]),
    ):
        server = _server()
        server.queue_command(9, command)

        kinds = _kinds(server.advance_tick())

        assert kinds == expected, label
        assert 0x4C not in kinds, label
