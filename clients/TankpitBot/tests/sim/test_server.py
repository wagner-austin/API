"""Tests for the sim server's command loop and movement law.

``test_server.py`` was 698 lines; combat and the per-action surfaces are
now siblings.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_DO,
    SUPERVISOR_ERROR_CANT_GO,
)
from tankpit_bot.sim.commands import (
    ClientCommandDict,
    SimError,
)
from tankpit_bot.sim.world import (
    make_sim_tank,
)
from tests.sim._server_harness import (
    _kinds,
    _move,
    _server,
    _shoot,
    _supervisors,
    _syncs,
)


def test_unsupported_kind_and_unknown_tank_raise() -> None:
    """Out-of-scope kinds and unknown/dead tanks fail loudly at queue time."""
    server = _server()
    unknown = ClientCommandDict(
        kind="other", command=90, x=0, y=0, target_id=0, slot=0, message_id=0, direction=0
    )
    with pytest.raises(SimError):
        server.queue_command(9, unknown)
    with pytest.raises(SimError):
        server.queue_command(404, _move(1, 1))
    server.world["tanks"][11]["alive"] = False
    with pytest.raises(SimError):
        server.queue_command(11, _move(1, 1))


def test_dead_client_commands_drop_silently() -> None:
    """A corpse's clicks are ignored, not refused — the real connection
    survives deactivation and the server simply drops them."""
    server = _server()
    server.world["tanks"][9]["alive"] = False
    server.queue_command(9, _move(12, 10))
    messages = server.advance_tick()
    assert [m["msg_type"] for m in messages if m["msg_type"] == 0x47] == []
    assert (server.world["tanks"][9]["x"], server.world["tanks"][9]["y"]) == (10, 10)


def test_move_tick_emits_echo_then_fuel_sync() -> None:
    """One move: 0x47 echo with the route, then the cadence syncs.

    Every living tank syncs every tick (the measured ~2 s broadcast,
    [[tank-freshness-model]]); only the client's own sync carries fuel.
    """
    server = _server()
    server.queue_command(9, _move(13, 12))
    messages = server.advance_tick()
    assert server.world["tick"] == 1
    echo = messages[0]
    assert echo["msg_type"] == 0x47
    assert echo["path"] == "sseee"
    syncs = _syncs(messages)
    assert [(sync["tank_id"], sync["fuel"]) for sync in syncs] == [(9, 995), (11, None)]


def test_rejected_moves_emit_supervisor_errors() -> None:
    """Out-of-window and cant_go rejections surface as 0x52 messages.

    The client's window is [2, 18) x [2, 18) (centered on (10, 10),
    map-clamped): a target one column past the edge draws code 0
    (CANT_DO) — the measured acceptance boundary
    ([[viewport-shift-protocol]]) — and an occupied in-window tile
    draws code 1 (CANT_GO).
    """
    server = _server()
    server.queue_command(9, _move(18, 10))
    outside = _supervisors(server.advance_tick())
    assert [record["error_code"] for record in outside] == [SUPERVISOR_ERROR_CANT_DO]
    server.queue_command(9, _move(15, 10))
    occupied = _supervisors(server.advance_tick())
    assert [record["error_code"] for record in occupied] == [SUPERVISOR_ERROR_CANT_GO]


def test_handshake_covers_client_and_living_tanks_only() -> None:
    """The join burst: OWN identity first, then live others.

    The first 0x21 of a session names the player's own tank — the
    archive convention ``validate.wire_timeline`` keys self-attribution
    on, so the sim must open the same way the real server does. The
    0x3E full status and the 0x5A-before-0x3D ordering are archived
    too: 285 of 285 real sessions open ``0x21, 0x3E, 0x5A, 0x3D``, and
    285 of the 286 archived enter-game sends draw exactly one 0x3F
    ([[session-state-deglobalisation]]).
    """
    server = _server()
    server.world["tanks"][12] = make_sim_tank(12, 3, 1, 20, 20, 100)
    server.world["tanks"][12]["alive"] = False
    burst = server.handshake()
    kinds = _kinds(burst)
    assert kinds == [0x21, 0x3E, 0x5A, 0x3D, 0x2E, 0x21, 0x49, 0x49, 0x74, 0x3F]
    own = burst[0]
    assert own["msg_type"] == 0x21
    assert own["tank_id"] == 9
    status = burst[1]
    assert status["msg_type"] == 0x3E
    assert status["tank_id"] == 9
    assert status["name"] == server.world["tanks"][9]["name"]
    viewport = burst[2]
    assert viewport["msg_type"] == 0x5A
    assert (viewport["viewport_left"], viewport["viewport_top"]) == (2, 2)
    assert viewport["entities"] == []
    own_position = burst[3]
    assert own_position["msg_type"] == 0x3D
    assert own_position["tank_id"] == 9
    own_sync = burst[4]
    assert own_sync["msg_type"] == 0x2E
    assert own_sync["tank_id"] == 9
    assert own_sync["fuel"] == server.world["tanks"][9]["fuel"]
    # Enemy 11 sits at (15, 10), INSIDE the client's [2, 18) window —
    # and still gets no position statement. The identity run is pure
    # 0x21 in 340 of 340 archived sessions ([[recipient-policy]]).
    assert burst[5]["msg_type"] == 0x21
    assert burst[5]["tank_id"] == 11
    assert 11 in server._viewport.visible


def test_handshake_tail_is_the_measured_inventory_pair_toggle_sync() -> None:
    """The burst tail: 0x49 twice, then 0x74, then 0x3F — and no 0x44.

    Measured 2026-09-01 over the archive ([[recipient-policy]]): the
    inventory arrives TWICE and sits AFTER the identity run, the 0x74
    equipment-enabled state follows it in 340 of 340 sessions, and no
    0x44 rides the burst at all — 293 of 340 sessions carry none in
    their first 90 received messages, because the burst's own 0x2E
    already carries fuel ([[decode-coverage]]).
    """
    server = _server()
    client = server.world["tanks"][9]
    client["counts"] = [1, 2, 3, 4, 5]
    client["enabled"] = [False, True, False, True, True]
    burst = server.handshake()

    assert _kinds(burst)[-4:] == [0x49, 0x49, 0x74, 0x3F]
    first, second = burst[-4], burst[-3]
    assert first["msg_type"] == 0x49
    assert second["msg_type"] == 0x49
    assert first["counts"] == [1, 2, 3, 4, 5]
    assert first["counts"] == second["counts"]
    assert first["enabled"] == second["enabled"]
    assert first["show"] is True

    toggle = burst[-2]
    assert toggle["msg_type"] == 0x74
    assert toggle["enabled"] == [False, True, False, True, True]

    assert 0x44 not in _kinds(burst)


def test_a_walk_draws_a_sync_and_a_standstill_does_not() -> None:
    """0x3F trails a walk that relocated the client, and only that.

    1,277 of the 1,528 archived syncs follow a move command, against
    zero after any of the 13,698 shoots; the JS handler is a view
    resync, which a standstill does not need
    ([[session-state-deglobalisation]]).
    """
    server = _server()
    server.queue_command(9, _move(12, 10))
    assert 0x3F in _kinds(server.advance_tick())

    server.queue_command(9, _move(12, 10))  # already there: empty path
    assert 0x3F not in _kinds(server.advance_tick())

    server.queue_command(9, _shoot(15, 10))
    assert 0x3F not in _kinds(server.advance_tick())
