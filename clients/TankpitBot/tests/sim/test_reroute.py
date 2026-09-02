"""Law 4 — viewport departure (0x58) and the homing reroute TTL.

The measured law ([[shoot-event-format]], run 2026-07-19 22:30): after
a pursued target's 0x58 TankRemove, id-targeted shots keep rerouting
to the departed tank as guaranteed homing hits (every one debited
ammo) until the TTL — measured boundary [11.0, 13.0] s, modeled as
``physics.combat.REROUTE_TTL_MS`` = 12 000 ms — after which the id no
longer resolves and the shot is a free single miss.
"""

from __future__ import annotations

from tankpit_bot.physics.combat import REROUTE_TTL_MS
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.sim.combat import (
    SLOT_HOMING,
    WEAPON_DUAL,
    WEAPON_HOMING,
    WEAPON_SINGLE,
    process_shot,
)
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOBODY: frozenset[int] = frozenset()


def _arena() -> SimWorldDict:
    """Shooter 9 (team 0) at (10, 10), enemy 11 (team 1) at (15, 10)."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][11] = make_sim_tank(11, 1, 1, 15, 10, 500)
    return world


def _server(world: SimWorldDict) -> SimServer:
    """A sim server for the arena with tank 9 as the client."""
    return SimServer(world, InMemoryTerrainMap(), client_id=9)


def _teleport(x: int, y: int) -> ClientCommandDict:
    """A decoded teleport command to (x, y)."""
    return ClientCommandDict(
        kind="teleport", command=116, x=x, y=y, target_id=0, slot=0, message_id=0, direction=0
    )


def _id_shot(x: int, y: int, target_id: int) -> ClientCommandDict:
    """A decoded id-targeted shoot command clicked at (x, y)."""
    return ClientCommandDict(
        kind="shoot", command=115, x=x, y=y, target_id=target_id, slot=0, message_id=0, direction=0
    )


def test_viewport_exit_emits_tank_remove_and_reentry_restates_position() -> None:
    """Leaving the viewport draws 0x58 once; re-entering draws 0x3D."""
    world = _arena()
    server = _server(world)
    assert server.session.viewport.visible == {11}
    server.queue_command(9, _teleport(40, 40))
    away = server.advance_tick()
    removes = [m for m in away if m["msg_type"] == 0x58]
    assert [m["tank_id"] for m in removes] == [11]
    assert server.session.viewport.removed_at == {11: 1}
    quiet = server.advance_tick()
    assert [m for m in quiet if m["msg_type"] == 0x58] == []
    server.queue_command(9, _teleport(15, 12))
    back = server.advance_tick()
    positions = [m for m in back if m["msg_type"] == 0x3D and m["tank_id"] == 11]
    assert len(positions) == 1
    assert server.session.viewport.visible == {11}
    assert server.session.viewport.removed_at == {}


def test_handshake_positions_are_self_only_even_inside_the_viewport() -> None:
    """Every other tank joins with identity only — near ones included.

    The join burst's position scope is SELF, not the viewport. Measured
    2026-09-01 across the archive ([[recipient-policy]]): the identity
    run is a pure 0x21 sequence in 340 of 340 sessions. That is not
    sampling — with ~36 tanks on a 256x256 map a 16x16 window should
    hold one about 13% of the time, so zero of 340 is the law. Other
    tanks' positions arrive from the in-play membership diff
    (``test_reroute_clock_starts_on_a_living_exit`` above), never from
    the burst.

    Tank 11 sits at (15, 10), INSIDE the client's window and in
    ``visible``; tank 12 is far outside. Neither gets a 0x3D.
    """
    world = _arena()
    world["tanks"][12] = make_sim_tank(12, 1, 1, 40, 40, 500)
    server = _server(world)
    burst = server.handshake()
    identities = [m["tank_id"] for m in burst if m["msg_type"] == 0x21]
    positions = [m["tank_id"] for m in burst if m["msg_type"] == 0x3D]
    assert identities == [9, 11, 12]
    assert positions == [9]
    assert server.session.viewport.visible == {11}


def test_deactivated_tank_drops_from_the_viewport_without_0x58() -> None:
    """Death is announced by 0x41 — the visible set just forgets it."""
    world = _arena()
    server = _server(world)
    world["tanks"][11]["fuel"] = 45
    world["tanks"][11]["counts"][0] = 0
    server.queue_command(9, _id_shot(15, 10, 11))
    messages = server.advance_tick()
    assert [m["msg_type"] for m in messages if m["msg_type"] in (0x41, 0x58)] == [0x41]
    assert server.session.viewport.visible == set()


def test_id_shot_reroutes_to_a_moved_targets_current_tile() -> None:
    """The queue-race conversion: a stale click still finds the mover.

    The enemy carries the LOWER tank id, so under the measured
    ascending-id round order its move processes before the client's
    shot in the same tick; the shot's coordinates point at the VACATED
    tile, but the id reroutes it to the tank's new position and the
    same-tick move draws homing.
    """
    world = make_sim_world("field01_r.gif")
    world["tanks"][7] = make_sim_tank(7, 1, 1, 15, 10, 500)
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][9]["counts"][SLOT_HOMING] = 2
    server = _server(world)
    enemy_move = ClientCommandDict(
        kind="move", command=112, x=15, y=12, target_id=0, slot=0, message_id=0, direction=0
    )
    server.queue_command(7, enemy_move)
    server.queue_command(9, _id_shot(15, 10, 7))
    messages = server.advance_tick()
    shots = [m for m in messages if m["msg_type"] == 0x53]
    assert [s["weapon"] for s in shots] == [WEAPON_HOMING]
    assert world["tanks"][9]["counts"][SLOT_HOMING] == 1
    assert world["tanks"][7]["fuel"] == 500 - 2 - 45


def test_id_shot_at_a_stationary_visible_target_is_a_dual() -> None:
    """An id-shot with true coordinates resolves as an ordinary dual."""
    world = _arena()
    world["tanks"][9]["counts"][1] = 3
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 11, None)
    assert outcome["weapon"] == WEAPON_DUAL
    assert outcome["victim_id"] == 11


def test_departed_target_draws_guaranteed_homing_within_the_ttl() -> None:
    """Every rerouted shot inside the TTL debits ammo and lands."""
    world = _arena()
    world["tanks"][9]["counts"][SLOT_HOMING] = 6
    for age_ticks in (0, 3, 6):
        age_ms = age_ticks * TICK_RATE_MS
        assert age_ms <= REROUTE_TTL_MS
        outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 11, age_ms)
        assert outcome["weapon"] == WEAPON_HOMING
        assert outcome["victim_id"] == 11
        assert outcome["ammo_slot"] == SLOT_HOMING
        assert outcome["shooter_debit"] == 10
    assert world["tanks"][9]["counts"][SLOT_HOMING] == 3
    assert world["tanks"][11]["fuel"] == 500 - 3 * 45


def test_past_the_ttl_the_id_no_longer_resolves() -> None:
    """The measured +13 s shot: a free single, nothing debited."""
    world = _arena()
    world["tanks"][9]["counts"][SLOT_HOMING] = 6
    outcome = process_shot(
        world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 11, REROUTE_TTL_MS + TICK_RATE_MS
    )
    assert outcome["weapon"] == WEAPON_SINGLE
    assert outcome["victim_id"] is None
    assert outcome["ammo_slot"] is None
    assert outcome["shooter_debit"] == 6
    assert world["tanks"][9]["counts"][SLOT_HOMING] == 6
    assert world["tanks"][11]["fuel"] == 500


def test_reroute_needs_a_ready_homing_slot() -> None:
    """No homing rounds means no reroute — the human analogue."""
    world = _arena()
    world["tanks"][9]["counts"][SLOT_HOMING] = 0
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 11, 0)
    assert outcome["weapon"] == WEAPON_SINGLE
    assert outcome["victim_id"] is None


def test_reroute_kill_deactivates_the_departed_tank() -> None:
    """A rerouted hit that zeroes fuel kills — 0x41 follows via the server."""
    world = _arena()
    world["tanks"][9]["counts"][SLOT_HOMING] = 2
    world["tanks"][11]["fuel"] = 45
    world["tanks"][11]["counts"][0] = 0
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 11, 0)
    assert outcome["victim_deactivated"] is True
    assert world["tanks"][11]["alive"] is False


def test_unknown_or_dead_target_ids_fall_back_to_the_tile() -> None:
    """An id that no longer resolves to a living tank shoots the tile."""
    world = _arena()
    ghost = process_shot(world, InMemoryTerrainMap(), 9, 12, 12, _NOBODY, 404, None)
    assert ghost["weapon"] == WEAPON_SINGLE
    world["tanks"][11]["alive"] = False
    corpse = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 11, None)
    assert corpse["weapon"] == WEAPON_SINGLE


def test_server_ticks_price_the_departure_age_for_the_shot() -> None:
    """The server converts ticks-since-0x58 into the reroute age.

    Departure at tick 1, shot processed at tick 2: age 2 000 ms —
    inside the TTL, so the wire shows a weapon-3 echo and the homing
    round debits server-side. No 0x49 rides the shot (the live law:
    firing costs never snapshot — response-shape differ 2026-08-01).
    """
    world = _arena()
    world["tanks"][9]["counts"][SLOT_HOMING] = 2
    server = _server(world)
    server.queue_command(9, _teleport(40, 40))
    away = server.advance_tick()
    assert [m["tank_id"] for m in away if m["msg_type"] == 0x58] == [11]
    server.queue_command(9, _id_shot(15, 10, 11))
    messages = server.advance_tick()
    shots = [m for m in messages if m["msg_type"] == 0x53]
    assert [s["weapon"] for s in shots] == [WEAPON_HOMING]
    assert [m for m in messages if m["msg_type"] == 0x49] == []
    assert world["tanks"][9]["counts"][SLOT_HOMING] == 1
    assert world["tanks"][11]["fuel"] == 455
