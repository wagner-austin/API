"""Tests for sim-server combat: shots, hits, kills, mines, and corpses."""

from __future__ import annotations

from tankpit_bot.sim.combat import (
    SLOT_DUAL,
    SLOT_HOMING,
)
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
    place_mine,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim._server_harness import (
    _command,
    _kinds,
    _move,
    _server,
    _shoot,
    _shots,
    _snapshots,
    _statistics_key,
    _syncs,
)


def test_arrival_pickup_and_mine_walk_emit_container_messages() -> None:
    """Pickups and destination mines ride the same tick's batch.

    Arrival auto-picks DOUBLE their container record — the measured
    duplicate-record law (2026-08-01: 129 move and 2,200+ teleport
    windows all read ``...pickup+pickup``).
    """
    server = _server()
    server.world["containers"].append(SimContainerDict(x=11, y=10, volume=50, dotted=True))
    place_mine(server.world, 11, 10, 1)
    server.queue_command(9, _move(11, 10))
    messages = server.advance_tick()
    assert _kinds(messages) == [
        0x47,
        0x45,
        "container_pickup",
        "container_pickup",
        0x3F,
        0x2E,
        0x2E,
    ]
    assert messages[2] == messages[3]
    pickup = messages[2]
    if pickup["msg_type"] != "container_pickup":
        raise AssertionError("expected the arrival's container_pickup record")
    assert pickup["pickups"][0]["remaining_volume"] == 0


def test_shot_bills_the_shooter_on_the_next_tick() -> None:
    """Charge latency: the firing cost lands one tick later."""
    server = _server()
    server.queue_command(9, _shoot(12, 12))
    first = server.advance_tick()
    assert [shot["weapon"] for shot in _shots(first)] == [0]
    assert server.world["tanks"][9]["fuel"] == 1000
    second = server.advance_tick()
    assert server.world["tanks"][9]["fuel"] == 994
    assert [(sync["tank_id"], sync["fuel"]) for sync in _syncs(second)] == [(9, 994), (11, None)]


def test_hit_victim_syncs_and_no_shot_snapshot() -> None:
    """A dual hit syncs the victim's fuel and sends NO 0x49.

    The archive's 11,051 live shot windows are 92.4% a bare 0x53
    echo (response-shape differ 2026-08-01): the real server never
    snapshots inventory for firing costs — counts re-sync on the
    next 0x49-bearing event. The count still decrements server-side.
    """
    server = _server()
    server.world["tanks"][9]["counts"][SLOT_DUAL] = 3
    server.queue_command(9, _shoot(15, 10))
    messages = server.advance_tick()
    assert [shot["weapon"] for shot in _shots(messages)] == [1]
    syncs = _syncs(messages)
    assert [(sync["tank_id"], sync["fuel"]) for sync in syncs] == [(9, 1000), (11, None)]
    assert server.world["tanks"][11]["fuel"] == 410
    assert _snapshots(messages) == []
    assert server.world["tanks"][9]["counts"][SLOT_DUAL] == 2


def test_same_tick_move_then_shot_selects_homing() -> None:
    """A same-round move resolving BEFORE the shot marks the mover
    for homing. Within-round order is ascending tank id (the
    2026-07-25 measured law), so the mover here is the LOWER id —
    the client (9) arrives, then the enemy (11) fires at the arrival
    tile."""
    server = _server()
    server.world["tanks"][11]["counts"][SLOT_HOMING] = 1
    server.queue_command(11, _shoot(11, 10))
    server.queue_command(9, _move(11, 10))
    messages = server.advance_tick()
    assert [shot["weapon"] for shot in _shots(messages)] == [3]


def test_armored_victim_marks_ammo_not_fuel() -> None:
    """A fully-absorbed hit changes the victim's shields, not fuel."""
    server = _server()
    server.world["tanks"][11]["counts"][0] = 5
    server.queue_command(9, _shoot(15, 10))
    messages = server.advance_tick()
    syncs = _syncs(messages)
    assert [(sync["tank_id"], sync["fuel"]) for sync in syncs] == [(9, 1000), (11, None)]
    assert server.world["tanks"][11]["counts"][0] == 4
    assert server.world["tanks"][11]["fuel"] == 500


def test_shot_mine_cascade_rides_the_batch() -> None:
    """Shooting a mine emits its 0x45 packets in the same tick."""
    server = _server()
    place_mine(server.world, 12, 12, 1)
    server.queue_command(9, _shoot(12, 12))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x53, 0x45, 0x2E, 0x2E]
    assert server.world["mines"] == {}


def test_corpse_window_closes_with_0x58_after_exactly_22_seconds() -> None:
    """The killed tank's 0x58 arrives 11 ticks after its 0x41.

    Corpus-swept 2026-07-22: 37 kill->remove pairs, min = median =
    exactly 22.0 s at the 2 s cadence.
    """
    from tankpit_bot.sim.server import CORPSE_WINDOW_TICKS

    server = _server()
    server.world["tanks"][9]["counts"][SLOT_DUAL] = 3
    server.world["tanks"][11]["fuel"] = 45
    server.queue_command(9, _shoot(15, 10))
    first = server.advance_tick()
    assert 0x41 in _kinds(first)
    assert 0x58 not in _kinds(first)
    for _ in range(CORPSE_WINDOW_TICKS - 1):
        assert 0x58 not in _kinds(server.advance_tick())
    closing = server.advance_tick()
    removes = [m for m in closing if m["msg_type"] == 0x58]
    assert [m["tank_id"] for m in removes] == [11]
    assert server.session.viewport.removed_at == {}


def test_kill_emits_deactivation_and_skips_the_deads_commands() -> None:
    """A killed tank's queued command is dropped, and 0x41 fires."""
    server = _server()
    server.world["tanks"][11]["fuel"] = 45
    server.queue_command(9, _shoot(15, 10))
    server.queue_command(11, _move(15, 12))
    messages = server.advance_tick()
    kinds = _kinds(messages)
    assert 0x41 in kinds
    assert 0x47 not in kinds
    assert server.world["tanks"][11]["alive"] is False


def test_radar_tick_emits_snapshot_then_scan_and_sync() -> None:
    """A scan with an extra: 0x49 FIRST, then 0x4F, 0x46, fuel syncs.

    Live radar windows are 84% ``49+4F+46`` (response-shape differ
    2026-08-01) — the extra-consumption snapshot LEADS the results.
    """
    server = _server()
    server.world["tanks"][9]["counts"][4] = 3
    server.queue_command(9, _command(("radar", 102)))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x49, 0x4F, 0x46, 0x2E, 0x2E]
    assert server.world["tanks"][9]["fuel"] == 990
    assert server.world["tanks"][9]["counts"][4] == 2


def test_mine_press_tick_emits_placement_and_trades() -> None:
    """A press: 0x4B placement, 0x45 for 1:1 trades, fuel sync."""
    server = _server()
    place_mine(server.world, 11, 11, 1)
    server.queue_command(9, _command(("mine", 107)))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x4B, 0x45, 0x2E, 0x2E]
    assert server.world["tanks"][9]["fuel"] == 990


def test_radar_without_extras_has_no_snapshot() -> None:
    """A built-in scan changes no counts, so no 0x49 follows."""
    server = _server()
    server.queue_command(9, _command(("radar", 102)))
    assert _kinds(server.advance_tick()) == [0x4F, 0x46, 0x2E, 0x2E]


def test_mine_press_on_sealed_ground_places_nothing() -> None:
    """A fully blocked 3x3 emits neither 0x4B nor 0x45 — just the bill."""
    world: SimWorldDict = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    ring = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    rocks = {(10 + dx, 10 + dy): "#" for dx, dy in ring}
    place_mine(world, 10, 10, 0)
    server = SimServer(world, InMemoryTerrainMap(terrain_data=rocks), client_id=9)
    server.queue_command(9, _command(("mine", 107)))
    assert _kinds(server.advance_tick()) == [0x2E]
    assert server.world["tanks"][9]["fuel"] == 990


def test_statistics_counts_the_clients_kills_not_the_rooms() -> None:
    """``destroyed`` is the client's own kill count."""
    server = _server()
    server.world["tanks"][9]["counts"][SLOT_DUAL] = 3
    server.world["tanks"][11]["fuel"] = 45
    server.queue_command(9, _shoot(15, 10))
    assert 0x41 in _kinds(server.advance_tick())

    server.queue_command(9, _statistics_key())
    reports = [m for m in server.advance_tick() if m["msg_type"] == 0x56]
    assert [(r["destroyed"], r["deactivated"]) for r in reports] == [(1, 0)]
