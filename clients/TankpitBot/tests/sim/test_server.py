"""Law 1 — the global queue, tick batching, and charge latency."""

from __future__ import annotations

import pytest

from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_DO,
    SUPERVISOR_ERROR_CANT_GO,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
)
from tankpit_bot.protocol.types import (
    BinaryMessage,
    InventoryDict,
    ShootEventDict,
    SupervisorDict,
    TankStatusSyncDict,
)
from tankpit_bot.sim.combat import SLOT_DUAL, SLOT_HOMING
from tankpit_bot.sim.commands import ClientCommandDict, SimError
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimMineDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _move(x: int, y: int) -> ClientCommandDict:
    """A decoded move command to (x, y)."""
    return ClientCommandDict(kind="move", command=112, x=x, y=y, target_id=0, slot=0)


def _shoot(x: int, y: int) -> ClientCommandDict:
    """A decoded shoot command at (x, y)."""
    return ClientCommandDict(kind="shoot", command=115, x=x, y=y, target_id=0, slot=0)


def _server() -> SimServer:
    """Client tank 9 at (10, 10) and enemy 11 at (15, 10)."""
    world: SimWorldDict = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][11] = make_sim_tank(11, 1, 1, 15, 10, 500)
    return SimServer(world, InMemoryTerrainMap(), client_id=9)


def _kinds(messages: list[BinaryMessage]) -> list[int | str]:
    """The msg_type of every message, in emission order."""
    return [message["msg_type"] for message in messages]


def _shots(messages: list[BinaryMessage]) -> list[ShootEventDict]:
    """All 0x53 echoes in the batch."""
    shots: list[ShootEventDict] = []
    for message in messages:
        if message["msg_type"] == 0x53:
            shots.append(message)
    return shots


def _syncs(messages: list[BinaryMessage]) -> list[TankStatusSyncDict]:
    """All 0x2E fuel syncs in the batch."""
    syncs: list[TankStatusSyncDict] = []
    for message in messages:
        if message["msg_type"] == 0x2E:
            syncs.append(message)
    return syncs


def _supervisors(messages: list[BinaryMessage]) -> list[SupervisorDict]:
    """All 0x52 command-result messages in the batch."""
    results: list[SupervisorDict] = []
    for message in messages:
        if message["msg_type"] == 0x52:
            results.append(message)
    return results


def _snapshots(messages: list[BinaryMessage]) -> list[InventoryDict]:
    """All 0x49 inventory snapshots in the batch."""
    snapshots: list[InventoryDict] = []
    for message in messages:
        if message["msg_type"] == 0x49:
            snapshots.append(message)
    return snapshots


def test_unsupported_kind_and_unknown_tank_raise() -> None:
    """Out-of-scope kinds and unknown/dead tanks fail loudly at queue time."""
    server = _server()
    unknown = ClientCommandDict(kind="other", command=90, x=0, y=0, target_id=0, slot=0)
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


def test_arrival_pickup_and_mine_walk_emit_container_messages() -> None:
    """Pickups and destination mines ride the same tick's batch."""
    server = _server()
    server.world["containers"].append(SimContainerDict(x=11, y=10, volume=50, dotted=True))
    server.world["mines"].append(SimMineDict(x=11, y=10, team=1))
    server.queue_command(9, _move(11, 10))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x47, 0x45, "container_pickup", 0x2E, 0x2E]
    pickup = messages[2]
    assert pickup["msg_type"] == "container_pickup"
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


def test_hit_victim_syncs_and_client_ammo_snapshot() -> None:
    """A dual hit syncs the victim's fuel and snapshots client ammo."""
    server = _server()
    server.world["tanks"][9]["counts"][SLOT_DUAL] = 3
    server.queue_command(9, _shoot(15, 10))
    messages = server.advance_tick()
    assert [shot["weapon"] for shot in _shots(messages)] == [1]
    syncs = _syncs(messages)
    assert [(sync["tank_id"], sync["fuel"]) for sync in syncs] == [(9, 1000), (11, None)]
    assert server.world["tanks"][11]["fuel"] == 410
    snapshots = _snapshots(messages)
    assert [snapshot["counts"][SLOT_DUAL] for snapshot in snapshots] == [2]


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
    server.world["mines"].append(SimMineDict(x=12, y=12, team=1))
    server.queue_command(9, _shoot(12, 12))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x53, 0x45, 0x2E, 0x2E]
    assert server.world["mines"] == []


def test_handshake_covers_client_and_living_tanks_only() -> None:
    """The join burst: OWN identity first, then live others.

    The first 0x21 of a session names the player's own tank — the
    archive convention ``validate.wire_timeline`` keys self-attribution
    on, so the sim must open the same way the real server does.
    """
    server = _server()
    server.world["tanks"][12] = make_sim_tank(12, 3, 1, 20, 20, 100)
    server.world["tanks"][12]["alive"] = False
    burst = server.handshake()
    kinds = _kinds(burst)
    assert kinds == [0x21, 0x3D, 0x5A, 0x44, 0x49, 0x21, 0x3D]
    own = burst[0]
    assert own["msg_type"] == 0x21
    assert own["tank_id"] == 9
    viewport = burst[2]
    assert viewport["msg_type"] == 0x5A
    assert (viewport["viewport_left"], viewport["viewport_top"]) == (2, 2)
    assert viewport["entities"] == []


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
    assert server._viewport.removed_at == {}


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


def _command(kind_command: tuple[str, int], x: int = 0, y: int = 0) -> ClientCommandDict:
    """A decoded client command of the given (kind, byte) pair."""
    kind, command = kind_command
    move_kind: ClientCommandDict = ClientCommandDict(
        kind="move", command=command, x=x, y=y, target_id=0, slot=0
    )
    if kind == "teleport":
        return ClientCommandDict(kind="teleport", command=command, x=x, y=y, target_id=0, slot=0)
    if kind == "radar":
        return ClientCommandDict(kind="radar", command=command, x=x, y=y, target_id=0, slot=0)
    if kind == "mine":
        return ClientCommandDict(kind="mine", command=command, x=x, y=y, target_id=0, slot=0)
    if kind == "map_open":
        return ClientCommandDict(kind="map_open", command=command, x=x, y=y, target_id=0, slot=0)
    if kind == "pickup_fuel":
        return ClientCommandDict(kind="pickup_fuel", command=command, x=x, y=y, target_id=0, slot=0)
    return move_kind


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
    assert _kinds(messages) == [
        "teleport_landed",
        0x3D,
        "container_pickup",
        0x5A,
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


def test_radar_tick_emits_scan_ack_sync_and_snapshot() -> None:
    """A scan with an extra: 0x4F, 0x46, fuel sync, ammo snapshot."""
    server = _server()
    server.world["tanks"][9]["counts"][4] = 3
    server.queue_command(9, _command(("radar", 102)))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x4F, 0x46, 0x2E, 0x2E, 0x49]
    assert server.world["tanks"][9]["fuel"] == 990
    assert server.world["tanks"][9]["counts"][4] == 2


def test_mine_press_tick_emits_placement_and_trades() -> None:
    """A press: 0x4B placement, 0x45 for 1:1 trades, fuel sync."""
    server = _server()
    server.world["mines"].append(SimMineDict(x=11, y=11, team=1))
    server.queue_command(9, _command(("mine", 107)))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x4B, 0x45, 0x2E, 0x2E]
    assert server.world["tanks"][9]["fuel"] == 990


def test_teleport_to_empty_ground_has_no_pickup_message() -> None:
    """A landing on bare ground emits no container message."""
    server = _server()
    server.queue_command(9, _command(("teleport", 116), 30, 30))
    assert _kinds(server.advance_tick()) == ["teleport_landed", 0x3D, 0x5A, 0x58, 0x2E, 0x2E]


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
    world["mines"].append(SimMineDict(x=10, y=10, team=0))
    server = SimServer(world, InMemoryTerrainMap(terrain_data=rocks), client_id=9)
    server.queue_command(9, _command(("mine", 107)))
    assert _kinds(server.advance_tick()) == [0x2E]
    assert server.world["tanks"][9]["fuel"] == 990


def test_equipment_toggle_flips_the_slot_and_answers_0x74() -> None:
    """A toggle press flips the slot server-side and reports all five."""
    server = _server()
    toggle = ClientCommandDict(kind="toggle_equipment", command=114, x=0, y=0, target_id=0, slot=2)
    server.queue_command(9, toggle)
    messages = server.advance_tick()
    assert _kinds(messages) == [0x74, 0x2E, 0x2E]
    toggled = messages[0]
    assert toggled["msg_type"] == 0x74
    assert toggled["enabled"] == [True, False, True, True, True]
    assert server.world["tanks"][9]["enabled"][1] is False
    out_of_range = ClientCommandDict(
        kind="toggle_equipment", command=114, x=0, y=0, target_id=0, slot=9
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
    """A pickup-fuel click walks to the container and drains it."""
    server = _server()
    server.world["containers"].append(SimContainerDict(x=12, y=10, volume=30, dotted=True))
    server.queue_command(9, _command(("pickup_fuel", 100), 12, 10))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x47, "container_pickup", 0x2E, 0x2E]
