"""Law 7 addendum — equipment containers and the archive-mined 0x67 grant.

The grant law comes from 1,149 exact-pre ``0x67 -> next 0x49`` pairs
(2026-07-22 corpus mining): one slot per grant, hard cap 25, weapon
stacks 5-9 / radar stacks 2-4, all-full rejected with 0x52 error 7.
The sim's deterministic approximation grants the most-deficient slot
with the measured midpoint stacks (7 weapons / 3 radar).
"""

from __future__ import annotations

from tankpit_bot.bot.tick_loop import _tick_once
from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_INVENTORY_FULL
from tankpit_bot.sim.combat import SLOT_DUAL, SLOT_HOMING, SLOT_RADAR
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.equipment import (
    EQUIPMENT_CAP,
    RADAR_STACK,
    WEAPON_STACK,
    resolve_equipment_pickup,
)
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.session import deliver_batch
from tankpit_bot.sim.world import (
    SimEquipmentDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)
from tankpit_bot.sniffer.world_state import get_world_service
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim.seam import boot_seam


def _arena(counts: list[int]) -> SimWorldDict:
    """Client tank 9 at (10, 10) standing on an equipment container."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][9]["counts"] = counts
    world["equipment"].append(SimEquipmentDict(x=10, y=10))
    return world


def test_grant_fills_the_most_deficient_slot_with_the_weapon_stack() -> None:
    """Dual at 12 is the neediest: +7 dual, container consumed."""
    world = _arena([25, 12, 25, 20, 15])
    grant = resolve_equipment_pickup(world, 9)
    if grant is None:
        raise AssertionError("the tile holds a container - a grant record is required")
    assert grant["kind"] == "granted"
    assert grant["gained"] == [0, WEAPON_STACK, 0, 0, 0]
    assert world["tanks"][9]["counts"][SLOT_DUAL] == 12 + WEAPON_STACK
    assert world["equipment"] == []


def test_grant_clips_at_the_cap() -> None:
    """Homing at 23 with everything else full: +2, exactly to 25."""
    world = _arena([25, 25, 25, 23, 25])
    grant = resolve_equipment_pickup(world, 9)
    if grant is None:
        raise AssertionError("the tile holds a container - a grant record is required")
    assert grant["gained"] == [0, 0, 0, 2, 0]
    assert world["tanks"][9]["counts"][SLOT_HOMING] == EQUIPMENT_CAP


def test_radar_grants_use_the_smaller_measured_stack() -> None:
    """Radar is the only deficient slot: +3 (the 2-4 roll midpoint)."""
    world = _arena([25, 25, 25, 25, 10])
    grant = resolve_equipment_pickup(world, 9)
    if grant is None:
        raise AssertionError("the tile holds a container - a grant record is required")
    assert grant["gained"] == [0, 0, 0, 0, RADAR_STACK]
    assert world["tanks"][9]["counts"][SLOT_RADAR] == 10 + RADAR_STACK


def test_full_inventory_keeps_the_container() -> None:
    """All slots at 25: inventory_full, nothing granted, container stays."""
    world = _arena([25, 25, 25, 25, 25])
    grant = resolve_equipment_pickup(world, 9)
    if grant is None:
        raise AssertionError("the tile holds a container - a grant record is required")
    assert grant["kind"] == "inventory_full"
    assert grant["gained"] == [0, 0, 0, 0, 0]
    assert len(world["equipment"]) == 1


def test_bare_tile_resolves_to_none() -> None:
    """No equipment under the tank: no grant record at all."""
    world = _arena([25, 25, 25, 25, 25])
    world["equipment"] = []
    assert resolve_equipment_pickup(world, 9) is None


def _pickup_equipment(x: int, y: int) -> ClientCommandDict:
    """A decoded pickup_equipment click at (x, y)."""
    return ClientCommandDict(kind="pickup_equipment", command=99, x=x, y=y, target_id=0, slot=0)


def test_pickup_walk_emits_gain_and_snapshot() -> None:
    """The wire shape: 0x47 echo, 0x67 gained, syncs, then the 0x49.

    The archive shows every 0x67 immediately followed by its
    inventory snapshot — the sim batch carries both.
    """
    world = _arena([25, 12, 25, 20, 15])
    world["equipment"][0] = SimEquipmentDict(x=12, y=10)
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    server.queue_command(9, _pickup_equipment(12, 10))
    messages = server.advance_tick()
    kinds = [m["msg_type"] for m in messages]
    assert kinds == [0x47, 0x5A, 0x67, 0x2E, 0x49]
    gain = messages[2]
    assert gain["msg_type"] == 0x67
    assert gain["gained"] == [0, WEAPON_STACK, 0, 0, 0]
    snapshot = messages[4]
    assert snapshot["msg_type"] == 0x49
    assert snapshot["counts"][SLOT_DUAL] == 12 + WEAPON_STACK


def test_full_inventory_click_draws_the_measured_error_7() -> None:
    """An explicit pickup at full inventory answers 0x52 error 7."""
    world = _arena([25, 25, 25, 25, 25])
    world["equipment"][0] = SimEquipmentDict(x=12, y=10)
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    server.queue_command(9, _pickup_equipment(12, 10))
    messages = server.advance_tick()
    errors = [m for m in messages if m["msg_type"] == 0x52]
    assert [e["error_code"] for e in errors] == [SUPERVISOR_ERROR_INVENTORY_FULL]
    assert len(server.world["equipment"]) == 1


def test_incidental_arrival_at_full_inventory_is_silent() -> None:
    """A plain move onto equipment at full inventory: no error, no grant."""
    world = _arena([25, 25, 25, 25, 25])
    world["equipment"][0] = SimEquipmentDict(x=12, y=10)
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    move = ClientCommandDict(kind="move", command=112, x=12, y=10, target_id=0, slot=0)
    server.queue_command(9, move)
    messages = server.advance_tick()
    assert [m["msg_type"] for m in messages if m["msg_type"] in (0x52, 0x67)] == []
    assert len(server.world["equipment"]) == 1


def test_teleport_landing_collects_equipment() -> None:
    """A hop onto an equipment tile grants on arrival."""
    world = _arena([25, 25, 25, 12, 25])
    world["equipment"][0] = SimEquipmentDict(x=30, y=30)
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    hop = ClientCommandDict(kind="teleport", command=116, x=30, y=30, target_id=0, slot=0)
    server.queue_command(9, hop)
    messages = server.advance_tick()
    gains = [m for m in messages if m["msg_type"] == 0x67]
    assert [g["gained"] for g in gains] == [[0, 0, 0, WEAPON_STACK, 0]]
    assert server.world["equipment"] == []


def test_radar_reveals_equipment_with_the_wire_marker() -> None:
    """0x4F carries equipment as the 0xFFFF -> -1 cache value."""
    world = _arena([25, 25, 25, 25, 15])
    world["equipment"][0] = SimEquipmentDict(x=14, y=12)
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    scan = ClientCommandDict(kind="radar", command=102, x=0, y=0, target_id=0, slot=0)
    server.queue_command(9, scan)
    messages = server.advance_tick()
    scans = [m for m in messages if m["msg_type"] == 0x4F]
    assert len(scans) == 1
    assert [(c["x"], c["y"], c["volume"]) for c in scans[0]["containers"]] == [(14, 12, -1)]


def test_production_bot_restocks_ammo_over_the_seam() -> None:
    """The full pipeline: radar reveal -> pickup -> 0x67 -> belief rises.

    The seam world starts the client at 8 extra radars (below the
    contract's collect-ASAP threshold) with an equipment container in
    viewport range. Over the rounds the PRODUCTION bot must discover
    it and collect it — the world model's last named gap, closed.
    """
    bot, server, link, _table = boot_seam(
        counts=(25, 25, 25, 25, 8),
        equipment=((103, 103), (98, 98)),
    )
    start_extras = 8
    for _ in range(14):
        _tick_once(bot)
        deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
    _tick_once(bot)
    truth = server.world["tanks"][9]["counts"]
    assert truth[SLOT_RADAR] > start_extras - 3
    inventory = get_world_service().inventory_state
    assert inventory["extra_radars"]["count"] == truth[SLOT_RADAR]
    assert len(server.world["equipment"]) < 2
