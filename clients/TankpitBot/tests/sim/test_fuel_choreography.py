"""The byte-mined fuel-pickup choreography (archive windows 2026-08-01).

Every explicit ``pickup_fuel`` command answers with one of four
measured shapes ([[fuel-system]], [[capture-differ]]): the duplicate
container records, the 0x44 in its gain (``is_free=True, flag=0``) or
no-gain (``is_free=False, flag=43``) form, and the typed 0x52 close —
code 5 (clamped SUCCESS) for a stocked container, code 4 for an empty
one, ``reset_action`` keyed to whether a walk consumed the action.
"""

from __future__ import annotations

from tankpit_bot.container.types import ContainerPickupDict
from tankpit_bot.protocol.types import BinaryMessage, FuelGainDict
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.fuel_pickup import resolve_fuel_pickup
from tankpit_bot.sim.narrate import narrate_fuel_pickup
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import SimContainerDict, SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _pickup(x: int, y: int) -> ClientCommandDict:
    return ClientCommandDict(
        kind="pickup_fuel", command=100, x=x, y=y, target_id=0, slot=0, message_id=0, direction=0
    )


def _server(fuel: int = 1000) -> SimServer:
    world: SimWorldDict = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, fuel)
    world["tanks"][11] = make_sim_tank(11, 1, 1, 20, 20, 500)
    return SimServer(world, InMemoryTerrainMap(), client_id=9)


def _kinds(messages: list[BinaryMessage]) -> list[int | str]:
    return [message["msg_type"] for message in messages]


def _records(messages: list[BinaryMessage]) -> list[ContainerPickupDict]:
    """Every container-pickup record message, in order."""
    records: list[ContainerPickupDict] = []
    for message in messages:
        if message["msg_type"] == "container_pickup":
            records.append(message)
    return records


def _remainings(messages: list[BinaryMessage]) -> list[int]:
    """The remaining volume of every record, in order."""
    return [record["pickups"][0]["remaining_volume"] for record in _records(messages)]


def _closes(messages: list[BinaryMessage]) -> list[tuple[int, int]]:
    """Every 0x52 close as ``(error_code, reset_action)``, in order."""
    closes: list[tuple[int, int]] = []
    for message in messages:
        if message["msg_type"] == 0x52:
            closes.append((message["error_code"], message["reset_action"]))
    return closes


def _gains(messages: list[BinaryMessage]) -> list[FuelGainDict]:
    """Every 0x44 fuel statement, in order."""
    gains: list[FuelGainDict] = []
    for message in messages:
        if message["msg_type"] == 0x44:
            gains.append(message)
    return gains


def test_clamped_pickup_emits_the_five_message_gain_shape() -> None:
    """Transfer that fills the tank: rec x2, 0x44 gain form, rec, code 5.

    The archive's dominant fuel-pickup shape (~50% of 1,944 windows):
    ``47+pickup+pickup+44+pickup+52c5`` with the 0x44 carrying the
    absolute post-pickup fuel as ``is_free=True, flag=0`` and the
    close at ``reset_action=0``.
    """
    server = _server(fuel=1000)  # rank 1: capacity 1100, headroom 100
    server.world["containers"].append(SimContainerDict(x=12, y=10, volume=500, dotted=True))
    server.queue_command(9, _pickup(12, 10))
    messages = server.advance_tick()
    assert _kinds(messages) == [
        0x47,
        "container_pickup",
        "container_pickup",
        0x44,
        "container_pickup",
        0x52,
        0x3F,
        0x2E,
        0x2E,
    ]
    # The 2-tile walk debited 2 first: headroom 102, remainder 398.
    assert _remainings(messages) == [398, 398, 398]
    assert _gains(messages) == [FuelGainDict(msg_type=0x44, fuel_total=1100, is_free=True, flag=0)]
    assert _closes(messages) == [(5, 0)]


def test_full_tank_own_tile_click_uses_the_no_gain_0x44_form() -> None:
    """No transfer, no walk, stocked: 0x44 no-gain form, rec, code 5.

    The measured no-walk shape (``44+pickup+52c5``): the 0x44 comes as
    ``is_free=False, flag=43`` with the unchanged fuel, one record,
    close ``reset_action=0``.
    """
    server = _server(fuel=1100)
    server.world["containers"].append(SimContainerDict(x=10, y=10, volume=300, dotted=True))
    server.queue_command(9, _pickup(10, 10))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x47, 0x44, "container_pickup", 0x52, 0x2E, 0x2E]
    assert _gains(messages) == [
        FuelGainDict(msg_type=0x44, fuel_total=1100, is_free=False, flag=43)
    ]
    assert _remainings(messages) == [300]
    assert _closes(messages) == [(5, 0)]


def test_empty_own_tile_click_closes_code_4_without_reset() -> None:
    """No transfer, no walk, drained: 0x44 no-gain form, rec 0, code 4."""
    server = _server(fuel=1000)
    server.world["containers"].append(SimContainerDict(x=10, y=10, volume=0, dotted=True))
    server.queue_command(9, _pickup(10, 10))
    messages = server.advance_tick()
    assert _kinds(messages) == [0x47, 0x44, "container_pickup", 0x52, 0x2E, 0x2E]
    assert _remainings(messages) == [0]
    assert _closes(messages) == [(4, 0)]


def test_walk_to_a_drained_container_still_executes_and_closes_code_4() -> None:
    """The walk happens for a KNOWN-empty container — no pre-refusal.

    Archive receipt (bot-20260730-224244 @1785476734979): fuel 783 ->
    783 across a 4-tile walk to a drained container, two remaining-0
    records, code 4 ``reset_action=1``. The old sim refused before
    moving — a sim invention.
    """
    server = _server(fuel=1000)
    server.world["containers"].append(SimContainerDict(x=13, y=10, volume=0, dotted=True))
    server.queue_command(9, _pickup(13, 10))
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
    tank = server.world["tanks"][9]
    assert (tank["x"], tank["y"]) == (13, 10)
    assert _remainings(messages) == [0, 0]
    assert _closes(messages) == [(4, 1)]


def test_other_tanks_pickups_broadcast_records_without_closes() -> None:
    """Observers see the records; the 0x44 and 0x52 are per-connection.

    All three choreography families exercised for a NON-client tank:
    the walked drain (2 records), the clamp (3 records, no 0x44), and
    the own-tile no-walk click (1 record) — never a 0x44 or 0x52 on
    our wire.
    """
    server = _server()
    server.world["containers"].append(SimContainerDict(x=22, y=20, volume=200, dotted=True))
    server.queue_command(11, _pickup(22, 20))
    messages = server.advance_tick()
    assert _remainings(messages) == [0, 0]
    assert _gains(messages) == []
    assert _closes(messages) == []
    # Clamp family: tank 11 (rank 1, capacity 1100) fills from a big
    # container, leaving a remainder -> the 3-record shape, closes
    # still silent.
    server.world["containers"].append(SimContainerDict(x=24, y=20, volume=900, dotted=True))
    server.queue_command(11, _pickup(24, 20))
    clamp = server.advance_tick()
    clamp_remainings = _remainings(clamp)
    assert len(clamp_remainings) == 3
    assert clamp_remainings[0] > 0
    assert _gains(clamp) == []
    assert _closes(clamp) == []
    # No-walk family: an own-tile click at the now-full tank -> the
    # single-record shape, closes still silent.
    tank = server.world["tanks"][11]
    server.world["containers"].append(
        SimContainerDict(x=tank["x"], y=tank["y"], volume=50, dotted=True)
    )
    server.queue_command(11, _pickup(tank["x"], tank["y"]))
    nowalk = server.advance_tick()
    assert len(_remainings(nowalk)) == 1
    assert _gains(nowalk) == []
    assert _closes(nowalk) == []


def test_bare_ground_pickup_still_pre_refuses_without_moving() -> None:
    """A click at a tile with NO container record draws the moveless
    code-4 refusal (the production belief-removal signal); another
    tank's identical click stays silent on our wire."""
    server = _server()
    server.queue_command(9, _pickup(12, 10))
    own = server.advance_tick()
    assert _closes(own) == [(4, 1)]
    assert (server.world["tanks"][9]["x"], server.world["tanks"][9]["y"]) == (10, 10)
    server.queue_command(11, _pickup(25, 20))
    other = server.advance_tick()
    assert _closes(other) == []


def test_choreography_defaults_to_remaining_zero_for_a_vanished_record() -> None:
    """A container record gone by emission time reads as drained.

    The server only invokes the choreography for tiles that held a
    record at validation, so this is the function's defensive
    contract: no record -> remaining 0, the empty close.
    """
    server = _server()
    outcome = resolve_fuel_pickup(server.world, 9, 12, 10, volume_before=0, walked=False)
    assert outcome["remaining"] == 0
    messages = narrate_fuel_pickup(outcome, 9)
    assert _kinds(messages) == [0x44, "container_pickup", 0x52]
    assert _remainings(messages) == [0]
    assert _closes(messages) == [(4, 0)]
