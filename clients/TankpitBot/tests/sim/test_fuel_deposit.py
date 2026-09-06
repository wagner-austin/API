"""The fuel-deposit law, cracked from the archive 2026-09-03.

``0x44`` spent months in the wiki's "observed but NOT modelled" table
as ``CMD_UNMODELLED_COMBAT`` — "type 6, shoot-shaped, four distinct
payloads, no law". Two readings settled it: the client class ``Wb``
serializes ``[type][0x44][amount_lo][amount_hi][x][y]``, so the AMOUNT
sits where the shot keeps its coordinates, and the six archived
deposits then read as one transfer law with a floor.

Every case below is one of those six windows or the arithmetic they
force. The literal wire bytes are asserted against the builder because
a fixture that agreed only with our own decoder would prove nothing
about the live client.
"""

from __future__ import annotations

from tankpit_bot.container.types import ContainerPickupRecordDict
from tankpit_bot.physics.capacity import DEPOSIT_FLOOR
from tankpit_bot.protocol.command_builders import build_deposit_fuel_command
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import ClientCommandDict, decode_client_command
from tankpit_bot.sim.fuel_deposit import resolve_fuel_deposit
from tankpit_bot.sim.narrate import narrate_fuel_deposit
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import SimContainerDict, SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

#: sniff-20260620-190228, the deposit that hit the floor: 294 units
#: requested at tile (174, 49), which is ``0xAE, 0x31``.
ARCHIVE_FLOOR_PAYLOAD = bytes.fromhex("06442601ae31")

#: ghost_visual / sniff-20260620-143345: 100 units at (170, 174).
ARCHIVE_PLAIN_PAYLOAD = bytes.fromhex("06446400aaae")


def _deposit(x: int, y: int, amount: int) -> ClientCommandDict:
    """One queued deposit command.

    Args:
        x: Destination tile X.
        y: Destination tile Y.
        amount: Units the client asks to deposit.

    Returns:
        The typed command the tick processor routes.
    """
    return ClientCommandDict(
        kind="deposit_fuel",
        command=68,
        x=x,
        y=y,
        target_id=0,
        slot=0,
        message_id=0,
        direction=0,
        amount=amount,
    )


def _server(fuel: int) -> SimServer:
    """A one-client server with the client tank at (10, 10).

    Args:
        fuel: The client tank's starting fuel.

    Returns:
        The server, client id 9.
    """
    world: SimWorldDict = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, fuel)
    return SimServer(world, InMemoryTerrainMap(), client_id=9)


def _kinds(messages: list[BinaryMessage]) -> list[int | str]:
    """The message types of a tick's batch, in emission order.

    Args:
        messages: The batch.

    Returns:
        Each message's ``msg_type``.
    """
    return [message["msg_type"] for message in messages]


def test_the_archive_bytes_are_what_the_builder_writes() -> None:
    """The live client's frame, byte for byte.

    ``06442601ae31`` is not four arbitrary bytes: it is "deposit 294
    at (174, 49)". Reproducing it from the amount and the tile is the
    only evidence that the layout was read correctly rather than
    fitted to one sample.
    """
    assert build_deposit_fuel_command(174, 49, 294)[2:] == b"!" + ARCHIVE_FLOOR_PAYLOAD
    assert build_deposit_fuel_command(170, 174, 100)[2:] == b"!" + ARCHIVE_PLAIN_PAYLOAD


def test_the_archive_bytes_decode_to_the_archive_deposit() -> None:
    """The sim's decoder reads the live payload the same way."""
    command = decode_client_command(ARCHIVE_FLOOR_PAYLOAD)

    assert command["kind"] == "deposit_fuel"
    assert (command["amount"], command["x"], command["y"]) == (294, 174, 49)


def test_a_deposit_well_above_the_floor_transfers_the_whole_request() -> None:
    """Archive window 1: 1,034 fuel, 100 requested, 934 left, tile 100."""
    server = _server(fuel=1034)

    outcome = resolve_fuel_deposit(server.world, 9, 12, 10, 100)

    assert outcome["deposited"] == 100
    assert outcome["fuel_total"] == 934
    assert outcome["tile_volume"] == 100
    assert server.world["tanks"][9]["fuel"] == 934


def test_the_floor_caps_the_transfer_at_a_hundred_left() -> None:
    """Archive window 3, the one that discriminates the law.

    294 requested while holding exactly 294 — the client clamps its
    own request to the tank's fuel (``Wb`` is constructed with a
    pre-clamped amount), so this window pins the fuel before the
    transfer. 194 landed and 100 stayed: the server keeps
    :data:`DEPOSIT_FLOOR` back, the same 100 measured at four ranks in
    July 2026 from the user's own max deposits.
    """
    server = _server(fuel=294)

    outcome = resolve_fuel_deposit(server.world, 9, 174, 49, 294)

    assert outcome["requested"] == 294
    assert outcome["deposited"] == 194
    assert outcome["fuel_total"] == DEPOSIT_FLOOR
    assert outcome["tile_volume"] == 194


def test_a_tank_at_the_floor_deposits_nothing() -> None:
    """The floor is a floor, not a licence to go negative.

    Never observed live — the client refuses to initiate a deposit at
    or below the floor — but the arithmetic is forced: there is no
    amount below zero to transfer, so the clamp holds rather than
    inverting the transfer into a withdrawal.
    """
    server = _server(fuel=DEPOSIT_FLOOR)

    outcome = resolve_fuel_deposit(server.world, 9, 12, 10, 500)

    assert outcome["deposited"] == 0
    assert outcome["fuel_total"] == DEPOSIT_FLOOR
    assert outcome["tile_volume"] == 0


def test_a_deposit_onto_bare_ground_creates_an_undotted_container() -> None:
    """The tile now holds fuel, so the pickup law must be able to see it.

    Undotted: ``dotted`` is exposure memory set by a radar reveal of
    at least ``MAP_DOT_MIN_VOLUME``, and depositing is not a reveal.

    A container on a DIFFERENT tile is seeded first, so the lookup has
    to walk past a real record before concluding the destination is
    bare — a world holding exactly one container would let a lookup
    that ignored coordinates pass.
    """
    server = _server(fuel=1100)
    server.world["containers"].append(SimContainerDict(x=8, y=8, volume=900, dotted=True))
    assert not [c for c in server.world["containers"] if (c["x"], c["y"]) == (12, 10)]

    resolve_fuel_deposit(server.world, 9, 12, 10, 300)

    created = [c for c in server.world["containers"] if (c["x"], c["y"]) == (12, 10)]
    assert created == [SimContainerDict(x=12, y=10, volume=300, dotted=False)]
    assert SimContainerDict(x=8, y=8, volume=900, dotted=True) in server.world["containers"]


def test_a_deposit_onto_a_stocked_tile_adds_to_what_is_there() -> None:
    """The record carries the tile's REMAINING volume, not the delta.

    Production reads a container record as the container's absolute
    remaining volume and overwrites its belief with it, so reporting
    the deposited amount on a tile that already held fuel would tell
    the client the tile had LOST volume. All six archived deposits
    landed on empty tiles, where the two numbers coincide — which is
    exactly the case that would have hidden this.
    """
    server = _server(fuel=1100)
    server.world["containers"].append(SimContainerDict(x=12, y=10, volume=400, dotted=True))

    outcome = resolve_fuel_deposit(server.world, 9, 12, 10, 300)

    assert outcome["deposited"] == 300
    assert outcome["tile_volume"] == 700
    stocked = [c for c in server.world["containers"] if (c["x"], c["y"]) == (12, 10)]
    assert [c["volume"] for c in stocked] == [700]
    assert [c["dotted"] for c in stocked] == [True]


def test_the_depositor_is_told_sync_then_deposit_then_one_record() -> None:
    """The measured three-message answer, in the measured order.

    0x2E carrying post-transfer fuel, 0x64 repeating it as the
    absolute level, then ONE container record with the tile's new
    remaining volume. The record count is the discriminator: every
    pickup doubles its record and the deposit does not.
    """
    server = _server(fuel=1034)
    outcome = resolve_fuel_deposit(server.world, 9, 170, 174, 100)

    messages = narrate_fuel_deposit(server.world, outcome, observer_id=9)

    assert _kinds(messages) == [0x2E, 0x64, "container_pickup"]
    sync, deposit, record = messages
    assert sync["msg_type"] == 0x2E and sync["fuel"] == 934
    assert deposit["msg_type"] == 0x64 and deposit["fuel_total"] == 934
    assert record["msg_type"] == "container_pickup"
    assert record["pickups"] == (ContainerPickupRecordDict(x=170, y=174, remaining_volume=100),)


def test_another_observer_is_told_nothing_at_all() -> None:
    """A third party's deposit is wire-invisible. MEASURED, not assumed.

    The 120-day atlas found zero cross-tank 0x64s and zero cross-tank
    refill records against hundreds of INFERRED refills
    ([[fuel-system]]): the record is a depositor-only send, not the
    field broadcast a container change looks like it should be. It
    would also be actively harmful — production reads any
    fuel-bearing message as SELF fuel.
    """
    server = _server(fuel=1034)
    outcome = resolve_fuel_deposit(server.world, 9, 170, 174, 100)

    assert narrate_fuel_deposit(server.world, outcome, observer_id=11) == []


def test_an_own_tile_deposit_answers_with_the_three_messages_alone() -> None:
    """End to end: no walk, so no echo, no 0x52 and no 0x3F.

    The three no-walk archive windows carry exactly the sync, the
    deposit and the record. The trailing 0x2E is the sim's per-tick
    roster sync, which every tick emits.
    """
    server = _server(fuel=1100)
    server.queue_command(9, _deposit(10, 10, 400))

    messages = server.advance_tick()

    assert _kinds(messages) == [0x2E, 0x64, "container_pickup", 0x2E]
    assert server.world["tanks"][9]["fuel"] == 700


def test_a_walked_deposit_echoes_the_walk_and_still_draws_no_sync() -> None:
    """The two walked archive windows: 0x47, then the three, and stop.

    A plain move of the same distance ends with a 0x3F view resync.
    Neither walked deposit does, which is why the deposit returns
    before the move law's sync rather than falling through to it.
    """
    server = _server(fuel=1100)
    server.queue_command(9, _deposit(13, 10, 400))

    messages = server.advance_tick()

    assert _kinds(messages) == [0x47, 0x2E, 0x64, "container_pickup", 0x2E]
    assert 0x3F not in _kinds(messages)
    assert (server.world["tanks"][9]["x"], server.world["tanks"][9]["y"]) == (13, 10)


def test_a_deposit_outside_the_stored_window_is_refused_like_any_click() -> None:
    """The window check precedes the law, as it does for every click.

    The client's own viewport bounds the command — ``ae()`` in the JS
    gates the deposit on the same 1..16 viewport test every other
    click passes — so an out-of-window tile draws the measured 0x52
    code 0 and no transfer happens.
    """
    server = _server(fuel=1100)
    server.queue_command(9, _deposit(200, 200, 400))

    messages = server.advance_tick()

    assert 0x52 in _kinds(messages)
    assert 0x64 not in _kinds(messages)
    assert server.world["tanks"][9]["fuel"] == 1100
