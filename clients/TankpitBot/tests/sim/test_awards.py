"""Law 11 — decorations, their packing, and the 0x4E that grants them."""

from __future__ import annotations

import pytest

from tankpit_bot.protocol.decoders import decode_tank_info
from tankpit_bot.protocol.encoders import encode_message_payload
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.awards import (
    DECORATION_SLOTS,
    HONOR_DEATHS,
    SLOT_HONOR,
    SLOT_STARS,
    SLOT_TANK,
    STAR_RANKS,
    TANK_KILLS,
    AwardLedger,
    pack_decorations,
)
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

#: Artax's own decoration bytes, lifted from the archive's 0x3E before
#: the Golden Tank landed: Double Star, Silver Tank, Combat Honor Medal.
_ARTAX_BEFORE_GOLD = b"\x1a\x00\x00\x00"


def test_the_packer_reproduces_artaxs_own_decoration_bytes() -> None:
    """The one real decoration_state in the archive round-trips.

    ``Stars=2, Tank=2, Honor=1`` is what every session up to
    2026-07-29 carried; the packer is the inverse of the JS ``yg``
    ([[decoration-encoding]]).
    """
    levels = [0] * DECORATION_SLOTS
    levels[SLOT_STARS] = 2
    levels[SLOT_TANK] = 2
    levels[SLOT_HONOR] = 1

    assert pack_decorations(tuple(levels)) == _ARTAX_BEFORE_GOLD


def test_the_packed_bytes_survive_the_production_codec() -> None:
    """What the sim packs is what the real 0x21 decoder reads back."""
    levels = [0] * DECORATION_SLOTS
    levels[SLOT_TANK] = 3
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 10, 10, 1100)

    from tankpit_bot.sim.wire_statements import identity_statement

    identity = identity_statement(world, 9, pack_decorations(tuple(levels)))
    payload = encode_message_payload(identity)

    assert decode_tank_info(payload)["decoration_state"] == pack_decorations(tuple(levels))


@pytest.mark.parametrize("bad", [(0,) * 8, (0,) * 10])
def test_the_packer_refuses_a_wrong_slot_count(bad: tuple[int, ...]) -> None:
    """Nine slots or nothing — a short pack would drop an award."""
    with pytest.raises(ValueError, match="9 slots"):
        pack_decorations(bad)


def test_the_packer_refuses_a_level_two_bits_cannot_hold() -> None:
    """Level 4 would silently bleed into the next slot."""
    levels = [0] * DECORATION_SLOTS
    levels[SLOT_TANK] = 4
    with pytest.raises(ValueError, match="outside 0-3"):
        pack_decorations(tuple(levels))


def test_a_fresh_account_carries_no_awards() -> None:
    """The sim's client starts where a new account starts."""
    assert AwardLedger(9).decoration_state == bytes(4)


def test_the_five_hundredth_kill_grants_the_golden_tank() -> None:
    """The exact grant the archive caught, on 2026-07-29."""
    ledger = AwardLedger(9)
    messages: list[BinaryMessage] = []

    ledger.advance(1, TANK_KILLS[2], 0, 0, messages)

    assert messages == [{"msg_type": 0x4E, "tank_id": 9, "slot": SLOT_TANK, "level": 3}]
    assert ledger.levels[SLOT_TANK] == 3


def test_each_threshold_grants_once() -> None:
    """Crossing grants; staying past it does not grant again."""
    ledger = AwardLedger(9)
    first: list[BinaryMessage] = []
    ledger.advance(1, TANK_KILLS[0], 0, 0, first)
    again: list[BinaryMessage] = []
    ledger.advance(1, TANK_KILLS[0] + 50, 0, 0, again)

    assert first == [{"msg_type": 0x4E, "tank_id": 9, "slot": SLOT_TANK, "level": 1}]
    assert again == []


def test_deaths_grant_the_combat_honor_medal() -> None:
    """20 deactivations — the medal Artax carries."""
    ledger = AwardLedger(9)
    messages: list[BinaryMessage] = []

    ledger.advance(1, 0, HONOR_DEATHS[0], 0, messages)

    assert messages == [{"msg_type": 0x4E, "tank_id": 9, "slot": SLOT_HONOR, "level": 1}]


def test_colonel_grants_the_double_star() -> None:
    """Rank 7 — the star Artax carries."""
    ledger = AwardLedger(9)
    messages: list[BinaryMessage] = []

    ledger.advance(STAR_RANKS[1], 0, 0, 0, messages)

    assert messages == [{"msg_type": 0x4E, "tank_id": 9, "slot": SLOT_STARS, "level": 2}]


def test_awards_never_go_back_down() -> None:
    """A demotion does not take a star back.

    The archive shows Artax keeping ``Tank=3`` in every session after
    the one that granted it.
    """
    ledger = AwardLedger(9)
    ledger.advance(STAR_RANKS[2], 0, 0, 0, [])
    messages: list[BinaryMessage] = []

    ledger.advance(0, 0, 0, 0, messages)

    assert messages == []
    assert ledger.levels[SLOT_STARS] == 3


def _world() -> SimWorldDict:
    """A world with just the client tank."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 10, 10, 1100)
    return world


def test_the_join_burst_carries_the_clients_awards() -> None:
    """0x21 and 0x3E both report what the client has earned."""
    from tankpit_bot.sim.server import SimServer

    server = SimServer(_world(), InMemoryTerrainMap(), client_id=9)
    server.session.awards.levels[SLOT_TANK] = 3

    burst = server.handshake()

    expected = server.session.awards.decoration_state
    assert expected != bytes(4)
    match burst[0], burst[1]:
        case (
            {"msg_type": 0x21, "decoration_state": bytes(identity_state)},
            {"msg_type": 0x3E, "decoration_state": bytes(status_state)},
        ):
            assert (identity_state, status_state) == (expected, expected)
        case _:
            raise AssertionError("the join burst opens 0x21 then 0x3E")


def test_the_server_grants_on_the_tick() -> None:
    """The tick processor drives the ledger, not just the unit.

    The kills are booked through the field's real recording API — the
    same call a resolved deactivation makes — rather than by writing a
    counter, so the award reads the kill book the 0x56 answer reads.
    Each victim is distinct because a deactivation opens that victim's
    corpse window, and one tank cannot die a hundred times at once.
    """
    from tankpit_bot.sim.server import SimServer

    world = _world()
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    for victim_id in range(1000, 1000 + TANK_KILLS[0]):
        server.combat.record_deactivation(9, victim_id)

    batch = server.advance_tick()

    grants = [message for message in batch if message["msg_type"] == 0x4E]
    assert grants == [{"msg_type": 0x4E, "tank_id": 9, "slot": SLOT_TANK, "level": 1}]
