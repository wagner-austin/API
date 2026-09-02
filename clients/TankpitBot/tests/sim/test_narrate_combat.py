"""Observer semantics for the shoot family and the corpse window.

Every test resolves a shot ONCE and narrates it for two connections,
for the reason :mod:`tests.sim.test_narrate` gives: a narrator that
ignored its ``observer_id`` passes every single-client test in the
suite and leaks private receipts the moment a second connection exists.

Combat is where that mattered most. Until 2026-09-02 the shot emitter
resolved the shot AND applied the kill reward AND appended the wire
messages in one call taking the client id, so narrating a second
connection would have fired the shot twice. The last test here is the
pin on that: narration cannot move a count, however many times it runs.
"""

from __future__ import annotations

from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.combat import SLOT_RADAR, ShotOutcomeDict, process_shot
from tankpit_bot.sim.equipment import MERCY_BUNDLE
from tankpit_bot.sim.narrate import narrate_corpse_removals, narrate_shot
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world, place_mine
from tests.in_memory_terrain_map import InMemoryTerrainMap

KILLER = 9
VICTIM = 11
OBSERVER = 12
_NOBODY: frozenset[int] = frozenset()


def _arena(victim_fuel: int) -> SimWorldDict:
    """Killer 9 and victim 11 in line, with an unrelated observer 12.

    Args:
        victim_fuel: The victim's starting fuel — 5 dies to one
            single shot, 1000 survives it.

    Returns:
        The world.
    """
    world = make_sim_world("field01_r.gif")
    world["tanks"][KILLER] = make_sim_tank(KILLER, 0, 1, 10, 10, 1000)
    world["tanks"][VICTIM] = make_sim_tank(VICTIM, 1, 1, 15, 10, victim_fuel)
    world["tanks"][OBSERVER] = make_sim_tank(OBSERVER, 1, 1, 40, 40, 1000)
    return world


def _kinds(messages: list[BinaryMessage]) -> list[int | str]:
    """The msg_type of every message, in emission order."""
    return [message["msg_type"] for message in messages]


def _kill() -> tuple[SimWorldDict, ShotOutcomeDict]:
    """Resolve one lethal shot and hand back the world and its outcome.

    Returns:
        The post-shot world and the shot's outcome.
    """
    world = _arena(victim_fuel=5)
    outcome = process_shot(world, InMemoryTerrainMap(), KILLER, 15, 10, _NOBODY, 0, None)
    assert outcome["victim_deactivated"] is True
    return world, outcome


def test_a_shot_that_hits_nothing_is_a_bare_echo() -> None:
    """No 0x49 answers a shot — 92.4% of 11,051 live windows are bare.

    The pre-split emitter snapshotted the shooter's inventory after
    every shot; the response-shape differ caught it as an invented law
    in 2026-08-01, and the narrator must not reintroduce it.
    """
    world = _arena(victim_fuel=1000)
    outcome = process_shot(world, InMemoryTerrainMap(), KILLER, 3, 3, _NOBODY, 0, None)

    assert outcome["victim_id"] is None
    assert _kinds(narrate_shot(outcome, KILLER)) == [0x53]
    assert _kinds(narrate_shot(outcome, OBSERVER)) == [0x53]


def test_the_echo_carries_the_shooters_own_team() -> None:
    """0x53 stamps the team from the outcome, not from a live lookup.

    Carrying it is what lets the narrator run after the world has
    moved on — the shooter's row is never re-read.
    """
    world = _arena(victim_fuel=1000)
    outcome = process_shot(world, InMemoryTerrainMap(), KILLER, 3, 3, _NOBODY, 0, None)
    world["tanks"][KILLER]["team"] = 3

    echo = narrate_shot(outcome, KILLER)[0]
    assert echo["msg_type"] == 0x53
    assert echo["team"] == 0
    assert echo["shooter_id"] == KILLER


def test_the_kill_announcement_reaches_the_room() -> None:
    """0x53 and 0x41 are room-wide; only the killer hears its 0x67.

    The bundle's 0x67 is per-recipient because production treats any
    0x67 as a SELF gain ([[recipient-policy]]).
    """
    _world, outcome = _kill()

    assert _kinds(narrate_shot(outcome, KILLER)) == [0x53, 0x41, 0x67]
    assert _kinds(narrate_shot(outcome, OBSERVER)) == [0x53, 0x41]


def test_the_deactivation_names_both_tanks() -> None:
    """The 0x41 credits the killer and names the victim, mine flag off."""
    _world, outcome = _kill()

    announcement = narrate_shot(outcome, OBSERVER)[1]
    assert announcement["msg_type"] == 0x41
    assert announcement["victim_id"] == VICTIM
    assert announcement["killer_id"] == KILLER
    assert announcement["is_mine_kill"] is False
    assert announcement["promo_eligible"] is False


def test_the_mercy_bundle_is_silent_and_carries_the_measured_stacks() -> None:
    """``show_message=False`` and the deterministic per-slot medians.

    Five of five radar-zero kills in the corpus granted the bundle and
    none of the 254 kills at radar > 0 did (archive-cracked
    2026-07-22).
    """
    _world, outcome = _kill()

    gain = narrate_shot(outcome, KILLER)[2]
    assert gain["msg_type"] == 0x67
    assert gain["show_message"] is False
    assert gain["gained"] == list(MERCY_BUNDLE)


def test_a_killer_holding_radars_earns_no_bundle() -> None:
    """Radar > 0 is the measured non-trigger: 0 of 254 such kills paid."""
    world = _arena(victim_fuel=5)
    world["tanks"][KILLER]["counts"][SLOT_RADAR] = 1
    outcome = process_shot(world, InMemoryTerrainMap(), KILLER, 15, 10, _NOBODY, 0, None)

    assert outcome["victim_deactivated"] is True
    assert outcome["mercy"] is None
    assert _kinds(narrate_shot(outcome, KILLER)) == [0x53, 0x41]
    assert world["tanks"][KILLER]["counts"] == [0, 0, 0, 0, 1]


def test_the_mine_cascade_rides_the_echo_for_everyone() -> None:
    """0x45 broadcasts: 296 detonations against 23 placements."""
    world = _arena(victim_fuel=1000)
    place_mine(world, 13, 10, 3)
    outcome = process_shot(world, InMemoryTerrainMap(), KILLER, 13, 10, _NOBODY, 0, None)

    assert outcome["mine_cascade"] == [[(13, 10)]]
    assert _kinds(narrate_shot(outcome, KILLER)) == [0x53, 0x45]
    assert _kinds(narrate_shot(outcome, OBSERVER)) == [0x53, 0x45]


def test_narrating_a_kill_twice_pays_the_bundle_once() -> None:
    """THE PIN. Narration is pure, so a second connection is free.

    The bundle lands in the killer's counts when the SHOT resolves.
    Narrating for two observers afterwards must leave those counts
    exactly where the shot put them — the pre-split emitter would have
    added a second bundle here, which is the whole reason the resolve
    and narrate halves were separated.
    """
    world, outcome = _kill()
    assert world["tanks"][KILLER]["counts"] == list(MERCY_BUNDLE)

    narrate_shot(outcome, KILLER)
    narrate_shot(outcome, OBSERVER)
    narrate_shot(outcome, VICTIM)

    assert world["tanks"][KILLER]["counts"] == list(MERCY_BUNDLE)


def test_corpse_removals_are_one_message_per_closed_window() -> None:
    """Each closed window draws its own 0x58, in the order given."""
    removals = narrate_corpse_removals([11, 12, 500])

    assert removals == [
        {"msg_type": 0x58, "tank_id": 11},
        {"msg_type": 0x58, "tank_id": 12},
        {"msg_type": 0x58, "tank_id": 500},
    ]


def test_a_tick_with_no_closed_window_says_nothing() -> None:
    """An empty expiry list narrates to an empty batch, not a stray 0x58."""
    assert narrate_corpse_removals([]) == []
