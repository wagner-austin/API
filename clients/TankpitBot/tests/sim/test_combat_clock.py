"""The field's combat clocks: charge latency, corpse windows, kill book.

These are FIELD facts, which is the point the module was split out to
make. A corpse clears once for the room, a firing cost is billed once
against the shooter, and a kill is scored once — however many
connections are watching. The clocks used to hang off the single
client's session, where "the client's corpse clocks" and "the room's
corpse clocks" were indistinguishable.

Nothing here emits: the clock reports what came due and
:mod:`tankpit_bot.sim.narrate.combat` turns that into wire.
"""

from __future__ import annotations

from tankpit_bot.sim.combat_clock import CORPSE_WINDOW_TICKS, CombatClock
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world

SHOOTER = 9
TARGET = 11


def _world() -> SimWorldDict:
    """Shooter 9 with 1000 fuel and target 11 with 40, at tick zero."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][SHOOTER] = make_sim_tank(SHOOTER, 0, 1, 10, 10, 1000)
    world["tanks"][TARGET] = make_sim_tank(TARGET, 1, 1, 15, 10, 40)
    return world


def test_a_deferred_debit_costs_nothing_until_it_is_applied() -> None:
    """Charge latency: the shot's fuel is billed one tick LATER.

    Measured law — the shooter's own cost never lands in the batch its
    shot did.
    """
    world = _world()
    clock = CombatClock(world)

    clock.defer_debit(SHOOTER, 6)
    assert world["tanks"][SHOOTER]["fuel"] == 1000

    clock.apply_pending_debits()
    assert world["tanks"][SHOOTER]["fuel"] == 994


def test_applying_twice_bills_once() -> None:
    """The pending list empties, so a quiet tick costs nothing."""
    world = _world()
    clock = CombatClock(world)
    clock.defer_debit(SHOOTER, 6)

    clock.apply_pending_debits()
    clock.apply_pending_debits()

    assert world["tanks"][SHOOTER]["fuel"] == 994


def test_every_shot_in_a_tick_is_billed() -> None:
    """Two shots the same tick defer two debits, both billed together."""
    world = _world()
    clock = CombatClock(world)
    clock.defer_debit(SHOOTER, 6)
    clock.defer_debit(SHOOTER, 10)

    clock.apply_pending_debits()

    assert world["tanks"][SHOOTER]["fuel"] == 984


def test_a_debit_cannot_drive_fuel_negative() -> None:
    """An empty tank floors at zero rather than going into debt."""
    world = _world()
    clock = CombatClock(world)
    clock.defer_debit(TARGET, 100)

    clock.apply_pending_debits()

    assert world["tanks"][TARGET]["fuel"] == 0


def test_the_corpse_window_holds_for_the_measured_22_seconds() -> None:
    """0x58 lands EXACTLY 11 ticks after the 0x41, never earlier.

    Corpus-swept 2026-07-22: 37 kill->remove pairs, min = median =
    22.0 s at the 2 s tick cadence.
    """
    world = _world()
    clock = CombatClock(world)
    clock.record_deactivation(SHOOTER, TARGET)

    for _ in range(CORPSE_WINDOW_TICKS - 1):
        world["tick"] += 1
        assert clock.expire_corpses() == []

    world["tick"] += 1
    assert clock.expire_corpses() == [TARGET]


def test_a_closed_window_is_reported_once() -> None:
    """The window is forgotten on expiry, so no corpse clears twice."""
    world = _world()
    clock = CombatClock(world)
    clock.record_deactivation(SHOOTER, TARGET)
    world["tick"] = CORPSE_WINDOW_TICKS

    assert clock.expire_corpses() == [TARGET]
    assert clock.expire_corpses() == []


def test_corpses_clear_in_ascending_tank_order() -> None:
    """Several windows closing together report deterministically."""
    world = _world()
    clock = CombatClock(world)
    for victim_id in (500, 11, 77):
        clock.record_deactivation(SHOOTER, victim_id)
    world["tick"] = CORPSE_WINDOW_TICKS

    assert clock.expire_corpses() == [11, 77, 500]


def test_the_kill_book_scores_both_sides_of_one_deactivation() -> None:
    """One event, two rows: the killer's kill and the victim's death."""
    world = _world()
    clock = CombatClock(world)

    clock.record_deactivation(SHOOTER, TARGET)
    clock.record_deactivation(SHOOTER, TARGET)

    assert clock.destroyed_by(SHOOTER) == 2
    assert clock.deactivations_of(TARGET) == 2
    assert clock.deactivations_of(SHOOTER) == 0
    assert clock.destroyed_by(TARGET) == 0


def test_an_unscored_tank_reads_zero_rather_than_missing() -> None:
    """The 0x56 answer needs a number for a tank that has done nothing."""
    clock = CombatClock(_world())

    assert clock.destroyed_by(4242) == 0
    assert clock.deactivations_of(4242) == 0
