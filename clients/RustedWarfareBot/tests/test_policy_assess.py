"""The assessment layer's first tenant: the match-wide air latch.

Moved from the expander's private state because the answer belongs to the
match, not to one channel -- and pinned here at the unit level so its
campaign consumers can trust the latch without re-testing it.
"""

from __future__ import annotations

from rw_bot.policy.assess import AirWatch
from tests.wire_fixtures import enemy, entity, sample


def test_a_hostile_aircraft_latches_and_the_fog_cannot_unlatch_it() -> None:
    """Sorties leave the viewport and come back; an answer that stands down
    between them arms anti-air that is never finished when one arrives."""
    watch = AirWatch()
    quiet = sample(entity(213, "commandCenter"))
    watch.observe(quiet)
    assert watch.seen() is False
    sortie = sample(entity(213, "commandCenter"), enemy(9, "c_helicopter", flying=True))
    watch.observe(sortie)
    assert watch.seen() is True
    watch.observe(quiet)
    assert watch.seen() is True


def test_a_grounded_hostile_arms_nothing() -> None:
    watch = AirWatch()
    watch.observe(sample(entity(213, "commandCenter"), enemy(9, "c_tank")))
    assert watch.seen() is False


def test_our_own_aircraft_arms_nothing() -> None:
    """The latch reads THEIR air: our own gunship overhead says nothing
    about whether anti-air cover will ever have a target."""
    watch = AirWatch()
    watch.observe(sample(entity(213, "commandCenter"), entity(7, "gunShip", flying=True)))
    assert watch.seen() is False
