"""The razed-pool memory, and the gate that times the walk back.

What is pinned: a pool counts as razed exactly when an owned extractor
stood there and no longer does, the memory clears the moment one stands
there again, and the embargo opens on the same wave-break signal the
strike release reads -- with the knob off reproducing today's behaviour
bit for bit ([[impossible-economy-problem]]).
"""

from __future__ import annotations

from rw_bot.policy.reclaim import EXTRACTOR_TYPES, Razed, embargoed
from rw_bot.wire.state import Sample
from tests.wire_fixtures import entity, sample


def _world(*extractors: tuple[int, float, float]) -> Sample:
    return sample(
        entity(1, "commandCenter"),
        *(entity(unit_id, "extractorT1", x=x, y=y) for unit_id, x, y in extractors),
    )


def test_nothing_is_razed_before_anything_was_owned() -> None:
    razed = Razed()
    razed.observe(_world())
    assert razed.positions() == ()


def test_a_lost_extractor_marks_where_it_stood() -> None:
    razed = Razed()
    razed.observe(_world((7, 120.0, 44.0), (8, 300.0, 90.0)))
    razed.observe(_world((8, 300.0, 90.0)))
    assert razed.positions() == ((120.0, 44.0),)


def test_a_reclaimed_pool_leaves_the_memory() -> None:
    """The memory answers "is this pool razed", not "was it ever"."""
    razed = Razed()
    razed.observe(_world((7, 120.0, 44.0)))
    razed.observe(_world())
    razed.observe(_world((9, 120.0, 44.0)))
    assert razed.positions() == ()


def test_float_jitter_within_a_tile_is_one_pool() -> None:
    """The stream carries floats; the identity is the rounded tile, so a
    re-observation half a unit off does not double the memory."""
    razed = Razed()
    razed.observe(_world((7, 120.2, 44.1)))
    razed.observe(_world())
    razed.observe(_world((9, 119.8, 43.9)))
    assert razed.positions() == ()


def test_an_upgraded_extractor_is_still_ours() -> None:
    """The tier chains are the roster: a T1 that converted to T2 did not
    get razed, and its later loss is tracked like any other."""
    razed = Razed()
    razed.observe(_world((7, 120.0, 44.0)))
    upgraded = sample(entity(1, "commandCenter"), entity(7, "extractorT2", x=120.0, y=44.0))
    razed.observe(upgraded)
    assert razed.positions() == ()
    razed.observe(_world())
    assert razed.positions() == ((120.0, 44.0),)


def test_a_rivals_extractor_is_not_ours_to_lose() -> None:
    razed = Razed()
    theirs = sample(
        entity(1, "commandCenter"),
        entity(7, "extractorT1", x=50.0, y=50.0, mine=False, hostile=True),
    )
    razed.observe(theirs)
    razed.observe(_world())
    assert razed.positions() == ()


def test_the_roster_is_the_tier_chain() -> None:
    assert "extractorT1" in EXTRACTOR_TYPES
    assert "extractorT3_reinforced" in EXTRACTOR_TYPES
    assert "landFactory" not in EXTRACTOR_TYPES


def test_the_embargo_is_off_at_zero() -> None:
    """Today's behaviour exactly: the walk back starts at once."""
    assert embargoed(((120.0, 44.0),), 0, 0) == ()


def test_the_embargo_holds_while_the_wave_holds() -> None:
    assert embargoed(((120.0, 44.0),), 2999, 3000) == ((120.0, 44.0),)


def test_the_embargo_opens_on_the_wave_break() -> None:
    assert embargoed(((120.0, 44.0),), 3000, 3000) == ()
