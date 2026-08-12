"""Tests for the shared distance rule behind every lock release.

:func:`is_lock_release_warranted` is the distance half of both release
paths -- the equipment lock uses it alone, and the fuel lock uses it as
path 1 before falling back to the value rule. Until 2026-08-11 it had
no direct tests: it was only ever reached through callers that never
constructed the case where its two conditions disagree, so a mutation
sweep found the "at most half" arm indistinguishable from absent.

The rule is a conjunction, and each arm rejects a different shape:

* the HALF arm rejects a candidate that is closer but not decisively
  so, which is what stops a lock oscillating between two comparable
  targets;
* the MIN-GAP arm rejects a candidate that is proportionally much
  closer but only trivially nearer in absolute tiles (2 tiles vs 5 is
  "half the distance" and worth nothing).

The identical-coordinates case is pinned here deliberately. Callers
rely on this predicate rejecting a candidate that sits on the locked
tile, so that property belongs in a test rather than in each caller's
assumptions.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.equipment import _LOCK_RELEASE_MIN_GAP, is_lock_release_warranted
from tankpit_bot.state.types import SelfStateDict, make_self_state


def _self_at_origin() -> SelfStateDict:
    """Return a self state at (0,0) so distances read as plain coordinates."""
    return make_self_state(
        tank_id=1,
        x=0,
        y=0,
        team=1,
        rank=1,
        fuel=500,
        leaderboard_position=40,
    )


class TestBothArmsMustAgree:
    """A release needs the candidate at most half the distance AND >= the gap."""

    def test_releases_when_markedly_and_absolutely_closer(self) -> None:
        """40 tiles away vs 15: within half, and 25 tiles of gap."""
        assert is_lock_release_warranted(_self_at_origin(), 40, 0, 15, 0) is True

    def test_half_arm_rejects_a_candidate_that_clears_the_gap(self) -> None:
        """The case no caller ever built: gap passes, half-rule refuses.

        Locked at 21, candidate at 11. The absolute gap is exactly
        ``_LOCK_RELEASE_MIN_GAP``, so the min-gap arm is satisfied --
        but ``11 * 2 > 21``, so the candidate is not within half the
        distance and the lock holds. Deleting the half arm flips this
        to True, which is a lock released for a 10-tile improvement on
        a 21-tile trip.
        """
        assert _LOCK_RELEASE_MIN_GAP == 10
        assert is_lock_release_warranted(_self_at_origin(), 21, 0, 11, 0) is False

    def test_half_arm_rejects_a_larger_instance_of_the_same_shape(self) -> None:
        """Locked at 30, candidate at 16: gap 14, but 32 > 30."""
        assert is_lock_release_warranted(_self_at_origin(), 30, 0, 16, 0) is False

    def test_min_gap_arm_rejects_a_proportionally_closer_candidate(self) -> None:
        """Locked at 12, candidate at 5: within half, but only 7 tiles."""
        assert is_lock_release_warranted(_self_at_origin(), 12, 0, 5, 0) is False

    def test_exact_half_with_a_sufficient_gap_releases(self) -> None:
        """Locked at 40, candidate at 20: ``40*2 > 40`` is false, gap 20."""
        assert is_lock_release_warranted(_self_at_origin(), 40, 0, 20, 0) is True


class TestCandidateOnTheLockedTile:
    """A candidate at the locked coordinates is never an improvement.

    Callers depend on this: it is what makes a separate
    "candidate is the locked target" check unnecessary.
    """

    def test_identical_coordinates_at_distance_hold_the_lock(self) -> None:
        """Same tile 30 away: ``60 > 30`` fires the half arm."""
        assert is_lock_release_warranted(_self_at_origin(), 30, 0, 30, 0) is False

    def test_identical_coordinates_underfoot_hold_the_lock(self) -> None:
        """Both at the tank's own tile: zero gap fails the min-gap arm.

        The distance-zero case skips the half arm (``0 > 0`` is false)
        and is caught by the gap arm instead, so the property holds
        across both branches rather than only the common one.
        """
        assert is_lock_release_warranted(_self_at_origin(), 0, 0, 0, 0) is False


class TestFartherCandidates:
    """A candidate farther away than the lock never releases it."""

    def test_farther_candidate_holds_the_lock(self) -> None:
        """Locked at 10, candidate at 25."""
        assert is_lock_release_warranted(_self_at_origin(), 10, 0, 25, 0) is False

    def test_equal_distance_on_a_different_tile_holds_the_lock(self) -> None:
        """Locked at (10,0), candidate at (0,10): same distance, no gain."""
        assert is_lock_release_warranted(_self_at_origin(), 10, 0, 0, 10) is False
