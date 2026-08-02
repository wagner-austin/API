"""Tests for value-aware fuel-lock release (the flag-13 dreg fix).

The 2026-07-29 flag-13 incident ([[flag-triage-20260729]], run
bot-20260729-232252 tick 1055): an 84-fuel remnant held its lock at
deficit 207 against a 462-volume container five tiles away, because
the release rule was distance-only. These tests pin the deliverable
score, both release paths, the hysteresis that prevents lock
ping-pong, and the lock-continuation integration in collect mode.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_locks import continue_or_release_fuel_lock
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.equipment import (
    fuel_deliverable_score,
    is_fuel_lock_release_warranted,
)
from tankpit_bot.state.types import (
    ContainerStateDict,
    SelfStateDict,
    make_container_state,
    make_self_state,
)
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def _fuel_container(x: int, y: int, volume: int) -> ContainerStateDict:
    """Return a fuel container belief at the given tile and volume."""
    return make_container_state(
        x=x,
        y=y,
        is_fuel=True,
        volume=volume,
        timestamp_ms=100000,
        failed_pickups=0,
    )


def _self_at(x: int, y: int, fuel: int) -> SelfStateDict:
    """Return a self state at the given tile with the given fuel."""
    return make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=1,
        rank=1,
        fuel=fuel,
        leaderboard_position=40,
    )


class TestFuelDeliverableScore:
    """Deliverable value net of travel."""

    def test_volume_beyond_deficit_is_clamped(self) -> None:
        """A 900-volume container at deficit 200 delivers only 200."""
        self_state = _self_at(100, 100, 900)

        score = fuel_deliverable_score(self_state, _fuel_container(103, 104, 900), 200)

        assert score == 200 - 7

    def test_negative_deficit_floors_at_zero(self) -> None:
        """At or above capacity nothing is deliverable; distance still costs."""
        self_state = _self_at(100, 100, 1100)

        score = fuel_deliverable_score(self_state, _fuel_container(103, 104, 900), -10)

        assert score == -7

    def test_small_volume_delivers_itself(self) -> None:
        """A dreg below the deficit delivers exactly its volume."""
        self_state = _self_at(242, 16, 893)

        score = fuel_deliverable_score(self_state, _fuel_container(249, 18, 84), 207)

        assert score == 84 - 9


class TestFuelLockRelease:
    """Both release paths + hysteresis."""

    def test_flag_13_dreg_releases_for_high_volume_candidate(self) -> None:
        """The exact flag-13 shape releases on the value path.

        Locked (249,18) vol 84 at dist 9 (deliverable score 75) vs
        candidate (242,21) vol 462 at dist 5 (score 202): the distance
        rule does NOT fire (5*2 > 9), the value rule does (202 >= 150).
        """
        self_state = _self_at(242, 16, 893)

        released = is_fuel_lock_release_warranted(
            self_state,
            _fuel_container(249, 18, 84),
            _fuel_container(242, 21, 462),
            207,
        )

        assert released

    def test_comparable_containers_hold_the_lock(self) -> None:
        """Near-equal deliverable scores stay locked (no ping-pong)."""
        self_state = _self_at(100, 100, 893)

        released = is_fuel_lock_release_warranted(
            self_state,
            _fuel_container(105, 105, 300),
            _fuel_container(104, 104, 350),
            207,
        )

        assert not released

    def test_markedly_closer_candidate_still_releases_on_distance(self) -> None:
        """The pre-existing distance rule remains a release path."""
        self_state = _self_at(100, 100, 893)

        released = is_fuel_lock_release_warranted(
            self_state,
            _fuel_container(100, 140, 300),
            _fuel_container(100, 110, 300),
            207,
        )

        assert released

    def test_worthless_dreg_releases_for_any_positive_candidate(self) -> None:
        """A locked target not worth its own walk floors at score 1."""
        self_state = _self_at(100, 100, 893)

        released = is_fuel_lock_release_warranted(
            self_state,
            _fuel_container(100, 130, 5),
            _fuel_container(100, 110, 50),
            207,
        )

        assert released


class TestLockContinuationIntegration:
    """The collect-mode lock path applies the value rule."""

    def test_dreg_lock_releases_when_high_volume_fuel_is_visible(self) -> None:
        """The flag-13 shape clears the lock in continue_or_release_fuel_lock."""
        world, self_state = make_world(
            self_x=242,
            self_y=16,
            fuel=893,
            containers={
                "249,18": _fuel_container(249, 18, 84),
                "242,21": _fuel_container(242, 21, 462),
            },
        )
        ai_state = make_scanned_ai_state()
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        decision, updated = continue_or_release_fuel_lock(
            ctx, ctx.base, _fuel_container(249, 18, 84)
        )

        assert decision is None
        assert updated["resource_target_kind"] == ""

    def test_comparable_lock_continues(self) -> None:
        """A lock on comparable-value fuel keeps dispatching."""
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=893,
            containers={
                "105,105": _fuel_container(105, 105, 300),
                "104,104": _fuel_container(104, 104, 350),
            },
        )
        ai_state = make_scanned_ai_state()
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        decision, _updated = continue_or_release_fuel_lock(
            ctx, ctx.base, _fuel_container(105, 105, 300)
        )

        if decision is None:
            raise AssertionError("a comparable fuel lock must continue, not release")
        assert decision["behavior"]["reason_kind"] == "fuel_locked"
        assert decision["behavior"]["target_x"] == 105
        assert decision["behavior"]["target_y"] == 105
