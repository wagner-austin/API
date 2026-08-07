"""Tests for the enemy-tracking probe's record building.

The probe exists to catch the bot abandoning a target it should still
hold, so these cover the two halves that make a divergence row
readable: pairing a wire-side tank to its minified JS registry entry,
and the summary that reports which side thought the tank was gone.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from tests.action_lab._enemy_tracking_harness import (
    _make_js_belief,
    _make_observation,
    _make_our_belief,
    _make_session,
    _make_snapshot,
    _make_tracked,
)
from tests.action_lab._replay_page import ClockAdvancingPage, ReplayClock

from tankpit_bot._test_hooks.bot import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.enemy_tracking_records import (
    _build_sample_observations,
    _build_tracked_records,
    _resolve_identity,
    _wait_for_shot_feedback,
    format_enemy_tracking_probe_summary,
)
from tankpit_bot.action_lab.enemy_tracking_types import TrackingObservationDict
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.bot.ai.world_types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state import WorldStateDict, make_empty_world_state, make_self_state
from tankpit_bot.state.types import make_tank_state


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore the action-lab hooks these tests swap."""
    original_get_time = action_hooks.get_current_time_ms
    original_drain = action_hooks.drain_buffered_messages
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.drain_buffered_messages = original_drain


class _DrainProbe(ProbeBase):
    """Probe whose only job here is to be drained."""

    def __init__(self) -> None:
        """Build a probe with no browser behind it."""
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=False)


def _install_noop_drain() -> None:
    """Make message draining a no-op for the feedback wait."""

    def _drain(source: BufferedMessageSourceProtocol, /) -> int:
        _ = source
        return 0

    action_hooks.drain_buffered_messages = _drain


def _snapshot_with(
    entries: list[dict[str, int | float | bool | str | None]],
) -> PageClientSnapshotDict:
    """Return a snapshot whose JS tank registry holds ``entries``.

    ``P.j`` is the registry key the pairing reads; anything else is
    invisible to it, which is worth stating because a wrong key makes
    the unpaired-enemy case pass for the wrong reason.
    """
    snapshot = _make_snapshot()
    snapshot["world_collections"] = {"P.j": entries}
    return snapshot


def _threat(tank_id: int, x: int, y: int, name: str) -> EnemyThreatDict:
    """Build an enemy threat at a tile."""
    return make_enemy_threat(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=1,
        damage_state=3,
        rank=2,
        team=1,
        name=name,
        is_bot=False,
        timestamp_ms=1000,
        last_wire_seen_ms=1000,
        last_position_update_ms=1000,
    )


def test_wait_for_shot_feedback_reports_a_hit() -> None:
    """A response carrying a confirmed hit ends the wait as a hit."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    _install_noop_drain()
    world_service = get_world_service()
    world_service.got_our_shot_response = True
    world_service.got_confirmed_hit = True

    assert _wait_for_shot_feedback(
        ClockAdvancingPage(clock),
        _DrainProbe(),
        timeout_ms=1000,
    ) == (True, True)


def test_wait_for_shot_feedback_reports_a_miss() -> None:
    """A response without a confirmed hit ends the wait as a miss."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    _install_noop_drain()
    world_service = get_world_service()
    world_service.got_our_shot_response = True
    world_service.got_confirmed_hit = False

    assert _wait_for_shot_feedback(
        ClockAdvancingPage(clock),
        _DrainProbe(),
        timeout_ms=1000,
    ) == (True, False)


def test_wait_for_shot_feedback_times_out_without_a_response() -> None:
    """No response inside the window reports a timeout, not a miss.

    The distinction matters: a miss is the server answering, a
    timeout is the server saying nothing, and only the second one
    means the shot may never have been dispatched.
    """
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    _install_noop_drain()
    get_world_service().got_our_shot_response = False

    assert _wait_for_shot_feedback(
        ClockAdvancingPage(clock),
        _DrainProbe(),
        timeout_ms=300,
    ) == (False, False)


def test_resolve_identity_finds_the_field_holding_our_tank_id() -> None:
    """Whichever minified field equals our tank id is the JS tank id."""
    entry: dict[str, int | float | bool | str | None] = {"a": 511, "b": 77, "c": 100}
    assert _resolve_identity(entry, 77) == ("b", "77")


def test_resolve_identity_reports_no_pairing_when_nothing_matches() -> None:
    """An unpairable entry records empty strings rather than guessing."""
    entry: dict[str, int | float | bool | str | None] = {"a": 511, "b": 77}
    assert _resolve_identity(entry, 12345) == ("", "")


def test_build_tracked_records_pairs_each_enemy_to_its_js_entry() -> None:
    """A position match resolves the cross-tick join key for the tank."""
    snapshot = _snapshot_with([{"x": 40, "y": 50, "id": 77}])

    records = _build_tracked_records([_threat(77, 40, 50, "red-77")], snapshot)

    assert records == [
        {
            "tank_id": 77,
            "name": "red-77",
            "team": 1,
            "rank": 2,
            "acquired_x": 40,
            "acquired_y": 50,
            "tracked_js_key": "id",
            "tracked_js_value": "77",
        }
    ]


def test_build_tracked_records_keeps_an_unpaired_enemy() -> None:
    """No JS entry at the tank's tile still yields a wire-side row.

    Dropping the row would hide exactly the case the probe was built
    to catch -- our side believing in a tank the JS client does not
    list -- so the record survives with empty tracking keys.
    """
    snapshot = _snapshot_with([{"x": 1, "y": 2, "id": 99}])

    records = _build_tracked_records([_threat(77, 40, 50, "red-77")], snapshot)

    assert [(r["tank_id"], r["tracked_js_key"], r["tracked_js_value"]) for r in records] == [
        (77, "", "")
    ]


def test_build_sample_observations_returns_one_row_per_tracked_tank() -> None:
    """Each sample produces exactly one row per tank under track."""
    world: WorldStateDict = make_empty_world_state()
    world["self_state"] = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=5,
    )
    world["tanks"] = {
        "77": make_tank_state(
            tank_id=77,
            x=40,
            y=50,
            team=1,
            rank=2,
            damage_state=3,
            name="red-77",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        )
    }

    rows: list[TrackingObservationDict] = _build_sample_observations(
        sample_index=4,
        sample_timestamp_ms=5000,
        tracked=[_make_tracked()],
        world=world,
        threats=[_threat(77, 40, 50, "red-77")],
        snapshot=_snapshot_with([{"x": 40, "y": 50, "id": 77}]),
        bot_combat_target_id=77,
        bot_mode_state="hunt/engaging",
    )

    assert [(r["sample_index"], r["bot_combat_target_id"], r["bot_mode_state"]) for r in rows] == [
        (4, 77, "hunt/engaging")
    ]


def _observation(*, ours: bool, js: bool) -> TrackingObservationDict:
    """Build an observation whose two sides believe what is asked."""
    observation = _make_observation()
    our_belief = _make_our_belief()
    our_belief["would_locked_target_return"] = ours
    observation["our_belief"] = our_belief
    observation["js_belief"] = _make_js_belief(present=js)
    return observation


def test_summary_counts_divergence_in_both_directions() -> None:
    """The summary separates which side believed the tank was still there.

    Both halves matter and mean opposite things: ours-present with
    JS-absent is a stale wire TTL, JS-present with ours-absent is the
    lock released too early -- the bug the probe was built to find.
    """
    session = _make_session()
    session["observations"] = [
        _observation(ours=True, js=True),
        _observation(ours=True, js=False),
        _observation(ours=False, js=True),
        _observation(ours=False, js=True),
    ]

    summary = format_enemy_tracking_probe_summary(session)

    assert "samples=4" in summary
    assert "divergence=3" in summary
    assert "our_present_js_absent=1" in summary
    assert "js_present_our_absent=2" in summary


def test_summary_reports_no_divergence_when_both_sides_agree() -> None:
    """Agreement on every row is the clean run the operator wants to see."""
    session = _make_session()
    session["observations"] = [
        _observation(ours=True, js=True),
        _observation(ours=False, js=False),
    ]

    summary = format_enemy_tracking_probe_summary(session)

    assert "divergence=0" in summary
    assert "our_present_js_absent=0" in summary
    assert "js_present_our_absent=0" in summary
    assert f"tracked={len(session['tracked'])}" in summary
