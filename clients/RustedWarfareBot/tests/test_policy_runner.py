"""Deciding whether an order already given is still being carried out.

The loop these rules used to live in is gone; one tick runs the whole match now
([[policy-loop]]). What is left is the judgement that was never about looping:
the world reports no acknowledgement of an order, so something has to tell
"in progress" from "silently refused", and this is the only place that happens.
"""

from __future__ import annotations

from typing import Literal

from rw_bot.policy.build_order import Decision
from rw_bot.policy.runner import AFFORD_STALL_SAMPLES, DEFAULT_STALL_SAMPLES, OrderTracker
from rw_bot.wire.state import Sample
from tests.wire_fixtures import entity, sample

_PLAN = ("landFactory", "c_tank")


def _build(type_name: str = "landFactory", *, unit_id: int = 214) -> Decision:
    """A decision to place a structure."""
    return Decision(
        action="build",
        reason=f"building {type_name}",
        type_name=type_name,
        unit_id=unit_id,
        x=10.0,
        y=20.0,
        deficit=0,
    )


def _produce(type_name: str = "c_tank", *, unit_id: int = 300) -> Decision:
    """A decision to queue a unit."""
    return Decision(
        action="produce",
        reason=f"producing {type_name}",
        type_name=type_name,
        unit_id=unit_id,
        x=0.0,
        y=0.0,
        deficit=0,
    )


def _plain(action: Literal["wait", "done", "blocked", "stalled"], reason: str) -> Decision:
    """A decision carrying no order.

    The verb is typed as the literal union the production type declares rather
    than as ``str``, so a test cannot construct a decision the policy could not.
    """
    return Decision(action=action, reason=reason, type_name="", unit_id=0, x=0.0, y=0.0, deficit=0)


def _saving(deficit: int, *, unit_id: int = 214) -> Decision:
    """A decision waiting to afford something, carrying its shortfall."""
    return Decision(
        action="wait",
        reason=f"landFactory costs 700, holding {700 - deficit}",
        type_name="",
        unit_id=unit_id,
        x=0.0,
        y=0.0,
        deficit=deficit,
    )


def test_two_entries_pending_at_one_count_do_not_share_a_slot() -> None:
    """The plan defers an entry with nowhere to stand and reaches past it, so
    two entries are pending at the same completed count.

    Keyed on the count alone they collided: the second target was never issued,
    the stall clock ran against an order that had never been sent, and the plan
    was declared stalled. One duel went from five extractors and an army of 25
    to none of either ([[policy-holding-ground]]).
    """
    tracker = OrderTracker(("extractorT1", "landFactory"))
    world = _world()
    first = tracker.assess(world, _build("extractorT1"), builder_moved=False, site_refused=False)
    second = tracker.assess(world, _build("landFactory"), builder_moved=False, site_refused=False)
    assert first["act"] is True
    assert second["act"] is True
    assert tracker.orders_sent == 2


def test_turning_to_another_entry_restarts_the_stall_clock() -> None:
    """Silence measured against one entry says nothing about the next.

    Without this the clock carried over, and an entry the plan had only just
    turned to inherited a window that was already nearly spent.
    """
    tracker = OrderTracker(("extractorT1", "landFactory"))
    world = _world()
    tracker.assess(world, _build("extractorT1"), builder_moved=False, site_refused=False)
    for _ in range(DEFAULT_STALL_SAMPLES - 1):
        tracker.assess(world, _build("extractorT1"), builder_moved=False, site_refused=False)

    # A different entry: freshly ordered, so it acts rather than inheriting the
    # near-exhausted clock.
    assert (
        tracker.assess(world, _build("landFactory"), builder_moved=False, site_refused=False)["act"]
        is True
    )
    # And back again, with the clock restarted rather than one sample from
    # declaring a stall.
    assert (
        tracker.assess(world, _build("extractorT1"), builder_moved=False, site_refused=False)[
            "outcome"
        ]
        == "building"
    )


def _world(*, queued: int = 0, rising: bool = False) -> Sample:
    """A world holding a builder and a finished factory, optionally mid-build."""
    roster = [entity(214, "builder"), entity(300, "landFactory", queued=queued)]
    if rising:
        roster.append(entity(301, "landFactory", complete=False))
    return sample(*roster)


def test_a_fresh_slot_is_ordered_once() -> None:
    """Re-deciding every observation would re-order what is already going up."""
    tracker = OrderTracker(_PLAN)
    assert tracker.assess(_world(), _build(), False, False)["act"] is True
    assert tracker.orders_sent == 1
    assert tracker.assess(_world(), _build(), False, False)["act"] is False
    assert tracker.orders_sent == 1


def test_a_walking_builder_holds_the_clock_open() -> None:
    """Travel is the whole of the delay, so movement is the order in flight.

    The window used to run from the moment the order was sent, which capped how
    far the bot could build: a perfectly good order to a pool 588 units away was
    declared refused while the builder was still walking to it.
    """
    tracker = OrderTracker(_PLAN, stall_samples=3)
    tracker.assess(_world(), _build(), False, False)
    for _ in range(20):
        assert tracker.assess(_world(), _build(), True, False)["outcome"] == "building"


def test_a_structure_going_up_holds_the_clock_open() -> None:
    """The builder stops moving the moment it arrives, so movement alone is not enough.

    The structure joins the roster unfinished at about the same time, which is
    what carries the evidence across the handover.
    """
    tracker = OrderTracker(_PLAN, stall_samples=3)
    tracker.assess(_world(), _build(), False, False)
    for _ in range(20):
        assert tracker.assess(_world(rising=True), _build(), False, False)["outcome"] == "building"


def test_a_quiet_build_waits_for_the_ledger_rather_than_stalling() -> None:
    """A silent refusal ends a SITE now, not the plan.

    The workforce's presumed-lost clock is what records the refusal, and the
    two clocks may disagree by an observation -- so until the ledger carries
    the site, the tracker keeps the plan alive rather than ruling on evidence
    it does not have."""
    tracker = OrderTracker(_PLAN, stall_samples=3)
    tracker.assess(_world(), _build(), False, False)
    outcomes = [tracker.assess(_world(), _build(), False, False)["outcome"] for _ in range(6)]
    assert outcomes == ["building"] * 6


def test_a_refused_site_reopens_the_slot_for_the_next_one() -> None:
    """The armyless-match fix: a refusal is the next ring slot, said loudly."""
    tracker = OrderTracker(_PLAN, stall_samples=3)
    assert tracker.assess(_world(), _build(), False, False)["act"] is True
    tracker.assess(_world(), _build(), False, False)
    tracker.assess(_world(), _build(), False, False)
    step = tracker.assess(_world(), _build(), False, True)
    assert step["act"] is False
    assert step["outcome"] == "building"
    assert "refused silently" in step["reason"]
    assert "trying the next site" in step["reason"]
    # The slot is open again: the very next order for it is cleared to go.
    assert tracker.assess(_world(), _build(), False, False)["act"] is True
    assert tracker.orders_sent == 2


def test_a_refusal_before_the_quiet_window_does_not_retry_early() -> None:
    """The ledger alone is not enough; the quiet window still has to run out,
    or a slow walk to a site that HAPPENS to sit near an old refusal would be
    re-sited mid-journey."""
    tracker = OrderTracker(_PLAN, stall_samples=3)
    tracker.assess(_world(), _build(), False, True)
    step = tracker.assess(_world(), _build(), False, True)
    assert step["outcome"] == "building"
    assert "trying the next site" not in step["reason"]


def test_the_stall_clock_restarts_on_visible_progress() -> None:
    """A builder that moves again is an order back in flight, not a slow refusal."""
    tracker = OrderTracker(_PLAN, stall_samples=3)
    tracker.assess(_world(), _build(), False, True)
    tracker.assess(_world(), _build(), False, True)
    tracker.assess(_world(), _build(), True, True)
    steps = [tracker.assess(_world(), _build(), False, True) for _ in range(3)]
    assert [step["outcome"] for step in steps] == ["building", "building", "building"]
    # The movement bought the order a fresh window: the retry lands a full
    # three quiet observations after it, not one.
    assert "trying the next site" not in steps[0]["reason"]
    assert "trying the next site" not in steps[1]["reason"]
    assert "trying the next site" in steps[2]["reason"]


def test_a_working_factory_is_not_stalled() -> None:
    """A factory never moves, so the movement test alone would condemn a busy one."""
    tracker = OrderTracker(_PLAN, stall_samples=2)
    tracker.assess(_world(), _produce(), False, False)
    for _ in range(10):
        assert tracker.assess(_world(queued=1), _produce(), False, False)["outcome"] == "building"


def test_an_idle_factory_with_nothing_queued_stalls() -> None:
    tracker = OrderTracker(_PLAN, stall_samples=2)
    tracker.assess(_world(), _produce(), False, False)
    outcomes = [tracker.assess(_world(), _produce(), False, False)["outcome"] for _ in range(2)]
    assert outcomes == ["building", "stalled"]


def test_a_producer_that_has_died_is_not_waited_on() -> None:
    """Nothing is being made, so the clock runs rather than waiting on a ghost."""
    tracker = OrderTracker(_PLAN, stall_samples=1)
    tracker.assess(_world(), _produce(unit_id=999), False, False)
    assert tracker.assess(_world(), _produce(unit_id=999), False, False)["outcome"] == "stalled"


def test_the_stall_message_names_what_was_waited_on() -> None:
    """The producer path still stalls, and its message still says why."""
    tracker = OrderTracker(_PLAN, stall_samples=1)
    tracker.assess(_world(), _produce(), False, False)
    reason = tracker.assess(_world(), _produce(), False, False)["reason"]
    assert "c_tank" in reason
    assert "refused it silently" in reason


def test_the_retry_message_names_the_refused_site() -> None:
    """The run log has to say WHERE the engine said no, or the refusals read
    as one repeated failure rather than a walk around the ring."""
    tracker = OrderTracker(_PLAN, stall_samples=1)
    tracker.assess(_world(), _build(), False, False)
    reason = tracker.assess(_world(), _build(), False, True)["reason"]
    assert "landFactory" in reason
    assert "(10, 20)" in reason


def test_a_waiting_decision_neither_acts_nor_stalls() -> None:
    """Waiting is the plan's own answer, and the match keeps being played."""
    tracker = OrderTracker(_PLAN)
    step = tracker.assess(_world(), _plain("wait", "cannot afford it yet"), False, False)
    assert step["act"] is False
    assert step["outcome"] == "building"
    assert step["reason"] == "cannot afford it yet"
    assert tracker.orders_sent == 0


def test_a_finished_plan_reports_done() -> None:
    step = OrderTracker(_PLAN).assess(_world(), _plain("done", "all satisfied"), False, False)
    assert step["act"] is False
    assert step["outcome"] == "done"


def test_an_unreachable_plan_reports_blocked() -> None:
    step = OrderTracker(_PLAN).assess(_world(), _plain("blocked", "nothing makes it"), False, False)
    assert step["act"] is False
    assert step["outcome"] == "blocked"


def test_a_stalled_decision_carries_through() -> None:
    """The build policy can reach this itself; the tracker does not second-guess it."""
    step = OrderTracker(_PLAN).assess(_world(), _plain("stalled", "already stalled"), False, False)
    assert step["outcome"] == "stalled"
    assert step["reason"] == "already stalled"


def test_progress_is_read_from_the_roster() -> None:
    """Counting from observation is what makes the plan resumable mid-match."""
    tracker = OrderTracker(_PLAN)
    assert tracker.completed(sample(entity(214, "builder"))) == 0
    assert tracker.completed(_world()) == 1


def test_the_default_window_is_the_measured_one() -> None:
    """45 observations, from timing a 609-unit build that took 52 to walk."""
    assert DEFAULT_STALL_SAMPLES == 45
    assert OrderTracker(_PLAN).stall_samples == 45


def test_a_shrinking_shortfall_is_a_save_in_progress() -> None:
    """Every new credit high-water restarts the clock: a save that is happening
    is allowed to be arbitrarily slow, because the window measures progress
    rather than duration."""
    tracker = OrderTracker(_PLAN, afford_samples=3)
    world = _world()
    for deficit in (500, 400, 300, 200, 100):
        assert tracker.assess(world, _saving(deficit), False, False)["outcome"] == "building"


def test_a_shortfall_that_never_shrinks_blocks_the_plan() -> None:
    """The amphib arm's disease: "waiting to afford" had no ceiling, and a goals
    entry priced beyond the economy's reach held the only worker hostage for
    the rest of the match -- plan-holds-only-worker reached 255 and climbing,
    twelve seeds, none ever affording it. While the plan waits it claims
    nothing, so production keeps spending the income; under attrition the save
    is not slow, it is impossible, and this clock is what says so out loud
    ([[policy-economy]])."""
    tracker = OrderTracker(_PLAN, afford_samples=3)
    world = _world()
    assert tracker.assess(world, _saving(500), False, False)["outcome"] == "building"
    outcomes = [tracker.assess(world, _saving(500), False, False)["outcome"] for _ in range(3)]
    assert outcomes == ["building", "building", "blocked"]


def test_the_blocked_ruling_lifts_when_saving_resumes() -> None:
    """Not latched, deliberately: the released worker grows the economy, and an
    economy that recovers enough to set a new high-water has earned the plan
    back."""
    tracker = OrderTracker(_PLAN, afford_samples=2)
    world = _world()
    tracker.assess(world, _saving(500), False, False)
    tracker.assess(world, _saving(500), False, False)
    assert tracker.assess(world, _saving(500), False, False)["outcome"] == "blocked"
    assert tracker.assess(world, _saving(499), False, False)["outcome"] == "building"


def test_plan_progress_restarts_the_savings_clock() -> None:
    """A shortfall measured against one entry says nothing about the next --
    the same rule the stall clock already follows across slots."""
    tracker = OrderTracker(("extractorT1", "landFactory"), afford_samples=2)
    bare = sample(entity(214, "builder"))
    tracker.assess(bare, _saving(500), False, False)
    tracker.assess(bare, _saving(500), False, False)
    # The extractor finishes; the plan turns to the factory with a fresh window.
    assert tracker.assess(_world(), _saving(500), False, False)["outcome"] == "building"


def test_a_wait_with_no_shortfall_never_trips_the_savings_clock() -> None:
    """A full ring and an occupied pool are waits the world can end on its own;
    only the price wait carries a deficit, and only a deficit is judged."""
    tracker = OrderTracker(_PLAN, afford_samples=1)
    world = _world()
    for _ in range(5):
        step = tracker.assess(world, _plain("wait", "all ring positions taken"), False, False)
        assert step["outcome"] == "building"


def test_an_order_between_waits_resets_the_savings_clock() -> None:
    """Credits arrived and were spent on the entry: whatever shortfall follows
    belongs to the next save, not the finished one."""
    tracker = OrderTracker(_PLAN, afford_samples=2)
    world = _world()
    tracker.assess(world, _saving(500), False, False)
    tracker.assess(world, _saving(500), False, False)
    tracker.assess(world, _build(), False, False)
    outcomes = [tracker.assess(world, _saving(500), False, False)["outcome"] for _ in range(2)]
    assert outcomes == ["building", "building"]


def test_the_blocked_reason_names_the_shortfall_and_the_release() -> None:
    tracker = OrderTracker(_PLAN, afford_samples=1)
    tracker.assess(_world(), _saving(500), False, False)
    reason = tracker.assess(_world(), _saving(500), False, False)["reason"]
    assert "500" in reason
    assert "worker is released" in reason


def test_the_afford_window_is_twice_the_refusal_window() -> None:
    """Savings are slower than orders: a genuine save sets a new high-water
    whenever anything is banked at all, so ninety samples without one is not
    slowness, it is income entirely spoken for."""
    assert AFFORD_STALL_SAMPLES == 90
    assert OrderTracker(_PLAN).afford_samples == 90
