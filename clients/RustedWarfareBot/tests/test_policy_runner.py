"""Deciding whether an order already given is still being carried out.

The loop these rules used to live in is gone; one tick runs the whole match now
([[policy-loop]]). What is left is the judgement that was never about looping:
the world reports no acknowledgement of an order, so something has to tell
"in progress" from "silently refused", and this is the only place that happens.
"""

from __future__ import annotations

from typing import Literal

from rw_bot.policy.build_order import Decision
from rw_bot.policy.runner import DEFAULT_STALL_SAMPLES, OrderTracker
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
    )


def _plain(action: Literal["wait", "done", "blocked", "stalled"], reason: str) -> Decision:
    """A decision carrying no order.

    The verb is typed as the literal union the production type declares rather
    than as ``str``, so a test cannot construct a decision the policy could not.
    """
    return Decision(action=action, reason=reason, type_name="", unit_id=0, x=0.0, y=0.0)


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
    first = tracker.assess(world, _build("extractorT1"), builder_moved=False)
    second = tracker.assess(world, _build("landFactory"), builder_moved=False)
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
    tracker.assess(world, _build("extractorT1"), builder_moved=False)
    for _ in range(DEFAULT_STALL_SAMPLES - 1):
        tracker.assess(world, _build("extractorT1"), builder_moved=False)

    # A different entry: freshly ordered, so it acts rather than inheriting the
    # near-exhausted clock.
    assert tracker.assess(world, _build("landFactory"), builder_moved=False)["act"] is True
    # And back again, with the clock restarted rather than one sample from
    # declaring a stall.
    assert (
        tracker.assess(world, _build("extractorT1"), builder_moved=False)["outcome"] == "building"
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
    assert tracker.assess(_world(), _build(), False)["act"] is True
    assert tracker.orders_sent == 1
    assert tracker.assess(_world(), _build(), False)["act"] is False
    assert tracker.orders_sent == 1


def test_a_walking_builder_holds_the_clock_open() -> None:
    """Travel is the whole of the delay, so movement is the order in flight.

    The window used to run from the moment the order was sent, which capped how
    far the bot could build: a perfectly good order to a pool 588 units away was
    declared refused while the builder was still walking to it.
    """
    tracker = OrderTracker(_PLAN, stall_samples=3)
    tracker.assess(_world(), _build(), False)
    for _ in range(20):
        assert tracker.assess(_world(), _build(), True)["outcome"] == "building"


def test_a_structure_going_up_holds_the_clock_open() -> None:
    """The builder stops moving the moment it arrives, so movement alone is not enough.

    The structure joins the roster unfinished at about the same time, which is
    what carries the evidence across the handover.
    """
    tracker = OrderTracker(_PLAN, stall_samples=3)
    tracker.assess(_world(), _build(), False)
    for _ in range(20):
        assert tracker.assess(_world(rising=True), _build(), False)["outcome"] == "building"


def test_a_stationary_builder_making_no_progress_stalls() -> None:
    """The engine refuses some orders silently: a builder cannot make a laboratory."""
    tracker = OrderTracker(_PLAN, stall_samples=3)
    tracker.assess(_world(), _build(), False)
    outcomes = [tracker.assess(_world(), _build(), False)["outcome"] for _ in range(3)]
    assert outcomes == ["building", "building", "stalled"]


def test_the_stall_clock_restarts_on_visible_progress() -> None:
    """A builder that moves again is an order back in flight, not a slow refusal."""
    tracker = OrderTracker(_PLAN, stall_samples=3)
    tracker.assess(_world(), _build(), False)
    tracker.assess(_world(), _build(), False)
    tracker.assess(_world(), _build(), True)
    outcomes = [tracker.assess(_world(), _build(), False)["outcome"] for _ in range(3)]
    assert outcomes == ["building", "building", "stalled"]


def test_a_working_factory_is_not_stalled() -> None:
    """A factory never moves, so the movement test alone would condemn a busy one."""
    tracker = OrderTracker(_PLAN, stall_samples=2)
    tracker.assess(_world(), _produce(), False)
    for _ in range(10):
        assert tracker.assess(_world(queued=1), _produce(), False)["outcome"] == "building"


def test_an_idle_factory_with_nothing_queued_stalls() -> None:
    tracker = OrderTracker(_PLAN, stall_samples=2)
    tracker.assess(_world(), _produce(), False)
    outcomes = [tracker.assess(_world(), _produce(), False)["outcome"] for _ in range(2)]
    assert outcomes == ["building", "stalled"]


def test_a_producer_that_has_died_is_not_waited_on() -> None:
    """Nothing is being made, so the clock runs rather than waiting on a ghost."""
    tracker = OrderTracker(_PLAN, stall_samples=1)
    tracker.assess(_world(), _produce(unit_id=999), False)
    assert tracker.assess(_world(), _produce(unit_id=999), False)["outcome"] == "stalled"


def test_the_stall_message_names_what_was_waited_on() -> None:
    tracker = OrderTracker(_PLAN, stall_samples=1)
    tracker.assess(_world(), _build(), False)
    reason = tracker.assess(_world(), _build(), False)["reason"]
    assert "landFactory" in reason
    assert "refused it silently" in reason


def test_a_waiting_decision_neither_acts_nor_stalls() -> None:
    """Waiting is the plan's own answer, and the match keeps being played."""
    tracker = OrderTracker(_PLAN)
    step = tracker.assess(_world(), _plain("wait", "cannot afford it yet"), False)
    assert step["act"] is False
    assert step["outcome"] == "building"
    assert step["reason"] == "cannot afford it yet"
    assert tracker.orders_sent == 0


def test_a_finished_plan_reports_done() -> None:
    step = OrderTracker(_PLAN).assess(_world(), _plain("done", "all satisfied"), False)
    assert step["act"] is False
    assert step["outcome"] == "done"


def test_an_unreachable_plan_reports_blocked() -> None:
    step = OrderTracker(_PLAN).assess(_world(), _plain("blocked", "nothing makes it"), False)
    assert step["act"] is False
    assert step["outcome"] == "blocked"


def test_a_stalled_decision_carries_through() -> None:
    """The build policy can reach this itself; the tracker does not second-guess it."""
    step = OrderTracker(_PLAN).assess(_world(), _plain("stalled", "already stalled"), False)
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
