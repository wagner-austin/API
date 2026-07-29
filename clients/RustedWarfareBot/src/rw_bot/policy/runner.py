"""Whether an order already given is still being carried out.

This module used to own a loop. It ran the build plan to completion and handed
over to a second loop that fought, and the split was the bot's largest
structural defect: while building there was no army and no economy, and once
fighting there was no build policy at all, so the factory count was frozen for
the rest of the match and a plan that stalled meant a match that never fought
([[policy-loop]]).

The loop moved to :mod:`rw_bot.policy.campaign`, which runs one tick for the
whole match. What stayed is the part that was never about looping: an order
takes time to take effect, the world reports no acknowledgement, and something
has to decide whether silence means "in progress" or "refused". That is
:class:`OrderTracker`, and it is the only place that judgement is made.

Pure in the sense that matters: it opens nothing and reads no clock. It counts
observations, which is a different thing from measuring time and is what makes
the window independent of price, distance and map size alike.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

from rw_bot.policy.build_order import Decision, completed_count
from rw_bot.policy.observation import is_rising
from rw_bot.wire.state import Sample

#: Observations a *stationary* builder may persist without progress before the
#: order is presumed refused.
#:
#: Stationary is the important word, and it was not always there. The window
#: used to run from the moment an order was sent, which quietly capped how far
#: the bot could build: at a measured 11.7 world units per sample, 45 samples
#: reach 527 units, and a perfectly good order to a resource pool 588 units away
#: was declared refused while the builder was still walking to it. It completed
#: seconds after the run gave up.
#:
#: Timing one far build settled the shape of the fix. Ordering an extractor 609
#: units away, the builder travelled for 52 samples and the structure appeared
#: on the very sample it stopped moving -- construction itself cost nothing
#: measurable. So travel is the whole of the delay, and a builder that is still
#: moving is an order still in flight. The clock therefore only runs while the
#: builder stands still, which needs no speed constant, no frame rate, and no
#: assumption about map size ([[mechanics-resource-pools]]).
DEFAULT_STALL_SAMPLES = 45


class BuildStep(TypedDict):
    """What the build plan wants done on this observation.

    Attributes:
        act: Whether to issue the decision's order now. False when the plan is
            finished, cannot proceed, or is waiting on an order already given.
        outcome: ``"building"`` while the plan is live, or one of ``"done"``,
            ``"blocked"`` and ``"stalled"`` once it is not.
        reason: Human-readable justification, for the run log.
    """

    act: bool
    outcome: Literal["building", "done", "blocked", "stalled"]
    reason: str


class OrderTracker:
    """Remembers which plan slots have been ordered, and whether they moved.

    An order is issued at most once per plan slot. Structures take time to
    appear and the roster is what reports them, so re-deciding every observation
    would re-order the same structure repeatedly while the first was still going
    up -- spending credits the planner believes it still has.

    That protection has a cost, and the stall window pays it back. An order the
    engine refuses produces no roster change and no error the planner can see,
    so a once-only order would leave the plan reporting "building X" forever
    while nothing happened. Observed for real: a builder cannot construct a
    laboratory, and the engine says so only in its own log.

    Attributes:
        plan: What to make, in order.
        stall_samples: Observations of no visible progress before the plan is
            called stalled.
        orders_sent: Orders this tracker has cleared for dispatch.
    """

    def __init__(self, plan: Sequence[str], stall_samples: int = DEFAULT_STALL_SAMPLES) -> None:
        """Open a tracker for one plan.

        Args:
            plan: What to make, in order.
            stall_samples: Observations of no visible progress before the plan
                is called stalled.
        """
        self.plan = tuple(plan)
        self.stall_samples = stall_samples
        self.orders_sent = 0
        self._ordered: set[tuple[int, str]] = set()
        self._watching: tuple[int, str] | None = None
        self._quiet = 0

    def assess(self, sample: Sample, decision: Decision, builder_moved: bool) -> BuildStep:
        """Judge one observation against the order already outstanding.

        Args:
            sample: One observation of the world.
            decision: What the build policy wants, for this observation.
            builder_moved: Whether the builder moved since the previous
                observation, which is what walking to a site looks like.

        Returns:
            Whether to act, and how the plan stands.
        """
        if decision["action"] == "done":
            return BuildStep(act=False, outcome="done", reason=decision["reason"])
        if decision["action"] == "blocked":
            return BuildStep(act=False, outcome="blocked", reason=decision["reason"])
        if decision["action"] == "stalled":
            return BuildStep(act=False, outcome="stalled", reason=decision["reason"])
        if decision["action"] == "wait":
            return BuildStep(act=False, outcome="building", reason=decision["reason"])

        # **Keyed by what is being built, not by the count alone.** The plan
        # defers an entry with nowhere to stand and reaches past it, so two
        # entries are pending at the same completed count -- an extractor with
        # no free pool and the factory after it. Keyed on the count by itself
        # they collided on one slot: the second target was never issued, the
        # clock below ran against an order that had never been sent, and the
        # plan was declared stalled. One duel went from five extractors and an
        # army of 25 to none of either ([[policy-holding-ground]]).
        slot = (completed_count(sample, self.plan), decision["type_name"])
        if slot not in self._ordered:
            self._ordered.add(slot)
            self._watching = slot
            self._quiet = 0
            self.orders_sent += 1
            return BuildStep(act=True, outcome="building", reason=decision["reason"])

        # The clock follows the plan. When it turns to a different entry, the
        # silence measured against the previous one says nothing about this one.
        if slot != self._watching:
            self._watching = slot
            self._quiet = 0

        # One rule for both verbs: the clock only runs while nothing observable
        # is happening. What counts as observable differs -- a builder walking to
        # its site, a building holding something in its queue -- but neither is a
        # guess about how long the work should take, which is what makes the
        # window independent of price, distance and map size alike.
        if _in_flight(sample, decision, builder_moved):
            self._quiet = 0
            return BuildStep(act=False, outcome="building", reason=decision["reason"])
        self._quiet += 1
        if self._quiet < self.stall_samples:
            return BuildStep(act=False, outcome="building", reason=decision["reason"])
        return BuildStep(
            act=False,
            outcome="stalled",
            reason=(
                f"{decision['type_name']} was ordered but never appeared after "
                f"{self.stall_samples} samples with the builder standing still; "
                "the engine refused it silently"
            ),
        )

    def completed(self, sample: Sample) -> int:
        """Return how many plan slots the world already shows finished.

        Args:
            sample: One observation of the world.

        Returns:
            Satisfied plan entries.
        """
        return completed_count(sample, self.plan)


def _in_flight(sample: Sample, decision: Decision, builder_moved: bool) -> bool:
    """Report whether the pending order is visibly still being carried out.

    Each verb has its own evidence, and both are things the world shows rather
    than deadlines the planner invents.

    A placed build is in flight while the builder is walking to the site, and
    then while the structure itself is going up. Both halves are needed: the
    builder stops moving the moment it arrives, and the structure joins the
    roster unfinished at about the same time, so movement alone stops being
    evidence exactly when construction starts.

    Production is in flight while the producing building holds the order in its
    queue. That is read straight off the entity, and it is what makes the rule
    uniform: a factory never moves, so the movement test alone would call a
    working factory stalled.

    The queue is the second thing tried here and the first that works. Elapsed
    time capped what the bot could afford, since production time scales with
    price. Watching credits fall does not work either -- measured through one
    production run the balance read 4243, 3678, 3813, 3849, *rising* through
    most of it as income outpaced the drain ([[policy-loop]]).

    Args:
        sample: The current observation.
        decision: The pending decision.
        builder_moved: Whether the builder moved since the previous sample.

    Returns:
        True while there is evidence the order is still being carried out.
    """
    if decision["action"] == "build":
        return builder_moved or is_rising(sample, decision["type_name"])
    for entity in sample["entities"]:
        if entity["unit_id"] == decision["unit_id"]:
            return entity["queued"] > 0
    # The producer is gone -- destroyed, or never in the roster. Nothing is
    # being made, so the clock runs rather than waiting on a building that no
    # longer exists.
    return False


__all__ = ["DEFAULT_STALL_SAMPLES", "BuildStep", "OrderTracker"]
