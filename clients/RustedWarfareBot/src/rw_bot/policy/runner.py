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
from rw_bot.policy.siting import is_refused
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

#: Observations a price wait may persist without the shortfall shrinking before
#: the plan is ruled blocked.
#:
#: A different clock from the refusal window, because it judges a different
#: silence. A refused order produces nothing forever; a save produces a new
#: credit high-water every few samples for as long as anything is being banked,
#: however slowly -- so the window measures *progress*, not duration, and a
#: genuinely slow save never trips it. Twice the refusal window is generous:
#: ninety samples without one new high-water is not slowness, it is income
#: entirely spoken for.
#:
#: Why it exists at all: the plan claims nothing while it waits
#: (:func:`~rw_bot.policy.spending.build_plan` claims only when acting), so
#: production and expansion keep spending the income every tick. Under army
#: attrition an expensive entry is then not slow, it is *impossible*, and the
#: amphib arm measured what that costs -- an 11,000-credit goal against a
#: 4,000-credit start held the only worker hostage for whole matches:
#: ``plan-holds-only-worker reached 255`` and climbing, twelve seeds, zero of
#: them ever affording it (log: 2026-07-29, the amphib arm). Escrowing the
#: claim instead would starve replacement, which the reserve regression
#: already priced ([[policy-economy]]); saying "this save is not happening"
#: out loud and releasing the worker is the honest fix.
AFFORD_STALL_SAMPLES = 90


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
        afford_samples: Observations a price wait may persist without its
            shortfall shrinking before the plan is ruled blocked.
        orders_sent: Orders this tracker has cleared for dispatch.
    """

    def __init__(
        self,
        plan: Sequence[str],
        stall_samples: int = DEFAULT_STALL_SAMPLES,
        afford_samples: int = AFFORD_STALL_SAMPLES,
    ) -> None:
        """Open a tracker for one plan.

        Args:
            plan: What to make, in order.
            stall_samples: Observations of no visible progress before the plan
                is called stalled.
            afford_samples: Observations a price wait may persist without its
                shortfall shrinking before the plan is ruled blocked.
        """
        self.plan = tuple(plan)
        self.stall_samples = stall_samples
        self.afford_samples = afford_samples
        self.orders_sent = 0
        self._ordered: set[tuple[int, str]] = set()
        #: Where each cleared BUILD order was aimed, by plan slot. Kept
        #: because the refusal question is about the site that was ORDERED:
        #: once the ledger carries it, every fresh decision already offers
        #: the next position, so testing the current decision's site against
        #: the ledger is a gate nothing can ever pass -- measured as a match
        #: that held its only worker for 1314 samples "building" an order
        #: the workforce had long written off.
        self._sites: dict[tuple[int, str], tuple[float, float]] = {}
        self._watching: tuple[int, str] | None = None
        self._quiet = 0
        self._deficit_floor: int | None = None
        self._deficit_quiet = 0
        self._saving_at: int | None = None

    def assess(
        self,
        sample: Sample,
        decision: Decision,
        builder_moved: bool,
        refused: Sequence[tuple[float, float]],
    ) -> BuildStep:
        """Judge one observation against the order already outstanding.

        Args:
            sample: One observation of the world.
            decision: What the build policy wants, for this observation.
            builder_moved: Whether the builder moved since the previous
                observation, which is what walking to a site looks like.
            refused: The workforce's refusal ledger. Judged against the site
                this tracker ORDERED, not the decision's current one: once
                the ledger carries the ordered site, every fresh decision
                already offers the next position, so a gate on the current
                site is one nothing can ever pass -- measured as a match
                holding its only worker for 1314 samples on an order the
                workforce had long written off. Reopening BEFORE the ledger
                carries the ordered site would be the opposite failure: the
                chooser would offer the same doomed position again.

        Returns:
            Whether to act, and how the plan stands.
        """
        told = self._verdict_without_an_order(sample, decision)
        if told is not None:
            return told
        # Any actual order means the last save either completed or was
        # abandoned by the plan itself; whatever shortfall follows belongs to
        # the next save, not the finished one.
        self._reset_saving()

        # **Keyed by what is being built, not by the count alone.** The plan
        # defers an entry with nowhere to stand and reaches past it, so two
        # entries are pending at the same completed count -- an extractor with
        # no free pool and the factory after it. Keyed on the count by itself
        # they collided on one slot: the second target was never issued, the
        # clock below ran against an order that had never been sent, and the
        # plan was declared stalled. One duel went from five extractors and an
        # army of 25 to none of either ([[policy-holding-ground]]).
        slot = (completed_count(sample, self.plan), decision["type_name"])
        reopened = self._reopen_if_refused(slot, decision, refused)
        if reopened is not None:
            return reopened
        if slot not in self._ordered:
            self._ordered.add(slot)
            if decision["action"] == "build":
                self._sites[slot] = (decision["x"], decision["y"])
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
        return self._quiet_verdict(decision)

    def _verdict_without_an_order(self, sample: Sample, decision: Decision) -> BuildStep | None:
        """Translate the decisions that carry their own verdict.

        Done, blocked, stalled and wait all resolve without an order leaving,
        so none of them touch the slot tracking below -- except the saving
        clock, which a credit wait runs and everything else resets.

        Args:
            sample: One observation of the world.
            decision: What the build policy wants, for this observation.

        Returns:
            The decision's own verdict, or ``None`` when it orders something
            and the slot tracking must judge it.
        """
        if decision["action"] == "done":
            return BuildStep(act=False, outcome="done", reason=decision["reason"])
        if decision["action"] == "blocked":
            return BuildStep(act=False, outcome="blocked", reason=decision["reason"])
        if decision["action"] == "stalled":
            return BuildStep(act=False, outcome="stalled", reason=decision["reason"])
        if decision["action"] == "wait":
            if decision["deficit"] > 0:
                return self._assess_saving(sample, decision)
            self._reset_saving()
            return BuildStep(act=False, outcome="building", reason=decision["reason"])
        return None

    def _reopen_if_refused(
        self,
        slot: tuple[int, str],
        decision: Decision,
        refused: Sequence[tuple[float, float]],
    ) -> BuildStep | None:
        """Reopen an ordered slot the tick the ledger carries its site.

        The agent's build watch reports the engine dropping an order's
        waypoint one sample after it happens, so waiting out the quiet window
        on a verdict that is already in means spending 45 samples confirming
        what is written down. Judged against the site this tracker ORDERED --
        the current decision has already moved past a ledgered site (wiki log
        2026-09-01, the detection design).

        Args:
            slot: The plan entry under judgement.
            decision: What the build policy wants, for this observation.
            refused: The workforce's refusal ledger.

        Returns:
            The reopened step, or ``None`` when the ledger says nothing about
            this slot's ordered site.
        """
        ordered_site = self._sites.get(slot)
        if slot not in self._ordered or ordered_site is None:
            return None
        if not is_refused(ordered_site, refused):
            return None
        self._ordered.discard(slot)
        del self._sites[slot]
        self._watching = None
        self._quiet = 0
        return BuildStep(
            act=False,
            outcome="building",
            reason=(
                f"{decision['type_name']} at "
                f"({ordered_site[0]:.0f}, {ordered_site[1]:.0f}) was refused by the "
                "engine; trying the next site"
            ),
        )

    def _quiet_verdict(self, decision: Decision) -> BuildStep:
        """Rule on an order whose quiet clock just advanced.

        Args:
            decision: What the build policy wants, for this observation.

        Returns:
            Whether to act, and how the plan stands.
        """
        if self._quiet < self.stall_samples:
            return BuildStep(act=False, outcome="building", reason=decision["reason"])
        # **A quiet BUILD keeps waiting; a quiet PRODUCE stalls.** A build's
        # refusal now arrives explicitly -- the agent's build watch reports
        # the engine dropping the waypoint, the ledger carries the site, and
        # the gate in :meth:`assess` reopens the slot the tick it lands. So
        # a build that is merely quiet has no verdict yet, and ruling on it
        # here would be the 45-sample guess the watch replaced; the
        # presumed-lost clock in the workforce remains the fallback that
        # writes the ledger when the watch cannot see (wiki log 2026-09-01,
        # the detection design). Produced units have no site and no watch,
        # so for them the quiet clock still means what it always did.
        if decision["action"] == "build":
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

    def _reset_saving(self) -> None:
        """Forget the savings clock; the plan is not waiting on a price."""
        self._deficit_floor = None
        self._deficit_quiet = 0
        self._saving_at = None

    def _assess_saving(self, sample: Sample, decision: Decision) -> BuildStep:
        """Judge whether a save the plan is waiting on is actually happening.

        The clock measures progress, not duration: every new credit high-water
        (a shortfall smaller than any seen for this entry) restarts it, so a
        slow save is allowed to be arbitrarily slow. What it refuses to allow
        is a shortfall that never shrinks at all -- income entirely spoken for
        by production and expansion, which the plan cannot see and used to
        wait on forever, worker in hand (see :data:`AFFORD_STALL_SAMPLES` for
        the measured cost).

        **Blocked, not latched.** The worker a blocked ruling releases goes to
        the economy, and an economy that recovers enough to set a new
        high-water lifts the ruling on its own -- the plan resumes and takes
        its worker back. Release-while-stuck, reclaim-when-converging.

        Args:
            sample: One observation of the world.
            decision: The price wait, carrying its shortfall.

        Returns:
            Building while the save progresses or the window is open, blocked
            once the shortfall has sat still for the whole window.
        """
        count = completed_count(sample, self.plan)
        if self._saving_at != count:
            # A different entry is saving now; its shortfall starts fresh.
            self._saving_at = count
            self._deficit_floor = None
            self._deficit_quiet = 0
        gap = decision["deficit"]
        if self._deficit_floor is None or gap < self._deficit_floor:
            self._deficit_floor = gap
            self._deficit_quiet = 0
            return BuildStep(act=False, outcome="building", reason=decision["reason"])
        self._deficit_quiet += 1
        if self._deficit_quiet < self.afford_samples:
            return BuildStep(act=False, outcome="building", reason=decision["reason"])
        return BuildStep(
            act=False,
            outcome="blocked",
            reason=(
                f"{decision['reason']}; the shortfall never shrank below "
                f"{self._deficit_floor} across {self.afford_samples} samples -- "
                "income is spoken for and this save is not happening; "
                "the worker is released"
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


__all__ = ["AFFORD_STALL_SAMPLES", "DEFAULT_STALL_SAMPLES", "BuildStep", "OrderTracker"]
