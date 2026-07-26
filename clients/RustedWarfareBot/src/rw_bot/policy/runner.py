"""Run a policy against a live game until it finishes or stalls.

The loop is deliberately thin: read a sample, ask the policy, act, repeat. All
judgement lives in :mod:`rw_bot.policy.build_order`, which is pure, so what is
tested here is the loop's own behaviour — when it stops, what it counts, and
that it does not re-order something already in flight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.build_order import Decision, completed_count, decide, find_builder
from rw_bot.wire.command import build_order, produce_order
from rw_bot.wire.state import Sample

#: Samples a *stationary* builder may persist without progress before the run
#: is called stalled.
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
#: assumption about map size (wiki: mechanics-resource-pools).
DEFAULT_STALL_SAMPLES = 45

#: World-unit displacement between samples below which a builder counts as
#: stationary. A parked unit reports byte-identical coordinates, so this only
#: has to survive float noise rather than distinguish slow movement.
_MOVEMENT_EPSILON = 0.5


class Scorecard(TypedDict):
    """What a run achieved, for judging one policy against another.

    Attributes:
        completed: Structures from the plan standing at the end.
        planned: Structures the plan asked for.
        orders_sent: Build orders issued. Higher than ``completed`` means
            orders were wasted — re-issued, rejected, or lost.
        samples_seen: World samples read.
        frames_elapsed: Engine frames between the first and last sample.
        credits_at_end: Credits held at the last sample.
        outcome: Why the run stopped: ``"done"``, ``"blocked"``,
            ``"stalled"``, or ``"sample_limit"``.
        last_reason: The policy's own words for its final decision.
    """

    completed: int
    planned: int
    orders_sent: int
    samples_seen: int
    frames_elapsed: int
    credits_at_end: int
    outcome: str
    last_reason: str


def run(
    channel: AgentChannel,
    plan: Sequence[str],
    catalogue: Mapping[str, UnitStats],
    placements: Mapping[str, TypePlacement],
    reaches: Mapping[str, float],
    max_samples: int,
    stall_samples: int = DEFAULT_STALL_SAMPLES,
) -> Scorecard:
    """Play the plan until it completes, stalls, or the sample budget runs out.

    An order is issued at most once per plan position. Structures take time to
    appear, and the roster is what reports them, so re-deciding every sample
    would re-order the same structure repeatedly while the first one was still
    being built — spending credits the policy believes it still has.

    That protection has a cost, and ``stall_samples`` pays it back. An order the
    engine refuses produces no roster change and no error the planner can see,
    so a once-only order would leave the run reporting "building X" forever
    while nothing happened. Observed for real: a builder cannot construct a
    laboratory, and the engine says so only in its own log. After
    ``stall_samples`` observations of a stationary builder making no progress,
    the run stops and says it stalled.

    The stall clock restarts on every sample in which the ordered builder has
    moved, because walking to the site is the order being carried out rather
    than the order failing (see :data:`DEFAULT_STALL_SAMPLES`). The run-level
    ``max_samples`` is what bounds a builder that somehow never settles.

    Args:
        channel: An open connection to the agent.
        plan: What to make, in order. Entries may be structures or units.
        catalogue: Unit stats by type name, for prices.
        placements: Placement rules by type name, for where each may stand.
        reaches: Attack range by type name, for the threat filter.
        max_samples: Stop after this many samples regardless of progress.
        stall_samples: Observations of a stationary builder without progress
            before the run is called stalled.

    Returns:
        The scorecard.

    Raises:
        ChannelError: When the agent closes the connection mid-run.
        OSError: When the connection fails.
    """
    orders_sent = 0
    samples_seen = 0
    first_frame = 0
    ordered_positions: set[int] = set()
    ordered_at = 0
    builder_was: tuple[float, float] | None = None
    sample: Sample | None = None
    decision: Decision | None = None

    while samples_seen < max_samples:
        sample = channel.next_sample()
        if samples_seen == 0:
            first_frame = sample["frame"]
        samples_seen += 1

        # Every exit from the body acknowledges the sample, including the ones
        # that break out. In lockstep the agent holds the simulation until this
        # arrives, so a path that skipped it would stall the game rather than
        # merely skip a message ([[policy-determinism]]).
        try:
            # Read before deciding, and unconditionally. The builder's travel is
            # what the stall clock measures, so it has to be sampled on every
            # observation rather than only on the ones that reach an order.
            builder = find_builder(sample)
            builder_now = None if builder is None else (builder["x"], builder["y"])
            moved = _has_moved(builder_was, builder_now)
            builder_was = builder_now

            decision = decide(sample, plan, catalogue, placements, reaches)
            if decision["action"] in ("done", "blocked", "stalled"):
                break
            if decision["action"] == "wait":
                continue

            position = completed_count(sample, plan)
            if position in ordered_positions:
                # One rule for both verbs: the clock only runs while nothing
                # observable is happening. What counts as observable differs --
                # a builder walking to its site, a building holding something in
                # its queue -- but neither is a guess about how long the work
                # should take, which is what makes the window independent of price,
                # distance and map size alike.
                if _in_flight(sample, decision, moved):
                    ordered_at = samples_seen
                elif samples_seen - ordered_at >= stall_samples:
                    decision = _stalled(decision, stall_samples)
                    break
                continue
            ordered_at = samples_seen
            ordered_positions.add(position)
            if decision["action"] == "produce":
                channel.send_produce(
                    produce_order(
                        unit_id=decision["unit_id"],
                        type_name=decision["type_name"],
                    )
                )
            else:
                channel.send_build(
                    build_order(
                        unit_id=decision["unit_id"],
                        type_name=decision["type_name"],
                        x=decision["x"],
                        y=decision["y"],
                    )
                )
            orders_sent += 1
        finally:
            channel.send_ack()

    return _score(sample, decision, plan, orders_sent, samples_seen, first_frame)


def _in_flight(sample: Sample, decision: Decision, moved: bool) -> bool:
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
    most of it as income outpaced the drain (wiki: policy-loop).

    Args:
        sample: The current observation.
        decision: The pending decision.
        moved: Whether the builder moved since the previous sample.

    Returns:
        True while there is evidence the order is still being carried out.
    """
    if decision["action"] == "build":
        return moved or _rising(sample, decision["type_name"])
    for entity in sample["entities"]:
        if entity["unit_id"] == decision["unit_id"]:
            return entity["queued"] > 0
    # The producer is gone -- destroyed, or never in the roster. Nothing is
    # being made, so the clock runs and the run stops rather than waiting on a
    # building that no longer exists.
    return False


def _rising(sample: Sample, type_name: str) -> bool:
    """Report whether an unfinished structure of this type is going up.

    Ownership is checked, or an opponent's half-built factory in view would
    keep this run's clock alive indefinitely.

    Args:
        sample: The current observation.
        type_name: The type that was ordered.

    Returns:
        True when the player owns an unfinished entity of that type.
    """
    for entity in sample["entities"]:
        if entity["mine"] and not entity["complete"] and entity["type_name"] == type_name:
            return True
    return False


def _has_moved(before: tuple[float, float] | None, after: tuple[float, float] | None) -> bool:
    """Report whether the builder moved between two samples.

    A builder that has died, or that was not in the roster to begin with, has
    not moved. Treating a missing builder as movement would keep the stall clock
    permanently reset and turn a lost builder into an infinite wait.

    Args:
        before: Position at the previous sample, if it was there.
        after: Position now, if it is there.

    Returns:
        True when both positions are known and differ by more than float noise.
    """
    if before is None or after is None:
        return False
    return abs(after[0] - before[0]) + abs(after[1] - before[1]) > _MOVEMENT_EPSILON


def _stalled(decision: Decision | None, stall_samples: int) -> Decision:
    """Convert the pending decision into a stalled one.

    Args:
        decision: The decision that was pending when progress stopped.
        stall_samples: How many samples passed without progress.

    Returns:
        A decision reporting the stall, naming what was being waited on.
    """
    waiting_on = decision["type_name"] if decision is not None else "an order"
    return Decision(
        action="stalled",
        reason=(
            f"{waiting_on} was ordered but never appeared after {stall_samples}"
            " samples with the builder standing still; the engine refused it"
            " silently"
        ),
        type_name="",
        unit_id=0,
        x=0.0,
        y=0.0,
    )


def _score(
    sample: Sample | None,
    decision: Decision | None,
    plan: Sequence[str],
    orders_sent: int,
    samples_seen: int,
    first_frame: int,
) -> Scorecard:
    """Assemble the scorecard from the final observation.

    Args:
        sample: The last sample read, or None when none arrived.
        decision: The last decision made, or None when none was made.
        plan: Structures the plan asked for.
        orders_sent: Build orders issued.
        samples_seen: World samples read.
        first_frame: Frame of the first sample.

    Returns:
        The scorecard.
    """
    if sample is None or decision is None:
        return Scorecard(
            completed=0,
            planned=len(plan),
            orders_sent=0,
            samples_seen=0,
            frames_elapsed=0,
            credits_at_end=0,
            outcome="sample_limit",
            last_reason="no sample was read",
        )
    outcome = (
        decision["action"]
        if decision["action"] in ("done", "blocked", "stalled")
        else "sample_limit"
    )
    return Scorecard(
        completed=completed_count(sample, plan),
        planned=len(plan),
        orders_sent=orders_sent,
        samples_seen=samples_seen,
        frames_elapsed=sample["frame"] - first_frame,
        credits_at_end=sample["credits"],
        outcome=outcome,
        last_reason=decision["reason"],
    )


def format_scorecard(card: Scorecard) -> tuple[str, ...]:
    """Render a scorecard as report lines.

    Args:
        card: The scorecard.

    Returns:
        One line per figure.
    """
    return (
        f"outcome        {card['outcome']} ({card['last_reason']})",
        f"completed      {card['completed']}/{card['planned']}",
        f"orders sent    {card['orders_sent']}",
        f"samples seen   {card['samples_seen']}",
        f"frames elapsed {card['frames_elapsed']}",
        f"credits left   {card['credits_at_end']}",
    )


__all__ = ["Scorecard", "format_scorecard", "run"]
