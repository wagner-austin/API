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
from rw_bot.policy.build_order import Decision, completed_count, decide
from rw_bot.wire.command import build_order
from rw_bot.wire.state import Sample

#: Samples an unchanged plan position may persist before the run is called
#: stalled. Generous enough that a slow structure is not mistaken for a refused
#: one: the laboratory that exposed this took longer than any completed build.
DEFAULT_STALL_SAMPLES = 45


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
    ``stall_samples`` observations with no progress, the run stops and says it
    stalled.

    Args:
        channel: An open connection to the agent.
        plan: Structures to build, in order.
        catalogue: Unit stats by type name, for prices.
        max_samples: Stop after this many samples regardless of progress.

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
    sample: Sample | None = None
    decision: Decision | None = None

    while samples_seen < max_samples:
        sample = channel.next_sample()
        if samples_seen == 0:
            first_frame = sample["frame"]
        samples_seen += 1

        decision = decide(sample, plan, catalogue)
        if decision["action"] in ("done", "blocked", "stalled"):
            break
        if decision["action"] == "wait":
            continue

        position = completed_count(sample, plan)
        if position in ordered_positions:
            if samples_seen - ordered_at >= stall_samples:
                decision = _stalled(decision, stall_samples)
                break
            continue
        ordered_at = samples_seen
        ordered_positions.add(position)
        channel.send_build(
            build_order(
                unit_id=decision["unit_id"],
                type_name=decision["type_name"],
                x=decision["x"],
                y=decision["y"],
            )
        )
        orders_sent += 1

    return _score(sample, decision, plan, orders_sent, samples_seen, first_frame)


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
            " samples; the engine refused it silently"
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
