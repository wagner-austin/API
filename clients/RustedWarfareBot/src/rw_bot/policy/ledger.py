"""What the match asked to buy, and what became of each request.

**The bot has always known this and always thrown it away.**
:meth:`~rw_bot.policy.budget.Budget.claim` records every request with its
purpose, its amount, whether it was granted and a sentence saying why not --
"expand:extractorT1 wanted 700 of 305 available past a 0 reserve; 1400 already
committed this tick". :func:`~rw_bot.policy.budget.format_ledger` renders it.
Both were written, tested and exported, and nothing outside the tests ever
called either: the loop reduced a whole tick of reasoning to
``sum(1 for claim in ledger if not claim["granted"])``. At roughly one refusal
per sample across four thousand samples, that is a count of about four thousand
sentences nobody kept.

The cost was paid twice in one session. An experiment that aimed defence at the
extractors read as a clean refutation -- until a *different* new report line
showed three turrets built across twelve full matches, which meant the arm had
never tested the policy, only the two or three ticks a match where it fired.
And a wipe read as a lost fight until the trace showed an army of zero beside
37,225 banked credits, which is a stalled plan rather than a battle
([[policy-holding-ground]]).

Two records, because they answer questions that look alike and are not:

* :class:`Outlays` -- **what was asked for, by purpose**. Where the credits went,
  and the last reason each purpose was refused.
* :class:`Reaches` -- **which spender was even asked**. "Defence declined three
  thousand times" and "defence was asked three times" are opposite diagnoses
  calling for opposite fixes, and a refusal count alone reads identically for
  both ([[policy-economy]]).

Pure and cumulative: both take what a tick already produced and add it up.
Nothing here decides anything, which is why it can be trusted as evidence about
the things that do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from rw_bot.policy.budget import Claim

#: Width of a report line's label column.
#:
#: The sweep keeps a result line only when the character at this offset is not a
#: space, which is how it tells a report figure from the planner's commentary
#: without duplicating the list of labels ([[harness-parallel-matches]]). A
#: block rendered one character wider is silently dropped from every filed
#: result, so the width is written down rather than counted by eye -- these two
#: renderers were both a character out on their first draft, and the sweep's own
#: guard test is what caught it.
_LABEL_WIDTH = 15


class Outlay(TypedDict):
    """What one purpose asked for across a whole match.

    Attributes:
        purpose: What the credits were for, e.g. ``"expand:extractorT1"``. The
            type name is part of it, so ``plan:landFactory`` and
            ``expand:landFactory`` stay apart -- they are different spenders
            buying the same thing and they fail for different reasons.
        asked: Claims made.
        granted: Claims met. A claim is never met in part, so the gap between
            this and ``asked`` is a straight count of refusals.
        spent: Credits actually committed.
        refusal: The last refusal's own words, empty when it was never refused.
            The *last* rather than the first, because a purpose that starts
            unaffordable and stays unaffordable says the same thing every time,
            while one that fails late has usually failed for a new reason.
    """

    purpose: str
    asked: int
    granted: int
    spent: int
    refusal: str


class Reach(TypedDict):
    """How often one spender was reached, and what it said.

    Attributes:
        stage: Which spender, or which gate returned before any spender ran.
        reached: Observations this stage was arrived at.
        acted: Observations it produced an order.
        reason: Its last word on declining, empty when it never declined.
    """

    stage: str
    reached: int
    acted: int
    reason: str


class Outlays:
    """Totals every claim a match makes, keyed by purpose.

    One per match rather than one per tick. The budget is deliberately per
    observation -- carrying it between ticks would be a second source of truth
    about a figure the sample already reports ([[policy-budget]]) -- so the
    accumulation lives here instead, where it can hold no authority over
    anything.
    """

    def __init__(self) -> None:
        """Open an empty record."""
        self._rows: dict[str, Outlay] = {}

    def add(self, ledger: Sequence[Claim]) -> None:
        """Fold one observation's claims into the running totals.

        Args:
            ledger: The claims that observation made, granted or not.
        """
        for claim in ledger:
            row = self._rows.setdefault(
                claim["purpose"],
                Outlay(purpose=claim["purpose"], asked=0, granted=0, spent=0, refusal=""),
            )
            row["asked"] += 1
            if claim["granted"]:
                row["granted"] += 1
                row["spent"] += claim["amount"]
                continue
            row["refusal"] = claim["reason"]

    def rows(self) -> tuple[Outlay, ...]:
        """Return the totals, dearest first.

        Ordered by credits committed and then by purpose, so two runs that spent
        the same way report it identically and a diff between them is a
        difference in the match rather than in dictionary iteration -- the same
        rule :func:`~rw_bot.policy.scoreboard.composition_of` follows.

        Returns:
            One entry per purpose claimed against.
        """

        def rank(row: Outlay) -> tuple[int, str]:
            return -row["spent"], row["purpose"]

        return tuple(sorted(self._rows.values(), key=rank))


class Reaches:
    """Counts how often each spender was arrived at, in the order declared.

    Insertion order rather than sorted, because the order **is** the policy: the
    chain is plan, then losses, then income, then defence, then throughput, and
    reading the counts down the page shows where it stops
    ([[policy-budget]]).
    """

    def __init__(self) -> None:
        """Open an empty census."""
        self._rows: dict[str, Reach] = {}

    def reached(self, stage: str, acted: bool, reason: str) -> None:
        """Record that a stage was arrived at on one observation.

        Args:
            stage: Which spender, or which gate returned before any ran.
            acted: Whether it produced an order.
            reason: Its words for declining. Ignored when it acted, because a
                stage that acted has no refusal to report.
        """
        row = self._rows.setdefault(stage, Reach(stage=stage, reached=0, acted=0, reason=""))
        row["reached"] += 1
        if acted:
            row["acted"] += 1
            return
        row["reason"] = reason

    def rows(self) -> tuple[Reach, ...]:
        """Return the census in the order the stages were first reached.

        Returns:
            One entry per stage arrived at.
        """
        return tuple(self._rows.values())


def format_outlays(outlays: Sequence[Outlay]) -> tuple[str, ...]:
    """Render the spending record as report lines.

    Args:
        outlays: The totals, dearest first.

    Returns:
        One line per purpose, or a single line naming the empty case rather than
        nothing at all -- a blank block reads as a measurement that failed to
        happen.
    """
    if not outlays:
        return (f"{'spend':<{_LABEL_WIDTH}}nothing was ever claimed",)
    return tuple(
        f"{'spend':<{_LABEL_WIDTH}}{row['purpose']:<24}"
        f"asked {row['asked']:>5}  got {row['granted']:>5}  spent {row['spent']:>7}"
        + (f"  held: {row['refusal']}" if row["refusal"] else "")
        for row in outlays
    )


def format_reaches(reaches: Sequence[Reach]) -> tuple[str, ...]:
    """Render the spender census as report lines.

    Args:
        reaches: The census, in declaration order.

    Returns:
        One line per stage, or a single line naming the empty case.
    """
    if not reaches:
        return (f"{'reach':<{_LABEL_WIDTH}}the economy never ran",)
    return tuple(
        f"{'reach':<{_LABEL_WIDTH}}{row['stage']:<24}"
        f"reached {row['reached']:>5}  acted {row['acted']:>5}"
        + (f"  last: {row['reason']}" if row["reason"] else "")
        for row in reaches
    )


__all__ = ["Outlay", "Outlays", "Reach", "Reaches", "format_outlays", "format_reaches"]
