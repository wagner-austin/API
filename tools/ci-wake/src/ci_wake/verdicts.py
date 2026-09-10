"""When an enrolled push has an answer worth waking somebody for.

FOUR STATES, AND THREE OF THEM ARE ANNOUNCED. The obvious one -- every run
finished -- is the least interesting to get right, because it announces
itself. The other two announced states exist because this bridge's failure
mode is silence, and a push that never produces a verdict is exactly as
silent as a bridge that is switched off:

  holding    still worth waiting for. Nothing is posted and the row stays.
  ripe       at least one run exists and every one of them is over.
  abandoned  no run has appeared and enough time has passed that none will.
             The ordinary cause is a push git refused as non-fast-forward,
             which ``pre-push`` enrolled before the transfer was attempted
             and no later hook could retract. The other causes matter more:
             a workflow whose path filters matched nothing, and an Actions
             outage. This workspace has already lost hours to the last one
             -- two commits shipped a barrel importing an untracked module
             and survived because Actions had started no job in two days --
             and a bridge that stayed quiet through it would have been the
             instrument that agreed nothing was wrong.
  stalled    runs exist, some have not finished, and far too long has
             passed. A wedged runner, a job the host ran out of memory
             under, a queue nobody is draining.

THE SETTLE WINDOW IS ABOUT RUN CREATION, NOT RUN COMPLETION. A repository
with several workflows creates them within seconds of a push, but not in the
same instant, and a very fast workflow can finish before a slower sibling has
been created. Announcing on "every run I can see is over" without waiting
would then report a one-workflow verdict for a commit that had three -- which
is the category error ``reading-ci-run-outcomes`` names, arriving through the
one door nobody watches. :data:`CREATION_SETTLE_SECONDS` costs a genuinely
solitary push two minutes and removes the case entirely.

WHY ``stalled`` CLOSES THE ROW RATHER THAN WATCHING FOREVER, since it is the
one honest trade here. A stalled push is announced once and its row is
marked, so the real verdict -- if one ever lands -- is never posted. The
alternative is a row that re-announces on every cycle, which trains its
reader to ignore the bridge, and a bridge nobody reads is the silence again
in a costume. What the post carries instead is every unfinished run's URL, so
the session it wakes can watch the run itself, which is the thing it wanted
in the first place.

Every function here is pure. Nothing reads a clock, a file or the board --
the caller supplies ``now_epoch`` -- so the whole policy is exercised over
simulated time without a fake, and the constants below are the only thing a
tuning argument needs to touch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from typing_extensions import TypedDict

from ci_wake.enrolment import PushAttempt
from ci_wake.runs import WorkflowRun, is_terminal

#: How long after a push every workflow it triggers has certainly been
#: created. Two minutes against an observed creation latency of seconds --
#: the margin is cheap because it delays only a push whose runs are ALREADY
#: finished, which is the rarest case this bridge sees.
CREATION_SETTLE_SECONDS: Final = 120

#: How long a push may show no run at all before it is called abandoned.
#: Fifteen minutes: long enough that a queued-but-uncreated run is not
#: mistaken for an absent one, short enough that a rejected push does not sit
#: in the record for an afternoon.
NO_RUN_SECONDS: Final = 900

#: How long a push may hold unfinished runs before it is called stalled.
#: Three hours against a measured worst case of about one -- a 62-minute job
#: that ended by exhausting the host's memory, 2026-09-09 -- so the horizon
#: is well clear of a slow run and well inside a wedged one.
STALL_SECONDS: Final = 10800


class PushVerdict(TypedDict):
    """One enrolled push and what this cycle decided about it.

    Attributes:
        attempt: The enrolment row.
        runs: Every run GitHub listed for the sha, in its order. Empty for
            ``abandoned``, and possibly empty for ``holding``.
        state: ``holding``, ``ripe``, ``abandoned`` or ``stalled``.
    """

    attempt: PushAttempt
    runs: tuple[WorkflowRun, ...]
    state: Literal["holding", "ripe", "abandoned", "stalled"]


def classify(
    attempt: PushAttempt, runs: Sequence[WorkflowRun], now_epoch: int
) -> Literal["holding", "ripe", "abandoned", "stalled"]:
    """Decide what to do about one enrolled push.

    Args:
        attempt: The enrolment row.
        runs: Every run GitHub listed for its sha.
        now_epoch: Unix seconds, supplied by the caller so this stays pure.

    Returns:
        The state. See this module's docstring for what each means and why
        three of the four are announced.
    """
    waited = now_epoch - attempt["attempted_unix"]
    if len(runs) == 0:
        return "abandoned" if waited >= NO_RUN_SECONDS else "holding"
    if not all(is_terminal(run) for run in runs):
        return "stalled" if waited >= STALL_SECONDS else "holding"
    return "ripe" if waited >= CREATION_SETTLE_SECONDS else "holding"


def is_announceable(verdict: PushVerdict) -> bool:
    """Report whether a verdict is one somebody should be woken for.

    Args:
        verdict: The classified push.

    Returns:
        True for every state but ``holding``. Written as a negation of the
        one silent state rather than as a list of the three loud ones: a
        state added later is then announced by default and its post is
        wrong in a way somebody notices, instead of being dropped in a way
        nobody does.
    """
    return verdict["state"] != "holding"


__all__ = [
    "CREATION_SETTLE_SECONDS",
    "NO_RUN_SECONDS",
    "STALL_SECONDS",
    "PushVerdict",
    "classify",
    "is_announceable",
]
