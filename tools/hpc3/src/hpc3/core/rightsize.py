"""Comparing what a project ASKS FOR against what its work actually takes.

NOTHING DID THIS, and the cost was measured. ``turkic-lstm`` declared
``minutes: 720``. Its members finished in 1618 and 1585 seconds -- 27 minutes,
3.7% of the request. Five members then sat unschedulable while the partition
turned over, because Slurm's backfill can only start a twelve-hour job in a
twelve-hour hole and those are rare on a busy free partition; ninety-minute
holes are constant.

HOW A 26x OVER-REQUEST PASSED EVERY CHECK. It was never measured, because it
could not be: the commit that created the project says LSTM had never run on
the cluster at all, so 720 was a placeholder for a number nobody could take
yet. It was inherited -- 720 is the value in the README's ``abl`` workspace
example, the block you copy when adding a project, while the README's own
``turkic-lstm`` example twenty lines below says 240. And the budget was then
fitted to it: the cap is 84.0 GPU-hours and 7 members x 12 hours is 84.0
exactly, so the one number positioned to contradict the request had been
derived from it. "84.00 GPU-hours against a declared 84.00 cap" reads as a
check passing and is a tautology.

So this is deliberately the reverse direction from the budget. A budget asks
"can we afford what you declared"; this asks "did you mean what you declared",
and only history can answer.

WHY THE CLOSURE AND NOT ``sacct``. Retention is finite, so a cluster-side
query answers this only for recent work and silently stops answering later.
The closure record is written the moment a job is observed to have ended and
is never rewritten, so the history it holds is the one that survives -- and
this check needs no cluster query at all.

WHAT IT WILL NOT SAY. Nothing, until a project has finished work to learn
from, and nothing about a request that is merely generous. Headroom is
correct: a resume from a checkpoint runs longer than the run that made it, a
slower node runs longer than a fast one, and a job killed at its limit loses
everything. The factor below is wide enough that only a request nobody chose
can trip it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from typing_extensions import TypedDict

from hpc3.contracts.closure import Closure
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.workspace import ProjectConfig

HEADROOM = 4
"""How many times its longest observed run a project may ask for, unremarked.

Four rather than two, because the reasons a run legitimately exceeds its
predecessors are real and compounding -- a resume replays a partial epoch, a
V100 is not an A100, a corpus grows. At four, ``turkic-lstm``'s 720 minutes
against a 27-minute measurement trips by a factor of six and a request of 108
minutes would not trip at all.

Deliberately not tunable from the workspace. A threshold a project can raise
is one that gets raised by the project it was about to catch.
"""

MINIMUM_OBSERVATIONS = 2
"""How many finished runs before this says anything.

One run is an anecdote, and the first run of a new experiment is exactly when
a generous limit is most defensible. Two is the smallest number that can show
a spread.
"""


class Oversized(TypedDict):
    """A project asking for far more wall clock than its work has ever used.

    Attributes:
        project: The project as declared in the workspace.
        requested_minutes: What its runs ask Slurm for.
        longest_seconds: The longest finished run on record.
        observations: How many finished runs that is drawn from.
        evidence: Job id of the longest run, so the claim can be checked
            against the cluster rather than taken on trust.
    """

    project: str
    requested_minutes: int
    longest_seconds: int
    observations: int
    evidence: str


EVIDENCE_STATE = "COMPLETED"
"""The only terminal state that says how long the work takes.

Both other cases were in the first live run of this check and both are
misleading. ``turkic-lstm.bases-uz-r2`` was CANCELLED before it started and
carries ``elapsed_seconds=0``; counted, it drags the observed maximum toward
zero and makes any request look oversized. ``PREEMPTED`` and ``FAILED`` are
the same defect more slowly -- they measure when a run STOPPED, not what it
needed. ``TIMEOUT`` is genuinely informative and in the opposite direction:
it says the limit was too SHORT, which is not this check's question, and
reading it as a duration would argue for shrinking a limit that already
truncated a run.
"""


def observed_runtimes(
    entries: Sequence[LedgerEntry], closures: Mapping[str, Closure]
) -> dict[str, list[tuple[int, str, str]]]:
    """Collect each project's completed runtimes, from local records only.

    Args:
        entries: The whole ledger, which is what maps a job to its project
            and records the partition it went to.
        closures: Closures by job id, which is what carries the runtime and
            the terminal state.

    Returns:
        Project name to a list of (seconds, job id, partition), in ledger
        order. Two exclusions, both learned from this check's first live run:

        A closure carrying ``None`` -- written before the field existed -- is
        skipped rather than counted as zero, which would make an unmeasured
        history look instantaneous and turn every project into a finding.

        A closure whose state is not :data:`EVIDENCE_STATE` is skipped, for
        the same reason in a different disguise: a cancelled job reports zero
        seconds and a preempted one reports when it was killed.
    """
    runtimes: dict[str, list[tuple[int, str, str]]] = {}
    for entry in entries:
        closure = closures.get(entry["job_id"])
        if closure is None or closure["state"] != EVIDENCE_STATE:
            continue
        elapsed = closure["elapsed_seconds"]
        if elapsed is None:
            continue
        runtimes.setdefault(entry["project"], []).append(
            (elapsed, entry["job_id"], entry["partition"])
        )
    return runtimes


def oversized_projects(
    projects: Mapping[str, ProjectConfig],
    entries: Sequence[LedgerEntry],
    closures: Mapping[str, Closure],
) -> list[Oversized]:
    """Find projects whose time limit dwarfs everything they have ever run.

    Args:
        projects: The workspace's declared projects.
        entries: The whole ledger.
        closures: Closures by job id.

    Returns:
        One entry per project whose request exceeds :data:`HEADROOM` times its
        longest completed run, in declared order. A project with fewer than
        :data:`MINIMUM_OBSERVATIONS` such runs is absent: there is nothing yet
        to be wrong about.

        ONLY RUNS ON THE PROJECT'S OWN PARTITION COUNT, which the first live
        run of this check needed and did not have. ``turkic-lstm`` declares
        ``free-gpu``; its image build ran on ``free`` for 885 seconds, under
        the two-hour limit ``build.sbatch`` renders for itself and never under
        ``minutes`` at all. Counted, it became the evidence for a claim about
        a resource line it had never used. A job that went to a different
        partition was submitted with a different resource line, so it says
        nothing about whether this one is right.
    """
    found: list[Oversized] = []
    runtimes = observed_runtimes(entries, closures)
    for name, config in projects.items():
        observations = [
            (seconds, job_id)
            for seconds, job_id, partition in runtimes.get(name, [])
            if partition == config["partition"]
        ]
        if len(observations) < MINIMUM_OBSERVATIONS:
            continue
        longest_seconds, evidence = max(observations)
        if config["minutes"] * 60 <= longest_seconds * HEADROOM:
            continue
        found.append(
            Oversized(
                project=name,
                requested_minutes=config["minutes"],
                longest_seconds=longest_seconds,
                observations=len(observations),
                evidence=evidence,
            )
        )
    return found


def describe(oversized: Oversized) -> str:
    """Render one finding as the sentence an operator needs.

    Args:
        oversized: The finding.

    Returns:
        A line naming the request, the measurement it is out of step with,
        and the consequence -- which is the part that makes it worth acting
        on rather than noting.

        The ceiling it suggests is derived from the MEASUREMENT, never from
        the request: ``HEADROOM`` times the longest run is the largest
        request this check would not have flagged. Scaling the request down
        by some factor would produce a smaller number with the same defect,
        since the request is the thing under suspicion.
    """
    minutes = oversized["longest_seconds"] / 60
    percent = oversized["longest_seconds"] * 100 / (oversized["requested_minutes"] * 60)
    ceiling = oversized["longest_seconds"] * HEADROOM // 60
    return (
        f"{oversized['project']} requests {oversized['requested_minutes']} min; its "
        f"longest of {oversized['observations']} finished run(s) took {minutes:.0f} min "
        f"({percent:.1f}% of the request, job {oversized['evidence']}). Slurm backfills "
        f"a job into a hole its size, so an oversized request waits for a hole it never "
        f"needed. {ceiling} min or less would not be remarked on."
    )


__all__ = [
    "EVIDENCE_STATE",
    "HEADROOM",
    "MINIMUM_OBSERVATIONS",
    "Oversized",
    "describe",
    "observed_runtimes",
    "oversized_projects",
]
