"""Reconciling what we submitted against what the cluster is actually doing.

This is the module that answers "is anything wrong that looks fine?". Four
conditions, all of which present as a healthy-looking queue:

* **Blocked.** ``PENDING`` on a reason that will never resolve. Measured on
  HPC3: 261 of 621 pending GPU jobs sat on ``DependencyNeverSatisfied``.
  They look exactly like the 3 waiting on ``Resources``.
* **Unaccounted.** In our ledger, and accounting has never heard of it. The
  submission returned an id and the job does not exist -- so nothing will
  ever report on it, and no query that starts from the cluster will find it.
* **Unclaimed.** Running under the account, and our ledger has never heard of
  *it*. The mirror of unaccounted, and the direction that went unbuilt.
* **Silent.** ``RUNNING``, holding GPUs, and its log has not grown. A job
  wedged on a download it will never finish reports ``RUNNING`` forever and
  bills GPU-hours the whole time.

THE RECONCILIATION RUNS IN BOTH DIRECTIONS, and for a long time only ran in
one. This module used to justify that in so many words -- "the reconciliation
starts from the LEDGER, not from the cluster; starting from ``squeue`` can
only ever find jobs the cluster already knows about, which by construction
excludes the ones that went missing." Every clause of that is true and it is
an argument for the ledger-first query, not against the cluster-first one.
What it obscured is that the two queries answer different questions. Asking
the cluster about ids we recorded finds a job that vanished. Asking the
cluster to enumerate itself finds a job we never recorded -- and no
ledger-first query can, because it starts from the record whose completeness
is the thing in doubt.

That gap had a real occupant. The image builds documented in this package's
own README are started with a raw ``ssh <host> sbatch``, and twenty-one of
them ran without leaving a ledger row, invisible to a triage command whose
whole purpose was to find jobs nobody was watching. A guard that only checks
the direction its author was thinking about is the shape this workspace's
PM-112 is about: the paired check is not optional, it is the other half.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hpc3.contracts.account import AccountJob
from hpc3.contracts.array import expand_job_id
from hpc3.contracts.closure import Closure
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.pending import PendingJob, is_blocked
from hpc3.contracts.status import JobStatus, is_terminal


class Finding:
    """One job that needs attention, and why.

    Attributes:
        job_id: The job in question.
        name: Its name, as submitted.
        kind: Which condition it hit -- ``blocked``, ``unaccounted``,
            ``unclaimed``, ``silent`` or ``oversized``.
        detail: Human-readable specifics, such as the scheduler's own reason.
    """

    __slots__ = ("detail", "job_id", "kind", "name")

    def __init__(self, job_id: str, name: str, kind: str, detail: str) -> None:
        """Record one finding.

        Args:
            job_id: The job in question.
            name: Its name, as submitted.
            kind: Which condition it hit.
            detail: Human-readable specifics.
        """
        self.job_id = job_id
        self.name = name
        self.kind = kind
        self.detail = detail


def blocked_jobs(pending: Sequence[PendingJob]) -> list[Finding]:
    """Find pending jobs waiting on something that will never resolve.

    Args:
        pending: Pending rows from ``squeue``, carrying their reasons.

    Returns:
        One finding per blocked job, in the order reported.
    """
    return [
        Finding(job["job_id"], job["name"], "blocked", f"pending on {job['reason']!r}")
        for job in pending
        if is_blocked(job["reason"])
    ]


def open_entries(
    entries: Sequence[LedgerEntry], closures: Mapping[str, Closure]
) -> list[LedgerEntry]:
    """Narrow the ledger to jobs not already known to have ended.

    Args:
        entries: Everything the local ledger holds.
        closures: Jobs previously observed in a terminal state.

    Returns:
        Only the entries still worth asking the cluster about. This is what
        keeps ``unaccounted`` honest as a ledger ages: ``sacct`` retention is
        finite, so a job that ran perfectly a month ago eventually has no
        accounting row, which is indistinguishable from a job that never
        existed. Once a closure is recorded the question is never asked again,
        so the finding cannot come back and the query does not grow without
        bound either.
    """
    return [entry for entry in entries if entry["job_id"] not in closures]


def closures_for(statuses: Sequence[JobStatus], *, closed_at: str) -> list[Closure]:
    """Build a closure for every job accounting now reports as finished.

    Args:
        statuses: Accounting rows just read.
        closed_at: ISO-8601 timestamp of this observation, supplied by the
            caller so this function reads no clock.

    Returns:
        One closure per terminal row, in the order reported. ``REQUEUED`` is
        not terminal and produces none: the job is going back to the queue,
        which is protection working rather than the run ending.
    """
    return [
        Closure(
            job_id=status["job_id"],
            state=status["state"],
            closed_at=closed_at,
            # Captured here because this is the last moment it is available:
            # sacct's retention is finite, and this is the only place the
            # package will ever be able to answer "how long does this
            # project's work actually take".
            elapsed_seconds=status["elapsed_seconds"],
        )
        for status in statuses
        if is_terminal(status["state"])
    ]


def unaccounted_jobs(
    entries: Sequence[LedgerEntry], statuses: Sequence[JobStatus]
) -> list[Finding]:
    """Find submitted jobs the cluster has never reported on.

    Args:
        entries: Ledger entries still open -- see :func:`open_entries`. Passing
            the whole ledger would report every job older than the cluster's
            ``sacct`` retention window, which is the same observation as a job
            that never existed and is not the same event.
        statuses: Accounting rows the cluster returned for those ids.

    Returns:
        One finding per open ledger entry with no accounting row. This is the
        condition no cluster-side query can detect, because the evidence is
        precisely the absence of a cluster-side record.
    """
    known = {status["job_id"] for status in statuses}
    return [
        Finding(
            entry["job_id"],
            entry["name"],
            "unaccounted",
            f"submitted {entry['submitted_at']} to {entry['host']}, accounting has no record of it",
        )
        for entry in entries
        if entry["job_id"] not in known
    ]


def unclaimed_jobs(entries: Sequence[LedgerEntry], account: Sequence[AccountJob]) -> list[Finding]:
    """Find jobs the cluster is holding that no ledger row claims.

    Args:
        entries: The WHOLE ledger, not the open subset. A closure records
            that a job ended, never that it stopped having been ours, so
            filtering by it here would report every finished job the cluster
            still happens to be holding -- which it does for minutes after
            the end -- as though nobody had submitted it.
        account: Every job the cluster reports for this account.

    Returns:
        One finding per job with no ledger row, in the order the cluster
        reported them. No filter on the name: matching only jobs called
        ``<project>.<something>`` would be the natural narrowing and would
        defeat the check outright, because a job submitted around this
        package is under no obligation to be named the way this package
        names things -- the image builds are called ``<project>-image-v<n>``
        and would pass such a filter untouched.

        An interactive session is a true positive here, not a false one: it
        is a job on the account that this machine did not submit and cannot
        trace. If those become common enough to be noise, the fix is to
        record them, not to teach this function to look away.
    """
    recorded = {entry["job_id"] for entry in entries}
    # An account row can stand for many ledger rows: squeue reports every
    # still-pending task of a job array as one aggregate id --
    # 55678543_[2-3%2], measured on probe job 55678543 -- while the ledger
    # records each task individually. The row is claimed when EVERY task it
    # stands for is; a partially-claimed aggregate is a genuine finding,
    # because some task on the cluster has no ledger row behind it.
    return [
        Finding(
            job["job_id"],
            job["name"],
            "unclaimed",
            f"{job['state']} on the cluster and no ledger row claims it; "
            "it was not submitted from this machine through hpc3",
        )
        for job in account
        if any(task_id not in recorded for task_id in expand_job_id(job["job_id"]))
    ]


def _allocation_phrase(status: JobStatus) -> str:
    """Describe what a running job is holding, in its own terms.

    Args:
        status: The job's accounting row.

    Returns:
        The GPU count for a job that holds GPUs, and the core count for one
        that does not. A CPU job described as holding "0 GPU(s)" reads as a
        broken allocation rather than as the allocation it asked for.
    """
    gpus = status["gpu_count"]
    if gpus > 0:
        return f"{gpus} GPU(s)"
    return f"{status['cpu_count']} core(s)"


def silent_jobs(
    statuses: Sequence[JobStatus], log_ages: dict[str, int], *, quiet_seconds: int
) -> list[Finding]:
    """Find running jobs whose output has stopped growing.

    Args:
        statuses: Accounting rows.
        log_ages: Seconds since each job's log was last written, keyed by job
            id. A job absent from this mapping is skipped rather than
            assumed silent: no reading is not the same as a bad reading.
        quiet_seconds: How long a running job may produce nothing before it
            is reported.

    Returns:
        One finding per running job that has been quiet too long. Not an
        error: a job legitimately can be quiet during a long epoch. It is
        reported so a human can look, which is the only thing that
        distinguishes a slow job from a wedged one.
    """
    findings: list[Finding] = []
    for status in statuses:
        if status["state"] != "RUNNING":
            continue
        age = log_ages.get(status["job_id"])
        if age is None or age <= quiet_seconds:
            continue
        findings.append(
            Finding(
                status["job_id"],
                status["name"],
                "silent",
                # Reported from AllocTRES, so a CPU job says cores rather than
                # "0 GPU(s)" -- which reads as a broken allocation on a job
                # that never asked for one.
                f"RUNNING on {_allocation_phrase(status)} but its log has not "
                f"been written for {age}s",
            )
        )
    return findings


def live_entries(
    entries: Sequence[LedgerEntry], statuses: Sequence[JobStatus]
) -> list[LedgerEntry]:
    """List ledger entries whose jobs accounting has not finished.

    Args:
        entries: Everything the local ledger holds.
        statuses: Accounting rows for those ids.

    Returns:
        Entries whose job is either still going or unaccounted for. These are
        the ones worth asking further questions about; a job that completed
        needs no triage.
    """
    finished = {status["job_id"] for status in statuses if is_terminal(status["state"])}
    return [entry for entry in entries if entry["job_id"] not in finished]


__all__ = [
    "Finding",
    "blocked_jobs",
    "closures_for",
    "live_entries",
    "open_entries",
    "silent_jobs",
    "unaccounted_jobs",
    "unclaimed_jobs",
]
