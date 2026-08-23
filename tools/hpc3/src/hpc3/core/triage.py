"""Reconciling what we submitted against what the cluster is actually doing.

This is the module that answers "is anything wrong that looks fine?". Three
conditions, all of which present as a healthy-looking queue:

* **Blocked.** ``PENDING`` on a reason that will never resolve. Measured on
  HPC3: 261 of 621 pending GPU jobs sat on ``DependencyNeverSatisfied``.
  They look exactly like the 3 waiting on ``Resources``.
* **Unaccounted.** In our ledger, and accounting has never heard of it. The
  submission returned an id and the job does not exist -- so nothing will
  ever report on it, and no query that starts from the cluster will find it.
* **Silent.** ``RUNNING``, holding GPUs, and its log has not grown. A job
  wedged on a download it will never finish reports ``RUNNING`` forever and
  bills GPU-hours the whole time.

The reconciliation starts from the LEDGER, not from the cluster. Starting
from ``squeue`` can only ever find jobs the cluster already knows about,
which by construction excludes the ones that went missing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hpc3.contracts.closure import Closure
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.pending import PendingJob, is_blocked
from hpc3.contracts.status import JobStatus, is_terminal


class Finding:
    """One job that needs attention, and why.

    Attributes:
        job_id: The job in question.
        name: Its name, as submitted.
        kind: Which condition it hit -- ``blocked``, ``unaccounted`` or
            ``silent``.
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
        Closure(job_id=status["job_id"], state=status["state"], closed_at=closed_at)
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
]
