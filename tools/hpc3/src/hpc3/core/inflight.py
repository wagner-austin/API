"""Which artifacts a job already running would overwrite.

WHAT THIS PREVENTS, and it is not hypothetical. On 2026-08-28 a ``uz`` member
was preempted, resubmitted as ``bases-uz-r2``, and then -- while r2 was still
queued -- resubmitted again as ``bases-uz-r3`` against a rebuilt image. Both
declared the same artifact:

    /pub/wagnera3/LSTM/checkpoints/uz_best.pt

Two jobs training the same language into the same checkpoint file, racing on
every epoch boundary, producing a file that is neither run's. It was caught by
eye and one was cancelled. Nothing in this package would have caught it: there
was no check of any kind, in submit, in sweep, or in the contracts.

That is the failure mode this module exists for, and it is worse than a crash
because it SUCCEEDS. Both jobs report COMPLETED, the checkpoint exists, its
provenance says one job made it, and nothing about the result looks wrong.

WHY IT ASKS THE CLUSTER. A ledger row alone cannot say whether its job is
still running -- closures are written by ``hpc3-triage``, so a ledger nobody
has triaged recently looks like every job is live, and one triaged a moment
ago looks like none are. Neither is a basis for refusing a submission. The
authoritative answer is the account enumeration :mod:`hpc3.contracts.account`
already performs, intersected with the ledger to learn what those live jobs
are writing -- which only the ledger knows, because ``squeue`` has never heard
of an artifact.

WHAT IT DELIBERATELY DOES NOT DO. It does not check whether the artifact
already EXISTS. Overwriting a finished checkpoint is a normal, intended act --
that is what a resume does, and what re-running a corrected experiment does.
The defect is concurrency, not overwriting, and a check that refused an
existing path would refuse every resume in the package.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.account import AccountJob
from hpc3.contracts.array import expand_job_id
from hpc3.contracts.ledger import LedgerEntry


def claimed_artifacts(
    entries: Sequence[LedgerEntry], account: Sequence[AccountJob]
) -> dict[str, str]:
    """Map each artifact a live job is writing to the job writing it.

    Args:
        entries: The whole ledger. Not the open subset: a closure says a job
            ended, and this asks the cluster that question directly instead.
        account: Every job the cluster currently holds for this account.

    Returns:
        Artifact path to the qualified job name holding it, for live jobs
        that declare one. A job with a null artifact claims nothing and is
        absent -- it writes no file of its own, which is a positive fact the
        contract requires it to state.

        Later rows win on collision, which cannot mislead: if two live jobs
        somehow already share an artifact, either name is a true answer to
        "something live is writing this", and the refusal that follows names
        a real job either way.
    """
    # Expanded, because the ledger records array TASK ids -- 55678543_2 --
    # while squeue reports every still-pending task of an array as one
    # aggregate row, 55678543_[2-3%2] (measured, probe job 55678543). An
    # unexpanded set would read every pending member as not-live, and the
    # refusal this module exists for would wave the double submission
    # through while the first copy sat in the queue.
    live = {task_id for job in account for task_id in expand_job_id(job["job_id"])}
    return {
        entry["artifact"]: entry["name"]
        for entry in entries
        if entry["job_id"] in live and entry["artifact"] is not None
    }


def check_artifact_is_free(artifact: str | None, claimed: dict[str, str], *, name: str) -> None:
    """Refuse a submission that would race a live job for its output file.

    Args:
        artifact: Path the new job would write, or None when it writes no
            file of its own.
        claimed: Artifacts live jobs are already writing, from
            :func:`claimed_artifacts`.
        name: Qualified name of the job being submitted, for the message.

    Raises:
        AppError: With ``ARTIFACT_ALREADY_IN_FLIGHT`` when a live job is
            already writing that path. Refused rather than warned: the two
            jobs both succeed, so a warning is read once and the corrupted
            checkpoint is discovered months later by whoever tries to
            reproduce a number from it.
    """
    if artifact is None:
        return
    holder = claimed.get(artifact)
    if holder is None:
        return
    raise AppError(
        Hpc3ErrorCode.ARTIFACT_ALREADY_IN_FLIGHT,
        f"{name} would write {artifact}, which {holder} is on the cluster writing "
        "right now. Two jobs writing one file race on every write and BOTH report "
        f"success, so nothing afterwards looks wrong. Cancel {holder} first, or "
        "give this run its own artifact.",
    )


__all__ = ["check_artifact_is_free", "claimed_artifacts"]
