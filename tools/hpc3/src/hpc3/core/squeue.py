"""Reading why jobs are pending, which ``sacct`` cannot tell you.

Accounting knows a job is ``PENDING``. It does not know why, and the why is
the whole question: waiting on ``Resources`` is the queue working, while
waiting on ``DependencyNeverSatisfied`` is a job that will never run and will
sit there until a human notices. Only ``squeue`` carries the reason.

So this is a second query rather than a field on the first. The alternative --
inferring blocked-ness from elapsed pending time -- would call a busy
afternoon a defect and a genuinely dead job merely slow.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue

from hpc3.contracts.pending import PendingJob, decode_pending_job

SQUEUE_FORMAT = "%i|%j|%r"
"""Job id, name, reason -- the three fields, pipe-delimited, no header."""

_EXPECTED_COLUMNS = 3


def squeue_command(job_ids: Sequence[str]) -> str:
    """Build the pending-reason query for one or more jobs.

    Args:
        job_ids: Slurm job ids. Never empty -- an id-less query would report
            every job on the cluster, which is 1,400 rows of other people's
            work.

    Returns:
        A ``squeue`` command line restricted to pending jobs. Running jobs
        have no meaningful reason and would only add rows to filter out.

    Raises:
        ValueError: If no job id is given.
    """
    if len(job_ids) == 0:
        raise ValueError("squeue_command requires at least one job id")
    return f"squeue -h -j {','.join(job_ids)} -t PD -o {SQUEUE_FORMAT!r}"


def parse_squeue_row(line: str) -> PendingJob:
    """Parse one pipe-delimited pending row.

    Args:
        line: A single ``squeue`` row carrying :data:`SQUEUE_FORMAT`.

    Returns:
        The validated pending job.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE`
            if the row does not hold exactly three columns.
        JSONTypeError: If the id or name is empty.
    """
    columns = line.split("|")
    if len(columns) != _EXPECTED_COLUMNS:
        raise AppError(
            Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE,
            f"squeue row has {len(columns)} columns, expected {_EXPECTED_COLUMNS} "
            f"({SQUEUE_FORMAT}); got {line!r}",
        )
    job_id, name, reason = columns
    record: dict[str, JSONValue] = {
        "job_id": job_id.strip(),
        "name": name.strip(),
        "reason": reason.strip(),
    }
    return decode_pending_job(record)


def parse_squeue_output(output: str) -> list[PendingJob]:
    """Parse every row of a pending-reason query.

    Args:
        output: The command's standard output. Empty when none of the named
            jobs is pending, which is the normal healthy case.

    Returns:
        One entry per pending job, in the order reported.

    Raises:
        AppError: If any row is malformed. One bad row fails the whole parse:
            a partial list reads as "these are the blocked jobs", and the
            missing one is the one worth knowing about.
    """
    return [parse_squeue_row(line) for line in output.splitlines() if line.strip() != ""]


__all__ = [
    "SQUEUE_FORMAT",
    "parse_squeue_output",
    "parse_squeue_row",
    "squeue_command",
]
