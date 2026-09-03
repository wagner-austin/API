"""The two questions only ``squeue`` can answer.

**Why a job is pending**, which ``sacct`` cannot tell you. Accounting knows a
job is ``PENDING``. It does not know why, and the why is the whole question:
waiting on ``Resources`` is the queue working, while waiting on
``DependencyNeverSatisfied`` is a job that will never run and will sit there
until a human notices. Only ``squeue`` carries the reason. So this is a second
query rather than a field on the first. The alternative -- inferring
blocked-ness from elapsed pending time -- would call a busy afternoon a defect
and a genuinely dead job merely slow.

**What the account is holding**, which nothing else can be asked. Every other
query here names ids we already have, so by construction none of them can
return a job we do not know about. Enumerating the account is the only
question whose answer can, and it is what turns the ledger from a list that
proves its own entries into a record that can be checked for completeness.
See :mod:`hpc3.contracts.account`.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue

from hpc3.contracts.account import AccountJob, decode_account_job
from hpc3.contracts.pending import PendingJob, decode_pending_job

SQUEUE_FORMAT = "%i|%j|%r"
"""Job id, name, reason -- the three fields, pipe-delimited, no header."""

ACCOUNT_FORMAT = "%i|%j|%T"
"""Job id, name, state -- what the account is holding, in every state.

State rather than reason, because this query is not restricted to pending
jobs: a running job's reason column is ``None``, which would make a live job
holding four GPUs indistinguishable from one the scheduler has not looked at.
"""

_EXPECTED_COLUMNS = 3

#: Environment prefix for every squeue whose ids get parsed. Slurm caps a
#: job array's task-id expression at SLURM_BITSTR_LEN bytes (default 64)
#: and TRUNCATES past it -- a sparse convergence resubmission with
#: scattered indices produced ``55732071_[99,101-103,111-114,12`` with
#: its bracket never closed, and the parser rightly refused it
#: (ARRAY_ID_UNPARSABLE, 2026-09-03, measured live: the same queue row
#: printed whole under this prefix). 4096 bytes holds any selection a
#: thousand-task array can express.
_BITSTR = "SLURM_BITSTR_LEN=4096 "


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
    return f"{_BITSTR}squeue -h -j {','.join(job_ids)} -t PD -o {SQUEUE_FORMAT!r}"


def account_command() -> str:
    """Build the query that enumerates every job held for this account.

    Takes no job ids, which is the point: an id-restricted query can only
    return jobs already known, and the condition this exists to find is a job
    that is *not* known. ``--me`` scopes it to the authenticated account, so
    the 1,400 rows of other people's work that made an unrestricted
    :func:`squeue_command` unacceptable are not in the result either.

    Returns:
        A ``squeue`` command line covering every state, header suppressed.
    """
    return f"{_BITSTR}squeue --me -h -o {ACCOUNT_FORMAT!r}"


def _split_row(line: str, fmt: str) -> tuple[str, str, str]:
    """Split one pipe-delimited row into its three columns.

    Args:
        line: A single ``squeue`` row.
        fmt: The format that produced it, named in the error so a reader
            knows which of the two queries returned an unreadable row.

    Returns:
        The three columns, each stripped of surrounding whitespace.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE`
            if the row does not hold exactly three columns.
    """
    columns = line.split("|")
    if len(columns) != _EXPECTED_COLUMNS:
        raise AppError(
            Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE,
            f"squeue row has {len(columns)} columns, expected {_EXPECTED_COLUMNS} "
            f"({fmt}); got {line!r}",
        )
    first, second, third = columns
    return first.strip(), second.strip(), third.strip()


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
    job_id, name, reason = _split_row(line, SQUEUE_FORMAT)
    record: dict[str, JSONValue] = {"job_id": job_id, "name": name, "reason": reason}
    return decode_pending_job(record)


def parse_account_row(line: str) -> AccountJob:
    """Parse one pipe-delimited account row.

    Args:
        line: A single ``squeue`` row carrying :data:`ACCOUNT_FORMAT`.

    Returns:
        The validated account job.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE`
            if the row does not hold exactly three columns.
        JSONTypeError: If any field is empty.
    """
    job_id, name, state = _split_row(line, ACCOUNT_FORMAT)
    record: dict[str, JSONValue] = {"job_id": job_id, "name": name, "state": state}
    return decode_account_job(record)


def parse_account_output(output: str) -> list[AccountJob]:
    """Parse every row of an account enumeration.

    Args:
        output: The command's standard output. Empty when the account holds
            no jobs at all, which is the normal state between runs.

    Returns:
        One entry per job, in the order reported.

    Raises:
        AppError: If any row is malformed. One bad row fails the whole parse,
            for the reason :func:`parse_squeue_output` gives and one more:
            a partial enumeration reads as a COMPLETE list of what the
            account holds, and the row that failed to parse is the one that
            would have been reported as unrecorded.
    """
    return [parse_account_row(line) for line in output.splitlines() if line.strip() != ""]


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
    "ACCOUNT_FORMAT",
    "SQUEUE_FORMAT",
    "account_command",
    "parse_account_output",
    "parse_account_row",
    "parse_squeue_output",
    "parse_squeue_row",
    "squeue_command",
]
