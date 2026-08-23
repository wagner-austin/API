"""Measuring how long a running job has been quiet.

A job wedged on a download it will never finish reports ``RUNNING`` for as
long as its wall clock allows, holds its GPUs the whole time, and looks
exactly like a job in the middle of a long epoch. The only cheap signal that
separates them is whether the job's output file is still growing.

Modification time is compared against the CLUSTER's clock, not this machine's.
The two are not synchronised, and a skew of a few minutes would either invent
staleness or hide it. One command reads both, so they are read from the same
instant.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.ledger import LedgerEntry
from hpc3.core import remote


def log_path(entry: LedgerEntry) -> str:
    """Build the absolute path of a job's stdout log.

    Args:
        entry: The ledger record naming the job and its log directory.

    Returns:
        The path ``sbatch`` was told to write, reconstructed from the same
        two pieces the submission used: the log directory and the job name
        with its id.
    """
    return f"{entry['log_dir']}/{entry['name']}-{entry['job_id']}.out"


def age_command(entries: Sequence[LedgerEntry]) -> str:
    """Build one command reading the cluster clock and every log's mtime.

    Args:
        entries: Jobs whose logs to measure. Never empty.

    Returns:
        A shell command emitting ``now <epoch>`` followed by one
        ``<job_id> <mtime>`` line per log that exists. A log that does not
        exist yet emits nothing, which is correct: a job whose output file
        has not appeared has not been quiet, it has not started writing.

    Raises:
        ValueError: If no entries are given.
    """
    if len(entries) == 0:
        raise ValueError("age_command requires at least one entry")
    parts = [r'echo "now $(date +%s)"']
    for entry in entries:
        path = log_path(entry)
        parts.append(f"test -f '{path}' && echo \"{entry['job_id']} $(stat -c %Y '{path}')\"")
    return "; ".join(parts)


def parse_ages(output: str) -> dict[str, int]:
    """Read log ages out of the age command's output.

    Args:
        output: The command's standard output.

    Returns:
        Seconds since each log was last written, keyed by job id. Jobs whose
        log does not exist are absent from the mapping rather than present
        with a zero, so a caller can tell "not writing yet" from "written
        just now".

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE`
            if the cluster clock is missing or a timestamp is not numeric.
            Guessing a clock would invent staleness or hide it.
    """
    now: int | None = None
    stamps: list[tuple[str, int]] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        left, right = parts
        if not right.isdigit():
            raise AppError(
                Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE,
                f"log age output has a non-numeric timestamp: {line!r}",
            )
        if left == "now":
            now = int(right)
        else:
            stamps.append((left, int(right)))

    if now is None:
        raise AppError(
            Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE,
            f"log age output carries no cluster clock; got {output.strip()!r}.",
        )
    return {job_id: now - mtime for job_id, mtime in stamps}


def log_ages(host: str, entries: Sequence[LedgerEntry]) -> dict[str, int]:
    """Measure how long each job's log has gone unwritten.

    Args:
        host: SSH destination.
        entries: Jobs whose logs to measure.

    Returns:
        Seconds since each log was last written, keyed by job id. Empty when
        no entries are given, which is the honest answer to a question about
        no jobs.

    Raises:
        AppError: If the remote command fails or its output cannot be read.
    """
    if len(entries) == 0:
        return {}
    return parse_ages(remote.run_remote(host, age_command(entries)))


__all__ = ["age_command", "log_ages", "log_path", "parse_ages"]
