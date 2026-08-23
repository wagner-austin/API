"""Reading Slurm accounting output into typed status rows.

``sacct`` is asked for pipe-delimited output with no header, which removes
the column-alignment guessing that makes whitespace parsing fragile. Two
shapes still need handling and both are recorded here because neither is
obvious from the field names:

* A state can carry a suffix. ``CANCELLED by 1880454`` is the state
  ``CANCELLED`` plus the uid that cancelled it, so only the first token is the
  state.
* ``AllocTRES`` is a comma-separated list in which ``billing`` is one entry
  among several, and it is absent entirely on a job that never allocated --
  a pending job has no resources to describe.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.status import JobStatus, decode_job_status

SACCT_FIELDS = ("JobID", "JobName", "Partition", "State", "ElapsedRaw", "AllocTRES", "NodeList")
"""Columns requested from ``sacct``, in the order this module parses them."""

_EXPECTED_COLUMNS = len(SACCT_FIELDS)


def sacct_command(job_ids: Sequence[str]) -> str:
    """Build the accounting query for one or more jobs.

    ``sacct`` takes a comma-separated id list, so a sweep of six jobs is one
    round trip rather than six. That also makes the reported rows mutually
    consistent: six separate calls observe six different moments.

    Args:
        job_ids: Slurm job ids. Never empty -- a query naming no job would
            return every job the user has ever run.

    Returns:
        A ``sacct`` command line. ``-X`` restricts output to each job's own
        row rather than also emitting its batch and extern steps, which
        otherwise triple every result and report the step's resources rather
        than the job's.

    Raises:
        ValueError: If no job id is given.
    """
    if len(job_ids) == 0:
        raise ValueError("sacct_command requires at least one job id")
    fields = ",".join(SACCT_FIELDS)
    return f"sacct -j {','.join(job_ids)} -n -P -o {fields} -X"


def parse_tres_int(alloc_tres: str, key: str) -> int:
    """Extract one integer resource from an ``AllocTRES`` field.

    One parser for every TRES rather than one per key: ``billing`` and
    ``gres/gpu`` differ only in which name is read, and two copies would
    drift the moment one of them learned about a malformed value.

    The match is exact. ``gres/gpu`` and ``gres/gpu:rtx6000`` are separate
    entries in the same list, and reading the typed one as the untyped one
    would double-count a GPU on the partitions that report both.

    Args:
        alloc_tres: The field's raw text, such as
            ``billing=11,cpu=11,gres/gpu=1,mem=64G,node=1``. May be empty for
            a job that has not been allocated resources.
        key: TRES name to read, matched exactly.

    Returns:
        The value, or 0 when the field carries no such entry. Zero is correct
        for a pending job: it holds nothing, and a rate cannot be inferred
        from an allocation that has not happened.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE`
            if the entry is present but its value is not a non-negative
            integer. Defaulting to zero there would report a billed job as
            free, or a GPU job as holding none.
    """
    for entry in alloc_tres.split(","):
        name, separator, value = entry.partition("=")
        if name.strip() != key or separator == "":
            continue
        text = value.strip()
        if not text.isdigit():
            raise AppError(
                Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE,
                f"AllocTRES {key} must be a non-negative integer, got {value!r}.",
            )
        return int(text)
    return 0


def parse_elapsed_seconds(raw: str) -> int:
    """Read the ``ElapsedRaw`` field.

    Args:
        raw: The field's text, a whole number of seconds.

    Returns:
        Elapsed seconds.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE` if the
            field is not a non-negative integer.
    """
    text = raw.strip()
    if not text.isdigit():
        raise AppError(
            Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE,
            f"ElapsedRaw must be a non-negative integer of seconds, got {raw!r}.",
        )
    return int(text)


def parse_state(raw: str) -> str:
    """Reduce a ``State`` field to the state itself.

    Args:
        raw: The field's text, which may carry a suffix such as
            ``CANCELLED by 1880454``.

    Returns:
        The leading token, uppercased. An empty field yields an empty string,
        which the status contract then rejects by name rather than here by
        guess.
    """
    head = raw.strip().split(maxsplit=1)
    return head[0].upper() if head else ""


def parse_sacct_row(line: str, cluster: ClusterFacts) -> JobStatus:
    """Parse one pipe-delimited accounting row into a status.

    Args:
        line: A single ``sacct -P`` row carrying :data:`SACCT_FIELDS`.
        cluster: The cluster whose partitions the row is checked against.

    Returns:
        The validated status.

    Raises:
        AppError: With
            :attr:`~platform_core.errors.Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE` if the
            row does not hold exactly the requested number of columns, or if
            a numeric column is not numeric.
        JSONTypeError: If a field is present but not a value the status
            contract accepts -- an unrecognised state or partition, or an
            empty id or name.
    """
    columns = line.split("|")
    if len(columns) != _EXPECTED_COLUMNS:
        raise AppError(
            Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE,
            f"sacct row has {len(columns)} columns, expected {_EXPECTED_COLUMNS} "
            f"({','.join(SACCT_FIELDS)}); got {line!r}",
        )
    job_id, name, partition, state, elapsed, alloc_tres, node_list = columns
    record: dict[str, JSONValue] = {
        "job_id": job_id.strip(),
        "name": name.strip(),
        "partition": partition.strip(),
        "state": parse_state(state),
        "elapsed_seconds": parse_elapsed_seconds(elapsed),
        "billing_tres": parse_tres_int(alloc_tres, "billing"),
        "gpu_count": parse_tres_int(alloc_tres, "gres/gpu"),
        "cpu_count": parse_tres_int(alloc_tres, "cpu"),
        "node_list": node_list.strip(),
    }
    return decode_job_status(record, cluster)


def parse_sacct_output(output: str, cluster: ClusterFacts) -> list[JobStatus]:
    """Parse every row of an accounting query.

    Args:
        output: The command's standard output. Blank lines are skipped;
            ``sacct`` emits a trailing newline and, with ``-n``, nothing else
            that is not a row.
        cluster: The cluster whose partitions each row is checked against.

    Returns:
        One status per row, in the order reported.

    Raises:
        AppError: If any row is malformed. One bad row fails the whole
            parse: a partial list would be read as "these are the jobs", and
            the missing one is exactly the one worth knowing about.
        JSONTypeError: If any row holds a value the status contract rejects.
    """
    return [parse_sacct_row(line, cluster) for line in output.splitlines() if line.strip() != ""]


__all__ = [
    "SACCT_FIELDS",
    "parse_elapsed_seconds",
    "parse_sacct_output",
    "parse_sacct_row",
    "parse_state",
    "parse_tres_int",
    "sacct_command",
]
