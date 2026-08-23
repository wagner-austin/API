"""The ledger contract: a local record of every job we ever submitted.

A job outlives the process that submitted it by hours. If the submitting
process is the only thing that knew the id -- because it printed it and
exited, or because the SSH link dropped mid-sweep -- then the job is running,
consuming a share of a shared machine, and nobody can find it. ``squeue -u``
finds it only while it is queued or running; after that it is a row in
accounting that nobody knows to ask about.

So every submission is appended here, on disk, before the operator ever sees
it on screen. The ledger is append-only JSON Lines: one record per line, never
rewritten, so a crash midway through a sweep truncates at a line boundary and
loses at most the record currently being written -- and the entries before it
are still readable by anything that can read a line.

This is deliberately NOT a database and deliberately NOT on the cluster. It
answers one question -- "what did we submit, and where" -- from the machine
that did the submitting, at a moment when the cluster may be unreachable.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.contracts.cluster import ClusterFacts, require_partition
from hpc3.contracts.layout import require_project


class LedgerEntry(TypedDict):
    """One submission, recorded at the moment it was made.

    Attributes:
        job_id: Id Slurm assigned.
        project: Which body of work it belongs to, so a ledger holding
            several projects can be filtered without parsing names.
        name: The QUALIFIED job name, ``<project>.<name>`` -- which is
            also the stem of its log filenames, so the logs stay findable
            from this record alone.
        host: SSH destination it was sent to, so a reader knows which
            cluster to ask.
        partition: Partition it went to.
        submitted_at: ISO-8601 timestamp of the submission, supplied by the
            caller rather than read from a clock here -- a contract that
            reads the clock cannot be tested for what it records.
        log_dir: Absolute directory holding the job's output, so the logs are
            findable without reconstructing the submission.
    """

    job_id: str
    project: str
    name: str
    host: str
    partition: str
    submitted_at: str
    log_dir: str


def _require_nonempty_str(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required string field that must not be empty.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty. Every
            field here is part of finding a job again; an empty one is a
            record that cannot do its job.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return value


def encode_ledger_entry(entry: LedgerEntry) -> dict[str, JSONValue]:
    """Encode a ledger entry to a JSON object.

    Args:
        entry: Entry to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "job_id": entry["job_id"],
        "project": entry["project"],
        "name": entry["name"],
        "host": entry["host"],
        "partition": entry["partition"],
        "submitted_at": entry["submitted_at"],
        "log_dir": entry["log_dir"],
    }


def decode_ledger_entry(value: JSONValue, cluster: ClusterFacts) -> LedgerEntry:
    """Decode and validate a JSON value into a ledger entry.

    Args:
        value: Value produced by the JSON loader.
        cluster: The cluster the workspace selected. A ledger written against
            one cluster and read against another fails here rather than
            reporting every job as unaccounted.

    Returns:
        Validated entry.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing,
            mistyped or empty.
        AppError: With ``PARTITION_UNKNOWN`` if the recorded partition is not
            one this cluster has.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"ledger entry must be a JSON object, got {type(value).__name__}")

    return LedgerEntry(
        job_id=_require_nonempty_str(value, "job_id"),
        project=require_project(value, "project"),
        name=_require_nonempty_str(value, "name"),
        host=_require_nonempty_str(value, "host"),
        partition=require_partition(cluster, value, "partition"),
        submitted_at=_require_nonempty_str(value, "submitted_at"),
        log_dir=_require_nonempty_str(value, "log_dir"),
    )


__all__ = ["LedgerEntry", "decode_ledger_entry", "encode_ledger_entry"]
