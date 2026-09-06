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
    require_bool,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.contracts.cluster import ClusterFacts, require_partition
from hpc3.contracts.experiment import encode_experiment, require_experiment
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
        deterministic: Whether the run was configured for kernel-level
            numerical determinism. Recorded because runs on either side of
            this setting are separate records: the deterministic loss is a
            different number from the nondeterministic one, so a comparison
            that crosses the boundary measures the setting rather than the
            thing under test. A ledger that did not carry it could not tell
            the two apart afterwards.
        experiment: What this run was -- corpus digest, seed, base model, or
            whatever identifies a run in this project. The fields above find
            the job; this one says which result it produced, which is the
            question asked months later when an outcome file needs tracing
            back to the run that made it.
        image_digest: Content digest of the image the payload was launched
            in. Recorded because the LEDGER is the index and the launcher is
            the only party that knows this: an image cannot compute its own
            digest from inside itself, so the digest exists here and in the
            job's ``--comment`` and nowhere else durable. Without it a reader
            can find which job produced a result and cannot say which
            software produced it, which is half an answer.

            THREE states, because there are three and collapsing any two
            would make the record assert something it does not know:

            - a digest -- the run was launched inside that image;
            - ``""`` -- the run was launched out of a directory environment,
              which is a positive fact, matching
              :data:`~platform_core.comparability.NO_VALUE` so it differs
              from every real digest rather than matching any of them;
            - ``None`` -- this row does not record it. Only rows written
              before the field existed carry this; :func:`~hpc3.core.submit`
              always writes one of the first two. A reader must not read it
              as "no image", because rows named ``...-v4`` and
              ``ka-probe-v5-...`` demonstrably ran inside images that
              nothing recorded.
        submitter: The agent-board label of the session that submitted the
            job, so a bridge announcing the job's terminal state on the
            board can tag the one party that is waiting for it. Recorded
            here because the ledger is the only durable record of who asked
            for a job: Slurm knows the cluster account, which is the same
            for every session on this machine.

            THREE states, mirroring ``image_digest`` for the same reason:

            - a label -- the submitting session declared its board label
              (``BOARD_AGENT_LABEL`` in its environment);
            - ``""`` -- the submitter was asked and declared no label,
              which is a positive fact: the job is announceable but there
              is nobody specific to tag;
            - ``None`` -- this row does not record it. Only rows written
              before the field existed carry this, backfilled in one
              auditable pass like the 122 pre-``artifact`` rows; every
              writer now records one of the first two.
        artifact: Where the run was TOLD to write its manifest, or None when
            the row does not name one -- whether because the run declared
            none or because it predates the field, which are the same thing
            to a reader: nobody said where the answer went. Request-side by
            construction, and that is the point of it living here: this
            ledger records what was asked for, while the manifest at that
            path records what happened. A reader follows job -> image ->
            artifact and then reads the fact.

            It is not taken on trust. ``decode_job_spec`` refuses a run whose
            declared artifact does not appear in its own command, because a
            declaration that drifts from the command points the index at a
            file nobody writes -- which is worse than no index, being a
            confident wrong answer.
    """

    job_id: str
    project: str
    name: str
    host: str
    partition: str
    submitted_at: str
    log_dir: str
    deterministic: bool
    experiment: dict[str, str]
    image_digest: str | None
    submitter: str | None
    artifact: str | None


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


def _require_str_or_null(obj: dict[str, JSONValue], key: str) -> str | None:
    """Read a field that MUST be present and may be null.

    Present-and-null is a row saying "I do not record this". Absent is a row
    that was never asked the question, and the two must not read alike: a
    writer that forgets the field would otherwise produce rows that decode
    as "unknown" forever, silently, which is how an index rots. So the key
    is required and only its VALUE may be null.

    The 122 rows written before this field existed were backfilled to
    explicit null in one auditable pass rather than tolerated here, because
    tolerance in the reader is permanent and a backfill is not.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The value, or None when the field is explicitly null.

    Raises:
        JSONTypeError: If the field is absent, or present and neither null
            nor a string.
    """
    if key not in obj:
        raise JSONTypeError(
            f"Field '{key}' is required; write null to record that this run does not name one"
        )
    value = obj[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise JSONTypeError(f"Field '{key}' must be a string or null, got {type(value).__name__}")
    return value


def _require_path_or_null(obj: dict[str, JSONValue], key: str) -> str | None:
    """Read a required path field whose value may be null but never empty.

    Null and empty are different claims and only one is meaningful here.
    Null says the row names no path. An empty string would say it names one
    and then names nowhere, which is a record that reads as an answer and is
    not one.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The value, or None when the field is explicitly null.

    Raises:
        JSONTypeError: If the field is absent, present and not a string or
            null, or present as the empty string.
    """
    value = _require_str_or_null(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must name a path or be null, not an empty string")
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
        "deterministic": entry["deterministic"],
        "experiment": encode_experiment(entry["experiment"]),
        "image_digest": entry["image_digest"],
        "submitter": entry["submitter"],
        "artifact": entry["artifact"],
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
        deterministic=require_bool(value, "deterministic"),
        experiment=require_experiment(value, "experiment"),
        image_digest=_require_str_or_null(value, "image_digest"),
        submitter=_require_str_or_null(value, "submitter"),
        artifact=_require_path_or_null(value, "artifact"),
    )


__all__ = ["LedgerEntry", "decode_ledger_entry", "encode_ledger_entry"]
