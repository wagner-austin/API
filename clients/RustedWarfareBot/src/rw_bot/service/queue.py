"""The job table and the lease table, and every statement that touches them.

One transaction shape carries the whole design ([[harness-match-service]]):
a worker claims the oldest queued job with ``FOR UPDATE SKIP LOCKED`` -- two
workers can never claim one job, and neither ever waits -- and inside the
same transaction takes a lease on a free clone index. The lease is what
makes "the engine slot" a resource instead of a discipline: the allocator
owns clone indices, not convention.

Rows go in through the harness's own codecs and come out through them, so a
job read back from the queue is validated exactly as a job parsed from a
sweep file would be. No JSON library object crosses this module's boundary
undecoded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from rw_bot import RwBotError
from rw_bot.harness.match import MatchConfig, decode_match_config, encode_match_config
from rw_bot.harness.runner import SweepConfig, decode_sweep_config, encode_sweep_config
from rw_bot.harness.sweep import SweepJob, decode_sweep_job, encode_sweep_job
from rw_bot.service._test_hooks import Connection
from rw_bot.wire.ndjson import parse_object, render_json


class MatchServiceError(RwBotError):
    """The queue answered with a shape the service cannot read.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description, naming what was malformed.
    """


class BatchStatus(TypedDict):
    """One batch's rows, counted by state.

    Attributes:
        batch: The batch asked about.
        queued: Matches not yet claimed.
        running: Matches a worker holds.
        done: Matches that produced their scorecard.
        failed: Matches that did not.
    """

    batch: str
    queued: int
    running: int
    done: int
    failed: int


class ClaimedJob(TypedDict):
    """One match a worker holds, with everything needed to play it.

    Attributes:
        job_id: The queue row, for heartbeats and the finish.
        batch: The sweep name artifacts file under.
        clone_index: The leased engine slot; the worker's to use until
            :func:`finish` releases it.
        config: The batch configuration, decoded and validated.
        job: The match, decoded and validated.
    """

    job_id: int
    batch: str
    clone_index: int
    config: SweepConfig
    job: SweepJob


#: The schema, applied idempotently at worker start. Text columns for the
#: encoded config and job rather than JSONB operators, because nothing
#: queries inside them -- the codecs are the read path.
_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS match_jobs (
        id BIGSERIAL PRIMARY KEY,
        batch TEXT NOT NULL,
        label TEXT NOT NULL,
        seed BIGINT NOT NULL,
        config TEXT NOT NULL,
        match TEXT NOT NULL DEFAULT '',
        job TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'queued',
        worker TEXT NOT NULL DEFAULT '',
        clone_index INT NOT NULL DEFAULT -1,
        heartbeat_at TIMESTAMPTZ,
        finished_at TIMESTAMPTZ,
        ok BOOLEAN,
        UNIQUE (batch, label, seed)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clone_leases (
        clone_index INT PRIMARY KEY,
        worker TEXT NOT NULL,
        job_id BIGINT NOT NULL,
        leased_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
)


def bootstrap(conn: Connection) -> None:
    """Apply the schema, idempotently.

    Args:
        conn: An open connection; committed on success.
    """
    cursor = conn.cursor()
    for statement in _DDL:
        cursor.execute(statement)
    conn.commit()


def submit(conn: Connection, batch: str, config: SweepConfig, jobs: tuple[SweepJob, ...]) -> int:
    """Queue a batch's matches, skipping any already queued.

    Resubmission is deliberately safe: the ``(batch, label, seed)`` key makes
    a second submission of one batch enqueue only what the first missed,
    which is the same resume semantics sweeps have on disk.

    Args:
        conn: An open connection; committed on success.
        batch: The sweep name artifacts will file under.
        config: The batch configuration every job shares.
        jobs: The matches.

    Returns:
        How many rows were newly queued.
    """
    cursor = conn.cursor()
    encoded_config = render_json(encode_sweep_config(config))
    match = config["match"]
    encoded_match = "" if match is None else render_json(encode_match_config(match))
    queued = 0
    for job in jobs:
        cursor.execute(
            "INSERT INTO match_jobs (batch, label, seed, config, match, job)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (batch, label, seed) DO NOTHING"
            " RETURNING id",
            (
                batch,
                job["label"],
                job["seed"],
                encoded_config,
                encoded_match,
                render_json(encode_sweep_job(job)),
            ),
        )
        if cursor.fetchone() is not None:
            queued += 1
    conn.commit()
    return queued


def claim(conn: Connection, worker: str, clone_pool: tuple[int, ...]) -> ClaimedJob | None:
    """Claim the oldest queued job and lease a clone for it, atomically.

    ``FOR UPDATE SKIP LOCKED`` on both tables is the whole trick: two
    workers never claim one job or one clone, and neither ever blocks on
    the other's transaction.

    Args:
        conn: An open connection; committed on a claim, rolled back when
            there is nothing to do.
        worker: This worker's name, recorded on the row and the lease.
        clone_pool: The clone indices this worker may use.

    Returns:
        The claimed match, or None when the queue is empty or every clone
        in the pool is leased.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` when a claimed row decodes to
            a shape the codecs reject -- a poisoned queue is a loud stop,
            not a skipped job.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, batch, config, match, job FROM match_jobs"
        " WHERE state = 'queued' ORDER BY id"
        " LIMIT 1 FOR UPDATE SKIP LOCKED"
    )
    row = cursor.fetchone()
    if row is None:
        conn.rollback()
        return None
    job_id, batch, config_text, match_text, job_text = _claim_columns(row)
    cursor.execute("SELECT clone_index FROM clone_leases FOR UPDATE")
    leased = {_lease_index(held) for held in cursor.fetchall()}
    free = tuple(index for index in clone_pool if index not in leased)
    if not free:
        conn.rollback()
        return None
    clone_index = free[0]
    cursor.execute(
        "INSERT INTO clone_leases (clone_index, worker, job_id) VALUES (%s, %s, %s)",
        (clone_index, worker, job_id),
    )
    cursor.execute(
        "UPDATE match_jobs SET state = 'running', worker = %s, clone_index = %s,"
        " heartbeat_at = now() WHERE id = %s",
        (worker, clone_index, job_id),
    )
    conn.commit()
    match: MatchConfig | None = None
    if match_text != "":
        match = decode_match_config(_match_payload(parse_object(match_text)))
    return ClaimedJob(
        job_id=job_id,
        batch=batch,
        clone_index=clone_index,
        config=decode_sweep_config(parse_object(config_text), match),
        job=decode_sweep_job(parse_object(job_text)),
    )


def batch_status(conn: Connection, batch: str) -> BatchStatus:
    """Count one batch's rows by state.

    Args:
        conn: An open connection; rolled back after the read.
        batch: The batch name.

    Returns:
        The counts, zero for states the batch has none of.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` when a count row is
            unreadable.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT state, count(*) FROM match_jobs WHERE batch = %s GROUP BY state",
        (batch,),
    )
    counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
    for row in cursor.fetchall():
        state, count = _status_columns(row)
        if state in counts:
            counts[state] = count
    conn.rollback()
    return BatchStatus(
        batch=batch,
        queued=counts["queued"],
        running=counts["running"],
        done=counts["done"],
        failed=counts["failed"],
    )


def heartbeat(conn: Connection, job_id: int) -> None:
    """Record that the worker holding a job is alive.

    Args:
        conn: An open connection; committed.
        job_id: The running job.
    """
    cursor = conn.cursor()
    cursor.execute("UPDATE match_jobs SET heartbeat_at = now() WHERE id = %s", (job_id,))
    conn.commit()


def finish(conn: Connection, job_id: int, ok: bool) -> None:
    """Record a job's outcome and release its clone lease.

    Args:
        conn: An open connection; committed.
        job_id: The finished job.
        ok: Whether the match produced its scorecard.
    """
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE match_jobs SET state = %s, ok = %s, finished_at = now() WHERE id = %s",
        ("done" if ok else "failed", ok, job_id),
    )
    cursor.execute("DELETE FROM clone_leases WHERE job_id = %s", (job_id,))
    conn.commit()


def reap(conn: Connection, stale_seconds: int) -> int:
    """Requeue every running job whose heartbeat went silent.

    A worker that died mid-match leaves a running row and a held lease; this
    returns both to the pool. The threshold is generous by design -- phase
    zero heartbeats once at claim, so it must exceed the longest match
    (a capped 10,000-sample match at realtime runs about 42 minutes).

    Args:
        conn: An open connection; committed.
        stale_seconds: Silence, in seconds, after which a job is orphaned.

    Returns:
        How many jobs were requeued.
    """
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE match_jobs SET state = 'queued', worker = '', clone_index = -1"
        " WHERE state = 'running'"
        " AND heartbeat_at < now() - make_interval(secs => %s)"
        " RETURNING id",
        (stale_seconds,),
    )
    orphaned = tuple(_reaped_id(row) for row in cursor.fetchall())
    for job_id in orphaned:
        cursor.execute("DELETE FROM clone_leases WHERE job_id = %s", (job_id,))
    conn.commit()
    return len(orphaned)


def _claim_columns(row: Sequence[str | int]) -> tuple[int, str, str, str, str]:
    """Validate the claim query's row shape.

    Args:
        row: What ``fetchone`` returned.

    Returns:
        The id, batch, encoded config, encoded match and encoded job.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if (
        len(row) == 5
        and isinstance(row[0], int)
        and isinstance(row[1], str)
        and isinstance(row[2], str)
        and isinstance(row[3], str)
        and isinstance(row[4], str)
    ):
        return row[0], row[1], row[2], row[3], row[4]
    raise MatchServiceError("RW-SERVICE-001", f"claim row has an unreadable shape: {row!r}")


def _match_payload(parsed: dict[str, str | int | float | bool]) -> dict[str, str | int]:
    """Narrow a parsed match payload to the types its codec reads.

    Args:
        parsed: The stored match object, parsed.

    Returns:
        The same mapping, with every value proven ``str`` or ``int``.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` when a value is neither --
            a stored match carries only paths and counts, so anything else
            is corruption.
    """
    narrowed: dict[str, str | int] = {}
    for key, value in parsed.items():
        if not isinstance(value, (str, int)):
            raise MatchServiceError(
                "RW-SERVICE-001", f"match field {key} has an unreadable type: {value!r}"
            )
        narrowed[key] = value
    return narrowed


def _status_columns(row: Sequence[str | int]) -> tuple[str, int]:
    """Validate one state-count row.

    Args:
        row: One row of the status query.

    Returns:
        The state and its count.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if len(row) == 2 and isinstance(row[0], str) and isinstance(row[1], int):
        return row[0], row[1]
    raise MatchServiceError("RW-SERVICE-001", f"status row has an unreadable shape: {row!r}")


def _lease_index(row: Sequence[str | int]) -> int:
    """Validate one lease row's clone index.

    Args:
        row: One row of the lease query.

    Returns:
        The leased clone index.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if len(row) == 1 and isinstance(row[0], int):
        return row[0]
    raise MatchServiceError("RW-SERVICE-001", f"lease row has an unreadable shape: {row!r}")


def _reaped_id(row: Sequence[str | int]) -> int:
    """Validate one requeued job id.

    Args:
        row: One row of the reap statement's RETURNING clause.

    Returns:
        The requeued job's id.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` on any other shape.
    """
    if len(row) == 1 and isinstance(row[0], int):
        return row[0]
    raise MatchServiceError("RW-SERVICE-001", f"reaped row has an unreadable shape: {row!r}")
