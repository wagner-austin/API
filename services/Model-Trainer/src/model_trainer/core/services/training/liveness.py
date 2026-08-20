"""Deciding whether the worker handling a run is still alive.

A run's stored status records what its worker last wrote. It cannot record that
the worker stopped existing: a container killed or recreated mid-training runs
no code on the way out, so nothing moves the status off ``processing``. RQ does
not help either -- an observed kill produced no completion, no failure and no
requeue -- so any scheme that depends on the dying process running something
cannot cover this case.

The only evidence that separates a live run from a dead one is the heartbeat,
which the trainer stamps with wall-clock time every ten steps. This module owns
the single predicate that reads it, so the status endpoint and the resume gate
cannot drift apart on what "dead" means.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from platform_core.job_types import JobStatusLiteral

WORKER_HEARTBEAT_TIMEOUT_SECONDS: Final[float] = 1800.0
"""How long a running job may go silent before its worker is presumed dead.

Thirty minutes, chosen against the measured legitimate silences rather than
picked round. A run is quiet only while it is doing something that does not
step: the trainer heartbeats every ten steps, so mid-epoch gaps are seconds.
The long ones are inter-epoch evaluation and the end-of-run artifact upload,
and the worst upload observed took 8 minutes (12:38:25 to 12:46:31 on run
``hf_lm-medium-...-97bc90d9``, a 1.32 GB artifact). Corpus fetch before the
first heartbeat is the other quiet window.

Thirty minutes is therefore roughly 3.75x the worst measured silence, which
leaves room for a larger artifact or a slower fetch without a false positive.
The failure it guards against was 297 minutes stale, so the band between "still
working" and "obviously dead" is wide and this threshold sits well inside it.

Raise it if a legitimate silence ever approaches it -- a false positive here
tells an operator a healthy run died, which is worse than noticing a dead one
late.
"""


def seconds_since_last_sign_of_life(
    *,
    last_heartbeat_ts: float | None,
    status_updated_at: datetime,
    now_ts: float,
) -> float:
    """Age of the most recent evidence that a run's worker was alive.

    The heartbeat is preferred when present. Before the first heartbeat -- the
    window covering corpus fetch and model setup -- the only evidence is the
    moment the status was last written, so that is used instead. Without this
    fallback a job killed during setup would look infinitely alive, because a
    heartbeat that was never written cannot go stale.

    Args:
        last_heartbeat_ts: Wall-clock stamp of the last heartbeat, or None when
            the run has not reached its first one.
        status_updated_at: When the job store last wrote this run's status.
        now_ts: Current wall-clock time, injected so the caller owns the clock.

    Returns:
        Seconds elapsed since the most recent sign of life. Never negative: a
        stamp in the future reads as zero age, because a clock skew between the
        worker and the API must not be reported as staleness.
    """
    if last_heartbeat_ts is not None:
        return max(0.0, now_ts - last_heartbeat_ts)
    updated_ts = status_updated_at.replace(tzinfo=UTC).timestamp()
    return max(0.0, now_ts - updated_ts)


def worker_has_died(
    *,
    status: JobStatusLiteral,
    last_heartbeat_ts: float | None,
    status_updated_at: datetime,
    now_ts: float,
    timeout_seconds: float,
) -> bool:
    """Whether a run is claiming to run while its worker has gone silent too long.

    Only ``processing`` runs can be in this state. A queued run has no worker
    yet and a terminal run does not need one, so neither is judged by the
    heartbeat.

    Args:
        status: The run's stored job status.
        last_heartbeat_ts: Wall-clock stamp of the last heartbeat, or None.
        status_updated_at: When the job store last wrote this run's status.
        now_ts: Current wall-clock time, injected so the caller owns the clock.
        timeout_seconds: Silence permitted before the worker is presumed dead.
            Pass :data:`WORKER_HEARTBEAT_TIMEOUT_SECONDS` in production.

    Returns:
        True when the run says it is processing but nothing has signalled life
        within the timeout.
    """
    if status != "processing":
        return False
    age = seconds_since_last_sign_of_life(
        last_heartbeat_ts=last_heartbeat_ts,
        status_updated_at=status_updated_at,
        now_ts=now_ts,
    )
    return age > timeout_seconds


def worker_death_message(*, run_id: str, silent_for_seconds: float) -> str:
    """Operator-facing explanation that a run's worker died.

    Phrased as worker death rather than training failure because the two have
    different remedies: training failure is a property of the run, worker death
    is a property of the machine, and a run killed this way usually has a
    checkpoint and can resume.

    Args:
        run_id: The run whose worker went silent.
        silent_for_seconds: Age of its last sign of life.

    Returns:
        A message naming the run, the silence, and the remedy.
    """
    minutes = silent_for_seconds / 60.0
    return (
        f"the worker training run '{run_id}' stopped signalling {minutes:.1f} minutes ago "
        f"and is presumed dead; the run did not fail on its own, so if a checkpoint "
        f"exists it can be resumed"
    )


__all__ = [
    "WORKER_HEARTBEAT_TIMEOUT_SECONDS",
    "seconds_since_last_sign_of_life",
    "worker_death_message",
    "worker_has_died",
]
