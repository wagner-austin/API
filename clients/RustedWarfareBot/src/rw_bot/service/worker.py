"""The match worker: claim, lease, play, file, repeat.

The half of [[harness-match-service]] that touches an engine. Everything the
worker does to a match it does through the harness seams sweeps already use
-- ``prepare_tree``, ``prepare_clone``, ``play_job`` -- so artifacts file
exactly where a sweep files them and every existing reader keeps working.
What changes is only who owns the clone index: the lease table, not a
convention.

Phase-zero heartbeat honesty: the worker beats once at claim and once per
loop turn, not during a match -- ``play_job`` blocks for the match's
duration. The reaper's threshold accounts for that; see
:func:`rw_bot.service.queue.reap`.
"""

from __future__ import annotations

from rw_bot.service import _test_hooks
from rw_bot.service.queue import bootstrap, claim, finish, reap

#: Seconds of heartbeat silence after which the reaper requeues a job. Must
#: exceed the longest match: a capped 10,000-sample match at realtime runs
#: about 42 minutes, so 90 minutes says "dead", not "slow".
STALE_SECONDS = 5400

#: Seconds between polls of an empty queue.
POLL_SECONDS = 15.0


def run_worker(
    dsn: str,
    worker: str,
    clone_pool: tuple[int, ...],
    max_jobs: int,
) -> int:
    """Play queued matches until the queue stays empty or a budget is spent.

    Args:
        dsn: The queue database, as a libpq connection string.
        worker: This worker's name, recorded on rows and leases.
        clone_pool: The clone indices this worker may lease.
        max_jobs: Stop after this many matches, or zero to run until the
            queue is empty.

    Returns:
        How many matches were played.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` when a claimed row is
            unreadable; the queue is poisoned and stopping loudly beats
            skipping quietly.
    """
    conn = _test_hooks.connect(dsn)
    bootstrap(conn)
    played = 0
    idle = False
    while True:
        reap(conn, STALE_SECONDS)
        held = claim(conn, worker, clone_pool)
        if held is None:
            if idle:
                break
            idle = True
            _test_hooks.sleep(POLL_SECONDS)
            continue
        idle = False
        _test_hooks.prepare_tree(held["config"])
        game_dir = _test_hooks.prepare_clone(held["clone_index"], held["config"])
        ok = _test_hooks.play_job(held["job"], game_dir, held["config"])
        finish(conn, held["job_id"], ok)
        played += 1
        if max_jobs and played >= max_jobs:
            break
    conn.close()
    return played
