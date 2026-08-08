"""The queue's statements, asserted against a store that actually stores.

What is tested: submission is idempotent on the batch-label-seed key, a
claim takes the oldest queued job and a free clone atomically, exhausted
pools and empty queues decline without claiming, the lifecycle rows move as
the worker reports, and every unreadable row stops the service loudly.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.match import decode_match_config
from rw_bot.harness.runner import decode_sweep_config
from rw_bot.harness.sweep import parse_job_line
from rw_bot.service.queue import (
    ClaimedJob,
    MatchServiceError,
    bootstrap,
    claim,
    finish,
    heartbeat,
    reap,
    submit,
)
from tests.service_fakes import FakeConnection

_MATCH = decode_match_config(
    {"map_path": "maps/skirmish/[p2]duel_lake.tmx", "opponents": 1, "difficulty": 2}
)

_CONFIG = decode_sweep_config(
    {
        "out_dir": "runs/sweeps/demo",
        "workers": 1,
        "lockstep": 75,
        "clone_prefix": ".game-w",
        "source_game_dir": ".game",
        "tree": "runs/sweeps/demo/.tree",
        "pin_delta": 3,
        "fast_forward": 10,
    },
    _MATCH,
)

_JOBS = (
    parse_job_line("alpha|12345|doctrines/flame-nocover.doctrine|400"),
    parse_job_line("alpha|777|doctrines/flame-nocover.doctrine|400"),
)


def _submitted() -> FakeConnection:
    conn = FakeConnection()
    bootstrap(conn)
    submit(conn, "demo", _CONFIG, _JOBS)
    return conn


def _claimed(conn: FakeConnection, worker: str, pool: tuple[int, ...]) -> ClaimedJob:
    """Claim, insisting the queue does not decline.

    Args:
        conn: The fake connection under test.
        worker: The claiming worker's name.
        pool: The clone indices offered.

    Returns:
        The claim.

    Raises:
        AssertionError: When the queue declines -- the narrowing the tests
            need, stated once instead of as a weak assertion per site.
    """
    held = claim(conn, worker, pool)
    if held is None:
        raise AssertionError("expected a claim and the queue declined")
    return held


def test_resubmission_queues_only_what_the_first_missed() -> None:
    """The batch-label-seed key is the same resume semantics sweeps have."""
    conn = _submitted()
    assert submit(conn, "demo", _CONFIG, _JOBS) == 0
    assert len(conn.store.jobs) == 2


def test_a_claim_takes_the_oldest_job_and_a_free_clone() -> None:
    conn = _submitted()
    held = _claimed(conn, "w1", (1, 2))
    assert held["job"]["seed"] == 12345
    assert held["clone_index"] == 1
    assert held["config"]["fast_forward"] == 10
    match = held["config"]["match"]
    if match is None:
        raise AssertionError("the stored match vanished on the round trip")
    assert match["difficulty"] == 2
    assert conn.store.jobs[0].state == "running"


def test_the_decoded_claim_round_trips_the_submitted_job() -> None:
    """What comes back is what went in, through the harness's own codecs."""
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    assert held["job"] == _JOBS[0]
    stored = dict(held["config"])
    submitted = dict(_CONFIG)
    assert stored == submitted


def test_two_claims_take_two_jobs_and_two_clones() -> None:
    conn = _submitted()
    first = _claimed(conn, "w1", (1, 2))
    second = _claimed(conn, "w2", (1, 2))
    assert {first["clone_index"], second["clone_index"]} == {1, 2}
    assert {first["job"]["seed"], second["job"]["seed"]} == {12345, 777}


def test_an_exhausted_clone_pool_declines_without_claiming() -> None:
    """A job must never run without a leased slot to run in."""
    conn = _submitted()
    _claimed(conn, "w1", (1,))
    assert claim(conn, "w2", (1,)) is None
    assert conn.store.jobs[1].state == "queued"
    assert conn.rollbacks == 1


def test_an_empty_queue_declines() -> None:
    conn = FakeConnection()
    bootstrap(conn)
    assert claim(conn, "w1", (1,)) is None
    assert conn.rollbacks == 1


def test_a_finish_releases_the_lease_and_records_the_outcome() -> None:
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    finish(conn, held["job_id"], True)
    assert conn.store.jobs[0].state == "done"
    assert conn.store.jobs[0].ok is True
    assert conn.store.leases == {}
    _claimed(conn, "w1", (1,))


def test_a_failed_match_files_as_failed() -> None:
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    finish(conn, held["job_id"], False)
    assert conn.store.jobs[0].state == "failed"
    assert conn.store.jobs[0].ok is False


def test_a_heartbeat_resets_the_staleness_clock() -> None:
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    conn.store.jobs[0].heartbeat_age = 9000
    heartbeat(conn, held["job_id"])
    assert reap(conn, 5400) == 0


def test_the_reaper_requeues_the_silent_and_frees_their_clones() -> None:
    """A worker that died mid-match returns its job and its slot."""
    conn = _submitted()
    _claimed(conn, "w1", (1,))
    conn.store.jobs[0].heartbeat_age = 9000
    assert reap(conn, 5400) == 1
    assert conn.store.jobs[0].state == "queued"
    assert conn.store.leases == {}


def test_a_fresh_claim_is_not_reaped() -> None:
    conn = _submitted()
    _claimed(conn, "w1", (1,))
    assert reap(conn, 5400) == 0


def test_a_batch_without_a_match_round_trips_none() -> None:
    """Sandbox batches carry no match, and the queue must not invent one."""
    config = decode_sweep_config(
        {
            "out_dir": "runs/sweeps/demo2",
            "workers": 1,
            "lockstep": 75,
            "clone_prefix": ".game-w",
            "source_game_dir": ".game",
            "tree": "runs/sweeps/demo2/.tree",
            "pin_delta": 0,
            "fast_forward": 0,
        },
        None,
    )
    conn = FakeConnection()
    bootstrap(conn)
    submit(conn, "demo2", config, (_JOBS[0],))
    held = _claimed(conn, "w1", (1,))
    assert held["config"]["match"] is None


def test_an_unreadable_claim_row_stops_loudly() -> None:
    conn = _submitted()
    conn.store.jobs[0].batch = 7
    with pytest.raises(MatchServiceError) as caught:
        claim(conn, "w1", (1,))
    assert caught.value.code == "RW-SERVICE-001"


def test_an_unreadable_match_field_stops_loudly() -> None:
    """A stored match carries only paths and counts; anything else is rot."""
    conn = _submitted()
    conn.store.jobs[0].match = '{"map_path":"m","opponents":1.5,"difficulty":2}'
    with pytest.raises(MatchServiceError) as caught:
        claim(conn, "w1", (1,))
    assert caught.value.code == "RW-SERVICE-001"


def test_an_unreadable_lease_row_stops_loudly() -> None:
    conn = _submitted()
    conn.store.leases["bad"] = ("w9", 1)
    with pytest.raises(MatchServiceError) as caught:
        claim(conn, "w1", (1,))
    assert caught.value.code == "RW-SERVICE-001"


def test_an_unreadable_reaped_row_stops_loudly() -> None:
    """The reaper's RETURNING rows are ids or the queue is rotten."""
    conn = _submitted()
    conn.store.poison_reap_row = ("seven",)
    with pytest.raises(MatchServiceError) as caught:
        reap(conn, 5400)
    assert caught.value.code == "RW-SERVICE-001"
