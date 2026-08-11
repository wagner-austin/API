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
    batch_results,
    batch_status,
    bootstrap,
    claim,
    finish,
    heartbeat,
    reap,
    reprioritize,
    retry_failed,
    submit,
)
from rw_bot.service.queue_rows import MatchServiceError
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


def test_a_lost_clone_race_takes_the_next_free_clone() -> None:
    """The production race: a rival's lease lands between read and insert.

    ``FOR UPDATE`` cannot lock a lease row that does not exist, so the
    insert itself arbitrates -- the loser reads back no row and moves to
    the next index instead of dying on the primary key, which is how
    navpair48's first minutes lost workers (log 2026-08-07).
    """
    conn = _submitted()
    conn.store.thief = (1, "rival", 99)
    held = _claimed(conn, "w1", (1, 2))
    assert held["clone_index"] == 2
    assert conn.store.leases[1] == ("rival", 99)
    assert conn.store.leases[2] == ("w1", held["job_id"])


def test_a_race_that_exhausts_the_pool_declines() -> None:
    """Losing the last free clone is a decline, never an exception."""
    conn = _submitted()
    conn.store.thief = (1, "rival", 99)
    assert claim(conn, "w1", (1,)) is None
    assert conn.store.jobs[0].state == "queued"
    assert conn.rollbacks == 1


def test_an_empty_queue_declines() -> None:
    conn = FakeConnection()
    bootstrap(conn)
    assert claim(conn, "w1", (1,)) is None
    assert conn.rollbacks == 1


def test_a_finish_releases_the_lease_and_records_the_outcome() -> None:
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    finish(conn, held["job_id"], True, "### alpha-s12345\nverdict        won (won)")
    assert conn.store.jobs[0].state == "done"
    assert conn.store.jobs[0].ok is True
    assert conn.store.jobs[0].card == "### alpha-s12345\nverdict        won (won)"
    assert conn.store.leases == {}
    _claimed(conn, "w1", (1,))


def test_a_failed_match_files_as_failed() -> None:
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    finish(conn, held["job_id"], False, "")
    assert conn.store.jobs[0].state == "failed"
    assert conn.store.jobs[0].ok is False
    assert conn.store.jobs[0].card == ""


def test_batch_results_read_the_mirrored_verdicts_in_order() -> None:
    """The paired-panel read: label, seed, state and verdict per match."""
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    finish(conn, held["job_id"], True, "### alpha-s12345\nverdict        wiped (wiped)")
    results = batch_results(conn, "demo")
    assert [r["seed"] for r in results] == [777, 12345]
    assert results[0] == {"label": "alpha", "seed": 777, "state": "queued", "verdict": ""}
    assert results[1] == {"label": "alpha", "seed": 12345, "state": "done", "verdict": "wiped"}


def test_a_card_without_a_verdict_line_reports_an_empty_verdict() -> None:
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    finish(conn, held["job_id"], True, "### alpha-s12345\nverdict")
    results = batch_results(conn, "demo")
    assert results[1]["verdict"] == ""


def test_an_unreadable_result_row_stops_loudly() -> None:
    conn = _submitted()
    conn.store.jobs[0].card = 7
    with pytest.raises(MatchServiceError) as caught:
        batch_results(conn, "demo")
    assert caught.value.code == "RW-SERVICE-001"


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


def test_an_unreadable_status_row_stops_loudly() -> None:
    conn = _submitted()
    conn.store.jobs[0].state = 7
    with pytest.raises(MatchServiceError) as caught:
        batch_status(conn, "demo")
    assert caught.value.code == "RW-SERVICE-001"


def test_a_state_the_service_never_writes_is_not_counted() -> None:
    """A foreign state neither crashes the count nor inflates a bucket."""
    conn = _submitted()
    conn.store.jobs[0].state = "paused"
    status = batch_status(conn, "demo")
    assert status["queued"] == 1
    assert status["running"] + status["done"] + status["failed"] == 0


def test_retry_failed_requeues_only_the_failed() -> None:
    """A failed row would stay failed forever under the resume key; the
    retry verb is how a bind collision's casualties play again."""
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    finish(conn, held["job_id"], False, "")
    assert retry_failed(conn, "demo") == 1
    row = conn.store.jobs[0]
    assert row.state == "queued"
    assert row.ok is None
    assert row.card == ""
    _claimed(conn, "w1", (1,))


def test_retry_failed_leaves_done_rows_alone() -> None:
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    finish(conn, held["job_id"], True, "### alpha-s12345\nverdict        won (won)")
    assert retry_failed(conn, "demo") == 0
    assert conn.store.jobs[0].state == "done"


def test_a_bumped_batch_claims_ahead_of_an_earlier_one() -> None:
    """The queue is first-come until priority says otherwise: a background
    data batch submitted an hour early must not make a strategic question
    wait behind it (log 2026-08-09). Lower runs sooner."""
    conn = _submitted()
    submit(
        conn, "urgent", _CONFIG, (parse_job_line("beta|999|doctrines/flame-nocover.doctrine|400"),)
    )
    assert reprioritize(conn, "urgent", 10) == 1
    held = _claimed(conn, "w1", (1,))
    assert held["batch"] == "urgent"


def test_reprioritize_moves_only_the_queued() -> None:
    conn = _submitted()
    held = _claimed(conn, "w1", (1,))
    assert reprioritize(conn, "demo", 10) == 1
    running = next(r for r in conn.store.jobs if r.job_id == held["job_id"])
    assert running.priority == 100


def test_a_label_scoped_bump_moves_only_that_arm() -> None:
    """The paired-panel case: the interesting half need not wait behind
    its own controls (log 2026-08-10)."""
    conn = _submitted()
    submit(
        conn, "demo", _CONFIG, (parse_job_line("beta|999|doctrines/flame-nocover.doctrine|400"),)
    )
    assert reprioritize(conn, "demo", 10, "beta") == 1
    held = _claimed(conn, "w1", (1,))
    assert held["job"]["label"] == "beta"
