"""The lease file's turn across processes (MCPs board task b1089284).

Every node runner is its own process on the hub and all of them share one
lease file, rewritten in place. On 2026-10-11 serendipity's runner read it
mid-write and its serving loop ended on ``InvalidJsonError``. Each case runs
against a real lease file and a real lock beside it: another process's turn
is a lock file this process did not create, and its writing is what the
case's sleep does while the reader waits.
"""

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Callable

import pytest
from platform_core.json_utils import dump_json_str

from fleet.contracts.lease import Lease, encode_lease
from fleet.core import _test_hooks, leases

_NOW = 1_791_745_210


def _lease(run_id: str) -> Lease:
    """A live lease of Model-Trainer on lavender.

    Args:
        run_id: The dispatch holding it.

    Returns:
        The lease.
    """
    return Lease(
        node="lavender",
        project="services/Model-Trainer",
        run_id=run_id,
        agent="opus-mcps-1008-3057",
        session_id="3057e09c-99bb-43fb-bd93-6573cf5f8c6e",
        acquired_unix=_NOW,
        expires_unix=_NOW + 600,
        resources=(),
    )


def _write(path: pathlib.Path, *held: Lease) -> None:
    """Write a whole lease file, as another process's rewrite does.

    Args:
        path: The lease file.
        *held: The leases it holds.
    """
    path.write_text(dump_json_str([encode_lease(entry) for entry in held]), encoding="utf-8")


def _sleeping(steps: list[Callable[[], None]], slept: list[int]) -> _test_hooks.SleepProtocol:
    """A sleep that records each wait and runs the next step in it.

    Args:
        steps: Run in turn, one per sleep.
        slept: Every wait asked for, appended in order.

    Returns:
        The sleep.
    """

    def sleep(seconds: int) -> None:
        slept.append(seconds)
        steps.pop(0)()

    return sleep


def test_a_read_leaves_no_lock_behind(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "leases.json"
    _write(path, _lease("run-1"))

    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-1"]
    assert not leases.lock_path(path).exists()
    assert leases.lock_path(path) == tmp_path / "leases.json.lock"


def test_a_read_waits_out_another_processs_rewrite_and_reads_it_whole(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The torn read of 2026-10-11, made the filesystem's: the file holds half
    a document while the other process's turn is held, and the reader reads
    only after that turn is given back."""
    path = tmp_path / "leases.json"
    path.write_text('[{"node": "lav', encoding="utf-8")
    lock = leases.lock_path(path)
    lock.write_text("fleet process 4242 pid 4242 turn writing", encoding="utf-8")
    clock = [int(time.time())]
    slept: list[int] = []

    def the_other_process_finishes_its_rewrite() -> None:
        _write(path, _lease("run-2"))
        lock.unlink()
        clock[0] += 1

    _test_hooks.now = lambda: clock[0]
    _test_hooks.sleep = _sleeping([the_other_process_finishes_its_rewrite], slept)
    with caplog.at_level("INFO"):
        held = leases.read_leases(path)

    assert [lease["run_id"] for lease in held] == ["run-2"]
    assert slept == [leases.LEASE_TURN["poll_seconds"]]
    assert (
        f"fleet process {os.getpid()} took its lease turn after waiting 1 s for another "
        "process reading or rewriting the lease file"
    ) in [record.getMessage() for record in caplog.records]


def test_a_turn_left_by_a_process_that_died_inside_it_is_taken_away(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "leases.json"
    _write(path, _lease("run-3"))
    lock = leases.lock_path(path)
    lock.write_text("fleet process 4243 pid 4243 turn dead", encoding="utf-8")
    died = time.time() - leases.LEASE_TURN["stale_seconds"] - 5
    os.utime(lock, (died, died))

    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-3"]
    assert not lock.exists()


def test_a_rewrite_reads_and_writes_in_one_turn_without_waiting_on_itself(
    tmp_path: pathlib.Path,
) -> None:
    """release_if_held reads, then release reads and writes, all nested in
    one turn: a turn that waited on its own lock would sleep here."""
    path = tmp_path / "leases.json"
    _write(path, _lease("run-4"), _lease("run-5"))
    slept: list[int] = []
    _test_hooks.sleep = _sleeping([], slept)

    said = leases.release_if_held(path, run_id="run-4", now_unix=_NOW)

    assert said == "lease released"
    assert slept == []
    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-5"]
    assert not leases.lock_path(path).exists()
