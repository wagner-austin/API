"""The lease file's turn across processes (MCPs board tasks b1089284 and 1e0ee821).

Every node runner is its own process on the hub and all of them share one
lease file. A rewrite reads, checks and writes in the file's turn; a read
takes none, since every rewrite renames a whole file over the path. Each
case runs against a real lease file and a real lock beside it: another
process's turn is a lock file this process did not create, and its writing
is what the case's sleep does while the rewrite waits.
"""

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Callable

import pytest
from platform_core.json_utils import dump_json_str

from fleet.contracts.lease import Lease, encode_lease
from fleet.core import _test_hooks, file_turn, leases

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


def _sleeping(steps: list[Callable[[], None]], slept: list[float]) -> _test_hooks.SleepProtocol:
    """A sleep that records each wait and runs the next step in it.

    Args:
        steps: Run in turn, one per sleep.
        slept: Every wait asked for, appended in order.

    Returns:
        The sleep.
    """

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        steps.pop(0)()

    return sleep


def test_a_read_leaves_no_lock_behind(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "leases.json"
    _write(path, _lease("run-1"))

    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-1"]
    assert not leases.lock_path(path).exists()
    assert leases.lock_path(path) == tmp_path / "leases.json.lock"


def test_a_read_takes_no_turn_while_another_process_holds_one(tmp_path: pathlib.Path) -> None:
    """Every rewrite renames a whole file over the path, so a read never waits
    (MCPs board task 1e0ee821): with another process's turn held it reads the
    last whole document at once and leaves that process's lock alone."""
    path = tmp_path / "leases.json"
    _test_hooks.write_text(path, dump_json_str([encode_lease(_lease("run-2"))]))
    lock = leases.lock_path(path)
    lock.write_text("fleet process 4242 pid 4242 turn writing", encoding="utf-8")
    slept: list[float] = []
    _test_hooks.sleep = _sleeping([], slept)

    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-2"]
    assert slept == []
    assert lock.read_text(encoding="utf-8") == "fleet process 4242 pid 4242 turn writing"


def test_a_rewrite_waits_out_another_processs_turn_and_rewrites_what_it_left(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A release reads, checks and writes in one turn, so it waits while
    another process holds the turn and rewrites the file that process left,
    losing neither process's change."""
    path = tmp_path / "leases.json"
    _write(path, _lease("run-6"))
    lock = leases.lock_path(path)
    lock.write_text("fleet process 4242 pid 4242 turn writing", encoding="utf-8")
    clock = [int(time.time())]
    slept: list[float] = []

    def the_other_process_finishes_its_rewrite() -> None:
        _write(path, _lease("run-6"), _lease("run-7"))
        lock.unlink()
        clock[0] += 1

    _test_hooks.now = lambda: clock[0]
    _test_hooks.sleep = _sleeping([the_other_process_finishes_its_rewrite], slept)
    with caplog.at_level("INFO"):
        leases.release(path, run_id="run-6", now_unix=_NOW)

    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-7"]
    assert slept == [file_turn.FIRST_LOOK_SECONDS]
    assert (
        f"fleet process {os.getpid()} took its lease turn after waiting 1 s for another "
        "process rewriting the lease file"
    ) in [record.getMessage() for record in caplog.records]
    assert not lock.exists()


def test_a_turn_left_by_a_process_that_died_inside_it_is_taken_away(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "leases.json"
    _write(path, _lease("run-3"), _lease("run-8"))
    lock = leases.lock_path(path)
    lock.write_text("fleet process 4243 pid 4243 turn dead", encoding="utf-8")
    died = time.time() - leases.LEASE_TURN["stale_seconds"] - 5
    os.utime(lock, (died, died))

    leases.release(path, run_id="run-3", now_unix=_NOW)

    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-8"]
    assert not lock.exists()


def test_a_rewrite_reads_and_writes_in_one_turn_without_waiting_on_itself(
    tmp_path: pathlib.Path,
) -> None:
    """release_if_held reads, then release reads and writes, all nested in
    one turn: a turn that waited on its own lock would sleep here."""
    path = tmp_path / "leases.json"
    _write(path, _lease("run-4"), _lease("run-5"))
    slept: list[float] = []
    _test_hooks.sleep = _sleeping([], slept)

    said = leases.release_if_held(path, run_id="run-4", now_unix=_NOW)

    assert said == "lease released"
    assert slept == []
    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-5"]
    assert not leases.lock_path(path).exists()
