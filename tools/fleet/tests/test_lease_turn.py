"""The lease file's turn across processes (MCPs board tasks b1089284, 1e0ee821 and 5280a25d).

Every node runner is its own process on the hub and all of them share one
lease file. A rewrite reads, checks and writes in the file's turn; a read
takes none, since every rewrite renames a whole file over the path. Each
case runs against a real lease file and a real lock beside it: another
process's turn is a real ``flock`` on that lock through a descriptor this
case opened (:class:`tests._held_turns.OtherTurn`), and its writing is what
that turn does before it lets go.
"""

from __future__ import annotations

import os
import pathlib
import re
import threading
import time

import pytest
from platform_core.json_utils import dump_json_str

from fleet.contracts.lease import Lease, encode_lease
from fleet.core import _test_hooks, leases
from tests._held_turns import OtherTurn

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


def test_a_read_takes_no_turn(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "leases.json"
    _write(path, _lease("run-1"))

    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-1"]
    assert not leases.lock_path(path).exists()
    assert leases.lock_path(path) == tmp_path / "leases.json.turn"


def test_a_read_takes_no_turn_while_another_process_holds_one(tmp_path: pathlib.Path) -> None:
    """Every rewrite renames a whole file over the path, so a read never waits
    (MCPs board task 1e0ee821): with another process's turn held it reads the
    last whole document at once."""
    path = tmp_path / "leases.json"
    _test_hooks.write_text(path, dump_json_str([encode_lease(_lease("run-2"))]))
    other = OtherTurn(leases.lock_path(path))
    asked = time.monotonic()

    read = [lease["run_id"] for lease in leases.read_leases(path)]

    took = time.monotonic() - asked
    other.end_after(0).join()
    assert read == ["run-2"]
    assert took < 0.5, f"a read took {took:.3f} s"


def test_a_rewrite_waits_out_another_processs_turn_and_rewrites_what_it_left(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A release reads, checks and writes in one turn, so it waits while
    another process holds the turn and rewrites the file that process left,
    losing neither process's change."""
    path = tmp_path / "leases.json"
    _write(path, _lease("run-6"))
    other = OtherTurn(leases.lock_path(path))
    ending = other.end_after(0.1, lambda: _write(path, _lease("run-6"), _lease("run-7")))

    with caplog.at_level("INFO"):
        leases.release(path, run_id="run-6", now_unix=_NOW)
    ending.join()

    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-7"]
    said = [record.getMessage() for record in caplog.records]
    assert len(said) == 1
    assert re.fullmatch(
        rf"fleet process {os.getpid()} took its lease turn after waiting \d+ ms for another "
        "process rewriting the lease file",
        said[0],
    ), said


def test_a_lock_file_a_day_old_that_nobody_holds_is_no_turn(tmp_path: pathlib.Path) -> None:
    """A lock file outlives every turn, so its age says nothing: a rewrite
    takes the turn at once whatever the file's age."""
    path = tmp_path / "leases.json"
    _write(path, _lease("run-3"), _lease("run-8"))
    lock = leases.lock_path(path)
    lock.touch()
    day_ago = time.time() - 86_400
    os.utime(lock, (day_ago, day_ago))
    asked = time.monotonic()

    leases.release(path, run_id="run-3", now_unix=_NOW)

    assert time.monotonic() - asked < 0.5
    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-8"]


def test_a_rewrite_reads_and_writes_in_one_turn_without_waiting_on_itself(
    tmp_path: pathlib.Path,
) -> None:
    """release_if_held reads, then release reads and writes, all nested in
    one turn: a turn that locked its file a second time would wait on its
    own ``flock`` for ever, so the release runs on a thread the case bounds."""
    path = tmp_path / "leases.json"
    _write(path, _lease("run-4"), _lease("run-5"))
    said: list[str] = []
    releasing = threading.Thread(
        target=lambda: said.append(leases.release_if_held(path, run_id="run-4", now_unix=_NOW)),
        daemon=True,
    )

    releasing.start()
    releasing.join(timeout=30)

    assert not releasing.is_alive()
    assert said == ["lease released"]
    assert [lease["run_id"] for lease in leases.read_leases(path)] == ["run-5"]
