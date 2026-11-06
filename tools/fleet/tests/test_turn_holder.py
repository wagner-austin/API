"""A holder lets go as it ends, and no live holder's turn is taken (MCPs tasks 86f70457, 5280a25d).

On diphtheria on 2026-10-22 systemd stopped a node runner with SIGKILL 20 s
after it took a mirror's fetch turn, and every export of that mirror waited
out the turn's 1020 s age bound; a probe of the holder's pid answered that
until a turn became an exclusive ``flock``, which the kernel lets go when
its holder's process ends. Each case holds a real ``flock`` on a real lock
file; the killed holder is a real process, and no command is run for any
holder (the command hook answers nothing).
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import pytest

from fleet.core import _test_hooks, file_turn
from tests._held_turns import OtherTurn
from tests.conftest import FakeRun

TURN = file_turn.TurnBounds(what="long turn", waited_for="another long turn")

#: A process that takes the turn's ``flock``, writes ``held`` to the pipe
#: it was handed, and holds the turn for ten minutes.
_HOLDER = (
    "import fcntl, os, sys, time\n"
    "descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)\n"
    "fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
    "os.write(int(sys.argv[2]), b'held')\n"
    "time.sleep(600)\n"
)


def test_a_holder_killed_inside_its_turn_lets_go_as_it_dies(tmp_path: pathlib.Path) -> None:
    """The stall of 2026-10-22, made a real process's: it holds the turn,
    a waiter is blocked behind it, and SIGKILL ends it. The waiter takes the
    turn within a second of the kill, asking nothing about the holder."""
    lock = tmp_path / "turn.turn"
    said, told = os.pipe()
    holder = subprocess.Popen(
        (sys.executable, "-c", _HOLDER, str(lock), str(told)), pass_fds=(told,)
    )
    os.close(told)
    _test_hooks.run = FakeRun([])
    taken: list[float] = []

    def waiter() -> None:
        with file_turn.held(lock, holder="the next export", bounds=TURN):
            taken.append(time.monotonic())

    try:
        assert os.read(said, 4) == b"held"
        waiting = threading.Thread(target=waiter)
        waiting.start()
        time.sleep(0.2)
        assert taken == []
        killed = time.monotonic()
    finally:
        os.kill(holder.pid, signal.SIGKILL)
        os.close(said)
    assert holder.wait(timeout=60) == -signal.SIGKILL
    waiting.join(timeout=30)

    assert len(taken) == 1
    assert taken[0] - killed < 1.0, f"took the turn {taken[0] - killed:.3f} s after the kill"


def test_a_live_holder_of_a_day_old_lock_is_waited_for_to_its_end(tmp_path: pathlib.Path) -> None:
    """No age bound: a lock file written a day ago, held now, is waited for
    until its holder lets go, never taken away while it is held."""
    lock = tmp_path / "turn.turn"
    other = OtherTurn(lock)
    day_ago = time.time() - 86_400
    os.utime(lock, (day_ago, day_ago))
    _test_hooks.run = FakeRun([])
    asked = time.monotonic()
    ending = other.end_after(0.3)

    with file_turn.held(lock, holder="the next export", bounds=TURN):
        waited = time.monotonic() - asked
    ending.join()

    assert waited >= 0.3, f"took the turn after {waited:.3f} s, inside its holder's"


def test_the_lock_file_stays_the_same_file_across_turns(tmp_path: pathlib.Path) -> None:
    """Nothing is created or removed per turn: the second turn opens the
    file the first made, which on the hub's NFS is what saves the server's
    synchronous create and remove."""
    lock = tmp_path / "turn.turn"

    with file_turn.held(lock, holder="the first turn", bounds=TURN):
        first = os.stat(lock).st_ino
    with file_turn.held(lock, holder="the second turn", bounds=TURN):
        second = os.stat(lock).st_ino

    assert lock.exists()
    assert first == second


def test_a_waiter_whose_lock_its_holder_removed_takes_the_path_s_next_file(
    tmp_path: pathlib.Path,
) -> None:
    """The holder removes the file inside its turn, as a retire turn does.
    The waiter, woken holding a file no path names, takes a new one at the
    path, which a later turn would see."""
    lock = tmp_path / "turn.turn"
    other = OtherTurn(lock)
    removed = os.stat(lock).st_ino
    ending = other.end_after(0.2, lock.unlink)

    with file_turn.held(lock, holder="the next turn", bounds=TURN):
        named = os.stat(lock).st_ino
    ending.join()

    assert named != removed


def test_a_waiter_whose_lock_was_replaced_takes_the_replacement(tmp_path: pathlib.Path) -> None:
    """The holder removes the file and another file is made at the path
    before the waiter wakes: the waiter takes the file the path names."""
    lock = tmp_path / "turn.turn"
    other = OtherTurn(lock)
    replacement: list[int] = []

    def replace() -> None:
        lock.unlink()
        lock.touch()
        replacement.append(os.stat(lock).st_ino)

    ending = other.end_after(0.2, replace)

    with file_turn.held(lock, holder="the next turn", bounds=TURN):
        named = os.stat(lock).st_ino
    ending.join()

    assert replacement == [named]


def test_a_block_that_raises_gives_the_turn_back(tmp_path: pathlib.Path) -> None:
    """The next turn takes the lock without waiting: :class:`OtherTurn`
    asks for it without blocking and would raise were it still held."""
    lock = tmp_path / "turn.turn"

    with (
        pytest.raises(RuntimeError, match="the turn's work failed"),
        file_turn.held(lock, holder="the failing turn", bounds=TURN),
    ):
        raise RuntimeError("the turn's work failed")
    OtherTurn(lock).end_after(0).join()
