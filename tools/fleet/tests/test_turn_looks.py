"""A waiter is woken as the turn is given back, never on a clock (MCPs board task 5280a25d).

Every look at a held lock used to come on a clock, at first a second apart
and then doubling from 10 ms up to a second, while an append, lease or claim
turn holds its lock for milliseconds, so each turn a launch found held cost
it up to a second between its job's claim and its first phase. A turn is now
an exclusive ``flock``, and the kernel wakes a waiter when its holder lets
go. Each case holds a real ``flock`` on a real lock file, and the sleep hook
fails the case if the code under test ever asks it for a look.
"""

from __future__ import annotations

import pathlib
import re
import threading
import time

import pytest

from fleet.core import _test_hooks, file_turn
from tests._held_turns import OtherTurn
from tests._thread_fakes import await_event

TURN = file_turn.TurnBounds(what="short turn", waited_for="another short turn")


def _no_look(seconds: float) -> None:
    """A sleep the code under test must never ask for.

    Args:
        seconds: The wait asked for.

    Raises:
        AssertionError: Always, naming the wait.
    """
    raise AssertionError(f"a turn asked for a look {seconds} s later")


def test_a_waiter_takes_the_turn_as_its_holder_lets_go_without_a_look(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Held 200 ms by another open file. Looked at on the old clock, the
    waiter would have slept at least once; it asks for no sleep at all."""
    lock = tmp_path / "turn.turn"
    other = OtherTurn(lock)
    _test_hooks.sleep = _no_look
    asked = time.monotonic()
    ending = other.end_after(0.2)

    with caplog.at_level("INFO"), file_turn.held(lock, holder="the waiting turn", bounds=TURN):
        waited = time.monotonic() - asked
    ending.join()

    assert 0.2 <= waited < 1.0, f"took the turn after {waited:.3f} s"
    said = [record.getMessage() for record in caplog.records]
    assert len(said) == 1
    assert re.fullmatch(
        r"the waiting turn took its short turn after waiting \d+ ms for another short turn",
        said[0],
    ), said


def test_a_free_turn_is_taken_at_once_and_says_nothing(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    lock = tmp_path / "turn.turn"
    _test_hooks.sleep = _no_look

    with caplog.at_level("INFO"), file_turn.held(lock, holder="the only turn", bounds=TURN):
        pass

    assert caplog.records == []


def test_two_threads_of_one_process_take_the_turn_one_at_a_time(tmp_path: pathlib.Path) -> None:
    """A ``flock`` belongs to the open file, so a node runner's watch thread
    and its fill pass exclude each other as two processes would."""
    lock = tmp_path / "turn.turn"
    inside: list[str] = []
    entered = threading.Event()
    leave = threading.Event()

    def first() -> None:
        with file_turn.held(lock, holder="the first thread", bounds=TURN):
            inside.append("first in")
            entered.set()
            await_event(leave, what="the second thread's start")
            time.sleep(0.1)
            inside.append("first out")

    thread = threading.Thread(target=first)
    thread.start()
    await_event(entered, what="the first thread's turn")
    leave.set()
    with file_turn.held(lock, holder="the second thread", bounds=TURN):
        inside.append("second in")
    thread.join()

    assert inside == ["first in", "first out", "second in"]
