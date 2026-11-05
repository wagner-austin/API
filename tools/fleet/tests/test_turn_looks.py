"""A turn held for milliseconds is taken in milliseconds (MCPs board task 5280a25d).

Every look at a lock another turn held used to come ``poll_seconds`` apart,
one second for every turn the hub takes, so each append, lease, claim or
mirror fetch turn a launch found held cost it a whole second between its
job's claim and its first phase. A waiter now looks again after
:data:`fleet.core.file_turn.FIRST_LOOK_SECONDS`, each wait twice the last up
to the turn's ``poll_seconds``. Each case holds a real lock file; the second
takes the turn behind a real thread's real turn on the real clock.
"""

from __future__ import annotations

import pathlib
import threading
import time
from collections.abc import Callable

from fleet.core import _test_hooks, file_turn
from tests._thread_fakes import await_event

#: A turn whose waits are capped at one second, as every turn the hub takes.
TURN = file_turn.TurnBounds(
    what="short turn",
    waited_for="another short turn",
    stale_seconds=30,
    poll_seconds=1,
)


def test_each_wait_doubles_from_the_first_look_up_to_the_turn_s_poll(
    tmp_path: pathlib.Path,
) -> None:
    """Held for nine looks: seven doublings from 10 ms, then the cap twice."""
    lock = tmp_path / "turn.lock"
    lock.write_text("another short turn pid 4246 turn holding", encoding="utf-8")
    slept: list[float] = []
    steps: list[Callable[[], None]] = [*([lambda: None] * 8), lambda: lock.unlink()]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        steps.pop(0)()

    _test_hooks.sleep = sleep

    with file_turn.held(lock, holder="the waiting turn", bounds=TURN):
        held = lock.read_text(encoding="utf-8")

    assert slept == [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1, 1]
    assert held.startswith("the waiting turn pid ")
    assert not lock.exists()


def test_a_turn_held_for_milliseconds_is_taken_in_milliseconds(tmp_path: pathlib.Path) -> None:
    """A real thread holds the turn for 50 ms. Looked at a second apart, as
    every turn was until this task, the wait would last at least 1 s."""
    lock = tmp_path / "turn.lock"
    holding = threading.Event()
    release = threading.Event()

    def the_other_turn() -> None:
        with file_turn.held(lock, holder="the other turn", bounds=TURN):
            holding.set()
            await_event(release, what="the waiting turn's first look")
            time.sleep(0.05)

    other = threading.Thread(target=the_other_turn)
    other.start()
    await_event(holding, what="the other turn")
    asked = time.monotonic()
    release.set()
    with file_turn.held(lock, holder="the waiting turn", bounds=TURN):
        waited = time.monotonic() - asked
    other.join()

    assert 0.05 <= waited < 0.9, f"took the turn after {waited:.3f} s"
    assert not lock.exists()
