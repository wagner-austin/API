"""A turn whose holder is gone is taken before its age bound (MCPs board task 86f70457).

On diphtheria on 2026-10-22 systemd stopped a node runner with SIGKILL 20 s
after it took a mirror's fetch turn, and every export of that mirror waited
out the turn's 1020 s bound. Each case holds a real lock file; the gone
holder is a real process that has exited, asked about with the real probe,
and the probe's other answers are scripted through the command hook.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence

import pytest

from fleet.core import _test_hooks, file_turn
from tests.conftest import FakeRun, failed, ok

#: A turn as long as a mirror's fetch, so only the probe can end a wait early.
LONG_TURN = file_turn.TurnBounds(
    what="long turn",
    waited_for="another long turn",
    stale_seconds=1020,
    poll_seconds=10,
)


def _gone_pid() -> int:
    """The pid of a process that ran and has exited.

    Returns:
        Its pid, which nothing runs under for the length of a case.
    """
    finished = subprocess.Popen((sys.executable, "-c", "pass"))
    assert finished.wait(timeout=60) == 0
    return finished.pid


def _held(lock: pathlib.Path, *, pid: int, host: str, age: int) -> str:
    """Write a lock as another process's turn wrote it, aged.

    Args:
        lock: The lock file.
        pid: The holder's pid.
        host: The holder's host.
        age: How long ago it was written, in seconds.

    Returns:
        Its text.
    """
    text = f"the export of it pid {pid} on {host} turn {uuid.uuid4().hex}"
    lock.write_text(text, encoding="utf-8")
    written = time.time() - age
    os.utime(lock, (written, written))
    return text


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


def test_a_lock_names_its_holder_s_pid_and_host(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / "turn.lock"

    with file_turn.held(lock, holder="the export of it", bounds=LONG_TURN):
        text = lock.read_text(encoding="utf-8")

    assert text.startswith(f"the export of it pid {os.getpid()} on {_test_hooks.hostname()} turn ")
    assert not lock.exists()


def test_a_holder_killed_inside_its_turn_is_taken_at_once(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The stall of 2026-10-22, made the filesystem's: the lock is past the
    probe's threshold and far inside the age bound, and its pid has exited."""
    lock = tmp_path / "turn.lock"
    gone = _gone_pid()
    text = _held(
        lock, pid=gone, host=_test_hooks.hostname(), age=file_turn.HOLDER_PROBE_SECONDS + 5
    )
    slept: list[int] = []
    _test_hooks.sleep = _sleeping([], slept)

    with caplog.at_level("INFO"), file_turn.held(lock, holder="the next export", bounds=LONG_TURN):
        assert lock.read_text(encoding="utf-8") != text

    assert slept == []
    assert (
        f"the next export took away a long turn whose holder, pid {gone} on "
        f"{_test_hooks.hostname()}, no longer runs: {text!r}"
    ) in [record.getMessage() for record in caplog.records]


def test_a_holder_that_still_runs_is_waited_for(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / "turn.lock"
    _held(
        lock, pid=os.getpid(), host=_test_hooks.hostname(), age=file_turn.HOLDER_PROBE_SECONDS + 5
    )
    slept: list[int] = []
    _test_hooks.sleep = _sleeping([lambda: lock.unlink()], slept)

    with file_turn.held(lock, holder="the next export", bounds=LONG_TURN):
        pass

    assert slept == [LONG_TURN["poll_seconds"]]


@pytest.mark.parametrize(
    ("host", "age"),
    [("elsewhere", file_turn.HOLDER_PROBE_SECONDS + 5), (None, 0)],
    ids=["another host's lock", "a young lock"],
)
def test_is_not_probed_for_another_host_s_lock_or_a_young_one(
    tmp_path: pathlib.Path, host: str | None, age: int
) -> None:
    """This machine cannot see another host's process, and a lock younger
    than the threshold belongs to a turn that is nearly always about to end."""
    lock = tmp_path / "turn.lock"
    _held(lock, pid=_gone_pid(), host=host or _test_hooks.hostname(), age=age)
    runner = FakeRun([])
    _test_hooks.run = runner
    slept: list[int] = []
    _test_hooks.sleep = _sleeping([lambda: lock.unlink()], slept)

    with file_turn.held(lock, holder="the next export", bounds=LONG_TURN):
        pass

    assert runner.calls == []
    assert slept == [LONG_TURN["poll_seconds"]]


def test_a_lock_written_before_hosts_were_named_is_left_to_its_age_bound(
    tmp_path: pathlib.Path,
) -> None:
    lock = tmp_path / "turn.lock"
    lock.write_text("the export of it pid 4242 turn fetching", encoding="utf-8")
    written = time.time() - file_turn.HOLDER_PROBE_SECONDS - 5
    os.utime(lock, (written, written))
    runner = FakeRun([])
    _test_hooks.run = runner
    slept: list[int] = []
    _test_hooks.sleep = _sleeping([lambda: lock.unlink()], slept)

    with file_turn.held(lock, holder="the next export", bounds=LONG_TURN):
        pass

    assert runner.calls == []
    assert slept == [LONG_TURN["poll_seconds"]]


def test_a_lock_rewritten_while_its_holder_was_probed_is_left(tmp_path: pathlib.Path) -> None:
    """The gone holder's lock was taken away and a live turn created its own
    while the probe ran, so the lock this turn read is not the lock there."""
    lock = tmp_path / "turn.lock"
    _held(
        lock, pid=_gone_pid(), host=_test_hooks.hostname(), age=file_turn.HOLDER_PROBE_SECONDS + 5
    )
    later = f"a live export pid {os.getpid()} on {_test_hooks.hostname()} turn {uuid.uuid4().hex}"

    def probe(
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        lock.write_text(later, encoding="utf-8")
        return failed(1, "")

    _test_hooks.run = probe
    seen: list[str] = []

    def the_live_turn_ends() -> None:
        seen.append(lock.read_text(encoding="utf-8"))
        lock.unlink()

    slept: list[int] = []
    _test_hooks.sleep = _sleeping([the_live_turn_ends], slept)

    with file_turn.held(lock, holder="the next export", bounds=LONG_TURN):
        pass

    assert seen == [later]


class TestRuns:
    def test_asks_ps_for_the_pid(self) -> None:
        runner = FakeRun([ok("   4242\n")])
        _test_hooks.run = runner

        assert file_turn.runs(4242) is True
        assert runner.calls == [("ps", "-p", "4242", "-o", "pid=")]
        assert runner.timeouts == [file_turn.PROBE_TIMEOUT_SECONDS]

    @pytest.mark.parametrize(
        ("answer", "running"),
        [
            (ok("   4242\n"), True),
            (failed(1, ""), False),
            (ok("  14242\n"), False),
        ],
        ids=["ps lists it", "ps lists none", "another pid"],
    )
    def test_reads_the_answer(self, answer: _test_hooks.CommandResult, running: bool) -> None:
        _test_hooks.run = FakeRun([answer])

        assert file_turn.runs(4242) is running

    @pytest.mark.parametrize(
        "answer",
        [failed(127, "ps: not found"), failed(1, "ps: unknown option")],
        ids=["no probe", "a probe that complained"],
    )
    def test_a_probe_that_cannot_answer_reads_as_running(
        self, answer: _test_hooks.CommandResult, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Running is the answer that changes nothing: the turn then waits
        out its age bound, as it did before it could ask."""
        _test_hooks.run = FakeRun([answer])

        with caplog.at_level("INFO"):
            assert file_turn.runs(4242) is True

        assert any(
            record.getMessage().startswith("could not ask whether pid 4242 runs")
            for record in caplog.records
        )

    def test_this_process_runs_and_an_exited_one_does_not(self) -> None:
        assert (file_turn.runs(os.getpid()), file_turn.runs(_gone_pid())) == (True, False)


def test_a_lock_given_back_before_it_is_read_has_no_text(tmp_path: pathlib.Path) -> None:
    assert file_turn._lock_text(tmp_path / "given-back.lock") is None
