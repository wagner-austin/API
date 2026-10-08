"""The real concurrent-run binding, against real children and a real make.

Nothing here is faked, for the reason :mod:`tests.test_hook_defaults`
gives: these cases are the evidence that the production binding does what
the recording hook in :mod:`tests.test_concurrent_run` claims it does.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from maketools import _test_hooks
from maketools.concurrent_run import run_targets_concurrently
from maketools.makefile_grammar import SHELL_INCLUDE

#: Wall for the children that exit by themselves. Generous so a loaded box
#: never fails a case that is not about timing.
PROBE_WALL_SECONDS = 120

#: The repository's shell prologue, which the real-make case's Makefile
#: includes as every tracked Makefile does, so its recipes run under the
#: shell the prologue names rather than whichever one the environment hands
#: make. Without it fleet c68f15c7 on loki failed with ``CreateProcess(NULL,
#: echo left ran, ...) failed``: Windows make ran ``echo`` as a program
#: (MCPs board task 90c135ac), which the same make on the hub never did.
SHELL_PROLOGUE = Path(__file__).resolve().parents[3] / SHELL_INCLUDE


def test_every_child_runs_and_answers_in_the_order_named(tmp_path: Path) -> None:
    outcomes = _test_hooks.run_concurrently(
        [
            [sys.executable, "-c", "import time, sys; time.sleep(2); print('slow'); sys.exit(3)"],
            [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr); sys.exit(0)",
            ],
        ],
        cwd=tmp_path,
        env=_test_hooks.environ(),
        timeout_seconds=PROBE_WALL_SECONDS,
    )
    assert [outcome["returncode"] for outcome in outcomes] == [3, 0]
    assert outcomes[0]["output"].splitlines() == ["slow"]
    assert sorted(outcomes[1]["output"].splitlines()) == ["err", "out"]
    # Each child's seconds are its own: the quick one is not charged the
    # slow one's sleep, which is what the summary line reports per target.
    assert outcomes[1]["seconds"] < outcomes[0]["seconds"]
    assert outcomes[0]["seconds"] >= 2


def test_a_child_writing_more_than_a_pipe_holds_is_captured_whole(tmp_path: Path) -> None:
    """Two megabytes, far past a pipe's buffer, from a child nobody reads while it runs."""
    outcomes = _test_hooks.run_concurrently(
        [
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 2_000_000)"],
            [sys.executable, "-c", "pass"],
        ],
        cwd=tmp_path,
        env=_test_hooks.environ(),
        timeout_seconds=PROBE_WALL_SECONDS,
    )
    assert len(outcomes[0]["output"]) == 2_000_000
    assert outcomes[1] == {"returncode": 0, "output": "", "seconds": outcomes[1]["seconds"]}


def test_a_batch_past_its_wall_kills_the_children_still_running(tmp_path: Path) -> None:
    """WATCH THE BOUND BITE: the sleeper is killed, not left behind."""
    pid_file = tmp_path / "sleeper.pid"
    sleeper = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        _test_hooks.run_concurrently(
            [[sys.executable, "-c", sleeper], [sys.executable, "-c", "pass"]],
            cwd=tmp_path,
            env=_test_hooks.environ(),
            timeout_seconds=5,
        )
    # The message names the command that had not exited and the bound.
    still_running = [f"{sys.executable} -c {sleeper}"]
    assert str(raised.value) == f"Command '{still_running}' timed out after 5 seconds"
    assert _test_hooks.process_alive(int(pid_file.read_text(encoding="utf-8"))) is False


def test_a_real_make_runs_each_target_and_reports_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The text is quoted so PowerShell's echo prints it as one line, as
    # /bin/sh's does.
    (tmp_path / "Makefile").write_text(
        f"include {SHELL_PROLOGUE.as_posix()}\n\n"
        "_left:\n\t@echo 'left ran'\n\n_right:\n\t@echo 'right ran'\n",
        encoding="utf-8",
    )
    assert run_targets_concurrently(tmp_path, ["_left", "_right"]) == 0
    printed = capsys.readouterr().out
    # Under a parent make (this suite run by ``make test``) a child make
    # also prints its "Entering directory" lines, so each block is read by
    # its header and its echo, not by line number.
    left_block, right_block = printed.split("=== concurrent: _right passed after ")
    assert "=== concurrent: _left passed after " in left_block
    assert "left ran" in left_block.splitlines()
    assert "right ran" in right_block.splitlines()
    assert "left ran" not in right_block
    assert (
        printed.rstrip("\n")
        .splitlines()[-1]
        .startswith("concurrent: every target succeeded (_left ")
    )
