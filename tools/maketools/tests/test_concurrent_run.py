"""``concurrent``: several of one package's make targets side by side (board task 28e47ae3).

The command builds one ``make TARGET`` per target, hands them to the
concurrent-run hook as one batch, prints each child's output as a block in
the order the targets were named, and fails with the first named failure's
code while naming every failure. A recording hook shows each of those; a
real make with real children (:mod:`tests.test_concurrent_children`) shows
the whole path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import pytest
from platform_core.errors import AppError

from maketools import _test_hooks
from maketools.cli import dispatch
from maketools.commands import ConcurrentOutcome
from maketools.concurrent_run import run_targets_concurrently
from maketools.workspace import FANOUT_WALL_SECONDS
from tests.conftest import World


class ConcurrentCall(TypedDict):
    """One recorded batch.

    Attributes:
        argvs: The commands, in order.
        cwd: Their shared directory.
        env: Their shared environment.
        timeout_seconds: The batch's wall.
    """

    argvs: list[tuple[str, ...]]
    cwd: Path
    env: dict[str, str]
    timeout_seconds: int


class RecordingConcurrent:
    """A concurrent-run hook that records its batch and answers scripted outcomes.

    Attributes:
        outcomes: What the batch answers, one per command.
        calls: Every batch it was handed.
    """

    def __init__(self, outcomes: list[ConcurrentOutcome]) -> None:
        """Answer ``outcomes`` to the next batch.

        Args:
            outcomes: One per command the batch will carry.
        """
        self.outcomes = outcomes
        self.calls: list[ConcurrentCall] = []

    def __call__(
        self,
        argvs: Sequence[Sequence[str]],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: int,
    ) -> list[ConcurrentOutcome]:
        self.calls.append(
            ConcurrentCall(
                argvs=[tuple(argv) for argv in argvs],
                cwd=cwd,
                env=dict(env),
                timeout_seconds=timeout_seconds,
            )
        )
        return self.outcomes


def bind(outcomes: list[ConcurrentOutcome]) -> RecordingConcurrent:
    """Bind a recording hook for one test.

    Every caller also takes the ``world`` fixture, whose teardown restores
    this hook's default along with every other.

    Args:
        outcomes: What the batch answers.

    Returns:
        The hook.
    """
    recorder = RecordingConcurrent(outcomes)
    _test_hooks.run_concurrently = recorder
    return recorder


def test_every_target_runs_as_one_make_in_one_batch(world: World, tmp_path: Path) -> None:
    recorder = bind(
        [
            ConcurrentOutcome(returncode=0, output="guard ok\n", seconds=41.7),
            ConcurrentOutcome(returncode=0, output="Success\n", seconds=55.2),
            ConcurrentOutcome(returncode=0, output="7290 passed\n", seconds=190.9),
        ],
    )
    assert run_targets_concurrently(tmp_path, ["_guard", "_mypy", "_suite"]) == 0
    assert recorder.calls == [
        ConcurrentCall(
            argvs=[("make", "_guard"), ("make", "_mypy"), ("make", "_suite")],
            cwd=tmp_path,
            env={"PATH": "/bin"},
            timeout_seconds=FANOUT_WALL_SECONDS,
        )
    ]
    assert world.lines == [
        "",
        "=== concurrent: _guard passed after 41s",
        "guard ok",
        "",
        "=== concurrent: _mypy passed after 55s",
        "Success",
        "",
        "=== concurrent: _suite passed after 190s",
        "7290 passed",
        "",
        "concurrent: 3 of 3 passed (_guard 41s, _mypy 55s, _suite 190s)",
    ]
    assert world.errors == []


def test_a_failure_returns_the_first_named_code_and_every_failure_is_named(
    world: World, tmp_path: Path
) -> None:
    bind(
        [
            ConcurrentOutcome(returncode=0, output="guard ok\n", seconds=40.0),
            ConcurrentOutcome(returncode=2, output="error: bad\n", seconds=50.0),
            ConcurrentOutcome(returncode=1, output="1 failed\n", seconds=180.0),
        ],
    )
    assert run_targets_concurrently(tmp_path, ["_guard", "_mypy", "_suite"]) == 2
    assert world.lines == [
        "",
        "=== concurrent: _guard passed after 40s",
        "guard ok",
        "",
        "=== concurrent: _mypy FAILED with exit 2 after 50s",
        "error: bad",
        "",
        "=== concurrent: _suite FAILED with exit 1 after 180s",
        "1 failed",
    ]
    assert world.errors == [
        "concurrent: 2 of 3 target(s) failed: _mypy, _suite (_guard 40s, _mypy 50s, _suite 180s)"
    ]


def test_the_command_runs_in_the_working_directory(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = bind(
        [
            ConcurrentOutcome(returncode=0, output="", seconds=1.0),
            ConcurrentOutcome(returncode=0, output="", seconds=2.0),
        ],
    )
    monkeypatch.chdir(tmp_path)
    assert dispatch(["concurrent", "_a", "_b"]) == 0
    assert recorder.calls[0]["cwd"] == tmp_path
    assert recorder.calls[0]["argvs"] == [("make", "_a"), ("make", "_b")]


def test_one_target_is_a_usage_error(world: World) -> None:
    recorder = bind([])
    with pytest.raises(AppError, match=r"concurrent needs at least 2 targets, got \['_a'\]"):
        dispatch(["concurrent", "_a"])
    assert recorder.calls == []


def test_a_flag_is_a_usage_error(world: World) -> None:
    recorder = bind([])
    with pytest.raises(AppError, match=r"target names only, not flags: \['-j4'\]"):
        dispatch(["concurrent", "_a", "-j4"])
    assert recorder.calls == []
