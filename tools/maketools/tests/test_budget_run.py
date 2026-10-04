"""``check-budget``: a package's whole check timed against five minutes (MCPs board task 1b152218).

The command runs the package's ``_check-unbudgeted`` and judges it by the
lifted rule: within the budget it says how much it used, past it a passing
check fails with exit 3 naming its time, and a failing check keeps its own
code. A real make run through the real hooks shows the whole path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from platform_core.errors import AppError

from maketools.budget_run import run_check_budget
from maketools.check_budget import OVER_BUDGET_EXIT_CODE, UNBUDGETED_TARGET, WHOLE_CHECK_SPLIT
from maketools.cli import dispatch, repository_root
from maketools.workspace import FANOUT_WALL_SECONDS
from tests.conftest import InheritingCall, World


class TimedWorld(World):
    """A world whose child make takes a set number of seconds."""

    def __init__(self, seconds: float) -> None:
        """Start quiet, with every child taking ``seconds``.

        Args:
            seconds: How far each ``run_inheriting`` moves the clock.
        """
        super().__init__()
        self.child_seconds = seconds

    def run_inheriting(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        new_session: bool,
        timeout_seconds: int,
    ) -> int:
        """Record the child as the base world does, then let its time pass.

        Returns:
            The base world's scripted exit code.
        """
        code = super().run_inheriting(
            argv, cwd=cwd, env=env, new_session=new_session, timeout_seconds=timeout_seconds
        )
        self.now_value += self.child_seconds
        return code


@pytest.fixture()
def timed(world: World) -> TimedWorld:
    """A bound world whose child takes 120 s; a test may change it.

    The ``world`` fixture is requested so its teardown restores the real
    hooks after this one has rebound them.

    Returns:
        The world.
    """
    fake = TimedWorld(120)
    fake.bind()
    return fake


def test_a_check_within_the_budget_passes_and_says_how_much_it_used(
    tmp_path: Path, timed: TimedWorld
) -> None:
    assert run_check_budget(tmp_path, "libs/covenant_ml") == 0
    assert timed.inheriting_calls == [
        InheritingCall(
            argv=("make", UNBUDGETED_TARGET),
            cwd=tmp_path,
            env={"PATH": "/bin"},
            new_session=False,
            timeout_seconds=FANOUT_WALL_SECONDS,
        )
    ]
    assert timed.lines == [
        f"CHECK BUDGET: libs/covenant_ml took 120s of 300s ({WHOLE_CHECK_SPLIT})."
    ]


def test_a_passing_check_past_the_budget_fails_naming_its_time(
    tmp_path: Path, timed: TimedWorld
) -> None:
    timed.child_seconds = 744
    assert run_check_budget(tmp_path, "libs/covenant_ml") == OVER_BUDGET_EXIT_CODE
    assert (
        "CHECK OVER BUDGET: libs/covenant_ml took 744s, over the 300s budget by 444s."
        in timed.lines
    )
    assert f"  split : {WHOLE_CHECK_SPLIT}" in timed.lines


def test_a_failing_check_keeps_its_own_code_and_still_says_its_time(
    tmp_path: Path, timed: TimedWorld
) -> None:
    timed.child_seconds = 400
    timed.inheriting_code = 2
    assert run_check_budget(tmp_path, "tools/fleet") == 2
    assert "CHECK OVER BUDGET: tools/fleet took 400s, over the 300s budget by 100s." in (
        timed.lines
    )


def test_the_command_names_the_package_from_the_repository_root(
    timed: TimedWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = repository_root() / "tools" / "maketools"
    monkeypatch.chdir(package)
    assert dispatch(["check-budget"]) == 0
    assert timed.inheriting_calls[0]["cwd"] == package
    assert timed.lines[0].startswith("CHECK BUDGET: tools/maketools took 120s of 300s")


def test_the_command_takes_no_arguments(world: World) -> None:
    with pytest.raises(AppError, match=r"check-budget takes no arguments, got \['lint'\]"):
        dispatch(["check-budget", "lint"])
    assert world.inheriting_calls == []


def test_a_real_make_runs_the_unbudgeted_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The real hooks, a real make and a real child: the target writes a
    # marker only that run could have written.
    (tmp_path / "Makefile").write_text(
        f"{UNBUDGETED_TARGET}:\n\t@echo ran > marker.txt\n", encoding="utf-8"
    )
    assert run_check_budget(tmp_path, "pkg") == 0
    printed = capsys.readouterr().out
    # The one number a real run cannot fix in advance is its own seconds,
    # read back as an integer so the rest of the line is held exactly.
    seconds = int(printed.removeprefix("CHECK BUDGET: pkg took ").split("s of ", 1)[0])
    assert printed == f"CHECK BUDGET: pkg took {seconds}s of 300s ({WHOLE_CHECK_SPLIT}).\n"
    assert (tmp_path / "marker.txt").read_text(encoding="utf-8").strip() == "ran"
