"""The two lint openers: the stale-venv probe and the guard bootstrap."""

from __future__ import annotations

from pathlib import Path

from maketools.guard_run import GUARD_ARGV, run_guard
from maketools.venv_check import PROBE_ARGV, check_venv
from tests.conftest import World, failed, ok


def test_no_venv_yet_is_not_stale(world: World, tmp_path: Path) -> None:
    assert check_venv(tmp_path) is False
    assert world.captured == []
    assert world.lines == ["venv-check: no .venv yet; poetry sync will create one"]


def test_an_answering_venv_is_kept(world: World, tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    world.capturing_answers[PROBE_ARGV] = ok("mypy 2.3.1\n")
    assert check_venv(tmp_path) is False
    assert world.removed_trees == []
    assert world.lines == ["venv-check: .venv answers (mypy 2.3.1)"]


def test_a_silent_venv_is_removed(world: World, tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    world.capturing_answers[PROBE_ARGV] = failed(1, "no such interpreter")
    assert check_venv(tmp_path) is True
    assert world.removed_trees == [tmp_path / ".venv"]
    assert world.lines[0].startswith(
        "venv-check: stale venv detected (poetry run mypy --version exited 1)"
    )


def test_guard_runs_the_shim_in_the_package_and_returns_its_status(
    world: World, tmp_path: Path
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "guard.py").write_text("", encoding="utf-8")
    world.inheriting_code = 2
    assert run_guard(tmp_path) == 2
    call = world.inheriting_calls[0]
    assert call["argv"] == GUARD_ARGV
    assert call["cwd"] == tmp_path
    assert call["new_session"] is False
    assert world.lines == ["guard: running scripts/guard.py"]


def test_guard_accepts_the_package_form_of_the_shim(world: World, tmp_path: Path) -> None:
    (tmp_path / "scripts" / "guard").mkdir(parents=True)
    (tmp_path / "scripts" / "guard" / "__main__.py").write_text("", encoding="utf-8")
    assert run_guard(tmp_path) == 0
    assert world.lines == ["guard: running scripts/guard/__main__.py"]


def test_guard_says_not_applicable_when_there_is_no_shim(world: World, tmp_path: Path) -> None:
    assert run_guard(tmp_path) == 0
    assert world.inheriting_calls == []
    assert world.lines == [
        f"guard: not applicable: {tmp_path.name} has no scripts/guard.py or "
        "scripts/guard/__main__.py, 0 rules run"
    ]
