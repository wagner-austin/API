from __future__ import annotations

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from platform_music.testing import hooks


def test_guard_main_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[tuple[Path, Path]] = []

    def _runner(monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 0

    def _loader(monorepo_root: Path) -> Callable[[Path, Path], int]:
        return _runner

    hooks.load_orchestrator = _loader

    from scripts import guard

    rc = guard.main(["-v"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "guard_exit_code code=0" in out
    if len(calls) != 1:
        raise AssertionError("expected 1 call")
    if not (calls[0][0].__class__.__name__.endswith("Path")):
        raise AssertionError("first arg is not a Path")
    if not (calls[0][1].__class__.__name__.endswith("Path")):
        raise AssertionError("second arg is not a Path")


def test_guard_main_root_override(tmp_path: Path) -> None:
    received: dict[str, Path] = {}

    def _runner(monorepo_root: Path, project_root: Path) -> int:
        received["monorepo_root"] = monorepo_root
        received["project_root"] = project_root
        return 0

    def _loader(monorepo_root: Path) -> Callable[[Path, Path], int]:
        return _runner

    hooks.load_orchestrator = _loader

    from scripts import guard

    override = tmp_path.resolve()
    rc = guard.main(["--root", str(override), "--verbose"])
    assert rc == 0
    assert received.get("project_root") == override
    mono = received.get("monorepo_root")
    if mono is None:
        raise AssertionError("monorepo_root not set")
    if not mono.__class__.__name__.endswith("Path"):
        raise AssertionError("monorepo_root is not a Path")
    if not (mono / "libs").is_dir():
        raise AssertionError("monorepo_root/libs is not a directory")


def test_guard_find_monorepo_root_error(tmp_path: Path) -> None:
    from scripts import guard

    with pytest.raises(RuntimeError):
        guard._find_monorepo_root(tmp_path)


def test_guard_load_orchestrator_returns_callable() -> None:
    from scripts import guard

    monorepo_root = guard._find_monorepo_root(Path(__file__).resolve())
    runner = guard._load_orchestrator_impl(monorepo_root)
    assert callable(runner)


def test_guard_main_unknown_and_quiet() -> None:
    def _runner(monorepo_root: Path, project_root: Path) -> int:
        return 0

    def _loader(monorepo_root: Path) -> Callable[[Path, Path], int]:
        return _runner

    hooks.load_orchestrator = _loader

    from scripts import guard

    rc = guard.main(["--unknown-token"])  # exercises else: idx += 1 and quiet branch
    assert rc == 0


class FakeOrchestratorModule(ModuleType):
    """Fake monorepo_guards.orchestrator module for testing."""

    run_for_project: Callable[[Path, Path], int]

    def __init__(self, name: str, runner: Callable[[Path, Path], int]) -> None:
        super().__init__(name)
        self.run_for_project = runner


def test_guard_entrypoint_runpy() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "guard.py"

    def _runner(monorepo_root: Path, project_root: Path) -> int:
        return 0

    m = FakeOrchestratorModule("monorepo_guards.orchestrator", _runner)
    sys.modules["monorepo_guards.orchestrator"] = m

    with pytest.raises(SystemExit) as ei:
        runpy.run_path(str(script_path), run_name="__main__")
    code_obj = ei.value.code
    code_num = 0 if code_obj is None else int(code_obj)
    assert code_num == 0
