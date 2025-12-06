from __future__ import annotations

import types
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest


class _RunForProject(Protocol):
    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int: ...


def test_guard_main_verbose(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    imp = __import__
    mod: types.ModuleType = imp("scripts.guard", fromlist=["main", "_load_orchestrator"])

    calls: list[tuple[Path, Path]] = []

    def _loader(monorepo_root: Path) -> _RunForProject:
        def _runner(*, monorepo_root: Path, project_root: Path) -> int:
            calls.append((monorepo_root, project_root))
            return 0

        return _runner

    monkeypatch.setattr(mod, "_load_orchestrator", _loader)

    main_fn: Callable[[list[str] | None], int] = mod.main
    rc = main_fn(["-v"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "guard_exit_code code=0" in out
    assert len(calls) == 1 and isinstance(calls[0][0], Path) and isinstance(calls[0][1], Path)


def test_guard_main_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imp = __import__
    mod: types.ModuleType = imp("scripts.guard", fromlist=["main", "_load_orchestrator"])

    received: dict[str, Path] = {}

    def _loader(monorepo_root: Path) -> _RunForProject:
        def _runner(*, monorepo_root: Path, project_root: Path) -> int:
            received["monorepo_root"] = monorepo_root
            received["project_root"] = project_root
            return 0

        return _runner

    monkeypatch.setattr(mod, "_load_orchestrator", _loader)

    override = tmp_path.resolve()
    main_fn: Callable[[list[str] | None], int] = mod.main
    rc = main_fn(["--root", str(override), "--verbose"])
    assert rc == 0
    assert received.get("project_root") == override
    mono = received.get("monorepo_root")
    assert isinstance(mono, Path) and (mono / "libs").is_dir()


def test_guard_find_monorepo_root_error(tmp_path: Path) -> None:
    imp = __import__
    mod: types.ModuleType = imp("scripts.guard", fromlist=["_find_monorepo_root"])
    find_root: Callable[[Path], Path] = mod._find_monorepo_root
    with pytest.raises(RuntimeError):
        find_root(tmp_path)


def test_guard_load_orchestrator_returns_callable() -> None:
    imp = __import__
    mod: types.ModuleType = imp(
        "scripts.guard", fromlist=["_find_monorepo_root", "_load_orchestrator"]
    )
    find_root: Callable[[Path], Path] = mod._find_monorepo_root
    load: Callable[[Path], _RunForProject] = mod._load_orchestrator
    monorepo_root = find_root(Path(__file__).resolve())
    runner = load(monorepo_root)
    assert callable(runner)


def test_guard_main_unknown_and_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    imp = __import__
    mod: types.ModuleType = imp("scripts.guard", fromlist=["main", "_load_orchestrator"])

    def _loader(_: Path) -> _RunForProject:
        def _runner(*, monorepo_root: Path, project_root: Path) -> int:
            _ = (monorepo_root, project_root)
            return 0

        return _runner

    monkeypatch.setattr(mod, "_load_orchestrator", _loader)
    main_fn: Callable[[list[str] | None], int] = mod.main
    rc = main_fn(["--unknown-token"])  # exercises else: idx += 1 and quiet branch
    assert rc == 0


def test_guard_entrypoint_runpy(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "guard.py"

    m = types.ModuleType("monorepo_guards.orchestrator")

    def _runner(*, monorepo_root: Path, project_root: Path) -> int:
        _ = (monorepo_root, project_root)
        return 0

    object.__setattr__(m, "run_for_project", _runner)
    sys.modules["monorepo_guards.orchestrator"] = m

    with pytest.raises(SystemExit) as ei:
        runpy.run_path(str(script_path), run_name="__main__")
    code_obj = ei.value.code
    code_num = 0 if code_obj is None else int(code_obj)
    assert code_num == 0
