from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture
from scripts.guard import main as guard_main

from scripts import guard


def test_guard_main_with_root(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    code = guard_main(["--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Guard rule summary:" in out or "Guard checks passed" in out


def test_guard_main_unrecognized_arg(tmp_path: Path) -> None:
    # Exercise the else path in the arg loop
    code = guard_main(["--root", str(tmp_path), "--unknown-flag"])  # unknown is ignored
    assert code == 0


def test_guard_run_as_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run via __main__ to fully cover the exit branch
    # Ensure guard scans an empty temporary project root
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "guard.py"
    resolved = module_path.resolve()
    # Monkeypatch sys.argv to use tmp_path as root (avoids full project scan timeout)
    monkeypatch.setattr("sys.argv", ["guard.py", "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(resolved), run_name="__main__")
    assert exc.value.code == 0


def test_guard_find_monorepo_root_raises(tmp_path: Path) -> None:
    start = tmp_path / "nested"
    start.mkdir()
    with pytest.raises(RuntimeError):
        _ = guard._find_monorepo_root(start)


def test_guard_verbose_prints_exit_code(
    capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Path] = {}

    def _fake_loader(monorepo_root: Path) -> guard._RunForProject:
        calls["monorepo_root"] = monorepo_root

        def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
            calls["project_root"] = project_root
            return 7

        return _RunForProjectWrapper(_run_for_project)

    class _RunForProjectWrapper:
        def __init__(self: _RunForProjectWrapper, fn: guard._RunForProject) -> None:
            self._fn = fn

        def __call__(
            self: _RunForProjectWrapper, *, monorepo_root: Path, project_root: Path
        ) -> int:
            return self._fn(monorepo_root=monorepo_root, project_root=project_root)

    monkeypatch.setattr(guard, "_load_orchestrator", _fake_loader)

    def _fake_find(_: Path) -> Path:
        return tmp_path

    monkeypatch.setattr(guard, "_find_monorepo_root", _fake_find)

    rc = guard_main(["--root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 7
    assert "guard_exit_code code=7" in out
    assert calls["monorepo_root"] == tmp_path
    assert calls["project_root"] == tmp_path
