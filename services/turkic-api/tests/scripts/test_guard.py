from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pytest
import scripts.guard as guard
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from scripts.guard import _RunForProject


def test_guard_main_verbose_and_unknown_token(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    # Use a temp root so guard scans an empty project (no violations)
    code = guard.main(["--verbose", "--root", str(tmp_path), "unknown-flag"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Guard rule summary:" in out
    # Verbose flag should print the exit code line
    assert "guard_exit_code code=0" in out


def test_guard_main_non_verbose_path(tmp_path: Path) -> None:
    # Exercise the non-verbose branch where no extra line is printed
    code = guard.main(["--root", str(tmp_path)])
    assert code == 0


def test_find_monorepo_root_traversal(tmp_path: Path) -> None:
    # Build nested structure to force while loop to climb before finding libs
    root = tmp_path / "repo"
    nested = root / "a" / "b" / "c"
    (root / "libs").mkdir(parents=True)
    nested.mkdir(parents=True)
    found = guard._find_monorepo_root(nested)
    assert found == root

    # If libs is never found, raise at filesystem root
    with pytest.raises(RuntimeError):
        guard._find_monorepo_root(tmp_path)


def test_load_orchestrator_uses_import(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # Track calls to run_for_project
    calls: list[tuple[Path, Path]] = []

    def mock_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 123

    # Create a module-like stub with the attribute we need
    class _StubModule(ModuleType):
        run_for_project: _RunForProject

    stub = _StubModule("monorepo_guards.orchestrator")
    stub.run_for_project = mock_run_for_project

    real_import = __import__

    def fake_import(
        name: str,
        globals_: Mapping[str, ModuleType] | None = None,
        locals_: Mapping[str, ModuleType] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "monorepo_guards.orchestrator":
            return stub
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    rc = guard._load_orchestrator(tmp_path)
    assert rc is mock_run_for_project


def test_main_invokes_overrides(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # Force deterministic paths and orchestrator behavior
    def fake_find_root(_p: Path) -> Path:
        return tmp_path

    monkeypatch.setattr(guard, "_find_monorepo_root", fake_find_root)

    calls: list[tuple[Path, Path]] = []

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 7

    def fake_load(_r: Path) -> _RunForProject:
        return fake_run_for_project

    monkeypatch.setattr(guard, "_load_orchestrator", fake_load)

    rc = guard.main(["--root", str(tmp_path / "alt"), "--verbose"])
    assert rc == 7
    assert calls == [(tmp_path, (tmp_path / "alt").resolve())]


def test_guard_main_entry_via_module(tmp_path: Path) -> None:
    # Test the if __name__ == "__main__" block (line 62)
    # by actually executing the script as __main__
    import runpy

    # Execute the script file with __name__ == "__main__"
    script_path = Path(__file__).parent.parent.parent / "scripts" / "guard.py"

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(script_path), run_name="__main__")

    # The script runs successfully and exits with code 0
    assert exc_info.value.code == 0
