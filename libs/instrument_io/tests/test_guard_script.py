"""Tests for scripts/guard.py."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest
import scripts.guard as guard


class _RunCall(TypedDict):
    monorepo_root: Path
    project_root: Path


def test_find_monorepo_root_locates_libs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    libs = root / "libs"
    libs.mkdir(parents=True, exist_ok=True)
    start = libs / "some" / "deep"
    start.mkdir(parents=True, exist_ok=True)

    found = guard._find_monorepo_root(start)
    assert found == root


def test_find_monorepo_root_raises_when_missing(tmp_path: Path) -> None:
    start = tmp_path / "no_libs"
    start.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError):
        _ = guard._find_monorepo_root(start)


def test_load_orchestrator_imports_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    libs_dir = tmp_path / "libs"
    mg_src = libs_dir / "monorepo_guards" / "src" / "monorepo_guards"
    mg_src.mkdir(parents=True, exist_ok=True)
    (mg_src / "__init__.py").write_text("", encoding="utf-8")
    # Write orchestrator that stores call info in a file and returns int
    (mg_src / "orchestrator.py").write_text(
        "from pathlib import Path\n"
        "def run_for_project(monorepo_root, project_root):\n"
        "    (monorepo_root / 'call_info.txt').write_text("
        "f'{project_root}', encoding='utf-8')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    run = guard._load_orchestrator(tmp_path)
    result = run(monorepo_root=tmp_path, project_root=tmp_path / "proj")
    assert result == 0
    call_info = (tmp_path / "call_info.txt").read_text(encoding="utf-8")
    assert call_info == str(tmp_path / "proj")


def test_main_invokes_run_and_supports_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 3

    def _fake_find_monorepo_root(start: Path) -> Path:
        return start

    def _fake_load_orchestrator(_: Path) -> guard._RunForProject:
        return _fake_run_for_project

    monkeypatch.setattr(guard, "_find_monorepo_root", _fake_find_monorepo_root)
    monkeypatch.setattr(guard, "_load_orchestrator", _fake_load_orchestrator)

    project_root = Path(__file__).resolve().parents[1]

    rc = guard.main(["--root", str(tmp_path), "-v"])
    assert rc == 3
    assert len(calls) == 1
    assert calls[0]["monorepo_root"] == project_root
    assert calls[0]["project_root"] == tmp_path


def test_main_uses_default_args_when_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 0

    def _fake_find(start: Path) -> Path:
        return start

    def _fake_load(_: Path) -> guard._RunForProject:
        return _fake_run_for_project

    monkeypatch.setattr(guard, "_find_monorepo_root", _fake_find)
    monkeypatch.setattr(guard, "_load_orchestrator", _fake_load)

    rc = guard.main(None)
    assert rc == 0
    assert calls[0]["project_root"].name == "instrument_io"


def test_main_skips_unknown_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 7

    def _fake_find(start: Path) -> Path:
        return start

    def _fake_load(_: Path) -> guard._RunForProject:
        return _fake_run_for_project

    monkeypatch.setattr(guard, "_find_monorepo_root", _fake_find)
    monkeypatch.setattr(guard, "_load_orchestrator", _fake_load)

    rc = guard.main(["--unknown", "--root", str(tmp_path)])
    assert rc == 7
    assert len(calls) == 1
    assert calls[0]["project_root"] == tmp_path


def test_main_verbose_flag_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 5

    def _fake_find(start: Path) -> Path:
        return start

    def _fake_load(_: Path) -> guard._RunForProject:
        return _fake_run_for_project

    monkeypatch.setattr(guard, "_find_monorepo_root", _fake_find)
    monkeypatch.setattr(guard, "_load_orchestrator", _fake_load)

    rc = guard.main(["--verbose"])
    assert rc == 5
    assert len(calls) == 1


def test_guard_main_entry_via_module() -> None:
    # Test the if __name__ == "__main__" block (line 62)
    # by actually executing the script as __main__
    import runpy

    # Execute the script file with __name__ == "__main__"
    script_path = Path(__file__).parent.parent / "scripts" / "guard.py"

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(script_path), run_name="__main__")

    # The script runs successfully and exits with code 0
    assert exc_info.value.code == 0


def test_run_local_rules_no_tests_dir(tmp_path: Path) -> None:
    """Test _run_local_rules when tests directory doesn't exist."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    rc = guard._run_local_rules(project_root)
    assert rc == 0


def test_run_local_rules_with_violation(tmp_path: Path) -> None:
    """Test _run_local_rules when mock violations are found."""
    project_root = tmp_path / "project"
    tests_dir = project_root / "tests"
    tests_dir.mkdir(parents=True)

    # Create a test file with a mock import
    test_file = tests_dir / "test_something.py"
    test_file.write_text("from unittest.mock import patch\n", encoding="utf-8")

    rc = guard._run_local_rules(project_root)
    assert rc == 2


def test_main_with_verbose_and_local_violations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test main with verbose flag when local rules have violations."""

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        return 0

    def _fake_find(start: Path) -> Path:
        return start

    def _fake_load(_: Path) -> guard._RunForProject:
        return _fake_run_for_project

    monkeypatch.setattr(guard, "_find_monorepo_root", _fake_find)
    monkeypatch.setattr(guard, "_load_orchestrator", _fake_load)

    # Create project with mock violation
    project_root = tmp_path / "project"
    tests_dir = project_root / "tests"
    tests_dir.mkdir(parents=True)
    test_file = tests_dir / "test_something.py"
    test_file.write_text("from unittest.mock import patch\n", encoding="utf-8")

    rc = guard.main(["--root", str(project_root), "--verbose"])
    assert rc == 2


def test_run_local_rules_skips_directories_named_py(tmp_path: Path) -> None:
    """Test _run_local_rules skips directories ending in .py.

    Covers branch 47->46 where path.is_file() returns False.
    """
    project_root = tmp_path / "project"
    tests_dir = project_root / "tests"
    tests_dir.mkdir(parents=True)

    # Create a directory named with .py extension (not a file)
    fake_py_dir = tests_dir / "fake_module.py"
    fake_py_dir.mkdir()

    # Create a real test file that's clean
    test_file = tests_dir / "test_real.py"
    test_file.write_text("import pytest\n", encoding="utf-8")

    rc = guard._run_local_rules(project_root)
    assert rc == 0
