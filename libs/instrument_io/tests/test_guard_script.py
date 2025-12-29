"""Tests for scripts/guard.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import scripts._test_hooks as test_hooks
import scripts.guard as guard


def test_find_monorepo_root_locates_libs(tmp_path: Path) -> None:
    """Test _find_monorepo_root finds directory containing libs."""
    root = tmp_path / "repo"
    libs = root / "libs"
    libs.mkdir(parents=True, exist_ok=True)
    start = libs / "some" / "deep"
    start.mkdir(parents=True, exist_ok=True)

    found = guard._find_monorepo_root(start)
    assert found == root


def test_find_monorepo_root_raises_when_missing(tmp_path: Path) -> None:
    """Test _find_monorepo_root raises when libs directory not found."""
    start = tmp_path / "no_libs"
    start.mkdir(parents=True, exist_ok=True)

    # Override is_dir to always return False
    original_is_dir = test_hooks.is_dir

    def _fake_is_dir(path: Path) -> bool:
        return False

    test_hooks.is_dir = _fake_is_dir

    with pytest.raises(RuntimeError):
        guard._find_monorepo_root(start)

    test_hooks.is_dir = original_is_dir


def test_load_orchestrator_imports_run(tmp_path: Path) -> None:
    """Test _load_orchestrator imports run_for_project from orchestrator."""
    libs_dir = tmp_path / "libs"
    mg_src = libs_dir / "monorepo_guards" / "src" / "monorepo_guards"
    mg_src.mkdir(parents=True, exist_ok=True)
    (mg_src / "__init__.py").write_text("", encoding="utf-8")
    (mg_src / "orchestrator.py").write_text(
        "from pathlib import Path\n"
        "def run_for_project(*, monorepo_root: Path, project_root: Path) -> int:\n"
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


def test_main_invokes_run_and_supports_flags(tmp_path: Path) -> None:
    """Test main invokes run_for_project and supports --root and -v flags."""
    # Use real monorepo - test that flags are parsed correctly
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)

    # main() uses real orchestrator which returns 0 for valid projects
    rc = guard.main(["--root", str(project_root), "-v"])
    assert rc == 0


def test_main_uses_default_args_when_none() -> None:
    """Test main uses default project root when argv is None."""
    rc = guard.main(None)
    assert rc == 0


def test_main_skips_unknown_flags() -> None:
    """Test main skips unknown flags without error."""
    rc = guard.main(["--unknown", "--verbose"])
    assert rc == 0


def test_main_verbose_flag_only() -> None:
    """Test main handles --verbose flag alone."""
    rc = guard.main(["--verbose"])
    assert rc == 0


def test_guard_main_entry_via_module() -> None:
    """Test if __name__ == '__main__' block executes correctly."""
    import runpy

    script_path = Path(__file__).parent.parent / "scripts" / "guard.py"

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(script_path), run_name="__main__")

    assert exc_info.value.code == 0
