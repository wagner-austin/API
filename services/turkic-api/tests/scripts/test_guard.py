from __future__ import annotations

from pathlib import Path

import pytest
import scripts.guard as guard
from _pytest.capture import CaptureFixture

from turkic_api import _test_hooks
from turkic_api._test_hooks import GuardRunForProjectProtocol


def test_guard_main_verbose_and_unknown_token(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    # Use a temp root so guard scans an empty project (no violations)
    orig_find = _test_hooks.guard_find_monorepo_root
    orig_load = _test_hooks.guard_load_orchestrator

    def fake_find_root(start: Path) -> Path:
        return tmp_path

    def fake_run(*, monorepo_root: Path, project_root: Path) -> int:
        return 0

    def fake_load(monorepo_root: Path) -> GuardRunForProjectProtocol:
        return fake_run

    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load
    try:
        code = guard.main(["--verbose", "--root", str(tmp_path), "unknown-flag"])
        out = capsys.readouterr().out
        assert code == 0
        assert "guard_exit_code code=0" in out
    finally:
        _test_hooks.guard_find_monorepo_root = orig_find
        _test_hooks.guard_load_orchestrator = orig_load


def test_guard_main_non_verbose_path(tmp_path: Path) -> None:
    # Exercise the non-verbose branch where no extra line is printed
    orig_find = _test_hooks.guard_find_monorepo_root
    orig_load = _test_hooks.guard_load_orchestrator

    def fake_find_root(start: Path) -> Path:
        return tmp_path

    def fake_run(*, monorepo_root: Path, project_root: Path) -> int:
        return 0

    def fake_load(monorepo_root: Path) -> GuardRunForProjectProtocol:
        return fake_run

    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load
    try:
        code = guard.main(["--root", str(tmp_path)])
        assert code == 0
    finally:
        _test_hooks.guard_find_monorepo_root = orig_find
        _test_hooks.guard_load_orchestrator = orig_load


def test_find_monorepo_root_traversal(tmp_path: Path) -> None:
    # Build nested structure to force while loop to climb before finding libs
    root = tmp_path / "repo"
    nested = root / "a" / "b" / "c"
    (root / "libs").mkdir(parents=True)
    nested.mkdir(parents=True)

    # Use the default implementation to test real traversal logic
    found = _test_hooks._default_guard_find_monorepo_root(nested)
    assert found == root

    # If libs is never found, raise at filesystem root
    with pytest.raises(RuntimeError):
        _test_hooks._default_guard_find_monorepo_root(tmp_path)


def test_load_orchestrator_uses_hook(tmp_path: Path) -> None:
    # Track calls to run_for_project
    orig_load = _test_hooks.guard_load_orchestrator
    calls: list[tuple[Path, Path]] = []

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 123

    def fake_load(monorepo_root: Path) -> GuardRunForProjectProtocol:
        return fake_run_for_project

    _test_hooks.guard_load_orchestrator = fake_load
    try:
        rc = guard._load_orchestrator(tmp_path)
        assert rc is fake_run_for_project
    finally:
        _test_hooks.guard_load_orchestrator = orig_load


def test_main_invokes_overrides(tmp_path: Path) -> None:
    # Force deterministic paths and orchestrator behavior
    orig_find = _test_hooks.guard_find_monorepo_root
    orig_load = _test_hooks.guard_load_orchestrator

    def fake_find_root(start: Path) -> Path:
        return tmp_path

    calls: list[tuple[Path, Path]] = []

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 7

    def fake_load(monorepo_root: Path) -> GuardRunForProjectProtocol:
        return fake_run_for_project

    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load
    try:
        rc = guard.main(["--root", str(tmp_path / "alt"), "--verbose"])
        assert rc == 7
        assert calls == [(tmp_path, (tmp_path / "alt").resolve())]
    finally:
        _test_hooks.guard_find_monorepo_root = orig_find
        _test_hooks.guard_load_orchestrator = orig_load


def test_guard_main_entry_via_module(tmp_path: Path) -> None:
    # Test the if __name__ == "__main__" block
    # by actually executing the script as __main__
    import runpy

    # Execute the script file with __name__ == "__main__"
    script_path = Path(__file__).parent.parent.parent / "scripts" / "guard.py"

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(script_path), run_name="__main__")

    # The script runs successfully and exits with code 0
    assert exc_info.value.code == 0
