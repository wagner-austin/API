"""Tests for scripts.guard entrypoint.

Uses _hooks_guard to inject fake orchestrator so tests don't scan the real monorepo.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.guard import _find_monorepo_root_impl, main

from cleargbm import _hooks_guard


def _install_fake_guard_hooks(tmp_path: Path) -> None:
    """Install fake guard hooks that use tmp_path as monorepo root."""

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _hooks_guard.RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    _hooks_guard.guard_find_monorepo_root = _FakeFindRoot()
    _hooks_guard.guard_load_orchestrator = _FakeLoader()


@pytest.fixture(autouse=True)
def _reset_guard_hooks() -> Generator[None, None, None]:
    """Reset guard hooks after each test."""
    yield
    _hooks_guard.guard_find_monorepo_root = None
    _hooks_guard.guard_load_orchestrator = None


def test_guard_entrypoint_runs_as_main(tmp_path: Path) -> None:
    """Guard module can be run as __main__."""
    _install_fake_guard_hooks(tmp_path)

    orig_argv = sys.argv
    sys.argv = ["guard", "--root", str(tmp_path)]

    try:
        if "scripts.guard" in sys.modules:
            del sys.modules["scripts.guard"]
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(
                str(Path(__file__).resolve().parents[1] / "scripts" / "guard.py"),
                run_name="__main__",
            )
        code = exc.value.code if isinstance(exc.value.code, int) else 0
        assert code == 0
    finally:
        sys.argv = orig_argv


def test_main_with_verbose_flag(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Guard main runs with verbose flag."""
    _install_fake_guard_hooks(tmp_path)
    rc = main(["--root", str(tmp_path), "--verbose"])
    captured = capsys.readouterr()
    assert captured.out.endswith(f"guard_exit_code code={rc}\n")
    assert rc == 0


def test_main_with_root_override(tmp_path: Path) -> None:
    """Guard main runs with root override."""
    _install_fake_guard_hooks(tmp_path)
    rc = main(["--root", str(tmp_path)])
    assert rc == 0


def test_main_with_short_verbose_flag(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Guard main runs with short verbose flag."""
    _install_fake_guard_hooks(tmp_path)
    rc = main(["--root", str(tmp_path), "-v"])
    captured = capsys.readouterr()
    assert captured.out.endswith(f"guard_exit_code code={rc}\n")
    assert rc == 0


def test_main_with_unknown_arg(tmp_path: Path) -> None:
    """Guard main ignores unknown arguments."""
    _install_fake_guard_hooks(tmp_path)
    rc = main(["--root", str(tmp_path), "--unknown-flag"])
    assert rc == 0


def test_find_monorepo_root_impl_raises_when_not_found(tmp_path: Path) -> None:
    """_find_monorepo_root_impl raises RuntimeError when root not found."""
    with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        _find_monorepo_root_impl(tmp_path)


def test_find_monorepo_root_impl_finds_libs_dir(tmp_path: Path) -> None:
    """_find_monorepo_root_impl finds directory with libs folder."""
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    nested = tmp_path / "services" / "project"
    nested.mkdir(parents=True)
    result = _find_monorepo_root_impl(nested)
    assert result == tmp_path


def test_find_monorepo_root_uses_impl_when_hook_is_none() -> None:
    """_find_monorepo_root uses impl when hook is None (production path)."""
    from scripts.guard import _find_monorepo_root

    _hooks_guard.guard_find_monorepo_root = None
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    result = _find_monorepo_root(project_root)
    assert (result / "libs").is_dir()


def test_load_orchestrator_impl_loads_module() -> None:
    """_load_orchestrator_impl loads the real orchestrator module."""
    from scripts.guard import _load_orchestrator_impl

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    monorepo_root = _find_monorepo_root_impl(project_root)
    run_for_project = _load_orchestrator_impl(monorepo_root)
    assert callable(run_for_project)


def test_load_orchestrator_uses_impl_when_hook_is_none() -> None:
    """_load_orchestrator uses impl when hook is None (production path)."""
    from scripts.guard import _load_orchestrator

    _hooks_guard.guard_load_orchestrator = None
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    monorepo_root = _find_monorepo_root_impl(project_root)
    run_for_project = _load_orchestrator(monorepo_root)
    assert callable(run_for_project)


def test_verbose_prints_nonzero_exit_code(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Verbose flag prints nonzero exit code."""

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _hooks_guard.RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 7

            return _run_for_project

    _hooks_guard.guard_find_monorepo_root = _FakeFindRoot()
    _hooks_guard.guard_load_orchestrator = _FakeLoader()

    rc = main(["--root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 7
    assert "guard_exit_code code=7" in out
