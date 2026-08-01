"""Tests for guard CLI script."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture
from scripts.guard import _find_monorepo_root_impl
from scripts.guard import main as guard_main

from platform_calendar.testing import RunForProjectProto, hooks, reset_hooks


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


def test_guard_main_with_root(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test guard main function with --root argument."""

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    hooks.guard_find_monorepo_root = _FakeFindRoot()
    hooks.guard_load_orchestrator = _FakeLoader()

    code = guard_main(["--root", str(tmp_path)])
    _ = capsys.readouterr()
    assert code == 0


def test_guard_main_unrecognized_arg(tmp_path: Path) -> None:
    """Test guard handles unrecognized arguments gracefully."""

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    hooks.guard_find_monorepo_root = _FakeFindRoot()
    hooks.guard_load_orchestrator = _FakeLoader()

    code = guard_main(["--root", str(tmp_path), "--unknown-flag"])
    assert code == 0


def test_guard_run_as_main(tmp_path: Path) -> None:
    """Test guard main function with no arguments."""

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    hooks.guard_find_monorepo_root = _FakeFindRoot()
    hooks.guard_load_orchestrator = _FakeLoader()

    code = guard_main(["--root", str(tmp_path)])
    assert code == 0


def test_guard_find_monorepo_root_raises(tmp_path: Path) -> None:
    """Test _find_monorepo_root_impl raises when no libs dir found."""
    start = tmp_path / "nested"
    start.mkdir()
    with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        _ = _find_monorepo_root_impl(start)


def test_guard_verbose_prints_exit_code(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test verbose mode prints exit code."""
    calls: dict[str, Path] = {}

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> RunForProjectProto:
            calls["monorepo_root"] = monorepo_root

            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                calls["project_root"] = project_root
                return 7

            return _run_for_project

    hooks.guard_find_monorepo_root = _FakeFindRoot()
    hooks.guard_load_orchestrator = _FakeLoader()

    rc = guard_main(["--root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 7
    assert "guard_exit_code code=7" in out
    assert calls["monorepo_root"] == tmp_path
    assert calls["project_root"] == tmp_path


def test_find_monorepo_root_impl_finds_libs_dir(tmp_path: Path) -> None:
    """Test _find_monorepo_root_impl finds directory with libs folder."""
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    nested = tmp_path / "services" / "model-trainer"
    nested.mkdir(parents=True)

    result = _find_monorepo_root_impl(nested)
    assert result == tmp_path


def test_load_orchestrator_impl_loads_module() -> None:
    """Test _load_orchestrator_impl loads the real orchestrator module."""
    from scripts.guard import _load_orchestrator_impl

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    monorepo_root = _find_monorepo_root_impl(project_root)

    run_for_project = _load_orchestrator_impl(monorepo_root)
    assert callable(run_for_project)


def test_find_monorepo_root_uses_impl_when_hook_is_none() -> None:
    """Test _find_monorepo_root uses impl when hook is None."""
    from scripts.guard import _find_monorepo_root

    hooks.guard_find_monorepo_root = None

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    result = _find_monorepo_root(project_root)
    assert (result / "libs").is_dir()


def test_load_orchestrator_uses_impl_when_hook_is_none() -> None:
    """Test _load_orchestrator uses impl when hook is None."""
    from scripts.guard import _load_orchestrator

    hooks.guard_load_orchestrator = None

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    monorepo_root = _find_monorepo_root_impl(project_root)

    run_for_project = _load_orchestrator(monorepo_root)
    assert callable(run_for_project)


def test_guard_main_entry_via_runpy(tmp_path: Path) -> None:
    """Test the if __name__ == '__main__' block via runpy."""
    import runpy
    import sys

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    hooks.guard_find_monorepo_root = _FakeFindRoot()
    hooks.guard_load_orchestrator = _FakeLoader()

    orig_argv = sys.argv
    sys.argv = ["guard", "--root", str(tmp_path)]

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(
                str(Path(__file__).parents[2] / "scripts" / "guard.py"),
                run_name="__main__",
            )
        assert exc_info.value.code == 0
    finally:
        sys.argv = orig_argv


def test_guard_short_verbose_flag(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test short -v verbose flag."""

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 3

            return _run_for_project

    hooks.guard_find_monorepo_root = _FakeFindRoot()
    hooks.guard_load_orchestrator = _FakeLoader()

    rc = guard_main(["--root", str(tmp_path), "-v"])
    out = capsys.readouterr().out
    assert rc == 3
    assert "guard_exit_code code=3" in out


def test_guard_main_no_argv_uses_sys_argv(tmp_path: Path) -> None:
    """Test guard main with None argv uses sys.argv."""
    import sys

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    hooks.guard_find_monorepo_root = _FakeFindRoot()
    hooks.guard_load_orchestrator = _FakeLoader()

    orig_argv = sys.argv
    sys.argv = ["guard", "--root", str(tmp_path)]

    try:
        code = guard_main(None)
        assert code == 0
    finally:
        sys.argv = orig_argv
