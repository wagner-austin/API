from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture
from scripts.guard import _find_monorepo_root_impl
from scripts.guard import main as guard_main

from model_trainer.core import _test_hooks


def test_guard_main_with_root(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    # Set up hooks to make guard work with tmp_path as monorepo root
    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    _test_hooks.guard_find_monorepo_root = _FakeFindRoot()
    _test_hooks.guard_load_orchestrator = _FakeLoader()

    code = guard_main(["--root", str(tmp_path)])
    _ = capsys.readouterr()
    assert code == 0


def test_guard_main_unrecognized_arg(tmp_path: Path) -> None:
    # Exercise the else path in the arg loop
    # Set up hooks to make guard work with tmp_path
    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    _test_hooks.guard_find_monorepo_root = _FakeFindRoot()
    _test_hooks.guard_load_orchestrator = _FakeLoader()

    code = guard_main(["--root", str(tmp_path), "--unknown-flag"])  # unknown is ignored
    assert code == 0


def test_guard_run_as_main(tmp_path: Path) -> None:
    # Cover __main__ entry point by calling main() directly with args
    # Set up hooks to make guard use tmp_path as root
    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    _test_hooks.guard_find_monorepo_root = _FakeFindRoot()
    _test_hooks.guard_load_orchestrator = _FakeLoader()

    code = guard_main(["--root", str(tmp_path)])
    assert code == 0


def test_guard_find_monorepo_root_raises(tmp_path: Path) -> None:
    start = tmp_path / "nested"
    start.mkdir()
    # Use the impl function directly to test the RuntimeError path
    with pytest.raises(RuntimeError):
        _ = _find_monorepo_root_impl(start)


def test_guard_verbose_prints_exit_code(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    calls: dict[str, Path] = {}

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProto:
            calls["monorepo_root"] = monorepo_root

            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                calls["project_root"] = project_root
                return 7

            return _run_for_project

    _test_hooks.guard_find_monorepo_root = _FakeFindRoot()
    _test_hooks.guard_load_orchestrator = _FakeLoader()

    rc = guard_main(["--root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 7
    assert "guard_exit_code code=7" in out
    assert calls["monorepo_root"] == tmp_path
    assert calls["project_root"] == tmp_path


def test_find_monorepo_root_impl_finds_libs_dir(tmp_path: Path) -> None:
    """Test _find_monorepo_root_impl finds directory with libs folder."""
    # Create a directory structure with libs folder
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    nested = tmp_path / "services" / "model-trainer"
    nested.mkdir(parents=True)

    # Start from nested directory and search upward
    result = _find_monorepo_root_impl(nested)
    assert result == tmp_path


def test_load_orchestrator_impl_loads_module() -> None:
    """Test _load_orchestrator_impl loads the real orchestrator module."""
    from scripts.guard import _load_orchestrator_impl

    # Use the actual monorepo root
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # tests/scripts -> project root
    monorepo_root = _find_monorepo_root_impl(project_root)

    # Load the orchestrator
    run_for_project = _load_orchestrator_impl(monorepo_root)
    # Should be callable
    assert callable(run_for_project)


def test_find_monorepo_root_uses_impl_when_hook_is_none() -> None:
    """Test _find_monorepo_root uses impl when hook is None (production path)."""
    from scripts.guard import _find_monorepo_root

    # Ensure hook is None
    orig_hook = _test_hooks.guard_find_monorepo_root
    _test_hooks.guard_find_monorepo_root = None

    try:
        # Use a path that will find the real monorepo
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]

        # This should use _find_monorepo_root_impl internally
        result = _find_monorepo_root(project_root)
        # Should have libs directory
        assert (result / "libs").is_dir()
    finally:
        _test_hooks.guard_find_monorepo_root = orig_hook


def test_load_orchestrator_uses_impl_when_hook_is_none() -> None:
    """Test _load_orchestrator uses impl when hook is None (production path)."""
    from scripts.guard import _load_orchestrator

    # Ensure hook is None
    orig_hook = _test_hooks.guard_load_orchestrator
    _test_hooks.guard_load_orchestrator = None

    try:
        # Get the real monorepo root
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]
        monorepo_root = _find_monorepo_root_impl(project_root)

        # This should use _load_orchestrator_impl internally
        run_for_project = _load_orchestrator(monorepo_root)
        # Should be callable
        assert callable(run_for_project)
    finally:
        _test_hooks.guard_load_orchestrator = orig_hook


def test_guard_main_entry_via_runpy(tmp_path: Path) -> None:
    """Test the if __name__ == '__main__' block via runpy."""
    import runpy
    import sys

    # Set up hooks so guard succeeds
    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                return 0

            return _run_for_project

    _test_hooks.guard_find_monorepo_root = _FakeFindRoot()
    _test_hooks.guard_load_orchestrator = _FakeLoader()

    # Save original argv
    orig_argv = sys.argv
    sys.argv = ["guard", "--root", str(tmp_path)]

    try:
        # Run the module as __main__ - this covers line 84
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(
                str(Path(__file__).parents[2] / "scripts" / "guard.py"),
                run_name="__main__",
            )
        assert exc_info.value.code == 0
    finally:
        sys.argv = orig_argv
