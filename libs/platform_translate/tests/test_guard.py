"""Tests for scripts/guard.py module."""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture
from scripts.guard import _find_monorepo_root_impl
from scripts.guard import main as guard_main

from platform_translate import _test_hooks


def test_guard_main_with_root(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Main with --root uses specified root."""

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
    """Main ignores unrecognized arguments."""

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

    code = guard_main(["--root", str(tmp_path), "--unknown-flag"])
    assert code == 0


def test_guard_run_as_main(tmp_path: Path) -> None:
    """Main runs successfully with hooks set."""

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
    """Raises RuntimeError when libs directory not found."""
    start = tmp_path / "nested"
    start.mkdir()
    with pytest.raises(RuntimeError, match="monorepo root"):
        _ = _find_monorepo_root_impl(start)


def test_guard_verbose_prints_exit_code(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Verbose flag prints exit code."""
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
    assert out == "guard_exit_code code=7\n"
    assert calls["monorepo_root"] == tmp_path
    assert calls["project_root"] == tmp_path


def test_find_monorepo_root_impl_finds_libs_dir(tmp_path: Path) -> None:
    """Finds directory with libs folder."""
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    nested = tmp_path / "services" / "translator"
    nested.mkdir(parents=True)

    result = _find_monorepo_root_impl(nested)
    assert result == tmp_path


def test_load_orchestrator_impl_loads_module() -> None:
    """Loads the real orchestrator module."""
    from scripts.guard import _load_orchestrator_impl

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    monorepo_root = _find_monorepo_root_impl(project_root)

    run_for_project = _load_orchestrator_impl(monorepo_root)
    assert callable(run_for_project)


def test_find_monorepo_root_uses_impl_when_hook_is_none() -> None:
    """Uses impl when hook is None (production path)."""
    from scripts.guard import _find_monorepo_root

    orig_hook = _test_hooks.guard_find_monorepo_root
    _test_hooks.guard_find_monorepo_root = None

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    result = _find_monorepo_root(project_root)
    assert (result / "libs").is_dir()

    _test_hooks.guard_find_monorepo_root = orig_hook


def test_load_orchestrator_uses_impl_when_hook_is_none() -> None:
    """Uses impl when hook is None (production path)."""
    from scripts.guard import _load_orchestrator

    orig_hook = _test_hooks.guard_load_orchestrator
    _test_hooks.guard_load_orchestrator = None

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    monorepo_root = _find_monorepo_root_impl(project_root)

    run_for_project = _load_orchestrator(monorepo_root)
    assert callable(run_for_project)

    _test_hooks.guard_load_orchestrator = orig_hook


def test_guard_main_entry_via_runpy(tmp_path: Path) -> None:
    """Running as script raises SystemExit."""
    import runpy
    import sys

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

    orig_argv = sys.argv
    sys.argv = ["guard", "--root", str(tmp_path)]

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(
            str(Path(__file__).parents[1] / "scripts" / "guard.py"),
            run_name="__main__",
        )
    assert exc_info.value.code == 0

    sys.argv = orig_argv


def test_guard_short_verbose_flag(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Short verbose flag -v prints exit code."""

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

    rc = guard_main(["--root", str(tmp_path), "-v"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out == "guard_exit_code code=0\n"


def test_guard_main_default_project_root(tmp_path: Path) -> None:
    """Main uses project root when no --root specified."""
    captured_roots: dict[str, Path] = {}

    class _FakeFindRoot:
        def __call__(self, start: Path) -> Path:
            captured_roots["start"] = start
            return tmp_path

    class _FakeLoader:
        def __call__(self, monorepo_root: Path) -> _test_hooks.RunForProjectProto:
            def _run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
                captured_roots["project_root"] = project_root
                return 0

            return _run_for_project

    _test_hooks.guard_find_monorepo_root = _FakeFindRoot()
    _test_hooks.guard_load_orchestrator = _FakeLoader()

    code = guard_main([])
    assert code == 0
    # When no --root, project_root should be script's parent directory
    assert captured_roots["project_root"] == captured_roots["start"]
