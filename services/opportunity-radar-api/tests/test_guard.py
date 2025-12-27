"""Tests for guard script."""

from __future__ import annotations

import io
import runpy
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import scripts.guard as guard_module

from opportunity_radar_api import _test_hooks


def test_main_with_verbose_flag() -> None:
    """Test main with verbose flag outputs exit code."""
    calls: list[tuple[Path, Path]] = []

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 0

    def fake_is_dir(path: Path) -> bool:
        return path.name == "libs"

    def fake_find_root(start: Path) -> Path:
        current = start
        while True:
            if fake_is_dir(current / "libs"):
                return current
            if current.parent == current:
                raise RuntimeError("monorepo root with 'libs' directory not found")
            current = current.parent

    def fake_load(monorepo_root: Path) -> guard_module._RunForProject:
        _ = monorepo_root
        return fake_run_for_project

    original_is_dir = _test_hooks.guard_is_dir
    original_find = _test_hooks.guard_find_monorepo_root
    original_load = _test_hooks.guard_load_orchestrator

    _test_hooks.guard_is_dir = fake_is_dir
    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load

    try:
        f = io.StringIO()
        with redirect_stdout(f):
            result = guard_module.main(["--verbose"])

        output = f.getvalue()
        assert "guard_exit_code code=0" in output
        assert result == 0
        assert len(calls) == 1
    finally:
        _test_hooks.guard_is_dir = original_is_dir
        _test_hooks.guard_find_monorepo_root = original_find
        _test_hooks.guard_load_orchestrator = original_load


def test_main_with_root_override(tmp_path: Path) -> None:
    """Test main with --root override."""
    calls: list[tuple[Path, Path]] = []

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 0

    def fake_is_dir(path: Path) -> bool:
        return path.name == "libs"

    def fake_find_root(start: Path) -> Path:
        current = start
        while True:
            if fake_is_dir(current / "libs"):
                return current
            if current.parent == current:
                raise RuntimeError("monorepo root with 'libs' directory not found")
            current = current.parent

    def fake_load(monorepo_root: Path) -> guard_module._RunForProject:
        _ = monorepo_root
        return fake_run_for_project

    original_is_dir = _test_hooks.guard_is_dir
    original_find = _test_hooks.guard_find_monorepo_root
    original_load = _test_hooks.guard_load_orchestrator

    _test_hooks.guard_is_dir = fake_is_dir
    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load

    try:
        result = guard_module.main(["--root", str(tmp_path)])

        assert result == 0
        assert len(calls) == 1
        _, project_root = calls[0]
        assert project_root == tmp_path.resolve()
    finally:
        _test_hooks.guard_is_dir = original_is_dir
        _test_hooks.guard_find_monorepo_root = original_find
        _test_hooks.guard_load_orchestrator = original_load


def test_main_with_short_verbose_flag() -> None:
    """Test main with -v short flag."""
    calls: list[tuple[Path, Path]] = []

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 0

    def fake_is_dir(path: Path) -> bool:
        return path.name == "libs"

    def fake_find_root(start: Path) -> Path:
        current = start
        while True:
            if fake_is_dir(current / "libs"):
                return current
            if current.parent == current:
                raise RuntimeError("monorepo root with 'libs' directory not found")
            current = current.parent

    def fake_load(monorepo_root: Path) -> guard_module._RunForProject:
        _ = monorepo_root
        return fake_run_for_project

    original_is_dir = _test_hooks.guard_is_dir
    original_find = _test_hooks.guard_find_monorepo_root
    original_load = _test_hooks.guard_load_orchestrator

    _test_hooks.guard_is_dir = fake_is_dir
    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load

    try:
        f = io.StringIO()
        with redirect_stdout(f):
            result = guard_module.main(["-v"])

        output = f.getvalue()
        assert "guard_exit_code code=0" in output
        assert result == 0
    finally:
        _test_hooks.guard_is_dir = original_is_dir
        _test_hooks.guard_find_monorepo_root = original_find
        _test_hooks.guard_load_orchestrator = original_load


def test_main_with_other_args() -> None:
    """Test main ignores unknown args."""
    calls: list[tuple[Path, Path]] = []

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 0

    def fake_is_dir(path: Path) -> bool:
        return path.name == "libs"

    def fake_find_root(start: Path) -> Path:
        current = start
        while True:
            if fake_is_dir(current / "libs"):
                return current
            if current.parent == current:
                raise RuntimeError("monorepo root with 'libs' directory not found")
            current = current.parent

    def fake_load(monorepo_root: Path) -> guard_module._RunForProject:
        _ = monorepo_root
        return fake_run_for_project

    original_is_dir = _test_hooks.guard_is_dir
    original_find = _test_hooks.guard_find_monorepo_root
    original_load = _test_hooks.guard_load_orchestrator

    _test_hooks.guard_is_dir = fake_is_dir
    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load

    try:
        result = guard_module.main(["--unknown", "arg"])

        assert result == 0
        assert len(calls) == 1
    finally:
        _test_hooks.guard_is_dir = original_is_dir
        _test_hooks.guard_find_monorepo_root = original_find
        _test_hooks.guard_load_orchestrator = original_load


def test_find_monorepo_root_impl_raises_when_not_found() -> None:
    """Test _find_monorepo_root_impl raises when libs dir not found."""
    original_is_dir = _test_hooks.guard_is_dir

    def fake_is_dir(path: Path) -> bool:
        _ = path
        return False  # Never find libs

    _test_hooks.guard_is_dir = fake_is_dir

    try:
        with pytest.raises(RuntimeError, match="libs"):
            guard_module._find_monorepo_root_impl(Path("/some/path"))
    finally:
        _test_hooks.guard_is_dir = original_is_dir


def test_default_is_dir() -> None:
    """Test _default_is_dir uses Path.is_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        assert guard_module._default_is_dir(path) is True
        assert guard_module._default_is_dir(path / "nonexistent") is False


def test_main_runpy() -> None:
    """Test guard can be run as __main__."""

    def fake_is_dir(path: Path) -> bool:
        return path.name == "libs"

    def fake_find_root(start: Path) -> Path:
        current = start
        while True:
            if fake_is_dir(current / "libs"):
                return current
            if current.parent == current:
                raise RuntimeError("monorepo root with 'libs' directory not found")
            current = current.parent

    calls: list[tuple[Path, Path]] = []

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append((monorepo_root, project_root))
        return 0

    def fake_load(monorepo_root: Path) -> guard_module._RunForProject:
        _ = monorepo_root
        return fake_run_for_project

    original_is_dir = _test_hooks.guard_is_dir
    original_find = _test_hooks.guard_find_monorepo_root
    original_load = _test_hooks.guard_load_orchestrator

    _test_hooks.guard_is_dir = fake_is_dir
    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load

    try:
        # Test the if __name__ == "__main__" branch
        result = guard_module.main(None)
        assert result == 0
    finally:
        _test_hooks.guard_is_dir = original_is_dir
        _test_hooks.guard_find_monorepo_root = original_find
        _test_hooks.guard_load_orchestrator = original_load


def test_load_orchestrator_impl_real() -> None:
    """Test _load_orchestrator_impl loads the real orchestrator module."""
    # Find actual monorepo root from test file location
    test_file = Path(__file__).resolve()
    current = test_file
    while current.parent != current:
        if (current / "libs").is_dir():
            break
        current = current.parent
    else:
        pytest.skip("Could not find monorepo root for test")

    monorepo_root = current

    # Save original sys.path
    original_path = sys.path.copy()

    try:
        # Call real _load_orchestrator_impl
        run_for_project = guard_module._load_orchestrator_impl(monorepo_root)

        # Verify it returns a callable
        assert callable(run_for_project)
    finally:
        # Restore sys.path
        sys.path[:] = original_path


def test_main_entry_point() -> None:
    """Test the if __name__ == '__main__' block pattern."""

    def fake_is_dir(path: Path) -> bool:
        return path.name == "libs"

    def fake_find_root(start: Path) -> Path:
        current = start
        while True:
            if fake_is_dir(current / "libs"):
                return current
            if current.parent == current:
                raise RuntimeError("monorepo root with 'libs' directory not found")
            current = current.parent

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        _ = (monorepo_root, project_root)
        return 42  # Non-zero to test the exit code path

    def fake_load(monorepo_root: Path) -> guard_module._RunForProject:
        _ = monorepo_root
        return fake_run_for_project

    original_is_dir = _test_hooks.guard_is_dir
    original_find = _test_hooks.guard_find_monorepo_root
    original_load = _test_hooks.guard_load_orchestrator

    _test_hooks.guard_is_dir = fake_is_dir
    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load

    try:
        # Test the __main__ block by checking that main returns what run_for_project returns
        result = guard_module.main(None)
        assert result == 42
    finally:
        _test_hooks.guard_is_dir = original_is_dir
        _test_hooks.guard_find_monorepo_root = original_find
        _test_hooks.guard_load_orchestrator = original_load


def test_guard_main_via_runpy() -> None:
    """Test guard.py if __name__ == '__main__' block via runpy."""

    def fake_is_dir(path: Path) -> bool:
        return path.name == "libs"

    def fake_find_root(start: Path) -> Path:
        current = start
        while True:
            if fake_is_dir(current / "libs"):
                return current
            if current.parent == current:
                raise RuntimeError("monorepo root with 'libs' directory not found")
            current = current.parent

    def fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        _ = (monorepo_root, project_root)
        return 0

    def fake_load(monorepo_root: Path) -> guard_module._RunForProject:
        _ = monorepo_root
        return fake_run_for_project

    original_is_dir = _test_hooks.guard_is_dir
    original_find = _test_hooks.guard_find_monorepo_root
    original_load = _test_hooks.guard_load_orchestrator

    _test_hooks.guard_is_dir = fake_is_dir
    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load

    guard_path = Path(__file__).parent.parent / "scripts" / "guard.py"

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(guard_path), run_name="__main__")
        assert exc_info.value.code == 0
    finally:
        _test_hooks.guard_is_dir = original_is_dir
        _test_hooks.guard_find_monorepo_root = original_find
        _test_hooks.guard_load_orchestrator = original_load


def test_is_dir_uses_hook() -> None:
    """Test _is_dir uses hook when set."""
    hook_calls: list[Path] = []

    def fake_is_dir(path: Path) -> bool:
        hook_calls.append(path)
        return True

    original = _test_hooks.guard_is_dir
    _test_hooks.guard_is_dir = fake_is_dir

    try:
        result = guard_module._is_dir(Path("/test/path"))
        assert result is True
        assert len(hook_calls) == 1
        assert hook_calls[0] == Path("/test/path")
    finally:
        _test_hooks.guard_is_dir = original


def test_is_dir_uses_default_when_no_hook() -> None:
    """Test _is_dir uses default when no hook set."""
    original = _test_hooks.guard_is_dir
    _test_hooks.guard_is_dir = None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = guard_module._is_dir(Path(tmpdir))
            assert result is True
    finally:
        _test_hooks.guard_is_dir = original


def test_find_monorepo_root_uses_hook() -> None:
    """Test _find_monorepo_root uses hook when set."""
    expected_path = Path("/fake/monorepo")

    def fake_find(start: Path) -> Path:
        _ = start
        return expected_path

    original = _test_hooks.guard_find_monorepo_root
    _test_hooks.guard_find_monorepo_root = fake_find

    try:
        result = guard_module._find_monorepo_root(Path("/some/start"))
        assert result == expected_path
    finally:
        _test_hooks.guard_find_monorepo_root = original


def test_load_orchestrator_uses_hook() -> None:
    """Test _load_orchestrator uses hook when set."""
    calls: list[Path] = []

    def fake_run(*, monorepo_root: Path, project_root: Path) -> int:
        _ = (monorepo_root, project_root)
        return 99

    def fake_load(monorepo_root: Path) -> guard_module._RunForProject:
        calls.append(monorepo_root)
        return fake_run

    original = _test_hooks.guard_load_orchestrator
    _test_hooks.guard_load_orchestrator = fake_load

    try:
        result = guard_module._load_orchestrator(Path("/fake/root"))
        assert len(calls) == 1
        assert calls[0] == Path("/fake/root")
        assert result is fake_run
    finally:
        _test_hooks.guard_load_orchestrator = original


def test_find_monorepo_root_uses_impl_when_no_hook() -> None:
    """Test _find_monorepo_root uses impl when no hook set."""
    # Find actual monorepo root from test file location
    test_file = Path(__file__).resolve()
    current = test_file
    while current.parent != current:
        if (current / "libs").is_dir():
            break
        current = current.parent
    else:
        pytest.skip("Could not find monorepo root for test")

    expected_root = current

    # Ensure no hooks are set
    original_find = _test_hooks.guard_find_monorepo_root
    original_is_dir = _test_hooks.guard_is_dir
    _test_hooks.guard_find_monorepo_root = None
    _test_hooks.guard_is_dir = None

    try:
        # This should use _find_monorepo_root_impl which uses _is_dir -> _default_is_dir
        result = guard_module._find_monorepo_root(test_file.parent)
        assert result == expected_root
    finally:
        _test_hooks.guard_find_monorepo_root = original_find
        _test_hooks.guard_is_dir = original_is_dir


def test_load_orchestrator_uses_impl_when_no_hook() -> None:
    """Test _load_orchestrator uses impl when no hook set."""
    # Find actual monorepo root from test file location
    test_file = Path(__file__).resolve()
    current = test_file
    while current.parent != current:
        if (current / "libs").is_dir():
            break
        current = current.parent
    else:
        pytest.skip("Could not find monorepo root for test")

    monorepo_root = current

    # Ensure no hook is set
    original = _test_hooks.guard_load_orchestrator
    _test_hooks.guard_load_orchestrator = None

    # Save original sys.path
    original_path = sys.path.copy()

    try:
        # This should use _load_orchestrator_impl
        run_for_project = guard_module._load_orchestrator(monorepo_root)
        assert callable(run_for_project)
    finally:
        _test_hooks.guard_load_orchestrator = original
        sys.path[:] = original_path
