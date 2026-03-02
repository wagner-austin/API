"""Tests for scripts/guard.py."""

from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts import _test_hooks, guard


@pytest.fixture()
def fake_is_dir_false() -> Generator[None, None, None]:
    """Override is_dir to always return False."""
    original = _test_hooks.is_dir

    def _fake(path: Path) -> bool:
        del path
        return False

    _test_hooks.is_dir = _fake
    try:
        yield
    finally:
        _test_hooks.is_dir = original


@pytest.fixture()
def fake_is_dir_for_libs(tmp_path: Path) -> Generator[Path, None, None]:
    """Override is_dir to return True only for the fake libs directory."""
    original = _test_hooks.is_dir
    fake_monorepo = tmp_path / "fake_monorepo"
    fake_libs = fake_monorepo / "libs"
    fake_libs.mkdir(parents=True)

    def _fake(path: Path) -> bool:
        return path == fake_libs

    _test_hooks.is_dir = _fake
    try:
        yield fake_monorepo
    finally:
        _test_hooks.is_dir = original


def test_find_monorepo_root_raises_when_not_found(fake_is_dir_false: None, tmp_path: Path) -> None:
    """_find_monorepo_root should raise RuntimeError when libs not found."""
    del fake_is_dir_false
    with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        guard._find_monorepo_root(tmp_path)


def test_find_monorepo_root_finds_libs(fake_is_dir_for_libs: Path) -> None:
    """_find_monorepo_root should find the monorepo root with libs directory."""
    fake_monorepo = fake_is_dir_for_libs
    child = fake_monorepo / "some" / "nested" / "path"
    child.mkdir(parents=True)

    result = guard._find_monorepo_root(child)
    assert result == fake_monorepo


def test_main_with_verbose_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main should print exit code when --verbose is passed."""
    # Create fake monorepo structure
    fake_libs = tmp_path / "libs"
    fake_libs.mkdir()
    fake_guards = fake_libs / "monorepo_guards" / "src" / "monorepo_guards"
    fake_guards.mkdir(parents=True)

    # Create fake orchestrator module
    orchestrator_code = """
def run_for_project(*, monorepo_root, project_root):
    return 0
"""
    (fake_guards / "__init__.py").write_text("")
    (fake_guards / "orchestrator.py").write_text(orchestrator_code)

    # Override is_dir to find our fake monorepo
    original_is_dir = _test_hooks.is_dir

    def _fake_is_dir(path: Path) -> bool:
        if path == fake_libs:
            return True
        return path.is_dir()

    _test_hooks.is_dir = _fake_is_dir

    # Override get_script_path to return our fake project
    original_get_script_path = _test_hooks.get_script_path
    fake_project = tmp_path / "project"
    fake_project.mkdir()
    fake_scripts = fake_project / "scripts"
    fake_scripts.mkdir()
    fake_guard = fake_scripts / "guard.py"
    fake_guard.write_text("")

    def _fake_get_script_path() -> Path:
        return fake_guard

    _test_hooks.get_script_path = _fake_get_script_path

    try:
        result = guard.main(["--verbose"])
        assert result == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == "guard_exit_code code=0"
    finally:
        _test_hooks.is_dir = original_is_dir
        _test_hooks.get_script_path = original_get_script_path
        # Clean up sys.path
        guards_src = str(fake_libs / "monorepo_guards" / "src")
        if guards_src in sys.path:
            sys.path.remove(guards_src)
        if str(fake_libs) in sys.path:
            sys.path.remove(str(fake_libs))


def test_main_with_root_override(tmp_path: Path) -> None:
    """main should use --root override when provided."""
    # Create fake monorepo structure
    fake_libs = tmp_path / "libs"
    fake_libs.mkdir()
    fake_guards = fake_libs / "monorepo_guards" / "src" / "monorepo_guards"
    fake_guards.mkdir(parents=True)

    # Create fake orchestrator module
    orchestrator_code = """
def run_for_project(*, monorepo_root, project_root):
    return 0
"""
    (fake_guards / "__init__.py").write_text("")
    (fake_guards / "orchestrator.py").write_text(orchestrator_code)

    # Override is_dir
    original_is_dir = _test_hooks.is_dir

    def _fake_is_dir(path: Path) -> bool:
        if path == fake_libs:
            return True
        return path.is_dir()

    _test_hooks.is_dir = _fake_is_dir

    # Override get_script_path
    original_get_script_path = _test_hooks.get_script_path
    fake_project = tmp_path / "project"
    fake_project.mkdir()
    fake_scripts = fake_project / "scripts"
    fake_scripts.mkdir()
    fake_guard = fake_scripts / "guard.py"
    fake_guard.write_text("")

    override_root = tmp_path / "override"
    override_root.mkdir()

    def _fake_get_script_path() -> Path:
        return fake_guard

    _test_hooks.get_script_path = _fake_get_script_path

    try:
        result = guard.main(["--root", str(override_root)])
        assert result == 0
    finally:
        _test_hooks.is_dir = original_is_dir
        _test_hooks.get_script_path = original_get_script_path
        guards_src = str(fake_libs / "monorepo_guards" / "src")
        if guards_src in sys.path:
            sys.path.remove(guards_src)
        if str(fake_libs) in sys.path:
            sys.path.remove(str(fake_libs))


def test_main_ignores_unknown_args(tmp_path: Path) -> None:
    """main should ignore unknown arguments."""
    # Create fake monorepo structure
    fake_libs = tmp_path / "libs"
    fake_libs.mkdir()
    fake_guards = fake_libs / "monorepo_guards" / "src" / "monorepo_guards"
    fake_guards.mkdir(parents=True)

    orchestrator_code = """
def run_for_project(*, monorepo_root, project_root):
    return 0
"""
    (fake_guards / "__init__.py").write_text("")
    (fake_guards / "orchestrator.py").write_text(orchestrator_code)

    original_is_dir = _test_hooks.is_dir

    def _fake_is_dir(path: Path) -> bool:
        if path == fake_libs:
            return True
        return path.is_dir()

    _test_hooks.is_dir = _fake_is_dir

    # Override get_script_path
    original_get_script_path = _test_hooks.get_script_path
    fake_project = tmp_path / "project"
    fake_project.mkdir()
    fake_scripts = fake_project / "scripts"
    fake_scripts.mkdir()
    fake_guard = fake_scripts / "guard.py"
    fake_guard.write_text("")

    def _fake_get_script_path() -> Path:
        return fake_guard

    _test_hooks.get_script_path = _fake_get_script_path

    try:
        result = guard.main(["--unknown", "arg"])
        assert result == 0
    finally:
        _test_hooks.is_dir = original_is_dir
        _test_hooks.get_script_path = original_get_script_path
        guards_src = str(fake_libs / "monorepo_guards" / "src")
        if guards_src in sys.path:
            sys.path.remove(guards_src)
        if str(fake_libs) in sys.path:
            sys.path.remove(str(fake_libs))


def test_main_entry_point(tmp_path: Path) -> None:
    """__main__ block should call main and exit with its return code."""
    # Create fake monorepo with orchestrator
    fake_libs = tmp_path / "libs"
    fake_libs.mkdir()
    fake_guards = fake_libs / "monorepo_guards" / "src" / "monorepo_guards"
    fake_guards.mkdir(parents=True)
    (fake_guards / "__init__.py").write_text("")
    (fake_guards / "orchestrator.py").write_text(
        "def run_for_project(*, monorepo_root, project_root):\n    return 42\n"
    )

    # Set up hooks
    original_is_dir = _test_hooks.is_dir
    original_get_script_path = _test_hooks.get_script_path

    fake_project = tmp_path / "project"
    fake_scripts = fake_project / "scripts"
    fake_scripts.mkdir(parents=True)
    fake_guard = fake_scripts / "guard.py"
    fake_guard.write_text("")

    def _fake_is_dir(path: Path) -> bool:
        if path == fake_libs:
            return True
        return path.is_dir()

    def _fake_get_script_path() -> Path:
        return fake_guard

    _test_hooks.is_dir = _fake_is_dir
    _test_hooks.get_script_path = _fake_get_script_path

    # Clear cached orchestrator so import finds the fake one in tmp_path
    cached_keys = [k for k in sys.modules if k.startswith("monorepo_guards")]
    cached_modules: dict[str, types.ModuleType] = {}
    for k in cached_keys:
        mod = sys.modules.pop(k)
        if mod is not None:
            cached_modules[k] = mod

    try:
        if "scripts.guard" in sys.modules:
            del sys.modules["scripts.guard"]
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(
                str(Path(guard.__file__).resolve()),
                run_name="__main__",
            )
        code = exc_info.value.code if isinstance(exc_info.value.code, int) else 0
        assert code == 42
    finally:
        _test_hooks.is_dir = original_is_dir
        _test_hooks.get_script_path = original_get_script_path
        # Restore cached orchestrator modules
        sys.modules.update(cached_modules)
        guards_src = str(fake_libs / "monorepo_guards" / "src")
        if guards_src in sys.path:
            sys.path.remove(guards_src)
        if str(fake_libs) in sys.path:
            sys.path.remove(str(fake_libs))
