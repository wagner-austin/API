"""Tests for scripts/guard.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts import _test_hooks, guard

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def fake_is_dir_false() -> Iterator[None]:
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


@pytest.fixture
def fake_is_dir_for_libs(tmp_path: Path) -> Iterator[Path]:
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


def test_find_monorepo_root_raises_when_not_found(
    fake_is_dir_false: None, tmp_path: Path
) -> None:
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


def test_main_with_verbose_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """main should print exit code when --verbose is passed."""
    # Create fake monorepo structure
    fake_libs = tmp_path / "libs"
    fake_libs.mkdir()
    fake_guards = fake_libs / "monorepo_guards" / "src" / "monorepo_guards"
    fake_guards.mkdir(parents=True)

    # Create fake orchestrator module
    orchestrator_code = '''
def run_for_project(*, monorepo_root, project_root):
    return 0
'''
    (fake_guards / "__init__.py").write_text("")
    (fake_guards / "orchestrator.py").write_text(orchestrator_code)

    # Override is_dir to find our fake monorepo
    original_is_dir = _test_hooks.is_dir

    def _fake_is_dir(path: Path) -> bool:
        if path == fake_libs:
            return True
        return path.is_dir()

    _test_hooks.is_dir = _fake_is_dir

    # Patch __file__ to be in our fake project
    fake_project = tmp_path / "project"
    fake_project.mkdir()
    fake_scripts = fake_project / "scripts"
    fake_scripts.mkdir()
    fake_guard = fake_scripts / "guard.py"
    fake_guard.write_text("")

    monkeypatch.setattr(guard, "__file__", str(fake_guard))

    try:
        result = guard.main(["--verbose"])
        assert result == 0
        captured = capsys.readouterr()
        assert "guard_exit_code code=0" in captured.out
    finally:
        _test_hooks.is_dir = original_is_dir
        # Clean up sys.path
        guards_src = str(fake_libs / "monorepo_guards" / "src")
        if guards_src in sys.path:
            sys.path.remove(guards_src)
        if str(fake_libs) in sys.path:
            sys.path.remove(str(fake_libs))


def test_main_with_root_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main should use --root override when provided."""
    # Create fake monorepo structure
    fake_libs = tmp_path / "libs"
    fake_libs.mkdir()
    fake_guards = fake_libs / "monorepo_guards" / "src" / "monorepo_guards"
    fake_guards.mkdir(parents=True)

    captured_project_root: list[Path] = []

    # Create fake orchestrator module
    orchestrator_code = '''
def run_for_project(*, monorepo_root, project_root):
    return 0
'''
    (fake_guards / "__init__.py").write_text("")
    (fake_guards / "orchestrator.py").write_text(orchestrator_code)

    # Override is_dir
    original_is_dir = _test_hooks.is_dir

    def _fake_is_dir(path: Path) -> bool:
        if path == fake_libs:
            return True
        return path.is_dir()

    _test_hooks.is_dir = _fake_is_dir

    # Patch __file__
    fake_project = tmp_path / "project"
    fake_project.mkdir()
    fake_scripts = fake_project / "scripts"
    fake_scripts.mkdir()
    fake_guard = fake_scripts / "guard.py"
    fake_guard.write_text("")

    override_root = tmp_path / "override"
    override_root.mkdir()

    monkeypatch.setattr(guard, "__file__", str(fake_guard))

    try:
        result = guard.main(["--root", str(override_root)])
        assert result == 0
    finally:
        _test_hooks.is_dir = original_is_dir
        guards_src = str(fake_libs / "monorepo_guards" / "src")
        if guards_src in sys.path:
            sys.path.remove(guards_src)
        if str(fake_libs) in sys.path:
            sys.path.remove(str(fake_libs))


def test_main_ignores_unknown_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main should ignore unknown arguments."""
    # Create fake monorepo structure
    fake_libs = tmp_path / "libs"
    fake_libs.mkdir()
    fake_guards = fake_libs / "monorepo_guards" / "src" / "monorepo_guards"
    fake_guards.mkdir(parents=True)

    orchestrator_code = '''
def run_for_project(*, monorepo_root, project_root):
    return 0
'''
    (fake_guards / "__init__.py").write_text("")
    (fake_guards / "orchestrator.py").write_text(orchestrator_code)

    original_is_dir = _test_hooks.is_dir

    def _fake_is_dir(path: Path) -> bool:
        if path == fake_libs:
            return True
        return path.is_dir()

    _test_hooks.is_dir = _fake_is_dir

    fake_project = tmp_path / "project"
    fake_project.mkdir()
    fake_scripts = fake_project / "scripts"
    fake_scripts.mkdir()
    fake_guard = fake_scripts / "guard.py"
    fake_guard.write_text("")

    monkeypatch.setattr(guard, "__file__", str(fake_guard))

    try:
        result = guard.main(["--unknown", "arg"])
        assert result == 0
    finally:
        _test_hooks.is_dir = original_is_dir
        guards_src = str(fake_libs / "monorepo_guards" / "src")
        if guards_src in sys.path:
            sys.path.remove(guards_src)
        if str(fake_libs) in sys.path:
            sys.path.remove(str(fake_libs))


def test_main_entry_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """__main__ block should call main and exit with its return code."""
    # Create fake monorepo structure
    fake_libs = tmp_path / "libs"
    fake_libs.mkdir()
    fake_guards = fake_libs / "monorepo_guards" / "src" / "monorepo_guards"
    fake_guards.mkdir(parents=True)

    orchestrator_code = '''
def run_for_project(*, monorepo_root, project_root):
    return 42
'''
    (fake_guards / "__init__.py").write_text("")
    (fake_guards / "orchestrator.py").write_text(orchestrator_code)

    original_is_dir = _test_hooks.is_dir

    def _fake_is_dir(path: Path) -> bool:
        if path == fake_libs:
            return True
        return path.is_dir()

    _test_hooks.is_dir = _fake_is_dir

    # Create a copy of guard.py in the fake project
    fake_project = tmp_path / "project"
    fake_project.mkdir()
    fake_scripts = fake_project / "scripts"
    fake_scripts.mkdir()

    # Copy _test_hooks
    hooks_content = Path(guard.__file__).parent / "_test_hooks.py"
    (fake_scripts / "__init__.py").write_text("")
    (fake_scripts / "_test_hooks.py").write_text(hooks_content.read_text())

    # Create guard with patched is_dir
    guard_source = Path(guard.__file__).read_text()
    # Patch the _test_hooks import to use our fake
    patched_guard = guard_source.replace(
        "from scripts import _test_hooks",
        f'''
import sys
sys.path.insert(0, r"{fake_libs / "monorepo_guards" / "src"}")
sys.path.insert(0, r"{fake_libs}")

from pathlib import Path as _Path

class _FakeHooks:
    @staticmethod
    def is_dir(path: _Path) -> bool:
        return path == _Path(r"{fake_libs}")

_test_hooks = _FakeHooks()
'''
    )
    (fake_scripts / "guard.py").write_text(patched_guard)

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(fake_scripts / "guard.py"), run_name="__main__")
        assert exc_info.value.code == 42
    finally:
        _test_hooks.is_dir = original_is_dir
