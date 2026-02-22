"""Tests for scripts.guard module."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pytest
from scripts import guard


class TestDefaultIsDir:
    """Tests for _default_is_dir function."""

    def test_returns_true_for_existing_directory(self, tmp_path: Path) -> None:
        """Return True for existing directory."""
        assert guard._default_is_dir(tmp_path) is True

    def test_returns_false_for_nonexistent_path(self, tmp_path: Path) -> None:
        """Return False for nonexistent path."""
        nonexistent = tmp_path / "does_not_exist"
        assert guard._default_is_dir(nonexistent) is False

    def test_returns_false_for_file(self, tmp_path: Path) -> None:
        """Return False for file path."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        assert guard._default_is_dir(file_path) is False


class TestFindMonorepoRoot:
    """Tests for _find_monorepo_root function."""

    def setup_method(self) -> None:
        """Store original _is_dir for restoration."""
        self._original_is_dir = guard._is_dir

    def teardown_method(self) -> None:
        """Restore original _is_dir."""
        guard._is_dir = self._original_is_dir

    def test_finds_root_with_libs_directory(self, tmp_path: Path) -> None:
        """Find root when libs directory exists."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        start = tmp_path / "libs" / "platform_langid"
        start.mkdir(parents=True)

        result = guard._find_monorepo_root(start)
        assert result == tmp_path

    def test_raises_when_no_libs_found(self, tmp_path: Path) -> None:
        """Raise RuntimeError when no libs directory found."""
        # Create a fake is_dir that always returns False
        guard._is_dir = lambda p: False

        with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
            guard._find_monorepo_root(tmp_path)


class TestLoadOrchestrator:
    """Tests for _load_orchestrator function."""

    def test_loads_run_for_project_function(self) -> None:
        """Load run_for_project from monorepo_guards."""
        # Use real monorepo root
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[1]
        monorepo_root = guard._find_monorepo_root(project_root)

        run_for_project = guard._load_orchestrator(monorepo_root)

        assert callable(run_for_project)


class TestMain:
    """Tests for main function."""

    def setup_method(self) -> None:
        """Store original _is_dir for restoration."""
        self._original_is_dir = guard._is_dir

    def teardown_method(self) -> None:
        """Restore original _is_dir."""
        guard._is_dir = self._original_is_dir

    def test_main_returns_exit_code(self) -> None:
        """Main returns integer exit code."""
        result = guard.main([])
        assert result == 0

    def test_main_with_verbose_flag(self) -> None:
        """Main with -v flag prints exit code."""
        captured = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured

        result = guard.main(["-v"])

        sys.stdout = original_stdout
        output = captured.getvalue()

        assert result == 0
        assert "guard_exit_code code=0" in output

    def test_main_with_verbose_long_flag(self) -> None:
        """Main with --verbose flag prints exit code."""
        captured = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured

        result = guard.main(["--verbose"])

        sys.stdout = original_stdout
        output = captured.getvalue()

        assert result == 0
        assert "guard_exit_code code=0" in output

    def test_main_with_root_override(self) -> None:
        """Main with --root flag uses override path."""
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[1]

        result = guard.main(["--root", str(project_root)])

        assert result == 0

    def test_main_ignores_unknown_args(self) -> None:
        """Main ignores unknown arguments."""
        result = guard.main(["--unknown", "value"])
        assert result == 0

    def test_main_with_none_argv_uses_sys_argv(self) -> None:
        """Main with None argv uses sys.argv[1:]."""
        original_argv = sys.argv
        sys.argv = ["guard.py"]

        result = guard.main(None)

        sys.argv = original_argv
        assert result == 0


class TestMainEntry:
    """Tests for __main__ entry point."""

    def test_script_raises_system_exit(self) -> None:
        """Running script as __main__ raises SystemExit."""
        import runpy

        # Clear module from sys.modules to avoid RuntimeWarning
        modules_to_remove = [k for k in sys.modules if k.startswith("scripts")]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("scripts.guard", run_name="__main__")

        # Guard passes on this project, so exit code is 0
        assert exc_info.value.code == 0
