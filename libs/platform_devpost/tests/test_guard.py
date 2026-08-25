"""Tests for scripts/guard.py module."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts import guard


class TestGuardEntrypoint:
    """Tests for guard.py entrypoint."""

    def test_guard_entrypoint_runs_as_main(self) -> None:
        """Test running guard.py as __main__."""
        # Ensure a clean module state to avoid runpy runtime warning
        if "scripts.guard" in sys.modules:
            del sys.modules["scripts.guard"]
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.guard", run_name="__main__")
        assert exc.value.code == 0


class TestFindMonorepoRoot:
    """Tests for _find_monorepo_root function."""

    def test_finds_root_with_libs_dir(self, tmp_path: Path) -> None:
        """Test finding monorepo root when libs directory exists."""
        # Create libs directory
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        # Create nested directory to start from
        nested = tmp_path / "some" / "nested" / "path"
        nested.mkdir(parents=True)

        # Override _is_dir to use real filesystem
        original_is_dir = guard._is_dir
        guard._is_dir = guard._default_is_dir

        try:
            result = guard._find_monorepo_root(nested)
            assert result == tmp_path
        finally:
            guard._is_dir = original_is_dir

    def test_raises_when_no_libs_dir(self, tmp_path: Path) -> None:
        """Test raising error when no libs directory found."""
        # Create a directory without libs
        nested = tmp_path / "no" / "libs" / "here"
        nested.mkdir(parents=True)

        # Use a fake _is_dir that always returns False
        original_is_dir = guard._is_dir

        def fake_is_dir(p: Path) -> bool:
            return False

        guard._is_dir = fake_is_dir

        try:
            with pytest.raises(RuntimeError, match=r"monorepo root.*not found"):
                guard._find_monorepo_root(nested)
        finally:
            guard._is_dir = original_is_dir


class TestDefaultIsDir:
    """Tests for _default_is_dir function."""

    def test_returns_true_for_directory(self, tmp_path: Path) -> None:
        """Test _default_is_dir returns True for existing directory."""
        result = guard._default_is_dir(tmp_path)
        assert result is True

    def test_returns_false_for_file(self, tmp_path: Path) -> None:
        """Test _default_is_dir returns False for file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        result = guard._default_is_dir(file_path)
        assert result is False

    def test_returns_false_for_nonexistent(self, tmp_path: Path) -> None:
        """Test _default_is_dir returns False for nonexistent path."""
        result = guard._default_is_dir(tmp_path / "nonexistent")
        assert result is False


class TestMain:
    """Tests for main function."""

    def test_main_with_verbose_flag(self) -> None:
        """Test main function with verbose flag outputs exit code."""
        # Run main with verbose flag - it will run the actual guards
        exit_code = guard.main(["--verbose"])
        assert exit_code == 0

    def test_main_with_root_override(self, tmp_path: Path) -> None:
        """Test main function with root override."""
        # Create minimal structure
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        # Run guards on the tmp_path - should succeed with 0 violations
        exit_code = guard.main(["--root", str(tmp_path)])
        assert exit_code == 0

    def test_main_with_unknown_argument(self) -> None:
        """Test main function with unknown argument (covers else branch)."""
        # Pass an unknown argument to trigger the else branch at line 107-108
        exit_code = guard.main(["--unknown-flag", "some-value"])
        assert exit_code == 0
