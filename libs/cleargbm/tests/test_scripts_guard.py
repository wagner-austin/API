"""Tests for scripts/guard.py module."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.guard import _find_monorepo_root, _load_orchestrator, main


class TestFindMonorepoRoot:
    """Tests for _find_monorepo_root."""

    def test_finds_root_from_project_dir(self) -> None:
        """Should find monorepo root from cleargbm directory."""
        # Start from the cleargbm lib directory
        cleargbm_dir = Path(__file__).parent.parent
        root = _find_monorepo_root(cleargbm_dir)

        # Root should contain 'libs' directory
        assert (root / "libs").is_dir()
        # And should be the actual monorepo root
        assert (root / "libs" / "cleargbm").is_dir()

    def test_finds_root_from_nested_dir(self) -> None:
        """Should find monorepo root from nested directory."""
        # Start from tests directory
        tests_dir = Path(__file__).parent
        root = _find_monorepo_root(tests_dir)

        assert (root / "libs").is_dir()

    def test_raises_when_no_libs_dir(self, tmp_path: Path) -> None:
        """Should raise RuntimeError when no libs directory found."""
        # Create isolated directory with no parents containing 'libs'
        isolated = tmp_path / "isolated" / "nested"
        isolated.mkdir(parents=True)

        with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
            _find_monorepo_root(isolated)


class TestLoadOrchestrator:
    """Tests for _load_orchestrator."""

    def test_loads_run_for_project(self) -> None:
        """Should load run_for_project function from monorepo_guards."""
        cleargbm_dir = Path(__file__).parent.parent
        root = _find_monorepo_root(cleargbm_dir)

        run_for_project = _load_orchestrator(root)

        # Should be callable
        assert callable(run_for_project)


class TestMain:
    """Tests for main function."""

    def test_runs_without_args(self) -> None:
        """main() should run and return an exit code."""
        # Run with empty args (uses project root)
        exit_code = main([])

        # Should return 0 (pass) or 2 (fail) depending on guard results
        assert exit_code in (0, 2)

    def test_verbose_flag(self) -> None:
        """main() should accept -v/--verbose flag."""
        # Just verify it doesn't crash with verbose and returns valid code
        exit_code_v = main(["-v"])
        assert exit_code_v in (0, 2)

        exit_code_verbose = main(["--verbose"])
        assert exit_code_verbose in (0, 2)

    def test_root_override(self) -> None:
        """main() should accept --root flag."""
        cleargbm_dir = Path(__file__).parent.parent

        # Run with explicit root - should return valid exit code
        exit_code = main(["--root", str(cleargbm_dir)])
        assert exit_code in (0, 2)

    def test_unknown_args_ignored(self) -> None:
        """main() should skip unknown arguments."""
        # Unknown args should be silently skipped
        exit_code = main(["--unknown-flag", "random-value"])
        assert exit_code in (0, 2)


def test_guard_entrypoint_runs_as_main() -> None:
    """Test that guard.py runs correctly when executed as __main__."""
    # Ensure a clean module state to avoid runpy runtime warning
    if "scripts.guard" in sys.modules:
        del sys.modules["scripts.guard"]
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    code = exc.value.code if isinstance(exc.value.code, int) else 0
    assert code in (0, 2)
