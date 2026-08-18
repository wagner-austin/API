"""Tests for the guard entry point.

``scripts.guard`` is inside the coverage target, so its argument parsing is
tested directly. :func:`scripts.guard.main` is exercised against the real
orchestrator over a real directory rather than a stand-in, because the point of
the module is that it reaches the shared rule set at all.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.guard import main, parse_target_root

#: The module file, executed directly by the ``__main__`` test.
GUARD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "guard.py"


class TestParseTargetRoot:
    """Tests for :func:`parse_target_root`."""

    def test_returns_the_default_when_no_flag_is_given(self) -> None:
        """An empty argument list selects the project root."""
        default = Path("/project")
        assert parse_target_root([], default) == default

    def test_root_flag_overrides_the_default(self, tmp_path: Path) -> None:
        """``--root`` selects a different directory."""
        assert parse_target_root(["--root", str(tmp_path)], Path("/project")) == tmp_path.resolve()

    def test_ignores_unrelated_arguments(self) -> None:
        """Unknown tokens advance the scan rather than failing it."""
        default = Path("/project")
        assert parse_target_root(["--verbose", "-x"], default) == default

    def test_root_without_a_value_is_ignored(self) -> None:
        """A trailing ``--root`` has no value to consume, so the default holds."""
        default = Path("/project")
        assert parse_target_root(["--root"], default) == default

    def test_last_root_flag_wins(self, tmp_path: Path) -> None:
        """A repeated flag resolves to the final occurrence."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        argv = ["--root", str(first), "--root", str(second)]
        assert parse_target_root(argv, Path("/project")) == second.resolve()


class TestMain:
    """Tests for :func:`scripts.guard.main`."""

    def test_reaches_the_shared_orchestrator_over_a_clean_tree(self, tmp_path: Path) -> None:
        """The real rule set runs and reports no violations on an empty tree.

        This asserts the direct import resolves and the orchestrator is
        callable with the arguments this module passes. An empty directory has
        nothing to violate, so a non-zero result here means the wiring is
        broken rather than the tree being dirty.
        """
        assert main(["--root", str(tmp_path)]) == 0

    def test_defaults_to_reading_sys_argv(self, tmp_path: Path) -> None:
        """Passing ``None`` reads the process arguments.

        Exercised through the ``--root`` path so the default-argument branch
        runs without the test depending on this project's own cleanliness.
        """
        original = sys.argv
        sys.argv = ["guard", "--root", str(tmp_path)]
        try:
            assert main(None) == 0
        finally:
            sys.argv = original


class TestModuleEntryPoint:
    """The ``if __name__ == "__main__"`` block."""

    def test_running_the_module_exits_with_the_guard_result(self, tmp_path: Path) -> None:
        """Executed as ``__main__``, the module raises ``SystemExit(0)``.

        Run through :mod:`runpy` rather than a subprocess so the executed lines
        are the ones under measurement. A subprocess would run the same code in
        a process the tracer is not attached to, leaving the entry point
        unexercised while appearing to test it.
        """
        original = sys.argv
        sys.argv = ["guard", "--root", str(tmp_path)]
        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_path(str(GUARD_PATH), run_name="__main__")
        finally:
            sys.argv = original
        assert caught.value.code == 0
