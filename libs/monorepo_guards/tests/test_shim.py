"""Tests for the guard entry point that all 41 packages share.

These cover the behaviour that used to be re-implemented, and separately
re-tested, in every package: the two flags, the root override, the verbose
line, and the layout assumption underneath the whole thing.

They run the REAL orchestrator against real trees rather than a stand-in.
That is what makes the passing and failing exit codes distinguishable here,
which is precisely what the 70 assertions this replaces never established.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from monorepo_guards import shim


class TestParseArguments:
    def test_no_arguments_checks_the_calling_package_quietly(self) -> None:
        parsed = shim.parse_arguments([])
        assert parsed.root_override is None
        assert parsed.verbose is False

    def test_root_takes_the_following_token(self, tmp_path: Path) -> None:
        parsed = shim.parse_arguments(["--root", str(tmp_path)])
        assert parsed.root_override == tmp_path.resolve()

    def test_both_verbose_spellings_are_accepted(self) -> None:
        assert shim.parse_arguments(["--verbose"]).verbose is True
        assert shim.parse_arguments(["-v"]).verbose is True

    def test_an_unknown_token_is_skipped_rather_than_rejected(self, tmp_path: Path) -> None:
        """One Makefile line serves 41 packages. A guard that rejected an
        unexpected argument would fail the build for a reason unrelated to
        the code it checks."""
        parsed = shim.parse_arguments(["bogus", "--root", str(tmp_path), "-v"])
        assert parsed.root_override == tmp_path.resolve()
        assert parsed.verbose is True

    def test_a_trailing_root_with_no_value_is_ignored(self) -> None:
        """`--root` as the final token has nothing to consume. Treating it as
        a flag rather than crashing keeps the ignore-unknown promise whole."""
        parsed = shim.parse_arguments(["--root"])
        assert parsed.root_override is None
        assert parsed.verbose is False

    def test_verbose_survives_a_later_flag(self, tmp_path: Path) -> None:
        """`verbose or ...` accumulates; a second token must not clear it."""
        parsed = shim.parse_arguments(["-v", "--root", str(tmp_path)])
        assert parsed.verbose is True

    def test_the_last_root_wins(self, tmp_path: Path) -> None:
        """Ported from NavProbe, which was the only package asserting it."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        parsed = shim.parse_arguments(["--root", str(first), "--root", str(second)])
        assert parsed.root_override == second.resolve()


class TestMonorepoRootOf:
    def test_it_is_two_levels_up_from_a_package(self) -> None:
        root = shim.monorepo_root_of(Path(__file__).resolve().parents[2] / "monorepo_guards")
        assert (root / "libs" / "monorepo_guards").is_dir()

    def test_a_package_outside_the_layout_raises(self, tmp_path: Path) -> None:
        """Raising beats searching: a guard that silently checks the wrong
        tree reports success for code it never read."""
        stray = tmp_path / "category" / "package"
        stray.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="is not <repo>/<category>/<package>"):
            shim.monorepo_root_of(stray)


class TestRunGuard:
    def test_a_clean_tree_passes(self, tmp_path: Path) -> None:
        project_root = Path(__file__).resolve().parents[1]
        assert shim.run_guard(["--root", str(tmp_path)], project_root=project_root) == 0

    def test_a_tree_with_violations_fails(self, tmp_path: Path) -> None:
        """The guard's whole purpose, asserted as an exit code that differs
        from the passing one -- the distinction 70 assertions in this repo
        used to blur by accepting both."""
        bad = tmp_path / "src" / "bad.py"
        bad.parent.mkdir(parents=True)
        any_kw = "An" + "y"
        bad.write_text(
            f"from typing import {any_kw}\nx: {any_kw} = 1\n",
            encoding="utf-8",
        )
        project_root = Path(__file__).resolve().parents[1]
        assert shim.run_guard(["--root", str(tmp_path)], project_root=project_root) == 2

    def test_verbose_writes_the_exit_code(self, tmp_path: Path) -> None:
        project_root = Path(__file__).resolve().parents[1]
        original = sys.stdout
        sys.stdout = io.StringIO()
        try:
            code = shim.run_guard(["--root", str(tmp_path), "--verbose"], project_root=project_root)
            written = sys.stdout.getvalue()
        finally:
            sys.stdout = original
        assert code == 0
        assert written.endswith(f"{shim.EXIT_CODE_PREFIX}0\n")

    def test_verbose_reports_a_failing_code_too(self, tmp_path: Path) -> None:
        """Ported from TankpitBot. The verbose line has to carry the failure,
        not just the success -- it is what a caller parses to learn which."""
        bad = tmp_path / "src" / "bad.py"
        bad.parent.mkdir(parents=True)
        banned = "An" + "y"
        bad.write_text(f"from typing import {banned}\nx: {banned} = 1\n", encoding="utf-8")
        project_root = Path(__file__).resolve().parents[1]
        original = sys.stdout
        sys.stdout = io.StringIO()
        try:
            code = shim.run_guard(["--root", str(tmp_path), "-v"], project_root=project_root)
            written = sys.stdout.getvalue()
        finally:
            sys.stdout = original
        assert code == 2
        assert written.endswith(f"{shim.EXIT_CODE_PREFIX}2\n")

    def test_quiet_omits_the_exit_code_line(self, tmp_path: Path) -> None:
        """Quiet suppresses the exit-code line, not the orchestrator's own
        rule summary -- that is written by `run_for_project` either way."""
        project_root = Path(__file__).resolve().parents[1]
        original = sys.stdout
        sys.stdout = io.StringIO()
        try:
            shim.run_guard(["--root", str(tmp_path)], project_root=project_root)
            written = sys.stdout.getvalue()
        finally:
            sys.stdout = original
        assert shim.EXIT_CODE_PREFIX not in written
        assert "Guard checks passed: no violations found." in written

    def test_none_argv_reads_the_process_arguments(self, tmp_path: Path) -> None:
        """`python -m scripts.guard --root X` arrives this way, so the None
        branch is the one the Makefile actually takes."""
        project_root = Path(__file__).resolve().parents[1]
        original = list(sys.argv)
        sys.argv[:] = ["prog", "--root", str(tmp_path)]
        try:
            assert shim.run_guard(None, project_root=project_root) == 0
        finally:
            sys.argv[:] = original


__all__ = [
    "TestMonorepoRootOf",
    "TestParseArguments",
    "TestRunGuard",
]
