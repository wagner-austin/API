"""Tests for the file-size guard rule.

The rule enforces the 400-600 line ceiling from the coding standards.
It has no allowlist by design -- the over-bar backlog is being split
rather than recorded -- so the tests pin the ceiling behaviour, the
worst-first report order, and the two ways a file escapes measurement
(a scanned root that does not exist, and ``__pycache__``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.file_size_rules import (
    LINE_CEILING,
    SCANNED_ROOTS,
    SOFT_TARGET,
    count_lines,
    evaluate,
    measure_tree,
    run_file_size_rules,
)


def _write(path: Path, lines: int) -> Path:
    """Create a Python file of a given line count.

    Args:
        path: File to create; parents are created as needed.
        lines: Number of newline-terminated lines to write.

    Returns:
        The created path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"x = {n}\n" for n in range(lines)), encoding="utf-8")
    return path


class TestCountLines:
    """Tests for the line counter."""

    def test_counts_newline_terminated_lines(self, tmp_path: Path) -> None:
        """A file of N terminated lines counts as N, matching ``wc -l``."""
        assert count_lines(_write(tmp_path / "m.py", 7)) == 7

    def test_empty_file_counts_zero(self, tmp_path: Path) -> None:
        """An empty file has no lines."""
        path = tmp_path / "empty.py"
        path.write_text("", encoding="utf-8")
        assert count_lines(path) == 0

    def test_final_line_without_newline_still_counts(self, tmp_path: Path) -> None:
        """A file whose last line lacks a trailing newline counts it."""
        path = tmp_path / "nonl.py"
        path.write_text("a = 1\nb = 2", encoding="utf-8")
        assert count_lines(path) == 2


class TestMeasureTree:
    """Tests for the tree walker."""

    def test_measures_every_scanned_root(self, tmp_path: Path) -> None:
        """Each scanned root contributes its modules, keyed POSIX-style."""
        for root in SCANNED_ROOTS:
            _write(tmp_path / root / "mod.py", 3)
        sizes = measure_tree(tmp_path)
        assert sizes == {f"{root}/mod.py": 3 for root in SCANNED_ROOTS}

    def test_absent_root_is_skipped(self, tmp_path: Path) -> None:
        """A project without one of the roots measures the others."""
        _write(tmp_path / "src" / "only.py", 5)
        assert measure_tree(tmp_path) == {"src/only.py": 5}

    def test_pycache_is_excluded(self, tmp_path: Path) -> None:
        """Compiled-artifact directories never count toward the ceiling."""
        _write(tmp_path / "src" / "real.py", 4)
        _write(tmp_path / "src" / "__pycache__" / "stale.py", 900)
        assert measure_tree(tmp_path) == {"src/real.py": 4}

    def test_nested_packages_use_posix_keys(self, tmp_path: Path) -> None:
        """Nested modules report a forward-slash path on every platform."""
        _write(tmp_path / "src" / "pkg" / "sub" / "deep.py", 2)
        assert measure_tree(tmp_path) == {"src/pkg/sub/deep.py": 2}


class TestEvaluate:
    """Tests for the ceiling comparison."""

    def test_file_at_the_ceiling_passes(self) -> None:
        """The ceiling is inclusive -- exactly 600 lines is legal."""
        assert evaluate({"src/a.py": LINE_CEILING}) == []

    def test_file_one_line_over_is_reported(self) -> None:
        """One line past the ceiling is a violation."""
        violations = evaluate({"src/a.py": LINE_CEILING + 1})
        assert len(violations) == 1
        assert violations[0].startswith(f"src/a.py is {LINE_CEILING + 1} lines")
        assert f"over the {LINE_CEILING}-line ceiling" in violations[0]
        assert f"target {SOFT_TARGET}-{LINE_CEILING}" in violations[0]

    def test_empty_tree_passes(self) -> None:
        """Nothing measured means nothing to report."""
        assert evaluate({}) == []

    def test_report_is_worst_first(self) -> None:
        """The longest file leads, so the worst offender is read first."""
        violations = evaluate(
            {
                "src/small.py": LINE_CEILING + 1,
                "src/huge.py": LINE_CEILING + 900,
                "src/mid.py": LINE_CEILING + 50,
            }
        )
        assert [v.split(" is ")[0] for v in violations] == [
            "src/huge.py",
            "src/mid.py",
            "src/small.py",
        ]

    def test_equal_lengths_break_ties_by_path(self) -> None:
        """Same-size files sort by path so the report is deterministic."""
        violations = evaluate({"src/b.py": LINE_CEILING + 5, "src/a.py": LINE_CEILING + 5})
        assert [v.split(" is ")[0] for v in violations] == ["src/a.py", "src/b.py"]


class TestRunFileSizeRules:
    """Tests for the guard entry point."""

    def test_clean_tree_returns_zero_and_prints_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A compliant tree is silent."""
        _write(tmp_path / "src" / "ok.py", SOFT_TARGET)
        assert run_file_size_rules(tmp_path) == 0
        assert capsys.readouterr().out == ""

    def test_violation_is_counted_and_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each over-bar file prints one tagged line and counts once."""
        _write(tmp_path / "src" / "big.py", LINE_CEILING + 2)
        assert run_file_size_rules(tmp_path) == 1
        out = capsys.readouterr().out
        assert out.startswith("file_size_violation ")
        assert f"src/big.py is {LINE_CEILING + 2} lines" in out
        assert out.count("\n") == 1

    def test_multiple_violations_each_get_a_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The count matches the number of reported lines."""
        _write(tmp_path / "src" / "big.py", LINE_CEILING + 1)
        _write(tmp_path / "tests" / "bigger.py", LINE_CEILING + 2)
        assert run_file_size_rules(tmp_path) == 2
        out = capsys.readouterr().out
        assert out.count("file_size_violation ") == 2
        assert out.index("tests/bigger.py") < out.index("src/big.py")

    def test_empty_project_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A tree with none of the scanned roots passes."""
        assert run_file_size_rules(tmp_path) == 0
        assert capsys.readouterr().out == ""
