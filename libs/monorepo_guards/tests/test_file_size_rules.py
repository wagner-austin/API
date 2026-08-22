"""Tests for the file-size guard rule."""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.file_size_rules import LINE_CEILING, SOFT_TARGET, FileSizeRule


def _write_file_with_lines(tmp_path: Path, name: str, line_count: int) -> Path:
    """Write a Python file containing exactly line_count lines."""
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    path = src_dir / name
    path.write_text("x = 1\n" * line_count, encoding="utf-8")
    return path


class TestFileSizeRuleThreshold:
    """Tests for the ceiling boundary."""

    def test_constants_pin_the_documented_band(self) -> None:
        assert LINE_CEILING == 600
        assert SOFT_TARGET == 400

    def test_file_at_ceiling_passes(self, tmp_path: Path) -> None:
        path = _write_file_with_lines(tmp_path, "at_ceiling.py", LINE_CEILING)

        violations = FileSizeRule().run([path])

        assert violations == []

    def test_file_one_over_ceiling_fails(self, tmp_path: Path) -> None:
        path = _write_file_with_lines(tmp_path, "over.py", LINE_CEILING + 1)

        violations = FileSizeRule().run([path])

        assert len(violations) == 1
        assert violations[0].file == path
        assert violations[0].kind == "file-over-ceiling"
        assert violations[0].line_no == LINE_CEILING + 1
        assert f"{LINE_CEILING + 1} lines" in violations[0].line
        assert "split it into cohesive modules by role" in violations[0].line

    def test_empty_file_passes(self, tmp_path: Path) -> None:
        path = _write_file_with_lines(tmp_path, "empty.py", 0)

        violations = FileSizeRule().run([path])

        assert violations == []

    def test_blank_lines_count_toward_the_ceiling(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        path = src_dir / "blanks.py"
        path.write_text("x = 1\n" + "\n" * LINE_CEILING, encoding="utf-8")

        violations = FileSizeRule().run([path])

        assert len(violations) == 1
        assert violations[0].line_no == LINE_CEILING + 1

    def test_file_without_trailing_newline_counts_last_line(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        path = src_dir / "no_trailing_newline.py"
        path.write_text("x = 1\n" * LINE_CEILING + "y = 2", encoding="utf-8")

        violations = FileSizeRule().run([path])

        assert len(violations) == 1
        assert violations[0].line_no == LINE_CEILING + 1


class TestFileSizeRuleReporting:
    """Tests for multi-file ordering and reporting."""

    def test_worst_offender_reported_first(self, tmp_path: Path) -> None:
        small = _write_file_with_lines(tmp_path, "small.py", LINE_CEILING + 10)
        big = _write_file_with_lines(tmp_path, "big.py", LINE_CEILING + 500)

        violations = FileSizeRule().run([small, big])

        assert [v.file for v in violations] == [big, small]

    def test_equal_lengths_ordered_by_path(self, tmp_path: Path) -> None:
        b_file = _write_file_with_lines(tmp_path, "b.py", LINE_CEILING + 5)
        a_file = _write_file_with_lines(tmp_path, "a.py", LINE_CEILING + 5)

        violations = FileSizeRule().run([b_file, a_file])

        assert [v.file for v in violations] == [a_file, b_file]

    def test_mixed_files_report_only_over_ceiling(self, tmp_path: Path) -> None:
        ok = _write_file_with_lines(tmp_path, "ok.py", SOFT_TARGET)
        over = _write_file_with_lines(tmp_path, "over.py", LINE_CEILING + 1)

        violations = FileSizeRule().run([ok, over])

        assert len(violations) == 1
        assert violations[0].file == over

    def test_rule_name_is_stable(self) -> None:
        assert FileSizeRule().name == "file-size"
