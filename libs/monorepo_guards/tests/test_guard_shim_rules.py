"""Tests for the rule that keeps all 41 guard shims byte-identical."""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.guard_shim_rules import GuardShimRule, first_difference
from monorepo_guards.guard_shim_template import CANONICAL_GUARD_SHIM


def _shim_at(root: Path, text: str) -> Path:
    """Write a guard shim into a package-shaped directory.

    Args:
        root: Directory to treat as a package root.
        text: Contents of the shim.

    Returns:
        Path to the written shim.
    """
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shim = scripts / "guard.py"
    shim.write_text(text, encoding="utf-8", newline="\n")
    return shim


class TestFirstDifference:
    def test_identical_texts_report_past_the_end(self) -> None:
        assert first_difference("a\nb\n", "a\nb\n") == 3

    def test_it_reports_the_changed_line(self) -> None:
        assert first_difference("a\nX\nc\n", "a\nb\nc\n") == 2

    def test_a_truncated_shim_reports_the_first_missing_line(self) -> None:
        assert first_difference("a\n", "a\nb\nc\n") == 2

    def test_an_extended_shim_reports_the_first_extra_line(self) -> None:
        assert first_difference("a\nb\nc\n", "a\n") == 2


class TestGuardShimRule:
    def test_the_canonical_shim_passes(self, tmp_path: Path) -> None:
        shim = _shim_at(tmp_path, CANONICAL_GUARD_SHIM)
        assert GuardShimRule().run([shim]) == []

    def test_a_drifted_shim_is_reported_with_the_line(self, tmp_path: Path) -> None:
        """This is the state the repo was actually in: 41 copies, 21 variants,
        every one of them passing its own tests."""
        drifted = CANONICAL_GUARD_SHIM.replace("here.parents[3]", "here.parents[2]")
        shim = _shim_at(tmp_path, drifted)

        violations = GuardShimRule().run([shim])

        assert len(violations) == 1
        assert violations[0].kind == "guard-shim-not-canonical"
        assert violations[0].file == shim
        assert "monorepo_guards/shim.py" in violations[0].line

    def test_the_reported_line_is_where_the_edit_is(self, tmp_path: Path) -> None:
        """A failure that names only the file makes the reader diff by hand."""
        lines = CANONICAL_GUARD_SHIM.splitlines(keepends=True)
        lines[4] = "changed\n"
        shim = _shim_at(tmp_path, "".join(lines))

        violations = GuardShimRule().run([shim])

        assert len(violations) == 1
        assert violations[0].line_no == 5

    def test_a_file_named_guard_py_elsewhere_is_not_a_shim(self, tmp_path: Path) -> None:
        """Only `scripts/guard.py` is the bootstrap. `src/.../guard.py` is
        ordinary code and has no business matching this text."""
        other = tmp_path / "src" / "pkg"
        other.mkdir(parents=True)
        elsewhere = other / "guard.py"
        elsewhere.write_text("x = 1\n", encoding="utf-8")
        assert GuardShimRule().run([elsewhere]) == []

    def test_a_non_guard_file_in_scripts_is_ignored(self, tmp_path: Path) -> None:
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        other = scripts / "benchmark.py"
        other.write_text("x = 1\n", encoding="utf-8")
        assert GuardShimRule().run([other]) == []

    def test_every_shim_in_the_repo_matches(self) -> None:
        """The end-to-end assertion: run the rule over the real tree rather
        than a fixture, so this catches a package added without its shim
        regenerated."""
        repo_root = Path(__file__).resolve().parents[3]
        shims = sorted(repo_root.glob("*/*/scripts/guard.py"))
        assert len(shims) == 41
        assert GuardShimRule().run(shims) == []


__all__ = ["TestFirstDifference", "TestGuardShimRule"]
