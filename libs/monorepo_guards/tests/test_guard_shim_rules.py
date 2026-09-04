"""Tests for the rule that keeps every guard shim byte-identical.

The end-to-end test at the bottom used to assert a hardcoded shim count. It
went red on 2026-09-04 for the only reason it ever could: `tools/fleet` was
added, correctly, with its shim, and 44 != 43. The number was not measuring
anything -- a package added WITHOUT its shim leaves the count right and the
package unguarded, which is the failure the assertion was there to prevent.

So the count is gone and the invariant it stood in for is derived: every
package has a shim, and every shim has a package. Adding a package the right
way now needs no edit here; adding one the wrong way fails.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.guard_shim_rules import (
    GuardShimRule,
    first_difference,
    orphaned_shims,
    package_roots,
    shim_paths,
    unshimmed_packages,
)
from monorepo_guards.guard_shim_template import CANONICAL_GUARD_SHIM

REPO_ROOT = Path(__file__).resolve().parents[3]


def _package(root: Path, name: str, *, manifest: bool, shim: bool) -> Path:
    """Build one package-shaped directory two levels under a fake repo root.

    Args:
        root: Directory to treat as the monorepo root.
        name: Package directory name, created under ``libs``.
        manifest: Whether to write a ``pyproject.toml``.
        shim: Whether to write a ``scripts/guard.py``.

    Returns:
        The package root, whether or not either file was written.
    """
    package = root / "libs" / name
    package.mkdir(parents=True, exist_ok=True)
    if manifest:
        package.joinpath("pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    if shim:
        _shim_at(package, CANONICAL_GUARD_SHIM)
    return package


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
        than a fixture, so this catches a shim edited in place."""
        assert GuardShimRule().run(shim_paths(REPO_ROOT)) == []


class TestTheInventoryIsDerived:
    """Package roots and shims are read off the tree, not listed anywhere."""

    def test_a_package_is_found_by_its_manifest(self, tmp_path: Path) -> None:
        package = _package(tmp_path, "widgets", manifest=True, shim=True)

        assert package_roots(tmp_path) == [package]

    def test_a_directory_without_a_manifest_is_not_a_package(self, tmp_path: Path) -> None:
        """`libs/x/scripts` and other plain directories must not be demanded
        to carry a bootstrap."""
        _package(tmp_path, "notes", manifest=False, shim=False)

        assert package_roots(tmp_path) == []

    def test_shims_are_found_beside_their_package(self, tmp_path: Path) -> None:
        package = _package(tmp_path, "widgets", manifest=True, shim=True)

        assert shim_paths(tmp_path) == [package / "scripts" / "guard.py"]

    def test_a_deeper_copy_of_a_package_tree_is_not_a_package(self, tmp_path: Path) -> None:
        """`clients/RustedWarfareBot/runs` holds ~180 staged copies of a
        package tree that were shipped to a cluster. They are run inputs, and
        a recursive walk would demand the canonical shim of every one."""
        staged = tmp_path / "clients" / "bot" / "runs" / "sweep-1" / "payload"
        staged.mkdir(parents=True)
        staged.joinpath("pyproject.toml").write_text("[project]\n", encoding="utf-8")
        _shim_at(staged, "print('a copy that drifted on the cluster')\n")

        assert package_roots(tmp_path) == []
        assert shim_paths(tmp_path) == []


class TestTheHalfTheRuleCannotSee:
    """A package with no shim runs no guards, and no guard run can say so."""

    def test_a_package_without_a_shim_is_named(self, tmp_path: Path) -> None:
        unshimmed = _package(tmp_path, "forgotten", manifest=True, shim=False)
        _package(tmp_path, "widgets", manifest=True, shim=True)

        assert unshimmed_packages(tmp_path) == [unshimmed]

    def test_a_fully_shimmed_tree_names_nothing(self, tmp_path: Path) -> None:
        _package(tmp_path, "widgets", manifest=True, shim=True)

        assert unshimmed_packages(tmp_path) == []

    def test_a_shim_whose_package_is_gone_is_named(self, tmp_path: Path) -> None:
        """Deleting a package and leaving its shim keeps a count assertion
        green while the tree is wrong, and the next package added then
        passes on the strength of the corpse."""
        orphan = _package(tmp_path, "deleted", manifest=False, shim=True)

        assert orphaned_shims(tmp_path) == [orphan / "scripts" / "guard.py"]

    def test_a_shim_beside_a_manifest_is_not_an_orphan(self, tmp_path: Path) -> None:
        _package(tmp_path, "widgets", manifest=True, shim=True)

        assert orphaned_shims(tmp_path) == []


class TestTheRealTree:
    """The invariant, over this repository, with no number to maintain."""

    def test_the_scan_has_subjects(self) -> None:
        # A derivation that finds nothing satisfies every equality below
        # forever -- the same silent-pass shape the guards exist to refuse.
        assert len(package_roots(REPO_ROOT)) >= 40

    def test_every_package_runs_guards(self) -> None:
        assert unshimmed_packages(REPO_ROOT) == []

    def test_no_shim_outlives_its_package(self) -> None:
        assert orphaned_shims(REPO_ROOT) == []

    def test_the_two_derivations_agree(self) -> None:
        # Stated as an equality rather than as two containments so that
        # neither direction can be satisfied by an empty set on one side.
        assert [shim.parent.parent for shim in shim_paths(REPO_ROOT)] == package_roots(REPO_ROOT)


__all__ = [
    "TestFirstDifference",
    "TestGuardShimRule",
    "TestTheHalfTheRuleCannotSee",
    "TestTheInventoryIsDerived",
    "TestTheRealTree",
]
