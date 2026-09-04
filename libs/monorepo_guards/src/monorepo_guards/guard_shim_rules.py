"""Rule keeping every copy of ``scripts/guard.py`` byte-identical.

The shim is a bootstrap: it puts ``monorepo_guards`` on ``sys.path`` and
hands over. Nothing in it is package-specific, so nothing in it should differ
between packages -- and yet it did. Forty-one hand-maintained copies of the
same sixty lines had drifted into twenty-one variants, differing in how they
located the monorepo root and whether their plumbing went through a
``_test_hooks`` module. The drift was invisible because every copy passed its
own tests, which had drifted alongside it.

Similarity cannot be enforced; equality can. This compares each shim against
``CANONICAL_GUARD_SHIM`` and reports the first line that differs, so a
failure names the edit rather than just the file.

THE RULE CANNOT SEE A PACKAGE THAT HAS NO SHIM. A ``Rule`` is handed the
files to check, and the guard entry point only ever collects ``.py`` files
under one project root -- so a package added without a shim runs no guards at
all, and nothing about that is visible from inside a guard run. That second
half of the invariant is the repository inventory below: every package has a
shim, and every shim has a package. It is checked once, over the real tree,
by this package's own suite.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.guard_shim_template import (
    CANONICAL_GUARD_SHIM,
    CANONICAL_GUARD_SHIM_TEST,
)

CANONICAL_FILES = {
    ("scripts", "guard.py"): CANONICAL_GUARD_SHIM,
    ("tests", "test_guard_shim.py"): CANONICAL_GUARD_SHIM_TEST,
}
"""(parent directory, filename) -> the only text that file may contain.

The test is here alongside the shim on purpose. The shim drifting is half
the failure; the tests drifting to accommodate it is the half that hid the
first one for as long as it did.
"""


PACKAGE_MANIFEST = "pyproject.toml"
"""What makes a directory a package, rather than a directory of files.

Read off the tree instead of listed here. A list of package names is the
thing that goes stale -- three separate hardcoded inventories in this
repository drifted the same way, and each was noticed only when a count
assertion happened to fail rather than when the tree actually changed.
"""

SHIM_SEGMENTS = ("scripts", "guard.py")
"""Where a package's guard bootstrap lives, relative to the package root."""

_PACKAGE_DEPTH = "*/*"
"""Packages sit exactly two levels down: ``libs/x``, ``services/x``, and so on.

Deliberately not a recursive walk. ``clients/RustedWarfareBot/runs`` holds
roughly a hundred and eighty staged copies of a package tree, each with its
own ``scripts/guard.py``; those are run inputs that were shipped to a cluster,
not packages in this repository, and a recursive glob would demand this
package's canonical shim of every one of them.
"""


def package_roots(repo_root: Path) -> list[Path]:
    """Find every package in the monorepo.

    Args:
        repo_root: The monorepo root.

    Returns:
        One directory per package manifest, sorted.
    """
    return sorted(path.parent for path in repo_root.glob(f"{_PACKAGE_DEPTH}/{PACKAGE_MANIFEST}"))


def shim_paths(repo_root: Path) -> list[Path]:
    """Find every guard bootstrap in the monorepo.

    Args:
        repo_root: The monorepo root.

    Returns:
        One path per ``scripts/guard.py``, sorted.
    """
    return sorted(repo_root.glob(f"{_PACKAGE_DEPTH}/{'/'.join(SHIM_SEGMENTS)}"))


def unshimmed_packages(repo_root: Path) -> list[Path]:
    """Find packages that run no guards at all.

    This is the defect a count assertion stands in for and does not actually
    catch: a package added without its shim is not checked by anything, and
    ``GuardShimRule`` cannot report it because the file it would report is
    the missing one.

    Args:
        repo_root: The monorepo root.

    Returns:
        The package roots carrying no ``scripts/guard.py``, sorted.
    """
    return [root for root in package_roots(repo_root) if not root.joinpath(*SHIM_SEGMENTS).exists()]


def orphaned_shims(repo_root: Path) -> list[Path]:
    """Find guard bootstraps left behind by a package that no longer exists.

    The mirror of :func:`unshimmed_packages`, and the reason the invariant is
    stated as an equality: a package deleted without its shim keeps the count
    right while the tree is wrong, and the next package added then passes.

    Args:
        repo_root: The monorepo root.

    Returns:
        The shims whose package root has no manifest, sorted.
    """
    return [
        shim
        for shim in shim_paths(repo_root)
        if not (shim.parent.parent / PACKAGE_MANIFEST).exists()
    ]


def first_difference(actual: str, expected: str) -> int:
    """Find the first line number at which two texts differ.

    Args:
        actual: Text as found on disk.
        expected: Canonical text.

    Returns:
        One-based line number of the first difference. When one text is a
        prefix of the other, this is the first line the shorter one lacks.
    """
    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    # strict=False on purpose: a truncated or extended shim is exactly the
    # case this reports, so unequal lengths are input, not an error.
    pairs = zip(actual_lines, expected_lines, strict=False)
    for index, (left, right) in enumerate(pairs, start=1):
        if left != right:
            return index
    return min(len(actual_lines), len(expected_lines)) + 1


class GuardShimRule:
    """Rule requiring every guard shim to match the canonical text exactly."""

    name = "guard-shim"

    def run(self, files: list[Path]) -> list[Violation]:
        """Compare every guard shim among the checked files to the canonical text.

        Args:
            files: Python files being checked.

        Returns:
            One violation per shim that differs, naming the first line that does.
        """
        violations: list[Violation] = []
        for path in files:
            expected = CANONICAL_FILES.get((path.parent.name, path.name))
            if expected is None:
                continue
            actual = path.read_text(encoding="utf-8")
            if actual == expected:
                continue
            line_no = first_difference(actual, expected)
            violations.append(
                Violation(
                    file=path,
                    line_no=line_no,
                    kind="guard-shim-not-canonical",
                    line=(
                        f"{path.parent.name}/{path.name} must match its entry in "
                        "monorepo_guards.guard_shim_template byte for byte; "
                        f"first difference at line {line_no}. Nothing in the "
                        "bootstrap is package-specific -- put the change in "
                        "monorepo_guards/shim.py, or in the template if the "
                        "bootstrap itself must change."
                    ),
                )
            )
        return violations


__all__ = [
    "CANONICAL_FILES",
    "PACKAGE_MANIFEST",
    "SHIM_SEGMENTS",
    "GuardShimRule",
    "first_difference",
    "orphaned_shims",
    "package_roots",
    "shim_paths",
    "unshimmed_packages",
]
