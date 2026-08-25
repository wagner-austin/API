"""Rule keeping all 41 copies of ``scripts/guard.py`` byte-identical.

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


__all__ = ["CANONICAL_FILES", "GuardShimRule", "first_difference"]
