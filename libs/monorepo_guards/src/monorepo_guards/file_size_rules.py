"""Guard rule: no monolithic files.

The coding standard sets a 400-600 line ceiling on every Python file,
src, scripts and tests alike. The rule was born in TankpitBot on
2026-07-31 with a documented backlog of 40 over-bar files and no
enforcing artifact; six days later the backlog was 77 and all 40
originals had grown (+6,272 lines, zero splits). Lifted monorepo-wide
on 2026-08-22 after the 137-file backlog was split away.

There is no allowlist and no baseline. A file over ``LINE_CEILING``
lines is a violation, full stop -- backlogs get split, not recorded.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.util import read_lines

LINE_CEILING = 600
SOFT_TARGET = 400


def _worst_first(violation: Violation) -> tuple[int, str]:
    """Order violations longest file first, then by path.

    Args:
        violation: A file-size violation whose line_no carries the count.

    Returns:
        Sort key placing the longest file first.
    """
    return (-violation.line_no, str(violation.file))


class FileSizeRule:
    """Rule enforcing the 600-line ceiling on every scanned Python file."""

    name = "file-size"

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every file over the ceiling, longest first.

        Args:
            files: Python files selected by the guard configuration.

        Returns:
            One violation per over-ceiling file. ``line_no`` is the
            file's line count so the report reads as a measurement.
        """
        violations: list[Violation] = []
        for path in files:
            lines = len(read_lines(path))
            if lines > LINE_CEILING:
                violations.append(
                    Violation(
                        file=path,
                        line_no=lines,
                        kind="file-over-ceiling",
                        line=(
                            f"{lines} lines, over the {LINE_CEILING}-line ceiling "
                            f"(target {SOFT_TARGET}-{LINE_CEILING}); "
                            "split it into cohesive modules by role"
                        ),
                    )
                )
        return sorted(violations, key=_worst_first)


__all__ = [
    "LINE_CEILING",
    "SOFT_TARGET",
    "FileSizeRule",
]
