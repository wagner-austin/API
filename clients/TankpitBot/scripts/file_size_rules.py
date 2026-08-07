"""Guard rule: no monolithic files.

[[coding-standards]] sets a 400-600 line ceiling on every Python file,
src and tests alike. The rule was documented on 2026-07-31 with a
backlog of 40 over-bar files and no enforcing artifact; six days later
the backlog was 77 and all 40 originals had grown (+6,272 lines, zero
splits). This module is the missing artifact.

There is no allowlist and no baseline. A file over ``LINE_CEILING``
lines is a violation, full stop -- the backlog is being split rather
than recorded. The rule is wired into ``scripts.guard`` once the last
over-bar file is gone, so it can never come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

LINE_CEILING = 600
SOFT_TARGET = 400
SCANNED_ROOTS: tuple[str, ...] = ("src", "tests", "scripts")


def count_lines(path: Path) -> int:
    """Count the lines in a source file.

    Args:
        path: File to measure.

    Returns:
        Number of newline-delimited lines, matching ``wc -l`` for files
        that end in a newline.
    """
    return len(path.read_text(encoding="utf-8").splitlines())


def measure_tree(project_root: Path) -> dict[str, int]:
    """Measure every scanned Python file under a project tree.

    Args:
        project_root: Project root containing the scanned roots.

    Returns:
        Mapping of repo-relative POSIX path to line count, for every
        ``*.py`` under ``SCANNED_ROOTS`` excluding ``__pycache__``.
    """
    sizes: dict[str, int] = {}
    for root_name in SCANNED_ROOTS:
        root = project_root / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            sizes[path.relative_to(project_root).as_posix()] = count_lines(path)
    return sizes


def _worst_first(entry: tuple[int, str]) -> tuple[int, str]:
    """Order over-bar files longest first, then by path.

    Args:
        entry: A ``(lines, path)`` pair.

    Returns:
        Sort key placing the longest file first.
    """
    lines, name = entry
    return (-lines, name)


def evaluate(sizes: dict[str, int]) -> list[str]:
    """Report every file over the ceiling.

    Args:
        sizes: Measured line counts, keyed by repo-relative POSIX path.

    Returns:
        Violation messages, longest file first so the worst offender
        leads the report.
    """
    over = [(lines, name) for name, lines in sizes.items() if lines > LINE_CEILING]
    return [
        f"{name} is {lines} lines, over the {LINE_CEILING}-line ceiling "
        f"(target {SOFT_TARGET}-{LINE_CEILING}); split it into cohesive modules"
        for lines, name in sorted(over, key=_worst_first)
    ]


def run_file_size_rules(project_root: Path) -> int:
    """Run the file-size guard rule over a project tree.

    Args:
        project_root: Project root containing ``src``, ``tests`` and
            ``scripts``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    violations = evaluate(measure_tree(project_root))
    for violation in violations:
        sys.stdout.write(f"file_size_violation {violation}\n")
    return len(violations)


__all__ = [
    "LINE_CEILING",
    "SCANNED_ROOTS",
    "SOFT_TARGET",
    "count_lines",
    "evaluate",
    "measure_tree",
    "run_file_size_rules",
]
