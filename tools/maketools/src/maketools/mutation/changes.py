"""The lines a commit adds to a package's Python sources (board task fec5851c).

The step mutates exactly these, so its cost scales with the change and a
commit already on main mutates nothing, as mcp-shared's TypeScript step does
(``mcp-shared/src/mutation/step.ts``, ``addedLineRanges``): ``git diff
--unified=0`` from the merge base of ``HEAD`` and ``origin/main``, new side
prefixed ``b/``. A file the commit only deletes lines from adds no code.
"""

from __future__ import annotations

import re

#: The ref a change is measured against: a fleet node stages it for a
#: package whose API registry entry declares ``base_branch``.
BASE_REF = "refs/remotes/origin/main"

#: The directory of a package whose files are mutated.
SOURCE_DIRECTORY = "src"

#: Where one file's diff starts.
_FILE_START = re.compile(r"^diff --git ", re.MULTILINE)

#: A hunk's header: where the new side's lines start, and how many. With no
#: context lines, every other line of a hunk opens with ``+`` or ``-``, so
#: no added line can read as a header.
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<first>\d+)(?:,(?P<count>\d+))? @@", re.MULTILINE)

#: The new side's file header, with the prefix the step asks git for. It
#: precedes the file's first hunk, so the first match in a file's diff is
#: it, even when an added line reads ``+++ b/...``.
_NEW_FILE = re.compile(r"^\+\+\+ b/(?P<path>.+)$", re.MULTILINE)


def diff_arguments(base: str) -> list[str]:
    """The arguments after ``git`` that list the change.

    Args:
        base: The merge base.

    Returns:
        A ``--unified=0`` diff of added, copied, modified and renamed files
        under :data:`SOURCE_DIRECTORY`, paths relative to the package.
    """
    return [
        "diff",
        "--unified=0",
        "--relative",
        "--diff-filter=ACMR",
        "--no-color",
        "--no-ext-diff",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        base,
        "HEAD",
        "--",
        SOURCE_DIRECTORY,
    ]


def _hunk_lines(hunk: re.Match[str]) -> range:
    """The new side's lines one hunk adds.

    Args:
        hunk: The hunk's header.

    Returns:
        Its new-side line numbers; none for a hunk that only deletes.
    """
    first = int(hunk["first"])
    return range(first, first + int(hunk["count"] or "1"))


def added_lines(diff: str) -> dict[str, frozenset[int]]:
    """Each Python source file's added lines.

    Args:
        diff: The output of ``git diff`` with :func:`diff_arguments`.

    Returns:
        Each ``.py`` file with at least one added line, to those lines,
        numbered from one.
    """
    added: dict[str, frozenset[int]] = {}
    for section in _FILE_START.split(diff):
        named = _NEW_FILE.search(section)
        path: str = "" if named is None else named["path"]
        if path.endswith(".py"):
            hunks = _HUNK.finditer(section)
            lines = frozenset(line for hunk in hunks for line in _hunk_lines(hunk))
            if lines:
                added[path] = lines
    return added


__all__ = ["BASE_REF", "SOURCE_DIRECTORY", "added_lines", "diff_arguments"]
