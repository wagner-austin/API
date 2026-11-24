"""The lines a commit adds to a package's Python sources (board task fec5851c).

The step mutates exactly these, so its cost scales with the change and a
commit already on main mutates nothing, as mcp-shared's TypeScript step does
(``mcp-shared/src/mutation/step.ts``, ``addedLineRanges``): ``git diff
--unified=0`` from the merge base of ``HEAD`` and ``origin/main``, new side
prefixed ``b/``. A file the commit only deletes lines from adds no code.

WHICH FILES ARE SOURCES (board task 9c456c33). Those the package's coverage
measures, as its ``pyproject.toml`` declares them under
``[tool.coverage.run]``: the directories ``source`` names, less the files
``omit`` matches. The code a package's 100 percent gate counts is the code
its mutants are made in, from one declaration, so a package whose code is
not under ``src`` (claude-hooks keeps its hooks at the package root, its
suites beside them as ``*.test.py``) is mutated where its code is. An omit
pattern is matched as coverage matches one against a file under the package
it runs in: ``fnmatch`` over the path relative to the package, whose ``*``
crosses a ``/``, so ``*.test.py`` drops a suite at any depth and
``tests/*`` drops everything under ``tests``. The declaration is read
through the package's codec (board task 9976dca1), as every document the
step reads is: its TOML reader and its narrowers, each refusing a value of
the wrong shape by its own error naming the field. API lifts this module
and allows one TOML reader, ``platform_core.toml_utils``, so the module
imports none.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from typing import NamedTuple

from .context import JsonCodec, JsonT, MutationErrorCode, MutationStepError

#: The ref a change is measured against: a fleet node stages it for a
#: package whose API registry entry declares ``base_branch``.
BASE_REF = "refs/remotes/origin/main"

#: The file, relative to the package, that declares what coverage measures.
PYPROJECT = "pyproject.toml"

#: Where in it the declaration is.
COVERAGE_TABLE = "[tool.coverage.run]"

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


class Sources(NamedTuple):
    """What a package's coverage measures, so what the step mutates.

    Attributes:
        roots: The directories ``source`` names, relative to the package.
        omit: The patterns ``omit`` names; none when it names none.
    """

    roots: tuple[str, ...]
    omit: tuple[str, ...]


def _undeclared(detail: str) -> MutationStepError:
    """The refusal of a package whose sources the step cannot read.

    Args:
        detail: What was found.

    Returns:
        ``MUTATION_SOURCES_UNDECLARED``, naming the declaration it needs.
    """
    return MutationStepError(
        MutationErrorCode.SOURCES_UNDECLARED,
        f"{PYPROJECT} {COVERAGE_TABLE} {detail}; the step mutates the files coverage "
        "measures, so the package declares them there as its 100 percent gate reads them",
    )


#: The tables on the way to the declaration, each with the name a refusal gives it.
_TABLES = (("tool", "[tool]"), ("coverage", "[tool.coverage]"), ("run", COVERAGE_TABLE))


def _strings(value: JsonT | None, name: str, codec: JsonCodec[JsonT]) -> tuple[str, ...]:
    """Require a TOML array of strings, or nothing where the table holds none.

    Args:
        value: What the table holds there, or None for nothing.
        name: Its key, for the refusal.
        codec: The package's codec.

    Returns:
        The strings, in order; none for nothing. The codec's own error,
        naming ``name``, is raised when it is not an array or a member is
        not a string.
    """
    if value is None:
        return ()
    return tuple(codec.text(item, name) for item in codec.sequence(value, name))


def declared_sources(text: str, codec: JsonCodec[JsonT]) -> Sources:
    """The sources a package's ``pyproject.toml`` declares for coverage.

    Args:
        text: The file's text.
        codec: The package's codec, whose TOML reader parses it.

    Returns:
        Its ``source`` and ``omit``.

    Raises:
        MutationStepError: ``MUTATION_SOURCES_UNDECLARED`` when a table on
            the way or ``source`` is absent, or ``source`` is empty; the
            codec's own error, naming the table or key, when a table on the
            way is not a table or ``source`` or ``omit`` is not an array of
            strings, and its TOML reader's own when the text is not TOML.
    """
    table = codec.mapping(codec.loads_toml(text), PYPROJECT)
    for key, name in _TABLES:
        member = table.get(key)
        if member is None:
            raise _undeclared("names no source")
        table = codec.mapping(member, name)
    roots = _strings(table.get("source"), "source", codec)
    if not roots:
        raise _undeclared("names no source")
    return Sources(roots=roots, omit=_strings(table.get("omit"), "omit", codec))


def diff_arguments(base: str, sources: Sources) -> list[str]:
    """The arguments after ``git`` that list the change.

    Args:
        base: The merge base.
        sources: The package's sources.

    Returns:
        A ``--unified=0`` diff of added, copied, modified and renamed files
        under the source roots, paths relative to the package.
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
        *sources.roots,
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


def added_lines(diff: str, sources: Sources) -> dict[str, frozenset[int]]:
    """Each Python source file's added lines.

    Args:
        diff: The output of ``git diff`` with :func:`diff_arguments`.
        sources: The package's sources, whose omit patterns drop a file.

    Returns:
        Each ``.py`` file no omit pattern matches with at least one added
        line, to those lines, numbered from one.
    """
    added: dict[str, frozenset[int]] = {}
    for section in _FILE_START.split(diff):
        named = _NEW_FILE.search(section)
        path: str = "" if named is None else named["path"]
        omitted = any(fnmatchcase(path, pattern) for pattern in sources.omit)
        if path.endswith(".py") and not omitted:
            hunks = _HUNK.finditer(section)
            lines = frozenset(line for hunk in hunks for line in _hunk_lines(hunk))
            if lines:
                added[path] = lines
    return added


__all__ = [
    "BASE_REF",
    "COVERAGE_TABLE",
    "PYPROJECT",
    "Sources",
    "added_lines",
    "declared_sources",
    "diff_arguments",
]
