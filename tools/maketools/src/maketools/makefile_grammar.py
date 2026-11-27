"""The recipe grammar, and the check that every Makefile keeps to it.

WHY THERE IS A GRAMMAR. Every recipe runs under ``/bin/sh -eu -c``, one
line at a time (``scripts/make/shell.mk``), and a recipe is written as one
plain command per line: ``@echo`` for output, make's own ``$(VAR)`` for
substitution. Everything that needs logic (a condition, a loop, a pipe, a
chain of commands, a redirection) lives in a script the recipe calls by
path, where it is linted and tested like any other code, rather than in a
line of a Makefile that no test runs.

``packages/maketools/src/maketools/makefile_grammar.py`` in ~/PROJECTS/MCPs
holds the same shell-logic and make-function rules (board task b835753b);
the two repositories share no package. This one writes its frames as
TypedDicts because this monorepo bans dataclasses under ``src``, and its
poetry ``--no-cache`` rule is :mod:`maketools.makefile_poetry_cache_rule`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

from maketools import _test_hooks

#: The shared prologue every Makefile includes, relative to the repository root.
SHELL_INCLUDE: Final[str] = "scripts/make/shell.mk"

#: The git pathspec that finds every tracked Makefile.
MAKEFILE_PATHSPEC: Final[str] = "*Makefile"

#: Shell logic a recipe line may not hold. Each entry is a regular
#: expression matched against a recipe line with make's prefixes stripped
#: (:func:`recipe_body`).
SHELL_LOGIC: Final[Sequence[tuple[str, str]]] = (
    (r"&&|\|\|", "&& or || chaining"),
    (r"\$\$", "shell variable or substitution ($$)"),
    (r"`", "backtick"),
    (r"(^|\s)\[\s", "[ test"),
    (r"^(if|for|while|case|until)\s", "sh control flow"),
    (r";", "; separator (one command per line)"),
    (r"[<>|]", "redirection or pipe"),
    (r"^cd\s", "cd (use make -C or a script)"),
)

#: Lines outside a recipe that reach the shell, banned anywhere in the file:
#: ``$(shell ...)`` runs one while make parses the file, outside every
#: recipe, and the shell is chosen in one place only.
MAKE_SHELL_FUNCTIONS: Final[Sequence[tuple[str, str]]] = (
    (r"\$\(shell\b", "$(shell ...) runs the shell while make parses"),
    (r"^\s*SHELL\s*[:?]?=", "SHELL is set only in " + SHELL_INCLUDE),
    (r"^\s*\.SHELLFLAGS\s*[:?]?=", ".SHELLFLAGS is set only in " + SHELL_INCLUDE),
)

#: A recipe command that prints text and runs nothing, read after
#: :func:`recipe_body` has stripped make's prefixes.
ECHO_LINE: Final[re.Pattern[str]] = re.compile(r"^echo(\s|$)")


class Violation(TypedDict):
    """One line of one Makefile outside the grammar.

    Attributes:
        path: The Makefile, absolute.
        line_number: 1-based line.
        rule: Which rule fired.
        text: The offending line, or a description when no single line is at fault.
    """

    path: Path
    line_number: int
    rule: str
    text: str


def render_violation(violation: Violation, repo_root: Path) -> str:
    """Format a violation for the terminal.

    Args:
        violation: The violation.
        repo_root: Paths are shown relative to this.

    Returns:
        One line.
    """
    relative = violation["path"].relative_to(repo_root).as_posix()
    return f"{relative}:{violation['line_number']}: {violation['rule']}: {violation['text']}"


def makefile_depth(repo_root: Path, makefile: Path) -> int:
    """How many directories below the root a Makefile sits.

    Args:
        repo_root: The repository root.
        makefile: The Makefile's path under it.

    Returns:
        0 for the root Makefile, 1 for ``libs/Makefile``, 2 for a package's.
    """
    return len(makefile.resolve().relative_to(repo_root.resolve()).parent.parts)


def expected_include(depth: int) -> str:
    """The exact include line a Makefile at this depth must carry.

    Args:
        depth: From :func:`makefile_depth`.

    Returns:
        For example ``include ../../scripts/make/shell.mk``.
    """
    return f"include {'../' * depth}{SHELL_INCLUDE}"


def recipe_body(line: str) -> str:
    """The command of a recipe line, without make's own prefixes.

    GNU make skips ``@``, ``-``, ``+`` and blanks in any order before it
    reads the command, so ``+cd x`` and ``@ -cd x`` reach the shell as
    ``cd x``; stripping only ``@`` and ``-`` once let both past every ``^``
    rule (MCPs board task 97a9e387).

    Args:
        line: A line beginning with a tab.

    Returns:
        The line with the tab and its whole run of ``@``, ``-``, ``+`` and
        blanks stripped, and trailing whitespace too.
    """
    return line[1:].lstrip("@-+ \t").strip()


def command_lines(text: str) -> list[tuple[int, str]]:
    """Every recipe line that runs a command, for the rules about what a recipe runs.

    A comment runs nothing, and neither does an ``echo``: it is help text
    naming a command. The poetry rules (:mod:`maketools.makefile_poetry_cache_rule`,
    :mod:`maketools.makefile_poetry_lock_rule`) read recipes through this one
    function so they agree on what counts as running a command.

    Args:
        text: A Makefile's contents.

    Returns:
        Each such line's 1-based number and :func:`recipe_body`, in order.
    """
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("\t"):
            continue
        body = recipe_body(line)
        if body != "" and not body.startswith("#") and ECHO_LINE.match(body) is None:
            found.append((number, body))
    return found


def check_recipe_line(path: Path, line_number: int, line: str) -> list[Violation]:
    """Apply the recipe grammar to one recipe line.

    Args:
        path: The Makefile.
        line_number: The line's number.
        line: The raw line, tab included.

    Returns:
        Every rule the line breaks.
    """
    body = recipe_body(line)
    if body == "" or body.startswith("#"):
        return []
    return [
        Violation(path=path, line_number=line_number, rule=rule, text=body)
        for pattern, rule in SHELL_LOGIC
        if re.search(pattern, body)
    ]


def first_code_line(lines: Sequence[str]) -> tuple[int, str] | None:
    """The first line that is neither blank nor a comment.

    Args:
        lines: The file's lines.

    Returns:
        Its 1-based number and text, or None for a file of nothing else.
    """
    for number, line in enumerate(lines, start=1):
        if line.strip() != "" and not line.lstrip().startswith("#"):
            return number, line
    return None


def check_makefile(repo_root: Path, path: Path, text: str) -> list[Violation]:
    """Apply every grammar rule to one Makefile's text.

    Args:
        repo_root: The repository root, for the include depth.
        path: The Makefile.
        text: Its contents.

    Returns:
        Every violation, in line order.
    """
    lines = text.splitlines()
    found: list[Violation] = []
    include = expected_include(makefile_depth(repo_root, path))
    first_code = first_code_line(lines)
    if first_code is None or first_code[1].strip() != include:
        where = 1 if first_code is None else first_code[0]
        found.append(
            Violation(
                path=path,
                line_number=where,
                rule="first line must be the shell prologue",
                text=include,
            )
        )
    for number, line in enumerate(lines, start=1):
        if line.startswith("\t"):
            found.extend(check_recipe_line(path, number, line))
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, rule in MAKE_SHELL_FUNCTIONS:
            if re.search(pattern, line):
                found.append(Violation(path=path, line_number=number, rule=rule, text=stripped))
    return found


def tracked_makefiles(repo_root: Path) -> Sequence[Path]:
    """Every Makefile git tracks under the root.

    The shell prologue itself is a ``.mk`` and never matches, so it is
    neither linted for an include of itself nor forbidden its ``SHELL``.

    Args:
        repo_root: The repository root.

    Returns:
        Absolute paths, in git's order.
    """
    return _test_hooks.tracked_files(repo_root, MAKEFILE_PATHSPEC)


def lint_grammar(repo_root: Path) -> tuple[int, list[Violation]]:
    """Apply the grammar to every tracked Makefile.

    Args:
        repo_root: The repository root.

    Returns:
        How many Makefiles were examined, and every violation across them.
    """
    found: list[Violation] = []
    makefiles = tracked_makefiles(repo_root)
    for makefile in makefiles:
        found.extend(check_makefile(repo_root, makefile, makefile.read_text(encoding="utf-8")))
    return len(makefiles), found


__all__ = [
    "ECHO_LINE",
    "MAKEFILE_PATHSPEC",
    "MAKE_SHELL_FUNCTIONS",
    "SHELL_INCLUDE",
    "SHELL_LOGIC",
    "Violation",
    "check_makefile",
    "check_recipe_line",
    "command_lines",
    "expected_include",
    "first_code_line",
    "lint_grammar",
    "makefile_depth",
    "recipe_body",
    "render_violation",
    "tracked_makefiles",
]
