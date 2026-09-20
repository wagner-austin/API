"""The portable recipe grammar, and the check that every Makefile keeps to it.

WHY THERE IS A GRAMMAR. A Makefile here runs under two shells: PowerShell
on Windows, where ``sh.exe`` is not on the PATH a PowerShell-launched make
inherits, and ``/bin/sh`` everywhere else. A recipe is therefore written in
the INTERSECTION of the two -- one plain command per line, ``@echo`` for
output, make's own ``$(VAR)`` for substitution -- and everything that needs
logic lives in a script the recipe calls by path. Both shells' private
syntax is banned outright, because a recipe that works on the machine it
was written on and dies on the other is exactly how 55 Makefiles came to
be Windows-only in the first place.

The two halves of the ban are deliberately symmetric. ``Write-Host`` and
``&&`` are the same defect: a line only one shell can read.

The same grammar, rule for rule, is ``packages/maketools/src/maketools/
makefile_grammar.py`` in ~/PROJECTS/MCPs, where it landed first (board task
b835753b); the two repositories share no package, so this is a deliberate
copy, with the frames written as TypedDicts because this monorepo bans
dataclasses under ``src``.
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

#: PowerShell-only syntax. Each entry is a regular expression matched
#: against a recipe line with its ``@`` and ``-`` prefixes stripped.
POWERSHELL_ONLY: Final[Sequence[tuple[str, str]]] = (
    (
        r"\b(Write|Get|Set|Test|New|Remove|Select|ForEach|Where|Join|Out|Split|Start|Invoke"
        r"|Resolve|Copy|Move|Add|Compare|Sort|Measure|Expand|Compress|Import|Export|Push|Pop)"
        r"-[A-Z]\w*",
        "cmdlet",
    ),
    (r"\$\$?env:", "$env: variable"),
    (r"-ForegroundColor|-ErrorAction|-NoProfile|-ExecutionPolicy", "cmdlet parameter"),
    (r"\bpowershell\b|\bpwsh\b", "powershell invocation"),
    (r"\.ps1\b", "PowerShell script"),
    (r"\$\$\?", "$? status variable"),
    (r"^(if|foreach|while|try)\s*\(", "PowerShell control flow"),
    (r"@\(", "PowerShell array literal"),
    # A backslash BETWEEN path-ish characters: ``scripts\tests``,
    # ``.\scripts``, ``backups\$(FILE)``. Not ``}}\t{{`` in a docker
    # ``--format`` string, which both shells pass through literally for
    # docker to expand, and not a trailing continuation backslash.
    (r"(?<=[\w.])\\(?=[\w$.])", "backslash path separator"),
    (r"`", "backtick"),
)

#: POSIX-shell-only syntax, matched the same way.
SH_ONLY: Final[Sequence[tuple[str, str]]] = (
    (r"&&|\|\|", "&& or || chaining"),
    (r"\$\$", "shell variable or substitution ($$)"),
    (r"/dev/null|\bNUL\b", "null device"),
    (r"(^|\s)\[\s", "[ test"),
    (r"^(if|for|while|case|until)\s", "sh control flow"),
)

#: Syntax the two shells read DIFFERENTLY, banned because it looks
#: portable and is not: PowerShell continues after a failing native command
#: inside ``a; b``, ``sh -e`` stops; ``>`` writes UTF-16 in PowerShell 5.1;
#: ``cd`` failing is a non-terminating error in one and fatal in the other.
DIVERGENT: Final[Sequence[tuple[str, str]]] = (
    (r";", "; separator (one command per line)"),
    (r"[<>|]", "redirection or pipe"),
    (r"^cd\s", "cd (use make -C or a script)"),
)

#: make functions that run the shell, banned anywhere in the file.
MAKE_SHELL_FUNCTIONS: Final[Sequence[tuple[str, str]]] = (
    (r"\$\(shell\b", "$(shell ...) runs the platform shell"),
    (r"^\s*SHELL\s*[:?]?=", "SHELL is set only in " + SHELL_INCLUDE),
    (r"^\s*\.SHELLFLAGS\s*[:?]?=", ".SHELLFLAGS is set only in " + SHELL_INCLUDE),
)

#: The one conditional that fences platform-specific recipes. Inside the
#: Windows arm a recipe may use PowerShell, because it runs nowhere else;
#: the ``else`` arm is portable and checked like any other line. A target
#: that has no meaning off Windows (Task Scheduler registration) lives in
#: such a block with an ``else`` arm that says so.
WINDOWS_CONDITION: Final[re.Pattern[str]] = re.compile(
    r"^(ifeq|ifneq)\s*\(\s*\$\(OS\)\s*,\s*Windows_NT\s*\)\s*(#.*)?$"
)

#: Any conditional directive, for the nesting stack.
CONDITIONAL_OPEN: Final[re.Pattern[str]] = re.compile(r"^(ifeq|ifneq|ifdef|ifndef)\b")


class ConditionalFrame(TypedDict):
    """One open conditional, and whether its current arm is the Windows arm.

    Attributes:
        windows_first: The first arm is the Windows arm (``ifeq``); for
            ``ifneq`` it is the ``else`` arm.
        in_else: Whether the ``else`` has been passed.
        fences_windows: Whether this conditional is the platform fence at
            all; any other conditional exempts nothing.
    """

    windows_first: bool
    in_else: bool
    fences_windows: bool


def frame_exempt(frame: ConditionalFrame) -> bool:
    """Whether lines in a frame's current arm are exempt from the recipe grammar.

    Args:
        frame: The frame.

    Returns:
        True inside a Windows arm.
    """
    return frame["fences_windows"] and (frame["windows_first"] != frame["in_else"])


class ConditionalStack:
    """Tracks conditional nesting while a Makefile is read line by line."""

    def __init__(self) -> None:
        """Start outside every conditional."""
        self._frames: list[ConditionalFrame] = []

    def observe(self, line: str) -> bool:
        """Feed one non-recipe line; consume it when it is a directive.

        Args:
            line: The line, without its leading tab.

        Returns:
            True when the line was a conditional directive.
        """
        stripped = line.strip()
        if WINDOWS_CONDITION.match(stripped) is not None:
            self._frames.append(
                ConditionalFrame(
                    windows_first=stripped.startswith("ifeq"), in_else=False, fences_windows=True
                )
            )
            return True
        if CONDITIONAL_OPEN.match(stripped):
            self._frames.append(
                ConditionalFrame(windows_first=False, in_else=False, fences_windows=False)
            )
            return True
        if stripped == "else" and self._frames:
            self._frames[-1]["in_else"] = True
            return True
        if stripped == "endif" and self._frames:
            self._frames.pop()
            return True
        return False

    @property
    def exempt(self) -> bool:
        """Whether the current line sits in a Windows arm at any depth."""
        return any(frame_exempt(frame) for frame in self._frames)


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

    Args:
        line: A line beginning with a tab.

    Returns:
        The line with the tab and any leading ``@`` / ``-`` stripped.
    """
    return line[1:].lstrip("@-").strip()


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
        for pattern, rule in (*POWERSHELL_ONLY, *SH_ONLY, *DIVERGENT)
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
    conditionals = ConditionalStack()
    for number, line in enumerate(lines, start=1):
        if line.startswith("\t"):
            if not conditionals.exempt:
                found.extend(check_recipe_line(path, number, line))
            continue
        stripped = line.strip()
        if stripped.startswith("#") or conditionals.observe(line):
            continue
        for pattern, rule in MAKE_SHELL_FUNCTIONS:
            if re.search(pattern, line) and not (
                conditionals.exempt and rule.startswith("$(shell")
            ):
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
    "CONDITIONAL_OPEN",
    "DIVERGENT",
    "MAKEFILE_PATHSPEC",
    "MAKE_SHELL_FUNCTIONS",
    "POWERSHELL_ONLY",
    "SHELL_INCLUDE",
    "SH_ONLY",
    "WINDOWS_CONDITION",
    "ConditionalFrame",
    "ConditionalStack",
    "Violation",
    "check_makefile",
    "check_recipe_line",
    "expected_include",
    "first_code_line",
    "frame_exempt",
    "lint_grammar",
    "makefile_depth",
    "recipe_body",
    "render_violation",
    "tracked_makefiles",
]
