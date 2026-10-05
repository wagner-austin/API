"""The lines every PowerShell script tools/fleet renders opens with.

Each rendered script is executed by a Pester suite under MCPs' PowerShell
harness from its committed copy under ``rendered/``
(:mod:`fleet.core.rendered_powershell`, MCPs board task d69786fa), and the
harness holds it to the rules it holds a hand-written script to. Two of
those rules shape every render, so they are written once here:

  STRICT HEADER  the first two statements after the param block are
                 ``Set-StrictMode -Version Latest`` and
                 ``$ErrorActionPreference = 'Stop'``, so an unset variable
                 and a failing cmdlet stop the script instead of passing.
  NATIVE TOOLS   every Windows binary a script runs is named by its absolute
                 System32 path, and that path is a parameter whose default
                 is a plain string. A suite passes a stand-in that records
                 its arguments and exits with the case's code, so the
                 script's own lines, the exit-code reads included, run as
                 rendered while nothing on the test machine changes. The
                 default is a string, not a script block, because the
                 harness counts a script block default's commands and one
                 that restarts a machine cannot be run to cover them; and
                 not ``$PSScriptRoot``, which Windows PowerShell 5.1 leaves
                 empty in an advanced script's defaults under ``-File``.
"""

from __future__ import annotations

import re

#: The strict header, one statement per line.
STRICT_HEADER = ("Set-StrictMode -Version Latest", "$ErrorActionPreference = 'Stop'")

#: What a parameter name and a System32 binary's file name may be.
_PARAMETER = re.compile(r"^[A-Z][A-Za-z]*$")
_BINARY = re.compile(r"^[a-z][a-z0-9]*\.exe$")

#: What a compiled type's namespace-qualified name may be.
_QUALIFIED_TYPE = re.compile(r"^[A-Z][A-Za-z]*\.[A-Z][A-Za-z]*$")


def add_type_lines(qualified: str, source: str) -> tuple[str, ...]:
    """Lines that compile a C# type into the session unless it is already there.

    A render reaches a Win32 call PowerShell has no cmdlet for through a type
    compiled by ``Add-Type``, and a type cannot be compiled twice into one
    session: a Pester suite runs a render in its own process once per case,
    so the second case finds the type already loaded and skips the compile,
    which is the guard's other arm.

    Args:
        qualified: The type's ``Namespace.Type`` name, which the guard
            looks up.
        source: The C# source declaring it, embedded in a single-quoted
            here-string.

    Returns:
        The lines, the here-string's terminator at column 0 as PowerShell
        requires.

    Raises:
        ValueError: When the name is not of that plain shape, or the source
            holds the here-string's own terminator.
    """
    if not _QUALIFIED_TYPE.match(qualified):
        raise ValueError(f"a compiled type must be named Namespace.Type, not {qualified!r}")
    if "\n'@" in f"\n{source}":
        raise ValueError(f"the source of {qualified} would end its here-string early")
    return (
        f"if ($null -eq ('{qualified}' -as [type])) {{",
        "    Add-Type -TypeDefinition @'",
        *source.splitlines(),
        "'@",
        "}",
    )


def system32_parameter(name: str, binary: str) -> str:
    """A param-block line naming one System32 binary by its absolute path.

    Args:
        name: The parameter's name, PascalCase.
        binary: The binary's file name, lowercase, ending in ``.exe``.

    Returns:
        ``[string]$<name> = "$env:SystemRoot\\System32\\<binary>"``, without
        a trailing comma.

    Raises:
        ValueError: When either value is not of that plain shape, since both
            are embedded verbatim.
    """
    if not _PARAMETER.match(name):
        raise ValueError(f"a PowerShell parameter name must be PascalCase letters, not {name!r}")
    if not _BINARY.match(binary):
        raise ValueError(f"a System32 binary must be a lowercase name ending .exe, not {binary!r}")
    return f'[string]${name} = "$env:SystemRoot\\System32\\{binary}"'


def indented(lines: tuple[str, ...], *, depth: int) -> tuple[str, ...]:
    """Lines nested ``depth`` blocks deep, four spaces a block.

    A blank line stays blank, so no line ends in whitespace, and a
    here-string's terminator stays at column 0, where PowerShell requires
    it; the lines between a here-string's markers are its text, which the
    indent does not change the meaning of for C#.

    Args:
        lines: The lines, as written at the top level.
        depth: How many blocks deep they go.

    Returns:
        The nested lines.
    """
    prefix = "    " * depth
    return tuple(line if line in ("", "'@") else f"{prefix}{line}" for line in lines)


__all__ = ["STRICT_HEADER", "add_type_lines", "indented", "system32_parameter"]
