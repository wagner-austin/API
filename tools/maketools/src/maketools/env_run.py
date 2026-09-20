"""``env``: run a command with named variables set, unset, defaulted or drawn.

The PowerShell recipes wrote ``$env:X = '1'; poetry run tool`` and
``if (-not $env:X) { $env:X = '300' }; ...``; sh would write ``X=1 tool``
and ``${X:-300}``. Neither reads in the other shell, so the assignment
moves here and the recipe stays one plain command::

    $(PYTHON) .../run.py env TANKPIT_BOT_SESSION_SECONDS=30 -- poetry run tankpit-bot
    $(PYTHON) .../run.py env TANKPIT_OUTPUT= -- poetry run tankpit-sniff
    $(PYTHON) .../run.py env SESSION?=300 --then "poetry run scorecard" -- poetry run bot
    $(PYTHON) .../run.py env --draw PORT=27600-27999 -- poetry run play --port @PORT@

``NAME=VALUE`` sets, ``NAME=`` (empty) unsets, ``NAME?=VALUE`` sets only
when the variable is absent or blank, and ``--draw NAME=LOW-HIGH`` sets
NAME to one random integer in the inclusive range. Every assigned name is
also substituted into the command's arguments as ``@NAME@`` with its final
value, which is how a drawn port reaches a flag the child reads from argv
rather than from the environment: the recipe that drew it with
``$(shell python -c ...)`` cannot, because ``$(shell`` runs the platform
shell and the grammar bans it outside the prologue.

``--then "<command>"`` runs a second command AFTER the first whatever the
first's status, and the exit status is the first non-zero of the two: a
scorecard must print even when the bot failed (operator flag 2026-09-03, a
teardown wedge exited before the scorecard ever printed), and the failure
must still be the recipe's verdict.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks

#: Separates the assignments from the command.
SEPARATOR: Final[str] = "--"

#: Introduces the command that runs after the first regardless of its status.
THEN_FLAG: Final[str] = "--then"

#: Introduces a ``NAME=LOW-HIGH`` draw.
DRAW_FLAG: Final[str] = "--draw"


class EnvRequest(TypedDict):
    """A parsed ``env`` invocation.

    Attributes:
        assignments: ``(name, value, only_if_unset)`` triples in order; an
            empty value means unset.
        draws: ``(name, low, high)`` triples in order.
        command: The argv to run.
        then: A second command, tokenised, or empty for none.
    """

    assignments: list[tuple[str, str, bool]]
    draws: list[tuple[str, int, int]]
    command: list[str]
    then: list[str]


def parse_draw(item: str) -> tuple[str, int, int]:
    """Parse one ``NAME=LOW-HIGH`` draw.

    Args:
        item: The text after ``--draw``.

    Returns:
        The name and the inclusive bounds.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` when the shape is wrong or the range
            is empty.
    """
    name, separator, span = item.partition("=")
    low_text, dash, high_text = span.partition("-")
    shaped = name != "" and separator != "" and dash != ""
    if not (shaped and low_text.isdigit() and high_text.isdigit()):
        raise AppError(MaketoolsErrorCode.USAGE, f"{DRAW_FLAG} {item!r} is not NAME=LOW-HIGH")
    low = int(low_text)
    high = int(high_text)
    if high < low:
        raise AppError(MaketoolsErrorCode.USAGE, f"{DRAW_FLAG} {item!r} has an empty range")
    return (name, low, high)


def parse_env_arguments(arguments: Sequence[str]) -> EnvRequest:
    """Split ``env``'s arguments into assignments, draws and commands.

    Args:
        arguments: Everything after ``env``.

    Returns:
        The request.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` when the separator is missing, an
            assignment has no ``=``, a flag has no value, or the command is
            empty.
    """
    if SEPARATOR not in arguments:
        raise AppError(
            MaketoolsErrorCode.USAGE,
            f"env needs '{SEPARATOR}' between the assignments and the command",
        )
    split = list(arguments).index(SEPARATOR)
    head = list(arguments[:split])
    command = list(arguments[split + 1 :])
    if not command:
        raise AppError(MaketoolsErrorCode.USAGE, f"env has no command after '{SEPARATOR}'")
    then: list[str] = []
    assignments: list[tuple[str, str, bool]] = []
    draws: list[tuple[str, int, int]] = []
    while head:
        item = head.pop(0)
        if item in (THEN_FLAG, DRAW_FLAG):
            if not head:
                needs = "a quoted command" if item == THEN_FLAG else "NAME=LOW-HIGH"
                raise AppError(MaketoolsErrorCode.USAGE, f"{item} needs {needs}")
            value = head.pop(0)
            if item == THEN_FLAG:
                then = shlex.split(value)
            else:
                draws.append(parse_draw(value))
            continue
        if "?=" in item:
            name, value = item.split("?=", 1)
            assignments.append((name, value, True))
        elif "=" in item:
            name, value = item.split("=", 1)
            assignments.append((name, value, False))
        else:
            raise AppError(
                MaketoolsErrorCode.USAGE,
                f"env assignment {item!r} is not NAME=VALUE or NAME?=VALUE",
            )
        if name == "":
            raise AppError(MaketoolsErrorCode.USAGE, f"env assignment {item!r} has no name")
    return EnvRequest(assignments=assignments, draws=draws, command=command, then=then)


def apply_assignments(
    environment: dict[str, str], assignments: Sequence[tuple[str, str, bool]]
) -> dict[str, str]:
    """Apply the assignments to a copy of an environment.

    Args:
        environment: The starting environment.
        assignments: From :func:`parse_env_arguments`.

    Returns:
        The child's environment.
    """
    result = dict(environment)
    for name, value, only_if_unset in assignments:
        if only_if_unset and result.get(name, "").strip() != "":
            continue
        if value == "":
            result.pop(name, None)
        else:
            result[name] = value
    return result


def substitute(argv: Sequence[str], names: Sequence[str], environment: dict[str, str]) -> list[str]:
    """Replace every ``@NAME@`` in an argv with the name's final value.

    Args:
        argv: The command.
        names: The assigned and drawn names.
        environment: The child's environment; a name it lacks (unset)
            substitutes as the empty string.

    Returns:
        The argv with every marker replaced.
    """
    result: list[str] = []
    for argument in argv:
        for name in names:
            argument = argument.replace(f"@{name}@", environment.get(name, ""))
        result.append(argument)
    return result


def run_env(arguments: Sequence[str], cwd: Path) -> int:
    """Run ``env``.

    Args:
        arguments: Everything after ``env``.
        cwd: The recipe's directory.

    Returns:
        The first non-zero status of the command and the ``--then`` command,
        or 0.
    """
    request = parse_env_arguments(arguments)
    drawn: list[tuple[str, str, bool]] = [
        (name, str(_test_hooks.draw(low, high)), False) for name, low, high in request["draws"]
    ]
    assignments = [*request["assignments"], *drawn]
    environment = apply_assignments(_test_hooks.environ(), assignments)
    names = [name for name, _value, _only in assignments]
    code = _test_hooks.run_inheriting(
        substitute(request["command"], names, environment),
        cwd=cwd,
        env=environment,
        new_session=False,
    )
    if request["then"]:
        then_code = _test_hooks.run_inheriting(
            substitute(request["then"], names, environment),
            cwd=cwd,
            env=environment,
            new_session=False,
        )
        if code == 0:
            code = then_code
    return code


__all__ = [
    "DRAW_FLAG",
    "SEPARATOR",
    "THEN_FLAG",
    "EnvRequest",
    "apply_assignments",
    "parse_draw",
    "parse_env_arguments",
    "run_env",
    "substitute",
]
