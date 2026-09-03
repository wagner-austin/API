"""Argument reading shared by this monorepo's command-line entry points.

Arguments are parsed by hand rather than with argparse because argparse's
namespace is untyped attribute access and these packages hold every
expression to a known type. The surface is small enough that the hand parser
is the simpler artifact.

In ``platform_core`` rather than beside any one command because more than one
package now has entry points -- the hpc3 submitter and the Model-Trainer
scorer -- and forty lines of flag parsing copied into the second is a fork
that drifts on exactly the question it exists to settle: whether a mistyped
flag is refused or ignored.

An unknown or malformed flag raises rather than being ignored. A job
submitted under a mistyped flag is a different job, and a staging run under
one is a different corpus; both would otherwise proceed and report success.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

HELP_FLAGS: Final[tuple[str, ...]] = ("--help", "-h")
"""Tokens that ask what the flags are instead of asking for work."""


class HelpRequestedError(Exception):
    """The command line asked for usage rather than for the command to run.

    Not a :class:`ValueError`, which is what the other raises here are. Those
    say the command line is wrong; this one says it is a question. A boundary
    that treats the two alike prints a refusal for a request that was
    correctly typed, and exits non-zero for a caller who did nothing wrong.

    Attributes:
        allowed: The flags the command accepts, in declaration order, so the
            boundary can render them without knowing the command.
    """

    def __init__(self, allowed: Sequence[str]) -> None:
        """Record which flags the command accepts.

        Args:
            allowed: Flags this command accepts, each taking one value.
        """
        self.allowed: tuple[str, ...] = tuple(allowed)
        super().__init__("expected one of " + ", ".join(self.allowed))


def usage_text(allowed: Sequence[str]) -> str:
    """Render the accepted flags as a usage line.

    Args:
        allowed: Flags the command accepts, each taking one value.

    Returns:
        A single line naming every flag and that each takes a value. Every
        flag in this parser is single-valued, so the rendering needs no
        per-flag metadata and cannot drift from the parser's own rules.
    """
    return "usage: " + " ".join(f"{flag} <value>" for flag in allowed)


def take_value(tokens: Sequence[str], index: int, flag: str) -> str:
    """Read the value following a flag.

    Args:
        tokens: All command-line tokens.
        index: Position of the value, one past the flag.
        flag: Flag being read, used in the error message.

    Returns:
        The value.

    Raises:
        ValueError: If the flag ends the command line, or the next token is
            itself a flag. ``--host --verbose`` would otherwise bind the
            string ``--verbose`` as a hostname.
    """
    if index >= len(tokens):
        raise ValueError(f"{flag} requires a value")
    value = tokens[index]
    if value.startswith("--"):
        raise ValueError(f"{flag} requires a value, got the flag {value!r}")
    return value


def parse_single_flags(tokens: Sequence[str], allowed: Sequence[str]) -> dict[str, str]:
    """Read single-valued flags from a command line.

    Args:
        tokens: Command-line tokens, excluding the program name.
        allowed: Flags this command accepts, each taking exactly one value.

    Returns:
        Flag names mapped to their values. Absent flags are absent from the
        mapping rather than present with a placeholder, so a caller
        distinguishes "not given" from "given as empty".

    Raises:
        HelpRequestedError: If ``--help`` or ``-h`` appears anywhere in the
            tokens. Checked before the unknown-flag refusal, because asking
            what the flags are is the one thing a caller who does not know
            them can type, and answering it with "unknown argument" is the
            interface refusing to describe itself.
        ValueError: If a token is not an allowed flag, a flag is repeated, or
            a flag's value is missing. Repetition raises because the intent
            is ambiguous: silently keeping the last would discard a value the
            caller typed deliberately.
    """
    parsed: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in HELP_FLAGS:
            raise HelpRequestedError(allowed)
        if token not in allowed:
            raise ValueError(f"unknown argument {token!r}; expected one of {list(allowed)}")
        if token in parsed:
            raise ValueError(f"{token} given more than once")
        parsed[token] = take_value(tokens, index + 1, token)
        index += 2
    return parsed


def require_flag(parsed: Mapping[str, str], flag: str) -> str:
    """Read a flag that the command cannot run without.

    Args:
        parsed: Flags already read from the command line.
        flag: Flag to read.

    Returns:
        The flag's value.

    Raises:
        ValueError: If the flag was not given. There is no default: a
            defaulted host or destination would send work somewhere the
            caller did not name.
    """
    if flag not in parsed:
        raise ValueError(f"{flag} is required")
    return parsed[flag]


__all__ = [
    "HELP_FLAGS",
    "HelpRequestedError",
    "parse_single_flags",
    "require_flag",
    "take_value",
    "usage_text",
]
