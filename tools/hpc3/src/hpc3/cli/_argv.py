"""Argument reading shared by the three entry points.

Arguments are parsed by hand rather than with argparse because argparse's
namespace is untyped attribute access and this package holds every expression
to a known type. The surface is small enough that the hand parser is the
simpler artifact.

An unknown or malformed flag raises rather than being ignored. A job
submitted under a mistyped flag is a different job, and a staging run under
one is a different corpus; both would otherwise proceed and report success.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


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
        ValueError: If a token is not an allowed flag, a flag is repeated, or
            a flag's value is missing. Repetition raises because the intent
            is ambiguous: silently keeping the last would discard a value the
            caller typed deliberately.
    """
    parsed: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
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


__all__ = ["parse_single_flags", "require_flag", "take_value"]
