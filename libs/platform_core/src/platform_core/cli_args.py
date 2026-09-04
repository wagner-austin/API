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

import argparse
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


def _wrong_type(key: str, value: str | int | bool | list[str] | None, expected: str) -> ValueError:
    """Build the refusal for a namespace attribute of the wrong type.

    Args:
        key: The attribute read.
        value: What was found there. Spelled as the union argparse can
            actually produce rather than as ``object``, which the workspace's
            typing guard refuses and which would let any value in here.
        expected: The type the command declared.

    Returns:
        The error to raise, naming the key, the expected type and the actual
        one. All three, because the cause is a parser declaration in another
        function and the reader has no other way to find it.
    """
    return ValueError(
        f"--{key.replace('_', '-')} parsed as {type(value).__name__}, expected {expected}; "
        "the parser's declaration for this flag disagrees with the type its "
        "reader wants"
    )


def namespace_str(ns: argparse.Namespace, key: str, default: str) -> str:
    """Read a string argument from an argparse namespace.

    Args:
        ns: The parsed namespace.
        key: The attribute to read.
        default: Value to use when the flag was not given.

    Returns:
        The flag's value, or ``default`` when the flag was not given --
        which argparse reports either by leaving the attribute off entirely
        or by setting it to None for a declared-but-unsupplied option. Both
        mean "not given"; neither is the wrong-type case below.

    Raises:
        ValueError: If the attribute is present with a non-string value. Both
            copies of this function that preceded it returned the default
            instead, and each carried a test pinning that silence. Absent and
            wrong-typed are different events: the first is a caller declining
            a flag, the second is a parser declaration contradicting its
            reader, and only the second is a defect.
    """
    value: str | int | bool | list[str] | None = getattr(ns, key, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value
    raise _wrong_type(key, value, "str")


def namespace_str_or_none(ns: argparse.Namespace, key: str) -> str | None:
    """Read an optional string argument from an argparse namespace.

    Args:
        ns: The parsed namespace.
        key: The attribute to read.

    Returns:
        The flag's value, or None when it was not given.

    Raises:
        ValueError: If the attribute is present with a non-string value.
    """
    value: str | int | bool | list[str] | None = getattr(ns, key, None)
    if value is None or isinstance(value, str):
        return value
    raise _wrong_type(key, value, "str or None")


def namespace_int(ns: argparse.Namespace, key: str, default: int) -> int:
    """Read an integer argument from an argparse namespace.

    Args:
        ns: The parsed namespace.
        key: The attribute to read.
        default: Value to use when the flag was not given.

    Returns:
        The flag's value, or ``default`` when the flag was not given --
        which argparse reports either by leaving the attribute off entirely
        or by setting it to None for a declared-but-unsupplied option. Both
        mean "not given"; neither is the wrong-type case below.

    Raises:
        ValueError: If the attribute is present with a non-integer value.
            ``bool`` is rejected as well: it is an ``int`` to Python, and a
            ``store_true`` flag read as a count would silently become 1.
    """
    value: str | int | bool | list[str] | None = getattr(ns, key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise _wrong_type(key, value, "int")
    return value


def namespace_bool(ns: argparse.Namespace, key: str, default: bool) -> bool:
    """Read a boolean argument from an argparse namespace.

    Args:
        ns: The parsed namespace.
        key: The attribute to read.
        default: Value to use when the flag was not given.

    Returns:
        The flag's value, or ``default`` when the flag was not given --
        which argparse reports either by leaving the attribute off entirely
        or by setting it to None for a declared-but-unsupplied option. Both
        mean "not given"; neither is the wrong-type case below.

    Raises:
        ValueError: If the attribute is present with a non-boolean value.
    """
    value: str | int | bool | list[str] | None = getattr(ns, key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise _wrong_type(key, value, "bool")


def namespace_str_tuple(ns: argparse.Namespace, key: str) -> tuple[str, ...]:
    """Read a repeatable string argument from an argparse namespace.

    Args:
        ns: The parsed namespace.
        key: The attribute to read, declared with ``action="append"``.

    Returns:
        The values given, empty when the flag was not given at all.

    Raises:
        ValueError: If the attribute is present but is not a list, or holds a
            non-string element. The version this replaces dropped offending
            elements silently, so a mistyped attachment path shortened the
            list and the mail went without it.
    """
    value: str | int | bool | list[str] | None = getattr(ns, key, None)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _wrong_type(key, value, "list of str")
    for element in value:
        if not isinstance(element, str):
            raise _wrong_type(key, element, "str")
    return tuple(value)


__all__ = [
    "HELP_FLAGS",
    "HelpRequestedError",
    "namespace_bool",
    "namespace_int",
    "namespace_str",
    "namespace_str_or_none",
    "namespace_str_tuple",
    "parse_single_flags",
    "require_flag",
    "take_value",
    "usage_text",
]
