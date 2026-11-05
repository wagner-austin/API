"""Read the numeric flags the weight-space commands take.

The three weight-space commands configure their runs by flag rather than by a
named plan table: each is run a handful of times, and its hpc3 run document,
which is committed, carries the exact command. A number on a command line is
still a value the record must be able to state, so each is parsed strictly
here and refused, not coerced, when it is not what the flag means.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from platform_core import cli_args


def require_int_flag(parsed: Mapping[str, str], flag: str, minimum: int) -> int:
    """Read a required integer flag at or above a floor.

    Args:
        parsed: The parsed flags.
        flag: The flag to read.
        minimum: The smallest value the flag accepts.

    Returns:
        The integer.

    Raises:
        ValueError: If the flag is absent, is not a plain decimal integer, or
            is below ``minimum``.
    """
    raw = cli_args.require_flag(parsed, flag)
    digits = raw[1:] if raw.startswith("-") else raw
    if not digits.isdigit():
        raise ValueError(f"{flag} must be an integer, got {raw!r}")
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{flag} must be at least {minimum}, got {value}")
    return value


def require_float_flag(parsed: Mapping[str, str], flag: str) -> float:
    """Read a required finite number.

    Args:
        parsed: The parsed flags.
        flag: The flag to read.

    Returns:
        The number.

    Raises:
        ValueError: If the flag is absent, does not parse as a number, or is
            an infinity or a NaN.
    """
    raw = cli_args.require_flag(parsed, flag)
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{flag} must be a finite number, got {raw!r}")
    return value


__all__ = ["require_float_flag", "require_int_flag"]
