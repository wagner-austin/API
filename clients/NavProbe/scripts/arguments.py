"""Command-line parsing shared by the measurement scripts.

Two scripts take a ``--device`` flag and several numeric positionals, and both
would otherwise declare that parsing themselves. One declaration means the flag
behaves identically in both, and that a fix to how an incomplete flag is
handled cannot land in one and miss the other.

Parsing raises rather than returning a sentinel. A mistyped sweep must stop
before it compiles anything -- a cold Warp compile costs minutes, and a run that
started from the wrong arguments is worse than one that never started -- and the
error names the argument that was wrong rather than printing a usage block and
leaving the caller to guess which field it meant.
"""

from __future__ import annotations

from collections.abc import Sequence

from navprobe import NavProbeError

#: The flag selecting a Warp device.
DEVICE_FLAG = "--device"

#: Device used when :data:`DEVICE_FLAG` is absent.
DEFAULT_DEVICE = "cuda:0"

#: The flag pinning the iterative line-search kernel's CUDA block size.
LINESEARCH_BLOCK_DIM_FLAG = "--linesearch-block-dim"


class ScriptArgumentError(NavProbeError):
    """A measurement script's command line could not be parsed.

    Args:
        code: Stable identifier in the ``NP-ARGS-<NNN>`` range.
        message: Human-readable description of what went wrong, naming the
            argument at fault.
    """


def split_device(args: Sequence[str]) -> tuple[str, list[str]]:
    """Split a ``--device`` flag out of an argument list.

    A flag rather than a positional because one script's world-count list is
    variadic and would swallow a trailing positional -- and because the argument
    order documented before the flag existed keeps working unchanged.

    Args:
        args: Arguments excluding the program name.

    Returns:
        The requested device and the remaining arguments, in order.

    Raises:
        ScriptArgumentError: When the flag is present with no value after it.
            Treating a dangling flag as "use the default" would run the whole
            sweep on the wrong card and label it with the right one.
    """
    remaining = list(args)
    if DEVICE_FLAG not in remaining:
        return DEFAULT_DEVICE, remaining
    index = remaining.index(DEVICE_FLAG)
    if index + 1 >= len(remaining):
        raise ScriptArgumentError(
            "NP-ARGS-002", f"{DEVICE_FLAG} needs a device identifier after it"
        )
    return remaining[index + 1], remaining[:index] + remaining[index + 2 :]


def require_count(raw: str, name: str) -> int:
    """Convert a positional argument to a count of zero or greater.

    Args:
        raw: The argument to convert.
        name: The argument's name, used in the error message.

    Returns:
        The argument as a non-negative integer.

    Raises:
        ScriptArgumentError: When the argument is not a base-ten non-negative
            integer.
    """
    if not raw.isdigit():
        raise ScriptArgumentError(
            "NP-ARGS-003", f"{name} must be a whole number of zero or greater, got {raw!r}"
        )
    return int(raw)


def require_positive_count(raw: str, name: str) -> int:
    """Convert a positional argument to a count of one or greater.

    Args:
        raw: The argument to convert.
        name: The argument's name, used in the error message.

    Returns:
        The argument as a positive integer.

    Raises:
        ScriptArgumentError: When the argument is not a base-ten integer above
            zero.
    """
    value = require_count(raw, name)
    if value < 1:
        raise ScriptArgumentError("NP-ARGS-004", f"{name} must be greater than zero, got {value}")
    return value


def split_linesearch_block_dim(args: Sequence[str]) -> tuple[int | None, list[str]]:
    """Split a ``--linesearch-block-dim`` flag out of an argument list.

    Absent means "leave the vendor's default in place", which is a different
    statement from any number and is why the return is optional rather than
    defaulted to 32. A report that recorded 32 when nobody asked for it would
    claim the run pinned a value it merely inherited, and the pin is the thing
    a cross-architecture comparison has to hold fixed.

    Args:
        args: Arguments excluding the program name.

    Returns:
        The requested block size or ``None``, and the remaining arguments.

    Raises:
        ScriptArgumentError: When the flag is present with no value after it,
            or with a value that is not a positive integer. A dangling flag is
            refused for the same reason ``--device`` refuses one: silently
            falling back would run the sweep under conditions the report then
            misstates.
    """
    remaining = list(args)
    if LINESEARCH_BLOCK_DIM_FLAG not in remaining:
        return None, remaining
    index = remaining.index(LINESEARCH_BLOCK_DIM_FLAG)
    if index + 1 >= len(remaining):
        raise ScriptArgumentError(
            "NP-ARGS-004", f"{LINESEARCH_BLOCK_DIM_FLAG} needs a block size after it"
        )
    value = require_positive_count(remaining[index + 1], LINESEARCH_BLOCK_DIM_FLAG)
    return value, remaining[:index] + remaining[index + 2 :]


__all__ = [
    "DEFAULT_DEVICE",
    "DEVICE_FLAG",
    "LINESEARCH_BLOCK_DIM_FLAG",
    "ScriptArgumentError",
    "require_count",
    "require_positive_count",
    "split_device",
    "split_linesearch_block_dim",
]
