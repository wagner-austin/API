"""The command line one headless match is started by.

Replaces the ``make play`` recipe and the PowerShell file behind it. A sweep
composes this command per job, so the flags here are the launch surface the
whole harness goes through -- which is why an unrecognised one is refused
rather than ignored: a match played under a mistyped flag is a DIFFERENT match
and would still file a scorecard.

Flag parsing is :mod:`platform_core.cli_args`, the monorepo's own, rather than
a parser written here. That module exists because the second copy of forty
lines of flag handling drifts on exactly the question it settles.

THE HELP IS GENERATED FROM THE FLAG TABLES, not written beside them. Eighteen
flags described by hand is eighteen chances for the description to outlive the
flag, and a launcher whose help lies is worse than one with none -- it is the
document someone reaches for precisely when they are unsure. Adding a flag
therefore adds its help, and removing one removes it, with no second edit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from platform_core.cli_args import parse_single_flags, require_flag

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks
from rw_bot.harness.launch import (
    CATALOGUE,
    TYPE_DUMP,
    LaunchConfig,
    decode_launch_config,
)
from rw_bot.harness.play_match import play

_NOT_A_WHOLE_NUMBER = "RW-LAUNCH-001"


class LaunchCommandError(RwBotError):
    """A match could not be started from the command line as written.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description naming the offending flag.
    """


#: Flags naming what must be stated.
#:
#: A match with no port has nothing to connect to, one with no game directory
#: has nothing to run, and one with no log has nowhere to say why it failed.
#: Nothing is defaulted here that a wrong guess would let a match run under:
#: the port especially, because a defaulted one would silently collide with a
#: live match's lease.
REQUIRED_FLAGS = ("--port", "--game-dir", "--play-log")

#: How long a sandbox map is left to settle before the planner connects.
DEFAULT_SETTLE_SECONDS = "22"

#: The planner run against a live match unless another is named.
DEFAULT_MODULE = "scripts.play"

#: Flags with a stated default, each of which turns a feature OFF or names a
#: fixed artifact. Zero and empty are what the engine and the agent read as
#: "not asked for", and an option a frozen tree predates must be absent rather
#: than zero -- see :func:`~rw_bot.harness.launch.agent_arguments`.
#:
#: The catalogue and type dump default to
#: :mod:`rw_bot.harness.launch`'s constants rather than being restated, so the
#: Makefile no longer carries its own copy of either path.
OPTIONAL_FLAGS = {
    "--seed": "0",
    "--lockstep": "0",
    "--settle": DEFAULT_SETTLE_SECONDS,
    "--display": "0",
    "--map": "",
    "--opponents": "1",
    "--difficulty": "0",
    "--tree": "",
    "--pin-delta": "0",
    "--fast-forward": "0",
    "--rng-tap": "0",
    "--extra-agent-args": "",
    "--module": DEFAULT_MODULE,
    "--play-args": "",
    "--catalogue": CATALOGUE,
    "--type-dump": TYPE_DUMP,
}

ALLOWED_FLAGS = (*REQUIRED_FLAGS, *OPTIONAL_FLAGS)

#: Flags whose value is a whole number. Named here rather than discovered by
#: trying ``int()`` and seeing what happens, so a non-numeric value is
#: reported against the flag that carried it.
NUMERIC_FLAGS = (
    "--port",
    "--seed",
    "--lockstep",
    "--settle",
    "--display",
    "--opponents",
    "--difficulty",
    "--pin-delta",
    "--fast-forward",
    "--rng-tap",
)

#: One line per flag, so the help says what each one is FOR rather than only
#: that it exists. Keyed by flag and checked against the flag tables by a
#: test, which is what stops a description outliving its flag.
FLAG_HELP = {
    "--port": "channel port the agent listens on; a clone's lease, never a guess",
    "--game-dir": "game directory to play in, relative to the repository root",
    "--play-log": "where the engine's log goes, relative to the repository root",
    "--seed": "engine random seed; 0 leaves the engine's own",
    "--lockstep": "engine frames between samples; 0 free-runs",
    "--settle": "seconds to let a sandbox map settle; ignored when a map is named",
    "--display": "X display to start a server on; 0 uses the machine's own",
    "--map": "skirmish map to play; empty plays the engine's sandbox",
    "--opponents": "AI opponents, when a map is named",
    "--difficulty": "AI difficulty, when a map is named",
    "--tree": "frozen code snapshot to import; empty imports the working tree",
    "--pin-delta": "constant frame delta in ms; 0 leaves the wall clock",
    "--fast-forward": "wall-clock multiple; 0 is realtime",
    "--rng-tap": "non-zero arms the engine's per-caller draw counter",
    "--extra-agent-args": "agent options no flag names, in key=value;key=value form",
    "--module": "planner module to run against the live match",
    "--play-args": "planner's positional tail: samples, doctrine, trace",
    "--catalogue": "unit catalogue the planner reads",
    "--type-dump": "unit type-flag dump the planner reads",
}

#: Asking for help is not an error, and a launcher that exits non-zero on it
#: makes every wrapper treat a puzzled human as a failed match.
EXIT_HELP = 0

#: What a caller types to get the help.
HELP_FLAG = "--help"


def render_usage() -> tuple[str, ...]:
    """Render the help, from the flag tables themselves.

    Returns:
        The lines to print: required flags first, then optional ones with the
        default each falls back to, every line carrying what the flag is for.
    """
    width = max(len(flag) for flag in ALLOWED_FLAGS)
    lines = [
        "usage: python -m rw_bot.harness.play_match_cli --port N --game-dir DIR "
        "--play-log PATH [options]",
        "",
        "Plays one headless match: builds or reuses the agent, starts the engine,",
        "waits for its channel, runs the planner, and tears everything down.",
        "",
        "required:",
    ]
    lines.extend(f"  {flag:<{width}}  {FLAG_HELP[flag]}" for flag in REQUIRED_FLAGS)
    lines.extend(("", "optional:"))
    for flag, default in OPTIONAL_FLAGS.items():
        shown = default if default else "(empty)"
        lines.append(f"  {flag:<{width}}  {FLAG_HELP[flag]} [default: {shown}]")
    return tuple(lines)


def _whole(values: Mapping[str, str], flag: str) -> int:
    """Read one flag's value as a whole number.

    Args:
        values: Flag values by name, defaults already applied.
        flag: The flag to read.

    Returns:
        The value.

    Raises:
        LaunchCommandError: ``RW-LAUNCH-001`` when the value is not a whole
            number, naming the flag. ``int()`` alone reports only the text it
            choked on, which for a launcher with nine numeric flags leaves the
            caller to guess which one they mistyped. Translated to a coded
            error rather than softened -- nothing is recovered here, the
            failure simply arrives with a name attached.
    """
    try:
        return int(values[flag])
    except ValueError as error:
        raise LaunchCommandError(
            _NOT_A_WHOLE_NUMBER, f"{flag} must be a whole number, got {values[flag]!r}"
        ) from error


def _payload(tokens: Sequence[str]) -> dict[str, str | int | float | bool]:
    """Turn command-line tokens into the flat payload a launch decodes from.

    Args:
        tokens: Arguments after the program name.

    Returns:
        Field values by name, numeric flags already whole.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
        LaunchCommandError: ``RW-LAUNCH-001`` when a numeric flag is not a
            number.
    """
    parsed = parse_single_flags(tokens, ALLOWED_FLAGS)
    for flag in REQUIRED_FLAGS:
        require_flag(parsed, flag)
    values = {**OPTIONAL_FLAGS, **parsed}
    # Converted from the declared list rather than one call per field, so a
    # numeric flag added without being declared fails here as a missing key
    # instead of silently reaching the engine as a string.
    numbers: dict[str, str | int | float | bool] = {
        _field(flag): _whole(values, flag) for flag in NUMERIC_FLAGS
    }
    text: dict[str, str | int | float | bool] = {
        _field(flag): value for flag, value in values.items() if flag not in NUMERIC_FLAGS
    }
    return {**text, **numbers}


def _field(flag: str) -> str:
    """Return the launch field one flag carries.

    Args:
        flag: A flag name, e.g. ``--pin-delta``.

    Returns:
        The field name, e.g. ``pin_delta``. Derived rather than tabulated: a
        second table mapping eighteen flags to eighteen fields is a second
        thing to keep in step, and the only difference is punctuation.
    """
    return flag.removeprefix("--").replace("-", "_")


def decode_launch(tokens: Sequence[str]) -> LaunchConfig:
    """Read a launch configuration from command-line tokens.

    The command line is turned into a payload and handed to
    :func:`~rw_bot.harness.launch.decode_launch_config`, so a launch composed
    by a sweep and one typed by a human are validated by the SAME rules. A
    second set of checks here would be a second thing to keep in step, and the
    first divergence would let one caller start a match the other could not.

    Args:
        tokens: Arguments after the program name.

    Returns:
        The configuration.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
        LaunchCommandError: ``RW-LAUNCH-001`` when a numeric flag is not a
            number.
        DecodeError: When a field is present but out of range -- a port that
            is not positive, or a blank name.
    """
    return decode_launch_config(_payload(tokens))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the match launcher.

    Args:
        argv: Argument list excluding the program name. ``None`` reads the
            process arguments.

    Returns:
        :data:`EXIT_HELP` when help was asked for, otherwise the planner's
        exit status or one of :mod:`rw_bot.harness.play_match`'s codes when
        the match did not get that far.

    Raises:
        ValueError: When the command line is malformed. Propagated rather than
            turned into an exit code: a malformed launch is a fault in
            whatever composed it, and a sweep that swallowed it would file the
            failure as if it were a match result.
        OSError: When a process cannot be started or a stream file opened.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    if HELP_FLAG in tokens or not tokens:
        for line in render_usage():
            _test_hooks.write_line(line)
        return EXIT_HELP
    return play(decode_launch(tokens))


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "ALLOWED_FLAGS",
    "DEFAULT_MODULE",
    "DEFAULT_SETTLE_SECONDS",
    "EXIT_HELP",
    "FLAG_HELP",
    "HELP_FLAG",
    "NUMERIC_FLAGS",
    "OPTIONAL_FLAGS",
    "REQUIRED_FLAGS",
    "decode_launch",
    "main",
    "render_usage",
]
