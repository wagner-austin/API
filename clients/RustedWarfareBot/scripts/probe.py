"""Running the game once for what it reports, rather than for a match.

Replaces ``scripts/make/probe.ps1`` and the inline PowerShell of ``make
boot-probe`` and ``make unit-dump``: every probe is one engine run from the
pinned game directory, differing only in whether the agent rides along, what
it is told, and the engine's own flags, so all of them are one composition
(:func:`~rw_bot.harness.launch.engine_command`) and six kinds.

* ``boot`` -- a headless boot and nothing else; the engine log is the result.
* ``units`` -- ``-printunits``, the stat catalogue the planner reads.
* ``sandbox`` -- the engine's own skirmish with the published agent attached
  (:data:`~rw_bot.harness.agent_build.BUILT_AGENT_JAR`), until interrupted.
* ``discover`` -- the same, snapshotting engine state at each ``--at`` offset
  into the log and exiting after the last.
* ``wire`` and ``type-flags`` -- the same again, the agent writing the world
  stream or the placement registry to ``--out``.

The agent options are composed HERE rather than passed in: they are joined by
``;``, which no Makefile recipe may carry (the portable grammar reads it
differently in the two shells), and a free-form option string is a probe the
caller can mistype into a different one.

``--out`` is made absolute before the agent sees it, because the engine runs
from the game directory and the agent writes where it is told. It is removed
before the run and its lines counted after, so the count is this run's.

THE ENGINE IS FELLED IF IT OUTLIVES THE PROBE. It leads its own session, so an
interrupt reaches this process and not the engine; the felling in the
``finally`` is what stops a ^C from leaving a JVM holding the game directory.

Run as ``python -m scripts.probe --kind wire --game-dir .game
--log runs/wire-capture.log --at 16,17,18 --out runs/world-sample.ndjson``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePath

from platform_core.cli_args import parse_single_flags, require_flag

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks
from rw_bot.harness.agent_build import BUILT_AGENT_JAR
from rw_bot.harness.launch import (
    AGENT_OPTION_SEPARATOR,
    DISPLAY_SIZE,
    GAME_ENVIRONMENT,
    HEADLESS_FLAGS,
    SANDBOX_FLAG,
    display_prefix,
    engine_command,
    java_agent_flag,
)
from rw_bot.harness.play_match import ENGINE_STDERR_SUFFIX, ENGINE_STDOUT_SUFFIX, EXIT_OK

_UNKNOWN_KIND = "RW-PROBE-001"
_OFFSETS_MISMATCHED = "RW-PROBE-002"
_OUT_MISMATCHED = "RW-PROBE-003"
_NOT_A_NUMBER = "RW-PROBE-004"

#: The engine exited non-zero. Its streams are beside the log.
EXIT_PROBE_FAILED = 6

#: The engine was still running at :data:`PROBE_WALL_SECONDS` and was felled.
EXIT_PROBE_TIMED_OUT = 7

#: How often the engine is asked whether it has exited.
PROBE_POLL_SECONDS = 1.0

#: How long a probe may run. A discovery probe exits by itself in seconds and
#: a sandbox probe runs until interrupted; four hours is far past either and
#: still finite, so a hung engine is felled rather than held forever.
PROBE_WALL_SECONDS = 14400.0

_SIZE = ("-width", DISPLAY_SIZE[0], "-height", DISPLAY_SIZE[1])
_WITH_AGENT = (*HEADLESS_FLAGS, SANDBOX_FLAG, *_SIZE)

#: The engine flags each kind runs with, after the main class.
KIND_FLAGS = {
    "boot": (*HEADLESS_FLAGS, *_SIZE),
    "units": (*HEADLESS_FLAGS, "-printunits"),
    "sandbox": _WITH_AGENT,
    "discover": _WITH_AGENT,
    "wire": _WITH_AGENT,
    "type-flags": _WITH_AGENT,
}

#: The kinds that attach the agent.
AGENT_KINDS = ("sandbox", "discover", "wire", "type-flags")

#: The kinds that snapshot at ``--at`` offsets and exit after the last.
DISCOVERY_KINDS = ("discover", "wire", "type-flags")

#: The kinds whose agent writes ``--out``, and the option naming it.
OUTPUT_OPTIONS = {"wire": "stateOutPath", "type-flags": "typeFlagsPath"}

REQUIRED_FLAGS = ("--kind", "--game-dir", "--log")
OPTIONAL_FLAGS = {"--at": "", "--out": "", "--display": "0"}


class ProbeCommandError(RwBotError):
    """A probe could not be run as asked.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description naming what was wrong.
    """


def agent_options(kind: str, at: str, out: str, root: PurePath) -> str:
    """Return what the agent is told for one kind of probe.

    Args:
        kind: One of :data:`AGENT_KINDS`.
        at: Comma-separated offsets in seconds, for a discovery kind.
        out: The output, relative to the root, for a kind that writes one.
        root: Absolute path of the repository root.

    Returns:
        The options, joined by the agent's own separator, or empty for the
        plain sandbox.
    """
    if kind not in DISCOVERY_KINDS:
        return ""
    options = [f"discoverAtSeconds={at}"]
    if kind in OUTPUT_OPTIONS:
        options.append(f"{OUTPUT_OPTIONS[kind]}={root / out}")
    options.append("exitAfterDiscovery=true")
    return AGENT_OPTION_SEPARATOR.join(options)


def check_probe(kind: str, at: str, out: str) -> None:
    """Raise unless the flags given are exactly the ones this kind reads.

    A flag a kind does not read is refused rather than ignored: the probe
    would report on a run that was not the one asked for.

    Args:
        kind: The kind asked for.
        at: The offsets given, or empty.
        out: The output given, or empty.

    Raises:
        ProbeCommandError: ``RW-PROBE-001`` for an unknown kind,
            ``RW-PROBE-002`` when ``--at`` is missing for a discovery kind or
            given to another, ``RW-PROBE-003`` likewise for ``--out``.
    """
    if kind not in KIND_FLAGS:
        raise ProbeCommandError(
            _UNKNOWN_KIND, f"--kind must be one of {', '.join(KIND_FLAGS)}, got {kind!r}"
        )
    if bool(at) != (kind in DISCOVERY_KINDS):
        raise ProbeCommandError(
            _OFFSETS_MISMATCHED,
            f"--at is required by {', '.join(DISCOVERY_KINDS)} and read by nothing else; "
            f"{kind!r} was given {at!r}",
        )
    if bool(out) != (kind in OUTPUT_OPTIONS):
        raise ProbeCommandError(
            _OUT_MISMATCHED,
            f"--out is required by {', '.join(OUTPUT_OPTIONS)} and read by nothing else; "
            f"{kind!r} was given {out!r}",
        )


def probe_command(
    kind: str, game_dir: str, log: str, at: str, out: str, display: int, root: PurePath
) -> tuple[str, ...]:
    """Return the command one probe runs.

    Args:
        kind: One of :data:`KIND_FLAGS`.
        game_dir: The pinned game directory, relative to the root.
        log: The engine's log, relative to the root.
        at: Offsets for a discovery kind, or empty.
        out: The agent's output for a kind that writes one, or empty.
        display: An X display to start a server on, or 0 for the machine's.
        root: Absolute path of the repository root.

    Returns:
        The argument vector.

    Raises:
        ProbeCommandError: As :func:`check_probe`.
        ValueError: When the display is negative.
    """
    check_probe(kind, at, out)
    agent: tuple[str, ...] = ()
    if kind in AGENT_KINDS:
        agent = (java_agent_flag(root / BUILT_AGENT_JAR, agent_options(kind, at, out, root)),)
    return (
        *display_prefix(display),
        *engine_command(root, game_dir, agent, KIND_FLAGS[kind], log),
    )


def wait_for_exit(game: _test_hooks.SpawnedMatchProto) -> int | None:
    """Wait for the engine to exit, up to :data:`PROBE_WALL_SECONDS`.

    Args:
        game: The engine.

    Returns:
        Its exit status, or None when it was still running at the wall.
    """
    started = _test_hooks.monotonic()
    while _test_hooks.monotonic() - started < PROBE_WALL_SECONDS:
        status = game.poll()
        if status is not None:
            return status
        _test_hooks.sleep(PROBE_POLL_SECONDS)
    return None


def run_probe(kind: str, game_dir: str, log: str, at: str, out: str, display: int) -> int:
    """Run one probe and report what it produced.

    Args:
        kind: One of :data:`KIND_FLAGS`.
        game_dir: The pinned game directory, relative to the root.
        log: The engine's log, relative to the root.
        at: Offsets for a discovery kind, or empty.
        out: The agent's output for a kind that writes one, or empty.
        display: An X display to start a server on, or 0.

    Returns:
        :data:`~rw_bot.harness.play_match.EXIT_OK`, :data:`EXIT_PROBE_FAILED`
        or :data:`EXIT_PROBE_TIMED_OUT`.

    Raises:
        ProbeCommandError: As :func:`check_probe`.
        OSError: When the engine cannot be started or an output read.
        ValueError: When the display is negative.
    """
    root = _test_hooks.resolve_root()
    command = probe_command(kind, game_dir, log, at, out, display, root)
    _test_hooks.make_dirs(Path(log).parent)
    if out:
        _test_hooks.remove_path(Path(out))
        _test_hooks.make_dirs(Path(out).parent)
    game = _test_hooks.spawn_game(
        command,
        root / game_dir,
        Path(f"{log}{ENGINE_STDOUT_SUFFIX}"),
        Path(f"{log}{ENGINE_STDERR_SUFFIX}"),
        {**_test_hooks.read_environment(), **GAME_ENVIRONMENT},
    )
    try:
        status = wait_for_exit(game)
    finally:
        if game.poll() is None:
            _test_hooks.kill_tree(game.pid)
    if status is None:
        _test_hooks.write_line(f"[probe] the engine was still running at {PROBE_WALL_SECONDS}s")
        return EXIT_PROBE_TIMED_OUT
    if status != 0:
        _test_hooks.write_line(
            f"[probe] the engine exited {status}; see {log}{ENGINE_STDERR_SUFFIX}"
        )
        return EXIT_PROBE_FAILED
    if out:
        lines = _test_hooks.read_text_lines(Path(out))
        _test_hooks.write_line(f"[probe] {len(lines)} line(s) written to {out}")
    _test_hooks.write_line(f"[probe] {kind} done, log {log}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the probe family of make targets.

    Args:
        argv: Flags, as :data:`REQUIRED_FLAGS` and :data:`OPTIONAL_FLAGS`.
            ``None`` reads the process arguments.

    Returns:
        What :func:`run_probe` returned.

    Raises:
        ProbeCommandError: ``RW-PROBE-004`` when ``--display`` is not a whole
            number, and as :func:`check_probe`.
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
        OSError: As :func:`run_probe`.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    parsed = parse_single_flags(tokens, (*REQUIRED_FLAGS, *OPTIONAL_FLAGS))
    for flag in REQUIRED_FLAGS:
        require_flag(parsed, flag)
    values = {**OPTIONAL_FLAGS, **parsed}
    if not values["--display"].isdigit():
        raise ProbeCommandError(
            _NOT_A_NUMBER, f"--display must be a whole number, got {values['--display']!r}"
        )
    return run_probe(
        values["--kind"],
        values["--game-dir"],
        values["--log"],
        values["--at"],
        values["--out"],
        int(values["--display"]),
    )


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "AGENT_KINDS",
    "DISCOVERY_KINDS",
    "EXIT_PROBE_FAILED",
    "EXIT_PROBE_TIMED_OUT",
    "KIND_FLAGS",
    "OPTIONAL_FLAGS",
    "OUTPUT_OPTIONS",
    "PROBE_POLL_SECONDS",
    "PROBE_WALL_SECONDS",
    "REQUIRED_FLAGS",
    "ProbeCommandError",
    "agent_options",
    "check_probe",
    "main",
    "probe_command",
    "run_probe",
    "wait_for_exit",
]
