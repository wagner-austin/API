"""A match a human watches, or joins: ``make watch`` and ``make host``.

Replaces ``scripts/make/watch.ps1`` and ``scripts/make/host.ps1``. Both are a
:class:`~rw_bot.harness.launch.LaunchConfig` like any headless match, with the
reproducibility machinery off -- no seed, no lockstep, no pinned delta,
because the simulation must track the wall clock a person is watching
([[multiplayer-portability-invariants]]) -- and they differ from ``play`` in
two places only:

* the engine's flags: ``watch`` opens a window (no ``-nodisplay``) at a size
  worth watching; ``host`` stays headless and hosts a LAN game the agent
  starts from the menu (``hostMap``), so neither passes ``-sandbox``;
* the wait for the agent's channel, which is a fixed deadline rather than
  ``play``'s progress rule: a hosted game's channel opens only once a human
  has joined, and a lobby writes nothing while it waits.

The build, the planner and the teardown are ``play``'s own
(:func:`~rw_bot.harness.play_match.build_agent`,
:func:`~rw_bot.harness.play_match.run_planner`), against a stamped jar so a
watched game never holds the jar a batch freezes.

Run as ``python -m rw_bot.harness.interactive_cli --mode watch --port N
--game-dir .game --play-log runs/watch.log --map MAP [--opponents N]
[--difficulty N] [--module M] [--play-args ARGS]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from platform_core.cli_args import parse_single_flags, require_flag

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks
from rw_bot.harness.agent_build import AGENT_BUILD_DIR
from rw_bot.harness.launch import (
    CATALOGUE,
    DISPLAY_SIZE,
    GAME_ENVIRONMENT,
    HEADLESS_FLAGS,
    PORT_POLL_SECONDS,
    TYPE_DUMP,
    LaunchConfig,
    agent_arguments,
    engine_command,
    java_agent_flag,
)
from rw_bot.harness.play_match import (
    ENGINE_STDERR_SUFFIX,
    ENGINE_STDOUT_SUFFIX,
    EXIT_AGENT_BUILD_FAILED,
    EXIT_NO_CHANNEL,
    TOOL_WALL_SECONDS,
    build_agent,
    run_planner,
)
from rw_bot.harness.play_match_cli import DEFAULT_MODULE

_UNKNOWN_MODE = "RW-INTERACTIVE-001"
_NO_ADDRESS = "RW-INTERACTIVE-002"
_NOT_A_NUMBER = "RW-INTERACTIVE-003"

WATCH = "watch"
HOST = "host"

#: The engine flags each mode runs with: a window worth watching, or a
#: headless host at the size ``-nodisplay`` still needs a display to be.
MODE_FLAGS = {
    WATCH: ("-nosound", "-width", "1280", "-height", "800"),
    HOST: (*HEADLESS_FLAGS, "-width", DISPLAY_SIZE[0], "-height", DISPLAY_SIZE[1]),
}

#: How long the agent's channel may take to open, and what it means when it
#: does not: a watched game boots in seconds; a hosted one waits on a human.
WAIT_SECONDS = {WATCH: 120.0, HOST: 900.0}
WAIT_REASON = {
    WATCH: "the agent never opened its channel",
    HOST: "nobody joined the lobby",
}

#: The port a hosted game listens on for the human, which the engine fixes.
LAN_GAME_PORT = 5123

#: The command naming this machine's addresses, for the join line.
ADDRESS_COMMAND = ("hostname", "-I")

REQUIRED_FLAGS = ("--mode", "--port", "--game-dir", "--play-log", "--map")
OPTIONAL_FLAGS = {
    "--opponents": "1",
    "--difficulty": "3",
    "--module": DEFAULT_MODULE,
    "--play-args": "",
}
NUMERIC_FLAGS = ("--port", "--opponents", "--difficulty")


class InteractiveCommandError(RwBotError):
    """A watched or hosted game could not be started as asked.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description naming what was wrong.
    """


def interactive_config(mode: str, values: dict[str, str]) -> LaunchConfig:
    """Return the launch a watched or hosted game is.

    Args:
        mode: :data:`WATCH` or :data:`HOST`.
        values: Flag values by flag, defaults applied, numbers checked.

    Returns:
        The launch. A watched game names its map, opponents and difficulty to
        the agent, which starts the skirmish; a hosted one names only the map
        it hosts, because the human is the opponent.

    Raises:
        InteractiveCommandError: ``RW-INTERACTIVE-001`` for an unknown mode.
    """
    if mode not in MODE_FLAGS:
        raise InteractiveCommandError(
            _UNKNOWN_MODE, f"--mode must be one of {', '.join(MODE_FLAGS)}, got {mode!r}"
        )
    hosted = mode == HOST
    return LaunchConfig(
        port=int(values["--port"]),
        game_dir=values["--game-dir"],
        seed=0,
        lockstep=0,
        settle=0,
        display=0,
        play_log=values["--play-log"],
        map="" if hosted else values["--map"],
        opponents=int(values["--opponents"]),
        difficulty=int(values["--difficulty"]),
        tree="",
        pin_delta=0,
        fast_forward=0,
        rng_tap=0,
        extra_agent_args=f"hostMap={values['--map']}" if hosted else "",
        module=values["--module"],
        catalogue=CATALOGUE,
        type_dump=TYPE_DUMP,
        play_args=values["--play-args"],
    )


def wait_for_channel_within(
    port: int, game: _test_hooks.SpawnedMatchProto, seconds: float, reason: str
) -> str | None:
    """Wait for the agent's channel until a fixed deadline.

    Args:
        port: The channel port.
        game: The engine, whose death ends the wait at once.
        seconds: How long the channel may take.
        reason: What a channel that never opens means, for the message.

    Returns:
        None once the channel opens, otherwise why the wait ended.
    """
    deadline = _test_hooks.monotonic() + seconds
    while True:
        failure = _test_hooks.probe_port(port, PORT_POLL_SECONDS)
        if failure is None:
            return None
        status = game.poll()
        if status is not None:
            return f"the engine exited with status {status} (last error: {failure})"
        if _test_hooks.monotonic() >= deadline:
            return f"nothing listened on port {port} within {seconds}s: {reason}"
        _test_hooks.sleep(PORT_POLL_SECONDS)


def announce(mode: str) -> None:
    """Tell the human what is about to happen, before anything is built.

    Args:
        mode: :data:`WATCH` or :data:`HOST`.

    Raises:
        InteractiveCommandError: ``RW-INTERACTIVE-002`` when a hosted game
            cannot name this machine's addresses: a human with no address to
            join cannot play it, so the game is not started.
    """
    if mode == WATCH:
        _test_hooks.write_line("==> A game window will open; the bot plays, you watch.")
        return
    status, output = _test_hooks.run_capture(ADDRESS_COMMAND, TOOL_WALL_SECONDS)
    addresses = " or ".join(" ".join(output).split())
    if status != 0 or not addresses:
        raise InteractiveCommandError(
            _NO_ADDRESS,
            f"{' '.join(ADDRESS_COMMAND)} exited {status} naming {addresses!r}: there is no "
            "address to give the human who would join",
        )
    _test_hooks.write_line(
        f"==> JOIN FROM YOUR GAME CLIENT: Multiplayer -> Join by IP -> {addresses} "
        f"(port {LAN_GAME_PORT}). The match starts the moment you join."
    )


def play_interactive(mode: str, config: LaunchConfig) -> int:
    """Build the agent, start the game, play it with the planner, stop it.

    Args:
        mode: :data:`WATCH` or :data:`HOST`.
        config: The launch (:func:`interactive_config`).

    Returns:
        :data:`~rw_bot.harness.play_match.EXIT_AGENT_BUILD_FAILED`,
        :data:`~rw_bot.harness.play_match.EXIT_NO_CHANNEL`, or the planner's
        own exit status.

    Raises:
        InteractiveCommandError: As :func:`announce`.
        OSError: When a process cannot be started.
        ValueError: When the agent source directory holds no Java.
    """
    announce(mode)
    root = _test_hooks.resolve_root()
    stamp = _test_hooks.new_stamp()
    jar = f"{AGENT_BUILD_DIR}/rw-agent-{mode}-{stamp}.jar"
    classes = f"{AGENT_BUILD_DIR}/{mode}-{stamp}"
    try:
        if not build_agent(jar, classes):
            return EXIT_AGENT_BUILD_FAILED
        log = Path(config["play_log"])
        agent = (java_agent_flag(root / jar, agent_arguments(config)),)
        game = _test_hooks.spawn_game(
            engine_command(root, config["game_dir"], agent, MODE_FLAGS[mode], config["play_log"]),
            root / config["game_dir"],
            Path(f"{log}{ENGINE_STDOUT_SUFFIX}"),
            Path(f"{log}{ENGINE_STDERR_SUFFIX}"),
            {**_test_hooks.read_environment(), **GAME_ENVIRONMENT},
        )
        try:
            failure = wait_for_channel_within(
                config["port"], game, WAIT_SECONDS[mode], WAIT_REASON[mode]
            )
            if failure is not None:
                _test_hooks.write_line(f"[{mode}] {failure}")
                return EXIT_NO_CHANNEL
            return run_planner(config, root)
        finally:
            _test_hooks.kill_tree(game.pid)
            _test_hooks.write_line(f"[{mode}] game stopped")
    finally:
        _test_hooks.remove_path(Path(classes))
        _test_hooks.remove_path(Path(jar))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``make watch`` and ``make host``.

    Args:
        argv: Flags, as :data:`REQUIRED_FLAGS` and :data:`OPTIONAL_FLAGS`.
            ``None`` reads the process arguments.

    Returns:
        What :func:`play_interactive` returned.

    Raises:
        InteractiveCommandError: ``RW-INTERACTIVE-003`` when a numeric flag is
            not a whole number, and as :func:`interactive_config`.
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
        OSError: As :func:`play_interactive`.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    parsed = parse_single_flags(tokens, (*REQUIRED_FLAGS, *OPTIONAL_FLAGS))
    for flag in REQUIRED_FLAGS:
        require_flag(parsed, flag)
    values = {**OPTIONAL_FLAGS, **parsed}
    for flag in NUMERIC_FLAGS:
        if not values[flag].isdigit():
            raise InteractiveCommandError(
                _NOT_A_NUMBER, f"{flag} must be a whole number, got {values[flag]!r}"
            )
    mode = values["--mode"]
    return play_interactive(mode, interactive_config(mode, values))


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "ADDRESS_COMMAND",
    "HOST",
    "LAN_GAME_PORT",
    "MODE_FLAGS",
    "NUMERIC_FLAGS",
    "OPTIONAL_FLAGS",
    "REQUIRED_FLAGS",
    "WAIT_REASON",
    "WAIT_SECONDS",
    "WATCH",
    "InteractiveCommandError",
    "announce",
    "interactive_config",
    "main",
    "play_interactive",
    "wait_for_channel_within",
]
