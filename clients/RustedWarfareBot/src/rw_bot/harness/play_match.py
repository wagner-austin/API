"""Playing one headless match: build, start, wait, plan, tear down.

The half of the old ``scripts/make/play.ps1`` that DOES things.
:mod:`rw_bot.harness.launch` decides what every command is; this runs them, in
order, reaching the operating system only through
:mod:`rw_bot.harness._test_hooks`.

LAUNCH PLUMBING RUNS FROM THE WORKING TREE. A batch freezes ``src/rw_bot``,
``scripts`` and its agent jar so a mid-batch edit cannot reach a running
experiment, and the planner is started with ``-P`` and a ``PYTHONPATH`` into
that snapshot. This module is not part of that: how a match is BOOTED belongs
to the harness, and only what the match RUNS is frozen ([[policy-loop]]). It
therefore keeps working when an old snapshot is resumed, which is the whole
point of the split.

TEARDOWN IS UNCONDITIONAL, IN TWO NESTED LAYERS. The engine is felled whatever
the planner did, and the build artifacts are removed whatever the engine did.
A match that dies holding its channel port kills the NEXT match at the bind
rather than itself (vhdoom96b, 2026-08-09), so the felling cannot be on a
success path.
"""

from __future__ import annotations

from pathlib import Path, PurePath

from rw_bot.harness import _test_hooks
from rw_bot.harness.agent_build import (
    AGENT_SOURCE_DIR,
    agent_jar,
    agent_sources,
    classes_dir,
    compile_command,
    package_command,
)
from rw_bot.harness.jvm import tool_path
from rw_bot.harness.launch import (
    PORT_POLL_SECONDS,
    PORT_WAIT_SECONDS,
    LaunchConfig,
    game_command,
    game_environment,
    planner_command,
    planner_pythonpath,
)
from rw_bot.harness.process_tree import (
    holder_is_an_orphaned_engine,
    parse_port_listener,
    parse_process_name,
    port_listener_command,
    process_name_command,
)
from rw_bot.platform_id import pure_path

#: The match played and the planner returned its own verdict.
EXIT_OK = 0

#: The agent could not be compiled or packaged.
EXIT_AGENT_BUILD_FAILED = 3

#: The engine started but never opened the channel. Distinct from a build
#: failure because they read identically in a transcript otherwise, and they
#: are diagnosed in completely different places.
EXIT_NO_CHANNEL = 4

#: How long to let file handles settle before deleting a per-match jar.
#: Windows keeps a handle briefly after the JVM exits and the delete fails
#: without it; on POSIX the wait is harmless and the branch it would save is
#: not worth owning.
TEARDOWN_SETTLE_SECONDS = 0.5

#: Where the engine's own log goes, relative to the play log.
ENGINE_STDOUT_SUFFIX = ".agent"

#: Where the JVM's crashes go. Kept apart from the engine's own log because a
#: crash interleaved into a merged stream is a crash nobody found.
ENGINE_STDERR_SUFFIX = ".err"


def build_agent(config: LaunchConfig, jar_path: str, classes: str, platform: str) -> bool:
    """Compile and package the agent for a match that is not using a snapshot.

    Args:
        config: The launch.
        jar_path: The jar to produce.
        classes: Directory for the compiled classes.
        platform: A ``sys.platform`` value.

    Returns:
        True when the jar was built.

    Raises:
        ValueError: When the agent source directory holds no Java.
        OSError: When a tool cannot be started.
    """
    # The stated platform decides the separator, not the interpreter that
    # happens to be composing: a launch composed on one platform for the
    # other would otherwise carry the composer's slashes.
    game_dir = pure_path(platform)(config["game_dir"])
    _test_hooks.make_dirs(Path(classes))
    sources = agent_sources(_test_hooks.list_names(Path(AGENT_SOURCE_DIR)))
    javac = str(game_dir / tool_path("javac", platform))
    status, output = _test_hooks.run_capture(compile_command(javac, classes, sources))
    if status != 0:
        for line in output:
            _test_hooks.write_line(line)
        _test_hooks.write_line("[play] javac failed")
        return False
    jar_tool = str(game_dir / tool_path("jar", platform))
    status, output = _test_hooks.run_capture(package_command(jar_tool, jar_path, classes))
    if status != 0:
        for line in output:
            _test_hooks.write_line(line)
        _test_hooks.write_line("[play] jar failed")
        return False
    return True


def clear_orphaned_engine(port: int, platform: str) -> None:
    """Fell a dead match's engine if it is still holding this match's port.

    Only a JVM may legitimately hold a match port, so a JVM holder is always
    an orphan of a worker that was killed without running its teardown.
    Anything else is left alone and the bind fails loudly, because felling a
    bystander is worse than failing to start: a match that does not start is
    reported, and a process that vanishes is not.

    Args:
        port: The channel port this match will bind.
        platform: A ``sys.platform`` value.

    Raises:
        OSError: When the listing command cannot be started.
    """
    _, listing = _test_hooks.run_capture(port_listener_command(platform))
    holder = parse_port_listener(listing, port, platform)
    if holder is None:
        return
    _, named = _test_hooks.run_capture(process_name_command(holder, platform))
    name = parse_process_name(named, platform)
    if not holder_is_an_orphaned_engine(name, platform):
        _test_hooks.write_line(
            f"[play] port {port} is held by {name!r} (pid {holder}), which is not an "
            "engine; leaving it alone so the bind fails loudly"
        )
        return
    _test_hooks.write_line(f"[play] clearing orphaned engine (pid {holder}) off port {port}")
    _test_hooks.kill_tree(holder)
    _test_hooks.sleep(TEARDOWN_SETTLE_SECONDS)


def run_planner(config: LaunchConfig, root: PurePath, platform: str) -> int:
    """Run the planner against a live match.

    Args:
        config: The launch.
        root: Absolute path of the repository root.
        platform: A ``sys.platform`` value.

    Returns:
        The planner's exit status.

    Raises:
        OSError: When the planner cannot be started.
    """
    environment = dict(_test_hooks.read_environment())
    if config["tree"]:
        environment["PYTHONPATH"] = planner_pythonpath(root, config["tree"], platform)
    command = planner_command(_test_hooks.read_executable(), config)
    return _test_hooks.run_inherited(command, environment)


def _teardown_build(config: LaunchConfig, jar_path: str, classes: str) -> None:
    """Remove what this launch compiled, leaving a snapshot's jar alone.

    Args:
        config: The launch.
        jar_path: The jar this launch may have built.
        classes: The class directory it may have used.
    """
    if config["tree"]:
        return
    _test_hooks.remove_path(Path(classes))
    _test_hooks.sleep(TEARDOWN_SETTLE_SECONDS)
    _test_hooks.remove_path(Path(jar_path))


def play(config: LaunchConfig) -> int:
    """Play one match and return what the planner made of it.

    Args:
        config: The launch.

    Returns:
        :data:`EXIT_AGENT_BUILD_FAILED` when the agent could not be built,
        :data:`EXIT_NO_CHANNEL` when the engine never opened its channel, and
        otherwise the planner's own exit status.

    Raises:
        OSError: When a process cannot be started or a stream file opened.
        ValueError: When the agent source directory holds no Java.
    """
    platform = _test_hooks.read_platform()
    root = _test_hooks.resolve_root()
    stamp = _test_hooks.new_stamp()
    jar_path = agent_jar(config, stamp)
    classes = classes_dir(stamp)
    try:
        if config["tree"]:
            _test_hooks.write_line(f"[play] frozen tree: {config['tree']}")
        elif not build_agent(config, jar_path, classes, platform):
            return EXIT_AGENT_BUILD_FAILED
        clear_orphaned_engine(config["port"], platform)
        log = Path(config["play_log"])
        game = _test_hooks.spawn_game(
            game_command(config, root, jar_path, platform),
            root / config["game_dir"],
            Path(f"{log}{ENGINE_STDOUT_SUFFIX}"),
            Path(f"{log}{ENGINE_STDERR_SUFFIX}"),
            # Inherited, plus what the dynamic linker needs to resolve the
            # engine's native GUI stack against itself on Linux.
            {**_test_hooks.read_environment(), **game_environment(platform)},
        )
        try:
            failure = _test_hooks.wait_for_port(
                config["port"], PORT_WAIT_SECONDS, PORT_POLL_SECONDS
            )
            if failure is not None:
                # The reason is printed, not swallowed: a refused connection
                # means the engine is up and the agent never bound, and a
                # timeout with no route means the engine died during boot.
                # Ninety seconds of silence used to report neither.
                _test_hooks.write_line(
                    f"[play] the agent never opened port {config['port']} within "
                    f"{PORT_WAIT_SECONDS}s (last error: {failure})"
                )
                return EXIT_NO_CHANNEL
            if config["map"]:
                _test_hooks.write_line(
                    "[play] channel open; the world is held at its first frame, no settle"
                )
            else:
                _test_hooks.write_line(
                    f"[play] channel open; letting the map settle {config['settle']}s"
                )
                _test_hooks.sleep(config["settle"])
            return run_planner(config, root, platform)
        finally:
            _test_hooks.kill_tree(game.pid)
            _test_hooks.write_line("[play] game stopped")
    finally:
        _teardown_build(config, jar_path, classes)


__all__ = [
    "ENGINE_STDERR_SUFFIX",
    "ENGINE_STDOUT_SUFFIX",
    "EXIT_AGENT_BUILD_FAILED",
    "EXIT_NO_CHANNEL",
    "EXIT_OK",
    "TEARDOWN_SETTLE_SECONDS",
    "build_agent",
    "clear_orphaned_engine",
    "play",
    "run_planner",
]
