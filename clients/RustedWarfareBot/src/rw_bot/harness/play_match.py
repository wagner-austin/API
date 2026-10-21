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
    HOST_JAR,
    HOST_JAVAC,
    agent_jar,
    agent_sources,
    classes_dir,
    compile_command,
    package_command,
)
from rw_bot.harness.launch import (
    GAME_ENVIRONMENT,
    PORT_POLL_SECONDS,
    PORT_QUIET_SECONDS,
    LaunchConfig,
    game_command,
    planner_command,
    planner_pythonpath,
)
from rw_bot.harness.process_tree import (
    PORT_LISTENER_COMMAND,
    holder_is_an_orphaned_engine,
    parse_port_listener,
    parse_process_name,
    process_name_command,
)

#: The match played and the planner returned its own verdict.
EXIT_OK = 0

#: The agent could not be compiled or packaged.
EXIT_AGENT_BUILD_FAILED = 3

#: The engine started but never opened the channel. Distinct from a build
#: failure because they read identically in a transcript otherwise, and they
#: are diagnosed in completely different places.
EXIT_NO_CHANNEL = 4

#: How long to let a felled orphan engine release its port before this
#: launch binds it. The fell is a signal, and the kernel frees the socket
#: when the process is reaped rather than when the signal is sent.
TEARDOWN_SETTLE_SECONDS = 0.5

#: Where the engine's own log goes, relative to the play log.
ENGINE_STDOUT_SUFFIX = ".agent"

#: Where the JVM's crashes go. Kept apart from the engine's own log because a
#: crash interleaved into a merged stream is a crash nobody found.
ENGINE_STDERR_SUFFIX = ".err"

#: Wall clock on the tools this module runs -- the agent compile, the jar
#: packaging, the port and process listings. The slowest of them (javac
#: over the agent sources) finishes in seconds; ten minutes is an order of
#: magnitude past a loaded box, and a tool still running then is hung, not
#: slow. The wall exists because a child that never returns defeats every
#: check above it (the 2026-09-09 five-hour driver wedge).
TOOL_WALL_SECONDS = 600.0

#: Wall clock on the planner, which plays a whole match rather than running a
#: tool. Four hours: a match is minutes and a batch of them runs unattended,
#: so this is far past any honest run and is still FINITE. It is separate
#: from :data:`TOOL_WALL_SECONDS` because ten minutes is right for javac and
#: wrong for a match, and one constant covering both would be sized for the
#: longer and stop bounding the shorter. The planner is NOT captured, so
#: without this nothing bounds it at all -- which is the shape of the
#: 2026-09-09 five-hour driver wedge (board task 0d891468).
PLANNER_WALL_SECONDS = 14400.0


def compile_agent(classes: str) -> bool:
    """Compile the agent's sources into a directory with the host JDK.

    :data:`~rw_bot.harness.agent_build.HOST_JAVAC`, because the depot the
    game runs on ships no compiler. The self-test stops here; every other
    caller packages the classes next (:func:`build_agent`).

    Args:
        classes: Directory for the compiled classes, created if absent.

    Returns:
        True when every source compiled. On failure the compiler's own lines
        are printed first, then which tool failed.

    Raises:
        ValueError: When the agent source directory holds no Java.
        OSError: When the compiler cannot be started -- including a host with
            no JDK on ``PATH``, which names ``javac`` rather than failing later.
    """
    _test_hooks.make_dirs(Path(classes))
    sources = agent_sources(_test_hooks.list_names(Path(AGENT_SOURCE_DIR)))
    status, output = _test_hooks.run_capture(
        compile_command(HOST_JAVAC, classes, sources), TOOL_WALL_SECONDS
    )
    if status != 0:
        for line in output:
            _test_hooks.write_line(line)
        _test_hooks.write_line("[agent] javac failed")
        return False
    return True


def build_agent(jar_path: str, classes: str) -> bool:
    """Compile and package the agent with the host JDK.

    The one build every jar-producing caller shares: a match that is not
    using a snapshot, ``make agent`` publishing the jar a batch freezes, and
    the watched and hosted games.

    Args:
        jar_path: The jar to produce, relative to the repository root.
        classes: Directory for the compiled classes.

    Returns:
        True when the jar was built.

    Raises:
        ValueError: When the agent source directory holds no Java.
        OSError: When a tool cannot be started.
    """
    if not compile_agent(classes):
        return False
    status, output = _test_hooks.run_capture(
        package_command(HOST_JAR, jar_path, classes), TOOL_WALL_SECONDS
    )
    if status != 0:
        for line in output:
            _test_hooks.write_line(line)
        _test_hooks.write_line("[agent] jar failed")
        return False
    return True


def clear_orphaned_engine(port: int) -> None:
    """Fell a dead match's engine if it is still holding this match's port.

    Only a JVM may legitimately hold a match port, so a JVM holder is always
    an orphan of a worker that was killed without running its teardown.
    Anything else is left alone and the bind fails loudly, because felling a
    bystander is worse than failing to start: a match that does not start is
    reported, and a process that vanishes is not.

    Args:
        port: The channel port this match will bind.

    Raises:
        OSError: When the listing command cannot be started.
    """
    _, listing = _test_hooks.run_capture(PORT_LISTENER_COMMAND, TOOL_WALL_SECONDS)
    holder = parse_port_listener(listing, port)
    if holder is None:
        return
    _, named = _test_hooks.run_capture(process_name_command(holder), TOOL_WALL_SECONDS)
    name = parse_process_name(named)
    if not holder_is_an_orphaned_engine(name):
        _test_hooks.write_line(
            f"[play] port {port} is held by {name!r} (pid {holder}), which is not an "
            "engine; leaving it alone so the bind fails loudly"
        )
        return
    _test_hooks.write_line(f"[play] clearing orphaned engine (pid {holder}) off port {port}")
    _test_hooks.kill_tree(holder)
    _test_hooks.sleep(TEARDOWN_SETTLE_SECONDS)


def run_planner(config: LaunchConfig, root: PurePath) -> int:
    """Run the planner against a live match.

    Args:
        config: The launch.
        root: Absolute path of the repository root.

    Returns:
        The planner's exit status.

    Raises:
        OSError: When the planner cannot be started.
    """
    environment = dict(_test_hooks.read_environment())
    if config["tree"]:
        environment["PYTHONPATH"] = planner_pythonpath(root, config["tree"])
    command = planner_command(_test_hooks.read_executable(), config)
    return _test_hooks.run_inherited(command, environment, PLANNER_WALL_SECONDS)


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
    _test_hooks.remove_path(Path(jar_path))


def wait_for_channel(
    port: int, game: _test_hooks.SpawnedMatchProto, engine_log: Path
) -> str | None:
    """Wait for the agent's channel while the engine demonstrably boots.

    Keyed on PROGRESS rather than on a total clock: the engine's stream files
    grow while it loads -- mods, maps, the asset tree -- and a boot that keeps
    writing keeps its wait alive, however long a contended filesystem makes it
    take. The budget spends only on SILENCE. A panel member died on the old
    total clock while mid-boot under a 22-way asset-read burst (one 56 second
    read against 4ms on the other twenty-three members), which a total budget
    cannot tell from a hang -- and a dead engine now fails as soon as it is
    seen rather than after the full wait.

    Args:
        port: The channel port the agent leases.
        game: The engine process whose boot is being waited on.
        engine_log: The launch's log path; the engine's two stream files are
            derived from it, and their combined growth is the progress signal.

    Returns:
        ``None`` once the channel opens. Otherwise why the wait ended: the
        engine exited, or nothing was written for :data:`PORT_QUIET_SECONDS`.
        A value rather than an exception -- a boot that never opens the
        channel is a match RESULT, not a fault in the harness.
    """
    stdout_path = Path(f"{engine_log}{ENGINE_STDOUT_SUFFIX}")
    stderr_path = Path(f"{engine_log}{ENGINE_STDERR_SUFFIX}")
    quiet_start = _test_hooks.monotonic()
    written = _test_hooks.file_size(stdout_path) + _test_hooks.file_size(stderr_path)
    while True:
        failure = _test_hooks.probe_port(port, PORT_POLL_SECONDS)
        if failure is None:
            return None
        status = game.poll()
        if status is not None:
            return f"the engine exited with status {status} (last error: {failure})"
        now = _test_hooks.monotonic()
        size = _test_hooks.file_size(stdout_path) + _test_hooks.file_size(stderr_path)
        if size != written:
            written = size
            quiet_start = now
        if now - quiet_start >= PORT_QUIET_SECONDS:
            return f"the engine wrote nothing for {PORT_QUIET_SECONDS}s (last error: {failure})"
        _test_hooks.sleep(PORT_POLL_SECONDS)


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
    root = _test_hooks.resolve_root()
    stamp = _test_hooks.new_stamp()
    jar_path = agent_jar(config, stamp)
    classes = classes_dir(stamp)
    try:
        if config["tree"]:
            _test_hooks.write_line(f"[play] frozen tree: {config['tree']}")
        elif not build_agent(jar_path, classes):
            return EXIT_AGENT_BUILD_FAILED
        clear_orphaned_engine(config["port"])
        log = Path(config["play_log"])
        game = _test_hooks.spawn_game(
            game_command(config, root, jar_path),
            root / config["game_dir"],
            Path(f"{log}{ENGINE_STDOUT_SUFFIX}"),
            Path(f"{log}{ENGINE_STDERR_SUFFIX}"),
            # Inherited, plus what the dynamic linker needs to resolve the
            # engine's native GUI stack against itself.
            {**_test_hooks.read_environment(), **GAME_ENVIRONMENT},
        )
        try:
            failure = wait_for_channel(config["port"], game, log)
            if failure is not None:
                # The reason is printed, not swallowed: a refused connection
                # means the engine is up and the agent never bound, an exit
                # status means it died during boot, and quiet means it hung.
                # Ninety seconds of silence used to report none of the three.
                _test_hooks.write_line(
                    f"[play] the agent never opened port {config['port']}: {failure}"
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
            return run_planner(config, root)
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
    "TOOL_WALL_SECONDS",
    "build_agent",
    "clear_orphaned_engine",
    "compile_agent",
    "play",
    "run_planner",
    "wait_for_channel",
]
