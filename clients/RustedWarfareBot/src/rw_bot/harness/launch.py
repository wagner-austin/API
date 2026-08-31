"""Every command a headless match is started by, composed and nothing else.

This is the half of the old ``scripts/make/play.ps1`` that decides WHAT to run.
The half that runs it lives in :mod:`rw_bot.harness.play_match`, reaching the
operating system only through :mod:`rw_bot.harness._test_hooks`, exactly as
:mod:`rw_bot.harness.clone` and :mod:`rw_bot.harness.runner` are split.

WHY IT MOVED OUT OF POWERSHELL AT ALL. Not preference: a launcher written in
PowerShell cannot start a match on a Linux compute node, and rewriting it as a
second shell script beside the first would make two launchers to keep in step
-- with the failure of a missed edit being a batch that runs the wrong
configuration and still reports a scorecard. The composition here is one
description of a launch that both platforms read, which is the only shape in
which "the workstation and the cluster ran the same experiment" is checkable
rather than hoped for.

WHAT IS DELIBERATELY STILL A STRING. The agent takes its options as one
``key=value;key=value`` argument, and that semicolon is the AGENT's separator,
not the operating system's -- it is identical on every platform and must not
be routed through :func:`~rw_bot.platform_id.path_list_separator`, which is
the character that genuinely does change. Two separators that look alike and
mean different things is exactly the confusion this package has already paid
for once, so they are named apart here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePath
from typing import TypedDict

from rw_bot.harness.jvm import game_classpath, has_module_system, tool_path
from rw_bot.platform_id import is_windows, path_list_separator
from rw_bot.validation import (
    require_int,
    require_non_empty_str,
    require_positive_int,
    require_str,
)

#: Heap ceiling for a headless match. The engine's own launcher scripts ship
#: this figure and a match holds about 430 MB of it, so it is headroom rather
#: than a target.
HEAP = "-Xmx1000M"

#: Modules the agent reflects into.
#:
#: Passed only to a JVM that HAS a module system. The engine is built for Java
#: 8 semantics; the Windows depot runs it on OpenJDK 13, which enforces module
#: boundaries and needs these or the agent's reflection fails at class-load.
#: The Linux depot ships a JRE 8, which has no module system and REJECTS the
#: option outright -- so passing it there is not harmless, it is a JVM that
#: never starts (:func:`~rw_bot.harness.jvm.has_module_system`).
ADD_OPENS = ("java.base/java.lang", "java.base/java.util")

#: Where the engine looks for its native libraries: the game directory, which
#: is also its working directory. Relative for that reason -- an absolute path
#: here would pin a clone to the machine that made it.
LIBRARY_PATH = "-Djava.library.path=."

#: What the dynamic linker reads to resolve a loaded library's OWN
#: dependencies, and the value the game's Linux launcher gives it. The JVM
#: option above covers the first load; this covers what that library links
#: against (see :func:`game_environment`).
LIBRARY_SEARCH_VAR = "LD_LIBRARY_PATH"
LIBRARY_SEARCH_VALUE = "."

#: The engine's entry point.
MAIN_CLASS = "com.corrodinggames.rts.java.Main"

#: The display the engine is told to open.
#:
#: ``-nodisplay`` is not a no-OpenGL mode: Slick2D still opens a display and
#: creates framebuffer objects during boot, and the 10x10 one it picks for
#: itself fails once in-game UI renders. An explicit size is what makes the
#: sandbox survive its first frame ([[agent-render-callback-noop]]).
DISPLAY_SIZE = ("800", "600")

#: The same size as one token, for the X server's ``-screen`` argument.
DISPLAY_SIZE_TEXT = f"{DISPLAY_SIZE[0]}x{DISPLAY_SIZE[1]}"

#: Flags that make a match headless and silent. ``-nosound`` does not skip the
#: audio load -- the engine substitutes a null sound factory -- so it is about
#: not opening a device, not about speed ([[harness-nodisplay]]).
HEADLESS_FLAGS = ("-nodisplay", "-nosound")

#: The engine's own hardcoded free-for-all, used when no map is named.
SANDBOX_FLAG = "-sandbox"


#: How long the engine may go WITHOUT DEMONSTRABLE PROGRESS before the wait
#: for its channel port gives up. A quiet budget rather than a total one: a
#: boot that keeps writing to its streams keeps its wait alive, however long
#: the filesystem makes it take. The panel member that died at 90 seconds of
#: TOTAL budget was mid-boot under a 22-way asset-read burst -- a 56 second
#: single read against 4ms on the other twenty-three members -- and a total
#: clock cannot tell that engine from a hung one. Silence can.
PORT_QUIET_SECONDS = 90

#: How long to wait between connection attempts while it boots.
PORT_POLL_SECONDS = 1.0

#: Separator between the agent's own options. The AGENT's, not the operating
#: system's -- identical everywhere, see the module docstring.
AGENT_OPTION_SEPARATOR = ";"

#: The wrapper that gives a match an X server on a machine with none.
#:
#: ``-nodisplay`` IS NOT A NO-OPENGL MODE. Slick2D still opens a display and
#: creates framebuffer objects during boot, and the 10x10 one the engine picks
#: for itself fails once in-game UI renders. On a desktop with a card that
#: costs nothing; on a headless compute node there is no server to open at all
#: ([[harness-nodisplay]]).
#:
#: ``xvfb-run`` starts one, runs the command under it, and takes it down again
#: -- so the server's lifetime is the match's by construction rather than by a
#: teardown that has to remember. It is also inside the match's process group,
#: so the fell that reaches the engine reaches the server with it.
XVFB_COMMAND = "xvfb-run"

#: The screen the virtual server offers, matching :data:`DISPLAY_SIZE`. Depth
#: 24 because the engine's framebuffer objects want a true-colour visual and a
#: lower depth fails at creation rather than degrading.
XVFB_SCREEN = f"-screen 0 {DISPLAY_SIZE_TEXT}x24"

#: The unit-stat oracle every match reads its catalogue from.
#:
#: Generated by the engine's own ``-printunits`` against the pinned build, so
#: it is an artifact of the game rather than of this package -- which is why it
#: lives under ``wiki/sources`` beside the other probe outputs.
CATALOGUE = "wiki/sources/m0-probe/printunits.log"

#: The unit type-flag dump, captured the same way.
TYPE_DUMP = "wiki/sources/m11-pools/type-flags.ndjson"

#: What those two are called inside a frozen tree.
#:
#: Flat basenames, because a freeze copies an entry under its own name --
#: the same reason :data:`~rw_bot.harness.agent_build.FROZEN_AGENT_JAR` is
#: ``rw-agent.jar`` and not ``agent/build/rw-agent.jar``.
#:
#: THESE HAD TO BE FROZEN AT ALL. They were left out on the reasoning that a
#: registry dump is an artifact of the game build rather than code that
#: changes between batches, which is true and was not the question. A match
#: READS them at launch, by a repository-relative path, and a compute node
#: has no repository: the first cluster member to reach the planner died on
#:
#:     FileNotFoundError: 'wiki/sources/m0-probe/printunits.log'
#:
#: having already patched the engine, seeded it and held the world at its
#: first frame. Same argument that put the job file and the agent jar in the
#: tree; it simply was not applied to these two.
FROZEN_CATALOGUE = "printunits.log"
FROZEN_TYPE_DUMP = "type-flags.ndjson"


class LaunchConfig(TypedDict):
    """Everything one match's launch is decided by.

    Attributes:
        port: Channel port the agent listens on and the planner connects to.
        game_dir: The game directory this match plays in, relative to the
            repository root. A worker's clone during a sweep, the pinned copy
            for a single match.
        seed: Engine random seed, or 0 to leave the engine's own.
        lockstep: Engine frames between samples, or 0 to free-run.
        settle: Seconds to let the sandbox map settle before the planner
            connects. Ignored when a map is named, because then the world is
            held at its first frame.
        display: X display this match's own server runs on, or
            :data:`~rw_bot.harness.clone.NO_DISPLAY` when the machine already
            has one. A lease from the clone's ordinal, never a draw -- two
            matches sharing a display number race exactly as two sharing a
            port do.
        play_log: Where the engine's log goes, relative to the repository
            root.
        map: Skirmish map to play, or empty for the engine's sandbox.
        opponents: How many AI opponents, when a map is named.
        difficulty: AI difficulty, when a map is named.
        tree: A frozen code snapshot to import, or empty to import the working
            tree.
        pin_delta: Constant frame delta in milliseconds, or 0 for the wall
            clock.
        fast_forward: Wall-clock multiple, or 0 for realtime.
        rng_tap: Non-zero to arm the engine's per-caller draw counter.
        extra_agent_args: Agent options no launch knob names, already in the
            agent's own ``key=value;key=value`` spelling.
        module: The planner module to run.
        catalogue: Unit catalogue path.
        type_dump: Type-flag dump path.
        play_args: The planner's positional tail, space separated.
    """

    port: int
    game_dir: str
    seed: int
    lockstep: int
    settle: int
    display: int
    play_log: str
    map: str
    opponents: int
    difficulty: int
    tree: str
    pin_delta: int
    fast_forward: int
    rng_tap: int
    extra_agent_args: str
    module: str
    catalogue: str
    type_dump: str
    play_args: str


def decode_launch_config(payload: Mapping[str, str | int | float | bool]) -> LaunchConfig:
    """Read a launch from a flat payload.

    Every field is narrowed by a ``require_*`` validator rather than read
    straight out of the mapping, so a launch that is wrong is wrong HERE --
    naming the field and carrying a traceable code -- instead of reaching the
    engine as a malformed argument and failing ninety seconds later as a
    channel that never opened.

    Args:
        payload: Field values by name.

    Returns:
        The launch.

    Raises:
        DecodeError: ``RW-DECODE-001`` when a field is absent, ``RW-DECODE-002``
            when one carries the wrong type, ``RW-DECODE-003`` when a required
            name is blank, ``RW-DECODE-004`` when the port is not positive.
    """
    return LaunchConfig(
        # Positive, not merely integral: a port of zero is what the old recipe
        # meant by "draw one at random", and there is no draw left to fall
        # back to -- an invented port collides with a live match's lease.
        port=require_positive_int(payload, "port"),
        game_dir=require_non_empty_str(payload, "game_dir"),
        seed=require_int(payload, "seed"),
        lockstep=require_int(payload, "lockstep"),
        settle=require_int(payload, "settle"),
        display=require_int(payload, "display"),
        play_log=require_non_empty_str(payload, "play_log"),
        map=require_str(payload, "map"),
        opponents=require_int(payload, "opponents"),
        difficulty=require_int(payload, "difficulty"),
        tree=require_str(payload, "tree"),
        pin_delta=require_int(payload, "pin_delta"),
        fast_forward=require_int(payload, "fast_forward"),
        rng_tap=require_int(payload, "rng_tap"),
        extra_agent_args=require_str(payload, "extra_agent_args"),
        module=require_non_empty_str(payload, "module"),
        catalogue=require_non_empty_str(payload, "catalogue"),
        type_dump=require_non_empty_str(payload, "type_dump"),
        play_args=require_str(payload, "play_args"),
    )


def encode_launch_config(config: LaunchConfig) -> dict[str, str | int]:
    """Write a launch back to a flat payload.

    Args:
        config: The launch.

    Returns:
        Field values by name, as :func:`decode_launch_config` reads them.
    """
    return {
        "port": config["port"],
        "game_dir": config["game_dir"],
        "seed": config["seed"],
        "lockstep": config["lockstep"],
        "settle": config["settle"],
        "display": config["display"],
        "play_log": config["play_log"],
        "map": config["map"],
        "opponents": config["opponents"],
        "difficulty": config["difficulty"],
        "tree": config["tree"],
        "pin_delta": config["pin_delta"],
        "fast_forward": config["fast_forward"],
        "rng_tap": config["rng_tap"],
        "extra_agent_args": config["extra_agent_args"],
        "module": config["module"],
        "catalogue": config["catalogue"],
        "type_dump": config["type_dump"],
        "play_args": config["play_args"],
    }


def agent_arguments(config: LaunchConfig) -> str:
    """Compose the single argument the java agent parses its options out of.

    Every option except the port is omitted when it is off rather than passed
    as a zero. That is not tidiness: a frozen tree predating an option REJECTS
    the unknown key outright, so a batch resumed against an old snapshot only
    works if the launcher does not mention what that snapshot has never heard
    of ([[policy-determinism]]).

    Args:
        config: The launch.

    Returns:
        The options, joined by :data:`AGENT_OPTION_SEPARATOR`.
    """
    options = [f"channelPort={config['port']}"]
    if config["seed"]:
        options.append(f"randomSeed={config['seed']}")
    if config["lockstep"]:
        options.append(f"lockstepFrames={config['lockstep']}")
    if config["map"]:
        options.append(f"matchMap={config['map']}")
        options.append(f"matchOpponents={config['opponents']}")
        options.append(f"matchDifficulty={config['difficulty']}")
    if config["pin_delta"]:
        options.append(f"pinDeltaMs={config['pin_delta']}")
    if config["fast_forward"]:
        options.append(f"fastForward={config['fast_forward']}")
    if config["rng_tap"]:
        options.append("rngTap=true")
    if config["extra_agent_args"]:
        options.append(config["extra_agent_args"])
    return AGENT_OPTION_SEPARATOR.join(options)


def game_environment(platform: str) -> dict[str, str]:
    """Return the environment overlay the engine needs, beyond what it inherits.

    ``-Djava.library.path`` tells the JVM where to find a library it is asked
    to load. It does NOT tell the dynamic linker where to find that library's
    OWN dependencies, and the engine's native GUI stack has them:
    ``librocketConnector.so`` links against ``libRocketCore.so.1`` beside it.
    Without this the first load fails as a missing shared object naming a file
    that is plainly there.

    Read off the game's own Linux launcher, which sets exactly this before
    invoking the JVM.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        The variables to add. Empty on Windows, which resolves a DLL's
        dependencies from the process's own directory and needs no help.
    """
    if is_windows(platform):
        return {}
    return {LIBRARY_SEARCH_VAR: LIBRARY_SEARCH_VALUE}


def display_prefix(display: int, platform: str) -> tuple[str, ...]:
    """Return the wrapper that gives a match an X server, if it needs one.

    Args:
        display: The X display this match owns, or
            :data:`~rw_bot.harness.clone.NO_DISPLAY` for none.
        platform: A ``sys.platform`` value.

    Returns:
        The wrapper tokens, or an empty tuple when no server is to be started.
        Empty on Windows whatever the display says: there is no X there, the
        engine draws on the desktop compositor, and a wrapper would be a
        command that does not exist.

    Raises:
        ValueError: When the display is negative. ``:0`` is already excluded
            by :data:`~rw_bot.harness.clone.DISPLAY_BASE` -- it is a physical
            console, and a batch match must never take somebody's desktop --
            so a number below zero is not a spare sentinel but a bug.
    """
    if display < 0:
        raise ValueError(f"an X display is named by a non-negative number, got {display}")
    if is_windows(platform) or display == 0:
        return ()
    # The server number is a lease, not a search. `xvfb-run -a` picks one by
    # scanning for a free lock file, which is the same shape as the random
    # port draw that collided the first time eight matches launched in one
    # instant (imp-creep12, 2026-08-08).
    return (XVFB_COMMAND, "-n", str(display), "-s", XVFB_SCREEN)


def game_command(
    config: LaunchConfig, root: PurePath, jar_path: str, platform: str
) -> tuple[str, ...]:
    """Return the command that starts the engine.

    The agent jar and the log are absolute because the engine runs with the
    GAME directory as its working directory, so a relative path would resolve
    against the clone rather than the repository.

    Args:
        config: The launch.
        root: Absolute path of the repository root.
        jar_path: The agent jar, relative to the root.
        platform: A ``sys.platform`` value, which decides the classpath
            separator and what ``java`` is called.

    Returns:
        The argument vector. The JVM is first unless the match brings its own
        X server, in which case :func:`display_prefix` comes first and the JVM
        follows it.
    """
    command = [
        *display_prefix(config["display"], platform),
        str(root / config["game_dir"] / tool_path("java", platform)),
        HEAP,
    ]
    if has_module_system(platform):
        for module in ADD_OPENS:
            command.extend(("--add-opens", f"{module}=ALL-UNNAMED"))
    command.extend(
        (
            LIBRARY_PATH,
            f"-javaagent:{root / jar_path}={agent_arguments(config)}",
            "-cp",
            game_classpath(platform),
            MAIN_CLASS,
            *HEADLESS_FLAGS,
        )
    )
    if not config["map"]:
        command.append(SANDBOX_FLAG)
    command.extend(("-width", DISPLAY_SIZE[0], "-height", DISPLAY_SIZE[1]))
    command.extend(("-log", str(root / config["play_log"])))
    return tuple(command)


def planner_command(interpreter: str, config: LaunchConfig) -> tuple[str, ...]:
    """Return the command that runs the planner against a live match.

    ``-P`` accompanies a frozen tree so the repository root cannot outrank the
    snapshot on ``sys.path``. The working directory stays the repository root
    either way, so catalogue, type-dump, doctrine and trace paths resolve
    unchanged.

    Args:
        interpreter: The Python to run the planner with. The harness's own,
            so the planner runs in the environment the harness was started in
            rather than in whatever a launcher script names -- which is what
            removes ``poetry`` from a path that has to work inside an image
            with no poetry in it.
        config: The launch.

    Returns:
        The argument vector, interpreter first.
    """
    command = [interpreter]
    if config["tree"]:
        command.append("-P")
    command.extend(("-m", config["module"], str(config["port"])))
    command.extend((config["catalogue"], config["type_dump"]))
    command.extend(part for part in config["play_args"].split(" ") if part)
    return tuple(command)


def planner_pythonpath(root: PurePath, tree: str, platform: str) -> str:
    """Return the ``PYTHONPATH`` a planner importing a frozen tree needs.

    Args:
        root: Absolute path of the repository root.
        tree: The frozen snapshot, relative to the root.
        platform: A ``sys.platform`` value, which decides the separator.

    Returns:
        The snapshot and its ``src`` directory, joined for the platform.

    Raises:
        ValueError: When no tree is given. A blank ``PYTHONPATH`` is not the
            same as an unset one, and setting it blank would put the current
            directory on the path of a planner that was told not to have it.
    """
    if not tree:
        raise ValueError(
            "a PYTHONPATH is only composed for a frozen tree; setting it blank would "
            "put the working directory back on a path that was told not to have it"
        )
    snapshot = root / tree
    return path_list_separator(platform).join((str(snapshot), str(snapshot / "src")))


__all__ = [
    "ADD_OPENS",
    "AGENT_OPTION_SEPARATOR",
    "CATALOGUE",
    "DISPLAY_SIZE",
    "DISPLAY_SIZE_TEXT",
    "FROZEN_CATALOGUE",
    "FROZEN_TYPE_DUMP",
    "HEADLESS_FLAGS",
    "HEAP",
    "LIBRARY_PATH",
    "LIBRARY_SEARCH_VALUE",
    "LIBRARY_SEARCH_VAR",
    "MAIN_CLASS",
    "PORT_POLL_SECONDS",
    "PORT_QUIET_SECONDS",
    "SANDBOX_FLAG",
    "TYPE_DUMP",
    "XVFB_COMMAND",
    "XVFB_SCREEN",
    "LaunchConfig",
    "agent_arguments",
    "decode_launch_config",
    "display_prefix",
    "encode_launch_config",
    "game_command",
    "game_environment",
    "planner_command",
    "planner_pythonpath",
]
