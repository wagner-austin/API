"""Typed launch configuration for one headless Rusted Warfare invocation.

The engine is a plain Java program whose behaviour headless is governed
entirely by its command line. This module makes that command line a validated
value rather than a string assembled at each call site.

The flag set encoded here is the one verified in
``wiki/pages/harness-nodisplay.md`` against game build 1.15 (code 176, build
#28): ``-nodisplay`` boots the full engine with no window, and an explicit
``-width``/``-height`` is required because ``-nodisplay`` alone selects a
10x10 display that fails once in-game UI renders.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypedDict

from rw_bot import RwBotError
from rw_bot.validation import (
    require_absolute_path,
    require_bool,
    require_non_empty_str,
    require_positive_int,
    require_str,
)

MAIN_CLASS: Final = "com.corrodinggames.rts.java.Main"
"""Entry-point class, stable across the obfuscation (see ``.game/fallback64.bat``)."""

CLASSPATH: Final = "game-lib.jar;libs/*"
"""Classpath relative to the game directory, lifted from the shipped launcher."""

JAVA_EXE_RELATIVE: Final = "jvm64/bin/java.exe"
"""Bundled 64-bit OpenJDK 13 shipped alongside the game."""

VERIFIED_WIDTH: Final = 800
"""Display width proven to reach a running skirmish headless."""

VERIFIED_HEIGHT: Final = 600
"""Display height proven to reach a running skirmish headless."""

_BOTH_MODES = "RW-LAUNCH-001"


class LaunchConfigError(RwBotError):
    """A :class:`LaunchConfig` is not usable as given.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending setting.
    """


class LaunchConfig(TypedDict):
    """One fully-specified headless game invocation.

    Attributes:
        game_dir: Directory holding ``game-lib.jar``, ``libs/`` and ``jvm64/``.
            The process runs with this as its working directory because the
            shipped launcher passes ``-Djava.library.path=.``.
        max_heap_mb: JVM maximum heap in megabytes.
        width: Virtual display width passed as ``-width``.
        height: Virtual display height passed as ``-height``.
        no_sound: Pass ``-nosound``; the engine substitutes a null sound
            factory rather than skipping audio load.
        sandbox: Pass ``-sandbox``, which boots directly into a skirmish and
            skips the menu entirely.
        print_units: Pass ``-printunits``, which emits the unit stat catalogue
            and exits before entering the game loop.
        log_path: Absolute path the engine writes its log to via ``-log``.
        agent_jar: Absolute path to the built javaagent, attached with
            ``-javaagent``. Always attached rather than optional: the agent is
            what keeps a headless engine alive past its first in-game frame
            (``wiki/pages/agent-render-callback-noop.md``), so a launch without
            it is not a supported mode, and modelling it as optional would add
            a branch whose only reachable outcome is a crash.
        agent_options: The agent's own ``;``-separated settings, appended after
            ``=``. Carried here rather than pasted onto the flag by each caller
            because it is the part that **contains spaces**: a map path like
            ``maps/skirmish/[p2]Lake (2p).tmx`` split the ``-javaagent`` flag in
            two when the launch was assembled as a shell string, and the JVM
            died with ``processing of -javaagent failed`` before the agent
            loaded. Rendered into one argv element here, quoting never arises.
    """

    game_dir: str
    max_heap_mb: int
    width: int
    height: int
    no_sound: bool
    sandbox: bool
    print_units: bool
    log_path: str
    agent_jar: str
    agent_options: str


def make_launch_config(
    *,
    game_dir: str,
    log_path: str,
    agent_jar: str,
    agent_options: str = "",
    max_heap_mb: int = 1000,
    width: int = VERIFIED_WIDTH,
    height: int = VERIFIED_HEIGHT,
    no_sound: bool = True,
    sandbox: bool = False,
    print_units: bool = False,
) -> LaunchConfig:
    """Build a validated :class:`LaunchConfig`.

    Args:
        game_dir: Directory holding the pinned game copy.
        log_path: Absolute path the engine writes its log to.
        agent_jar: Absolute path to the built javaagent.
        agent_options: The agent's ``;``-separated settings, empty for none.
        max_heap_mb: JVM maximum heap in megabytes.
        width: Virtual display width.
        height: Virtual display height.
        no_sound: Whether to pass ``-nosound``.
        sandbox: Whether to boot straight into a skirmish.
        print_units: Whether to dump the unit catalogue and exit.

    Returns:
        A validated configuration.

    Raises:
        DecodeError: When a field is blank, non-positive, or a relative path.
        LaunchConfigError: ``RW-LAUNCH-001`` when ``sandbox`` and
            ``print_units`` are both set. ``-printunits`` exits before the game
            loop, so it can never reach the skirmish ``-sandbox`` asks for;
            requesting both is a contradiction rather than a precedence
            question.
    """
    payload: Mapping[str, str | int | bool] = {
        "game_dir": game_dir,
        "max_heap_mb": max_heap_mb,
        "width": width,
        "height": height,
        "no_sound": no_sound,
        "sandbox": sandbox,
        "print_units": print_units,
        "log_path": log_path,
        "agent_jar": agent_jar,
        "agent_options": agent_options,
    }
    return decode_launch_config(payload)


def decode_launch_config(payload: Mapping[str, str | int | bool]) -> LaunchConfig:
    """Decode an untyped mapping into a :class:`LaunchConfig`.

    Args:
        payload: Untyped mapping carrying every configuration field.

    Returns:
        The validated configuration.

    Raises:
        DecodeError: When any field is absent, mistyped, blank, non-positive, or
            a relative path where an absolute one is required.
        LaunchConfigError: ``RW-LAUNCH-001`` when ``sandbox`` and
            ``print_units`` are both true.
    """
    sandbox = require_bool(payload, "sandbox")
    print_units = require_bool(payload, "print_units")
    if sandbox and print_units:
        raise LaunchConfigError(
            _BOTH_MODES,
            "sandbox and print_units are mutually exclusive: -printunits exits "
            "before the game loop and never reaches the -sandbox skirmish",
        )
    return LaunchConfig(
        game_dir=require_non_empty_str(payload, "game_dir"),
        max_heap_mb=require_positive_int(payload, "max_heap_mb"),
        width=require_positive_int(payload, "width"),
        height=require_positive_int(payload, "height"),
        no_sound=require_bool(payload, "no_sound"),
        sandbox=sandbox,
        print_units=print_units,
        log_path=require_absolute_path(payload, "log_path"),
        agent_jar=require_absolute_path(payload, "agent_jar"),
        # Deliberately not require_non_empty_str: no options is an ordinary
        # launch, and the probes that predate the agent taking any pass none.
        agent_options=require_str(payload, "agent_options"),
    )


def encode_launch_config(config: LaunchConfig) -> dict[str, str | int | bool]:
    """Encode a :class:`LaunchConfig` back to a plain mapping.

    Round-trips with :func:`decode_launch_config`.

    Args:
        config: The configuration to encode.

    Returns:
        A plain mapping suitable for JSON serialisation or run-artifact
        recording.
    """
    return {
        "game_dir": config["game_dir"],
        "max_heap_mb": config["max_heap_mb"],
        "width": config["width"],
        "height": config["height"],
        "no_sound": config["no_sound"],
        "sandbox": config["sandbox"],
        "print_units": config["print_units"],
        "log_path": config["log_path"],
        "agent_jar": config["agent_jar"],
        "agent_options": config["agent_options"],
    }


def build_argv(config: LaunchConfig) -> tuple[str, ...]:
    """Render a configuration as the exact process argument vector.

    The first element is the bundled ``java.exe`` path relative to
    ``game_dir``; the process must be spawned with ``game_dir`` as its working
    directory so ``-Djava.library.path=.`` resolves the shipped native
    libraries.

    ``-javaagent`` is emitted among the JVM options, ahead of ``-cp`` and the
    main class, because the JVM stops parsing its own options at the main class
    name and would otherwise hand the flag to the engine as a game argument.

    **The flag and its options are one element, and that is the point of
    building an argv rather than a command line.** A map path carries spaces --
    ``maps/skirmish/[p2]Lake (2p).tmx`` -- and assembled as a string it split
    the flag in two, so the JVM aborted with ``processing of -javaagent
    failed`` before the agent loaded. A list has no quoting to get wrong.

    Args:
        config: The validated configuration to render.

    Returns:
        The argument vector, ready to spawn.
    """
    agent = f"-javaagent:{config['agent_jar']}"
    if config["agent_options"]:
        agent = f"{agent}={config['agent_options']}"
    argv: list[str] = [
        JAVA_EXE_RELATIVE,
        f"-Xmx{config['max_heap_mb']}M",
        # Reflective access to java.lang.Math's generator, which the agent pins
        # so the opponents' placement repeats. Without it the agent throws at
        # premain rather than leaving a generator silently unseeded
        # ([[policy-determinism]]).
        "--add-opens",
        "java.base/java.lang=ALL-UNNAMED",
        "-Dfile.encoding=UTF-8",
        "-Djava.library.path=.",
        agent,
        "-cp",
        CLASSPATH,
        MAIN_CLASS,
        "-nodisplay",
        "-width",
        str(config["width"]),
        "-height",
        str(config["height"]),
    ]
    if config["no_sound"]:
        argv.append("-nosound")
    if config["sandbox"]:
        argv.append("-sandbox")
    if config["print_units"]:
        argv.append("-printunits")
    argv.extend(["-log", config["log_path"]])
    return tuple(argv)


__all__ = [
    "CLASSPATH",
    "JAVA_EXE_RELATIVE",
    "MAIN_CLASS",
    "VERIFIED_HEIGHT",
    "VERIFIED_WIDTH",
    "LaunchConfig",
    "LaunchConfigError",
    "build_argv",
    "decode_launch_config",
    "encode_launch_config",
    "make_launch_config",
]
