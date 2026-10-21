"""Building the agent jar a batch freezes, and certifying it against the game.

Replaces ``scripts/make/agent.ps1`` and ``scripts/make/selftest.ps1``, the two
PowerShell recipes behind ``make agent`` and ``make agent-selftest``, which
compiled with the Windows depot's own JDK. The depot the game runs on now is a
JRE, so both compile with the host's JDK through the one compile the match
launcher already uses (:func:`~rw_bot.harness.play_match.compile_agent`), at
the release the depot's Java 8 loads.

PUBLISHING IS ONE RENAME. The jar is assembled under a stamped name and moved
over :data:`~rw_bot.harness.agent_build.BUILT_AGENT_JAR` in a single rename,
so a game attaching it sees the old jar or the new one and never a
half-written one, and two builds at once cannot interleave class files. The
rename is then read back: a publish that reported success over a jar it never
replaced is exactly what the PowerShell recipe once did, silently, while a
game held the jar open.

THE SELF-TEST COMPILES ITS OWN CLASSES rather than reading the published jar,
so it runs while a game holds that jar, and runs them on the game's bundled
runtime, because the verifier a patch must satisfy is the one the engine
loads it under.

Run as ``python -m rw_bot.harness.agent_cli build`` or
``python -m rw_bot.harness.agent_cli selftest --game-dir .game``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from platform_core.cli_args import parse_single_flags, require_flag

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks
from rw_bot.harness.agent_build import AGENT_BUILD_DIR, BUILT_AGENT_JAR, selftest_command
from rw_bot.harness.play_match import (
    EXIT_AGENT_BUILD_FAILED,
    EXIT_OK,
    TOOL_WALL_SECONDS,
    build_agent,
    compile_agent,
)

_NOT_PUBLISHED = "RW-AGENT-001"
_BAD_COMMAND = "RW-AGENT-002"

#: The self-test ran and at least one check failed. Distinct from a build
#: failure: the code compiled and the game's verifier refused what it wrote.
EXIT_SELFTEST_FAILED = 5

#: The commands this entry point runs.
BUILD_COMMAND = "build"
SELFTEST_COMMAND = "selftest"

#: The one flag the self-test takes: where the pinned game is.
GAME_DIR_FLAG = "--game-dir"


class AgentCommandError(RwBotError):
    """The agent could not be published, or was asked for something unknown.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what went wrong.
    """


def check_published(staged: str) -> None:
    """Raise unless the rename left the published jar and no staged one.

    Args:
        staged: The stamped jar the build wrote.

    Raises:
        AgentCommandError: ``RW-AGENT-001`` when the published jar is absent
            or the staged one survived the rename -- either way the jar a batch
            would freeze is not the one this build produced.
    """
    if not _test_hooks.path_exists(Path(BUILT_AGENT_JAR)) or _test_hooks.path_exists(Path(staged)):
        raise AgentCommandError(
            _NOT_PUBLISHED,
            f"{BUILT_AGENT_JAR} was not replaced by {staged}: the jar a batch freezes is "
            "not the one this build wrote",
        )


def publish_agent() -> int:
    """Build the agent and publish it as the jar batches freeze.

    Returns:
        :data:`~rw_bot.harness.play_match.EXIT_OK` once the jar is published
        and read back, or
        :data:`~rw_bot.harness.play_match.EXIT_AGENT_BUILD_FAILED` when the
        compiler or the packager refused.

    Raises:
        AgentCommandError: ``RW-AGENT-001`` when the published jar does not
            read back as replaced.
        OSError: When a tool cannot be started or the rename is refused.
        ValueError: When the agent source directory holds no Java.
    """
    stamp = _test_hooks.new_stamp()
    classes = f"{AGENT_BUILD_DIR}/classes-{stamp}"
    staged = f"{BUILT_AGENT_JAR}.{stamp}.new"
    try:
        if not build_agent(staged, classes):
            return EXIT_AGENT_BUILD_FAILED
        _test_hooks.replace_path(Path(staged), Path(BUILT_AGENT_JAR))
        check_published(staged)
        _test_hooks.write_line(f"[agent] built {BUILT_AGENT_JAR}")
        return EXIT_OK
    finally:
        _test_hooks.remove_path(Path(classes))
        _test_hooks.remove_path(Path(staged))


def run_selftest(game_dir: str) -> int:
    """Compile the agent fresh and run its self-checks on the pinned game.

    Args:
        game_dir: The pinned game directory, relative to the repository root.

    Returns:
        :data:`~rw_bot.harness.play_match.EXIT_OK` when every check passed,
        :data:`~rw_bot.harness.play_match.EXIT_AGENT_BUILD_FAILED` when the
        sources did not compile, and :data:`EXIT_SELFTEST_FAILED` when the
        game's verifier refused a patch. The self-test's own lines are printed
        either way, because they name the check that failed.

    Raises:
        OSError: When a tool cannot be started.
        ValueError: When the agent source directory holds no Java.
    """
    classes = f"{AGENT_BUILD_DIR}/verify-{_test_hooks.new_stamp()}"
    try:
        if not compile_agent(classes):
            return EXIT_AGENT_BUILD_FAILED
        status, output = _test_hooks.run_capture(
            selftest_command(game_dir, classes), TOOL_WALL_SECONDS
        )
        for line in output:
            _test_hooks.write_line(line)
        if status != 0:
            _test_hooks.write_line(f"[agent-selftest] failed: the self-test exited {status}")
            return EXIT_SELFTEST_FAILED
        _test_hooks.write_line("[agent-selftest] passed")
        return EXIT_OK
    finally:
        _test_hooks.remove_path(Path(classes))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``make agent`` and ``make agent-selftest``.

    Args:
        argv: ``build``, or ``selftest --game-dir DIR``. ``None`` reads the
            process arguments.

    Returns:
        What :func:`publish_agent` or :func:`run_selftest` returned.

    Raises:
        AgentCommandError: ``RW-AGENT-002`` when the command word is missing
            or unknown.
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    command = tokens[0] if tokens else ""
    if command == BUILD_COMMAND:
        parse_single_flags(tokens[1:], ())
        return publish_agent()
    if command == SELFTEST_COMMAND:
        parsed = parse_single_flags(tokens[1:], (GAME_DIR_FLAG,))
        return run_selftest(require_flag(parsed, GAME_DIR_FLAG))
    raise AgentCommandError(
        _BAD_COMMAND,
        f"expected {BUILD_COMMAND!r} or {SELFTEST_COMMAND!r} first, got {command!r}",
    )


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "BUILD_COMMAND",
    "EXIT_SELFTEST_FAILED",
    "GAME_DIR_FLAG",
    "SELFTEST_COMMAND",
    "AgentCommandError",
    "check_published",
    "main",
    "publish_agent",
    "run_selftest",
]
