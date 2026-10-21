"""Decompiling the pinned jar into a readable tree: ``make decompile``.

Replaces the recipe's inline PowerShell. Disassembly (javap) is ground truth
but reads one method at a time, and cross-referencing it by grep has already
produced one false negative here -- the real writer of the tick counter was
missed because the increment compiles against the subclass as a bare field
name. A whole-jar decompile answers "who calls this" and "what else writes
this" directly.

CFR IS PINNED BY VERSION and fetched once into ``.tools``: a decompiler is a
heuristic, and an unpinned one would silently change the source reasoned
about between runs. It runs on the game's own bundled runtime, so nothing
beyond the pinned copy and ``curl`` is needed.

The output is DERIVED and gitignored; anything a wiki page cites is copied
into ``wiki/sources/`` as an excerpt. The tree is removed before each run so
it is this run's.

Run as ``python -m rw_bot.harness.decompile_cli --game-dir .game
[--cfr-version 0.152] [--out .decompiled]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from platform_core.cli_args import parse_single_flags, require_flag

from rw_bot.harness import _test_hooks
from rw_bot.harness.jvm import JAVA_PATH
from rw_bot.harness.play_match import EXIT_OK, TOOL_WALL_SECONDS

#: A tool refused: the fetch, CFR itself, or the count.
EXIT_DECOMPILE_FAILED = 8

#: Where pinned tools are kept, relative to the repository root.
TOOLS_DIR = ".tools"

#: Wall on CFR over the whole jar. A decompile of a few thousand classes takes
#: minutes; an hour is far past that and still finite.
DECOMPILE_WALL_SECONDS = 3600.0

#: How CFR is told to write a quiet, comment-free tree.
CFR_OPTIONS = ("--comments", "false", "--silent", "true")

REQUIRED_FLAGS = ("--game-dir",)
OPTIONAL_FLAGS = {"--cfr-version": "0.152", "--out": ".decompiled"}


def cfr_jar(version: str) -> str:
    """Return where a CFR version is kept.

    Args:
        version: The CFR release, e.g. ``0.152``.

    Returns:
        The jar's path, relative to the repository root.
    """
    return f"{TOOLS_DIR}/cfr-{version}.jar"


def cfr_url(version: str) -> str:
    """Return where a CFR version is published.

    Args:
        version: The CFR release.

    Returns:
        Its Maven Central URL.
    """
    return f"https://repo1.maven.org/maven2/org/benf/cfr/{version}/cfr-{version}.jar"


def _ran(argv: tuple[str, ...], wall: float, what: str) -> tuple[str, ...] | None:
    """Run one tool, printing its output and naming it when it fails.

    Args:
        argv: The command.
        wall: Its wall-clock bound.
        what: What the tool was doing, for the failure line.

    Returns:
        Its output lines, or None when it exited non-zero.

    Raises:
        OSError: When the tool cannot be started.
    """
    status, output = _test_hooks.run_capture(argv, wall)
    if status == 0:
        return output
    for line in output:
        _test_hooks.write_line(line)
    _test_hooks.write_line(f"[decompile] {what} failed: {argv[0]} exited {status}")
    return None


def decompile(game_dir: str, version: str, out: str) -> int:
    """Decompile the pinned jar with a pinned CFR into a fresh tree.

    Args:
        game_dir: The pinned game directory, relative to the root.
        version: The CFR release to use.
        out: The tree to write, relative to the root; removed first.

    Returns:
        :data:`~rw_bot.harness.play_match.EXIT_OK`, or
        :data:`EXIT_DECOMPILE_FAILED` naming the tool that refused.

    Raises:
        OSError: When a tool cannot be started or the old tree removed.
    """
    jar = cfr_jar(version)
    if not _test_hooks.path_exists(Path(jar)):
        _test_hooks.write_line(f"  fetching {cfr_url(version)}")
        fetch = ("curl", "-fsSL", "--create-dirs", "-o", jar, cfr_url(version))
        if _ran(fetch, TOOL_WALL_SECONDS, "fetching CFR") is None:
            return EXIT_DECOMPILE_FAILED
    _test_hooks.remove_path(Path(out))
    java = f"{game_dir}/{JAVA_PATH}"
    command = (java, "-Xmx2g", "-jar", jar, f"{game_dir}/game-lib.jar", "--outputdir", out)
    if _ran((*command, *CFR_OPTIONS), DECOMPILE_WALL_SECONDS, "CFR") is None:
        return EXIT_DECOMPILE_FAILED
    sources = _ran(("find", out, "-type", "f", "-name", "*.java"), TOOL_WALL_SECONDS, "counting")
    if sources is None:
        return EXIT_DECOMPILE_FAILED
    _test_hooks.write_line(f"[decompile] {len(sources)} source files in {out}")
    _test_hooks.write_line("  Reminder: decompiled Java is a RECONSTRUCTION. Pin load-bearing")
    _test_hooks.write_line("  claims to javap bytecode or observed runtime behaviour.")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``make decompile``.

    Args:
        argv: Flags, as :data:`REQUIRED_FLAGS` and :data:`OPTIONAL_FLAGS`.
            ``None`` reads the process arguments.

    Returns:
        What :func:`decompile` returned.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
        OSError: As :func:`decompile`.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    parsed = parse_single_flags(tokens, (*REQUIRED_FLAGS, *OPTIONAL_FLAGS))
    values = {**OPTIONAL_FLAGS, **parsed}
    return decompile(require_flag(parsed, "--game-dir"), values["--cfr-version"], values["--out"])


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "CFR_OPTIONS",
    "DECOMPILE_WALL_SECONDS",
    "EXIT_DECOMPILE_FAILED",
    "OPTIONAL_FLAGS",
    "REQUIRED_FLAGS",
    "TOOLS_DIR",
    "cfr_jar",
    "cfr_url",
    "decompile",
    "main",
]
