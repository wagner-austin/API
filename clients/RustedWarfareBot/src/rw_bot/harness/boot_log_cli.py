"""Command-line summary of an engine boot log.

Exposed as the ``rw-boot-log`` console script. Reads one archived engine log
and prints its build identity, subsystem count, recovered class mappings,
loaded maps, and any crashes.

Exit status is meaningful so the command can gate a probe run:

* ``0`` — the log parsed and recorded no crash.
* ``1`` — the log parsed but recorded at least one crash.
* ``2`` — the command was invoked with the wrong number of arguments.

A log that cannot be decoded raises
:class:`~rw_bot.harness.boot_log.BootLogError` rather than mapping to an exit
code. A malformed engine log means either the run died before writing its
header or the decoder has drifted from the engine's format; both need a stack
trace, not a quiet status code.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rw_bot.harness import _test_hooks
from rw_bot.harness.boot_log import BootLog, parse_boot_log

EXIT_OK = 0
EXIT_CRASHED = 1
EXIT_BAD_USAGE = 2

_USAGE = "usage: rw-boot-log <path-to-engine-log>"


def render_summary(log: BootLog, source: Path) -> tuple[str, ...]:
    """Render a parsed log as human-readable lines.

    Args:
        log: The parsed log.
        source: Path the log was read from, echoed in the header.

    Returns:
        The summary lines, without trailing newlines.
    """
    version = log["version"]
    lines: list[str] = [
        f"{source}",
        f"  build          {version['version']} (code {version['game_code']}, "
        f"build {version['build_number']})",
        f"  subsystems     {len(log['subsystems'])}",
        f"  class mappings {len(log['class_mappings'])}",
        f"  maps           {len(log['maps'])}",
        f"  crashes        {len(log['crashes'])}",
    ]
    for mapping in log["class_mappings"]:
        lines.append(
            f"    class L{mapping['line_number']}  "
            f"{mapping['subsystem']} -> {mapping['java_class']}"
        )
    for loaded in log["maps"]:
        lines.append(f"    map   L{loaded['line_number']}  {loaded['map_file']}")
    for crash in log["crashes"]:
        lines.append(
            f"    CRASH L{crash['line_number']}  {crash['exception_type']} at {crash['top_frame']}"
        )
    return tuple(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``rw-boot-log`` console script.

    Args:
        argv: Argument list excluding the program name. ``None`` reads
            ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when the log parsed with no crashes, ``EXIT_CRASHED`` when
        it parsed but recorded a crash, ``EXIT_BAD_USAGE`` when the argument
        count is wrong.

    Raises:
        BootLogError: When the log exists but cannot be decoded.
        OSError: When the log cannot be read.
    """
    args = list(argv) if argv is not None else _test_hooks.read_argv()
    if len(args) != 1:
        _test_hooks.write_line(_USAGE)
        return EXIT_BAD_USAGE

    source = Path(args[0])
    log = parse_boot_log(_test_hooks.read_text_lines(source))
    for line in render_summary(log, source):
        _test_hooks.write_line(line)
    return EXIT_CRASHED if log["crashes"] else EXIT_OK


__all__ = [
    "EXIT_BAD_USAGE",
    "EXIT_CRASHED",
    "EXIT_OK",
    "main",
    "render_summary",
]
