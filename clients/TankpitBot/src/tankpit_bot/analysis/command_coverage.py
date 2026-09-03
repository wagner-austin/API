"""Does the sim handle every command a real client actually sends?

The response-shape differ asks whether the sim ANSWERS correctly. This
asks the prior question: does the sim survive the command at all.

``SimServer.queue_command`` refuses any kind outside
``SUPPORTED_KINDS``, and every client frame reaches it through
``route_client_payload``, so a byte the decoder does not map takes the
server down the moment a real client sends it. Three such bytes were
found on 2026-09-03 — the keep-alive at 11,871 archived sends, the
enter-game at 343, the inventory request at 4 — and all three were
found by hand-written one-off sweeps.

They share a cause worth stating plainly: **our bot is the only client
that does not send them.** A sim validated against our own bot cannot
see a command our bot never emits, however long it soaks. That is a
property of the corpus, not of the sim, and no amount of sim testing
fixes it. Reading the archive does.

The session walk is not re-implemented — :mod:`analysis.scan` owns it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from tankpit_bot.analysis.command_coverage_types import (
    STATUS_CRASHES,
    STATUS_DECLARED_UNMODELLED,
    STATUS_HANDLED,
    CommandByteRowDict,
    CommandCoverageDict,
)
from tankpit_bot.analysis.scan import scan_archive
from tankpit_bot.protocol import commands as vocabulary
from tankpit_bot.protocol.commands import CMD_UNMODELLED_COMBAT, COMMAND_PREFIX
from tankpit_bot.sim.commands import decode_client_command
from tankpit_bot.sim.server import SUPPORTED_KINDS
from tankpit_bot.wire.helpers import DecodeError


def _constant_names() -> dict[int, str]:
    """Map every ``CMD_*`` constant's value to its name.

    Returns:
        Command byte to constant name. Read from the module rather
        than from a second hand-written table, so a constant added to
        the protocol cannot be missing here.
    """
    names: dict[int, str] = {}
    for name in dir(vocabulary):
        if not name.startswith("CMD_"):
            continue
        # Annotated AT ASSIGNMENT so the declared type overrides the
        # ``Any`` ``getattr`` returns — the documented pattern for
        # reading a dynamically-named attribute under these strict
        # rules ([[coding-standards]]).
        value: int = getattr(vocabulary, name)
        names[value] = name
    return names


def _status(byte: int, kind: str) -> str:
    """Classify one command byte against what the sim can do with it.

    Args:
        byte: The command byte.
        kind: The kind the decoder resolved it to.

    Returns:
        One of the ``STATUS_*`` values.
    """
    if byte == CMD_UNMODELLED_COMBAT:
        return STATUS_DECLARED_UNMODELLED
    if kind in SUPPORTED_KINDS:
        return STATUS_HANDLED
    return STATUS_CRASHES


def analyze_command_coverage(directories: list[Path]) -> CommandCoverageDict:
    """Mine every client command in an archive and classify each byte.

    Args:
        directories: Directories holding ``*.capture_session.json``.

    Returns:
        The coverage audit.

    Raises:
        OSError: If a session file cannot be read.
        InvalidJsonError: If a session file is not valid JSON.
        JSONTypeError: If a session file is not a capture session.
    """
    sessions = 0
    sends: Counter[int] = Counter()
    kinds: dict[int, str] = {}
    for directory in directories:
        for result in scan_archive(directory):
            if result["kind"] != "scanned":
                continue
            sessions += 1
            for frame in result["frames"]:
                if frame["direction"] != "sent" or frame["msg_type"] != COMMAND_PREFIX:
                    continue
                try:
                    command = decode_client_command(frame["body"])
                except DecodeError:
                    # A frame the decoder cannot read at all is archive
                    # noise, not a command byte to classify; the
                    # response-shape differ counts those separately.
                    continue
                sends[command["command"]] += 1
                kinds[command["command"]] = command["kind"]

    names = _constant_names()
    rows = [
        CommandByteRowDict(
            byte=byte,
            constant=names.get(byte, ""),
            kind=kinds[byte],
            sends=count,
            status=_status(byte, kinds[byte]),
        )
        for byte, count in sends.most_common()
    ]
    return CommandCoverageDict(
        sessions=sessions,
        rows=rows,
        unsent_constants=sorted(name for byte, name in names.items() if byte not in sends),
    )


def crashing_rows(coverage: CommandCoverageDict) -> list[CommandByteRowDict]:
    """The rows that would take a hosted server down.

    Args:
        coverage: The audit to read.

    Returns:
        Every row whose byte the sim does not map, descending by sends.
    """
    return [row for row in coverage["rows"] if row["status"] == STATUS_CRASHES]


def format_command_coverage(coverage: CommandCoverageDict) -> str:
    """Format the audit as a readable report.

    Args:
        coverage: The audit to format.

    Returns:
        Multi-line human-readable summary.
    """
    lines = [f"sessions={coverage['sessions']} distinct_command_bytes={len(coverage['rows'])}", ""]
    lines.append(f"{'byte':>6}  {'sends':>7}  {'constant':<24} {'kind':<18} status")
    lines.append("-" * 78)
    for row in coverage["rows"]:
        name = row["constant"] or "-- NO CONSTANT --"
        lines.append(
            f"  0x{row['byte']:02X}  {row['sends']:>7}  {name:<24} "
            f"{row['kind']:<18} {row['status']}"
        )
    crashing = crashing_rows(coverage)
    lines.append("")
    if crashing:
        lines.append(f"WOULD CRASH A HOSTED SERVER: {len(crashing)}")
        for row in crashing:
            lines.append(f"  0x{row['byte']:02X} ({row['sends']} sends) decodes to {row['kind']!r}")
    else:
        lines.append("Every command byte in this archive is handled or declared unmodelled.")
    if coverage["unsent_constants"]:
        lines.append("")
        lines.append("Defined but never sent in this archive:")
        lines.append("  " + ", ".join(coverage["unsent_constants"]))
    return "\n".join(lines)


__all__ = [
    "analyze_command_coverage",
    "crashing_rows",
    "format_command_coverage",
]
