"""Shared loading helpers for JSONL runtime-event artifacts.

Both diagnostic analyzers (:mod:`tankpit_bot.diagnostics.issue_report`
and :mod:`tankpit_bot.diagnostics.self_map`) consume the same artifact
format: one :class:`tankpit_bot.runtime_records.RuntimeEventRecordDict`
per line, written by the runtime logging handlers during ``make bot`` /
``make <name>-probe`` runs. This module owns the load-and-decode step
plus the CLI source-path resolution so the analyzers never fork it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.logging import stdlib_logging
from platform_core.rich_logging import setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.runtime_records import (
    RuntimeEventRecordDict,
    decode_runtime_event_record,
)

_DEFAULT_BOT_ARTIFACT: Path = Path("runs") / "bot" / "latest.events.jsonl"

_ReportT = TypeVar("_ReportT")


def decode_event_lines(lines: list[str]) -> list[RuntimeEventRecordDict]:
    """Decode JSONL event lines, skipping blank ones.

    The one place event text becomes event records, shared by the
    whole-file loader and the incremental tail reader
    (:mod:`tankpit_bot.diagnostics.event_tail`) so the two can never
    disagree about what a line means.

    Args:
        lines: Raw JSONL lines in file order.

    Returns:
        Decoded :class:`RuntimeEventRecordDict` rows in the same order.

    Raises:
        JSONTypeError: When any non-blank line fails strict event
            decoding; malformed artifacts are surfaced instead of
            silently dropped.
    """
    records: list[RuntimeEventRecordDict] = []
    for line in lines:
        if not line.strip():
            continue
        records.append(decode_runtime_event_record(narrow_json_to_dict(load_json_str(line))))
    return records


def load_event_records(source_path: Path) -> list[RuntimeEventRecordDict]:
    """Load and decode every event line from a JSONL artifact.

    Args:
        source_path: JSONL events path to read.

    Returns:
        Decoded :class:`RuntimeEventRecordDict` rows in file order.

    Raises:
        JSONTypeError: When any line fails strict event decoding;
            malformed artifacts are surfaced instead of silently dropped.
    """
    return decode_event_lines(_test_hooks.read_text(source_path).splitlines())


def scan_diagnostic_records(
    records: list[RuntimeEventRecordDict],
    diagnostic_kind: str,
) -> tuple[str, list[RuntimeEventRecordDict]]:
    """Return the artifact's runtime mode and all records of one kind.

    Args:
        records: Decoded event records in file order.
        diagnostic_kind: ``diagnostic_kind`` value to collect.

    Returns:
        ``(mode, matches)`` where ``mode`` is the latest non-empty mode
        string observed across ALL records (``"unconfigured"`` when none
        carries one) and ``matches`` is every ``DIAGNOSTIC`` record
        whose ``diagnostic_kind`` equals ``diagnostic_kind``.
    """
    mode = "unconfigured"
    matches: list[RuntimeEventRecordDict] = []
    for record in records:
        if record["mode"]:
            mode = record["mode"]
        if record["channel"] != "DIAGNOSTIC":
            continue
        if record["fields"].get("diagnostic_kind") != diagnostic_kind:
            continue
        matches.append(record)
    return (mode, matches)


def run_analyzer_cli(
    build: Callable[[Path], _ReportT],
    render: Callable[[_ReportT], str],
    log: stdlib_logging.Logger,
) -> int:
    """Run the shared analyzer CLI flow: resolve path, build, render, print.

    Every diagnostics analyzer entrypoint (`tankpit-issue-report`,
    `tankpit-self-map`, `tankpit-entity-map`) shares this exact flow;
    centralizing it here keeps the CLIs from forking argv handling or
    output conventions.

    Args:
        build: Report builder taking the resolved artifact path.
        render: Renderer producing the human-readable report string.
        log: Module logger of the calling entrypoint, so console output
            is attributed to the concrete analyzer.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.

    Raises:
        FileNotFoundError: When the resolved artifact path is absent.
    """
    setup_rich_logging(level="INFO")
    full_argv = list(_test_hooks.get_argv())
    user_args = full_argv[1:] if full_argv else []
    source = resolve_source_path(user_args)
    report = build(source)
    log.info("%s", render(report))
    return 0


def resolve_source_path(argv: list[str]) -> Path:
    """Return the JSONL path an analyzer CLI should read.

    Args:
        argv: Process argv with the entry-point name already removed.

    Returns:
        Concrete artifact path the user wants analyzed; defaults to
        ``runs/bot/latest.events.jsonl`` when no argument is supplied.

    Raises:
        FileNotFoundError: When the resolved path is not present on disk.
            Failing fast keeps reports from rendering an empty result
            when the artifact is missing.
    """
    source = Path(argv[0]) if argv else _DEFAULT_BOT_ARTIFACT
    if not _test_hooks.path_exists(source):
        raise FileNotFoundError(
            f"events artifact not found: {source}. "
            "Pass the path as the first argument, or run `make bot` / "
            "`make <name>-probe` first."
        )
    return source


__all__ = [
    "decode_event_lines",
    "load_event_records",
    "resolve_source_path",
    "run_analyzer_cli",
    "scan_diagnostic_records",
]
