"""Compact per-run digest: the 40-line truth table for one session.

Born 2026-08-05 after a night of misread logs (kills double-counted by
tailing two mirrored files; a crashed run diagnosed by freestyle grep
over 193 MB of JSONL). The digest distills one events artifact into a
small pre-computed table — kills, deaths, shots, displacement
histogram, clearance-shot conversions, release reasons, account rank,
inventory arc, activity timeline — so a reader consumes computed
counts instead of re-deriving them ad hoc. It works from the events
stream alone, so a crashed run with no teardown scorecard still gets a
digest.

This module is now just the file-shaped entry points; the arithmetic
lives in :mod:`tankpit_bot.diagnostics.run_digest_fold`, which folds
records one at a time so a live-run reader can resume instead of
re-reading (2026-09-01).

CLI: ``tankpit-run-digest [events.jsonl]`` (defaults to the latest bot
artifact) prints the table and writes ``<stem>.digest.json`` beside
the source.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from tankpit_bot.diagnostics.event_stream import load_event_records, run_analyzer_cli
from tankpit_bot.diagnostics.run_digest_fold import RunDigestAccumulator
from tankpit_bot.diagnostics.run_digest_render import render_run_digest
from tankpit_bot.diagnostics.run_digest_types import RunDigestDict

log = get_logger(__name__)


def build_run_digest(source_path: Path) -> RunDigestDict:
    """Distill one events artifact into the digest table.

    Args:
        source_path: JSONL events path.

    Returns:
        The computed digest.

    Raises:
        ValueError: If the artifact holds no events.
    """
    records = load_event_records(source_path)
    if not records:
        raise ValueError(f"no events in {source_path}")
    accumulator = RunDigestAccumulator(str(source_path))
    accumulator.absorb(records)
    return accumulator.snapshot()


def build_and_persist_run_digest(source_path: Path) -> RunDigestDict:
    """Build the digest and persist its JSON beside the source artifact.

    The persisted ``<stem>.digest.json`` is the machine-readable twin
    of the rendered table, so later sessions read computed counts
    instead of re-grepping the raw events.

    Args:
        source_path: JSONL events path.

    Returns:
        The computed digest.
    """
    digest = build_run_digest(source_path)
    out_path = source_path.with_suffix("").with_suffix(".digest.json")
    out_path.write_text(dump_json_str(dict(digest), indent=1), encoding="utf-8")
    log.info("digest written: %s", out_path)
    return digest


def main() -> int:
    """Run the ``tankpit-run-digest`` CLI entrypoint.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.
    """
    return run_analyzer_cli(build_and_persist_run_digest, render_run_digest, log)


__all__ = [
    "build_and_persist_run_digest",
    "build_run_digest",
    "main",
]
