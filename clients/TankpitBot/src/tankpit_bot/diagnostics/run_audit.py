"""Run audit CLI: deterministic verdicts over one run's artifacts.

``tankpit-run-audit [events.jsonl]`` -- defaults to
``runs/bot/latest.events.jsonl``; the capture artifact is resolved as
the sibling ``*.capture_session.json``. Wired into ``make analyze`` so
every run is audited by the same code path, with the same thresholds,
producing the same verdicts -- interpretation lives here, not in
whoever happens to read the raw JSONL.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.capture_audit import audit_capture
from tankpit_bot.diagnostics.event_stream import load_event_records, run_analyzer_cli
from tankpit_bot.diagnostics.ledger_audit import audit_ledger
from tankpit_bot.diagnostics.run_audit_types import (
    FindingDict,
    RunAuditReportDict,
    make_finding,
    make_run_audit_report,
    render_run_audit,
)
from tankpit_bot.runtime_logging import RuntimeEventRecordDict
from tankpit_bot.types import decode_capture_session

log = get_logger(__name__)

_EVENTS_SUFFIX = ".events.jsonl"
_CAPTURE_SUFFIX = ".capture_session.json"


def capture_path_for(events_path: Path) -> Path:
    """Return the capture artifact path that pairs with an events path.

    Args:
        events_path: Events JSONL path (``<stem>.events.jsonl``).

    Returns:
        The sibling ``<stem>.capture_session.json`` path. An events
        path without the canonical suffix maps to ``<name>.capture_session.json``
        beside it, which the audit then reports as missing.
    """
    name = events_path.name
    if name.endswith(_EVENTS_SUFFIX):
        return events_path.with_name(name[: -len(_EVENTS_SUFFIX)] + _CAPTURE_SUFFIX)
    return events_path.with_name(name + _CAPTURE_SUFFIX)


def _capture_findings(
    capture_path: Path,
    records: list[RuntimeEventRecordDict],
) -> list[FindingDict]:
    """Load the capture artifact and run the replay audit against it.

    Args:
        capture_path: Capture artifact path.
        records: The run's decoded event records (the ledger side).

    Returns:
        Capture-audit findings, or the ``capture_missing`` finding when
        the artifact is absent.
    """
    if not _test_hooks.path_exists(capture_path):
        return [
            make_finding(
                "capture_missing",
                "warning",
                "no capture artifact beside the events file -- replay audit skipped",
            )
        ]
    capture = decode_capture_session(
        narrow_json_to_dict(load_json_str(_test_hooks.read_text(capture_path)))
    )
    return audit_capture(capture, records)


def build_run_audit(events_path: Path) -> RunAuditReportDict:
    """Audit one run: ledger checks plus capture replay cross-validation.

    Args:
        events_path: Events JSONL artifact to audit.

    Returns:
        The assembled audit report.
    """
    records = load_event_records(events_path)
    capture_path = capture_path_for(events_path)
    findings = audit_ledger(records)
    findings.extend(_capture_findings(capture_path, records))
    return make_run_audit_report(str(events_path), str(capture_path), findings)


def main() -> int:
    """Entry point for the ``tankpit-run-audit`` CLI.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.
    """
    return run_analyzer_cli(build_run_audit, render_run_audit, log)


__all__ = [
    "build_run_audit",
    "capture_path_for",
    "main",
]
