"""Tests for the run-audit finding types, report assembly, and renderer."""

from __future__ import annotations

from tankpit_bot.diagnostics.run_audit_types import (
    CheckName,
    FindingDict,
    Severity,
    make_finding,
    make_run_audit_report,
    render_run_audit,
)


def test_make_finding_builds_full_dict() -> None:
    """make_finding assembles the check, severity, summary, and evidence."""
    finding = make_finding(
        CheckName.STALL_TIMEOUT,
        Severity.CRITICAL,
        "collect hit the stall timeout",
        action_kind="collect",
        timestamp="2026-07-19T00:48:31",
    )
    assert finding == FindingDict(
        check=CheckName.STALL_TIMEOUT,
        severity=Severity.CRITICAL,
        summary="collect hit the stall timeout",
        evidence={"action_kind": "collect", "timestamp": "2026-07-19T00:48:31"},
    )


def test_report_sorts_by_severity_then_check() -> None:
    """Findings order critical -> warning -> info regardless of input order."""
    info = make_finding(CheckName.SESSION_EXIT, Severity.INFO, "session ended: completed")
    warn = make_finding(CheckName.CAPTURE_MISSING, Severity.WARNING, "no capture artifact")
    crit = make_finding(CheckName.STALL_TIMEOUT, Severity.CRITICAL, "scan stalled")
    report = make_run_audit_report("events.jsonl", "capture.json", [info, crit, warn])
    assert [f["severity"] for f in report["findings"]] == [
        Severity.CRITICAL,
        Severity.WARNING,
        Severity.INFO,
    ]
    assert report["critical_count"] == 1
    assert report["warning_count"] == 1
    assert report["info_count"] == 1
    assert report["events_path"] == "events.jsonl"
    assert report["capture_path"] == "capture.json"


def test_render_lists_findings_with_evidence() -> None:
    """The renderer prints the verdict tally and one tagged line per finding."""
    report = make_run_audit_report(
        "events.jsonl",
        "capture.json",
        [
            make_finding(
                CheckName.STALL_TIMEOUT, Severity.CRITICAL, "scan stalled", action_kind="scan"
            ),
            make_finding(CheckName.SESSION_EXIT, Severity.INFO, "session ended: completed"),
        ],
    )
    rendered = render_run_audit(report)
    assert "TANKPIT RUN AUDIT" in rendered
    assert "Verdict: 1 critical, 0 warnings, 1 info" in rendered
    assert "[CRIT] stall_timeout: scan stalled  [action_kind=scan]" in rendered
    assert "[INFO] session_exit: session ended: completed" in rendered
    # The evidence-free finding gets no trailing bracket block.
    assert "session ended: completed  [" not in rendered


def test_render_empty_report_says_no_findings() -> None:
    """An empty findings list renders the explicit no-findings marker."""
    rendered = render_run_audit(make_run_audit_report("e", "c", []))
    assert "(no findings)" in rendered
    assert "Verdict: 0 critical, 0 warnings, 0 info" in rendered
