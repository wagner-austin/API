"""Typed findings for the deterministic run audit.

The audit exists so that "what happened in this run" is a computed
verdict, not a per-session interpretation: the same artifacts always
produce the same findings, each carrying the evidence (ids, counts,
timestamps) needed to verify it against the raw JSONL without
trusting the tool. Sessions that used to hand-read the ledger start
from this report instead; anything a session still has to interpret by
hand is a missing check (the ratchet rule, see wiki
[[self-observing-architecture]]).
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

Severity = Literal["critical", "warning", "info"]
"""How loudly a finding should be treated.

``critical`` -- the run misbehaved or an audit invariant broke;
``warning`` -- suspicious or wasteful but the run survived it;
``info`` -- a verified fact worth surfacing (exits, matched channels).
"""

CheckName = Literal[
    "empty_run",
    "kill_double_registration",
    "unresolved_decision",
    "stall_timeout",
    "command_rejection",
    "rejection_retry_loop",
    "executor_discards",
    "superseded_churn",
    "tick_cadence_gap",
    "session_exit",
    "capture_missing",
    "capture_unreadable",
    "decode_error",
    "unknown_container_subtypes",
    "deactivation_channel_diff",
    "supervisor_channel_diff",
    "human_episode",
    "turret_exchange",
    "dom_witness_diff",
]
"""Closed set of audit checks; every finding names the check that produced it."""

_SEVERITY_ORDER: dict[Severity, int] = {"critical": 0, "warning": 1, "info": 2}

_SEVERITY_TAG: dict[Severity, str] = {
    "critical": "CRIT",
    "warning": "WARN",
    "info": "INFO",
}


class FindingDict(TypedDict):
    """One audit verdict with its evidence.

    Attributes:
        check: Which audit check produced the finding.
        severity: How loudly the finding should be treated.
        summary: One-sentence human statement of the verdict.
        evidence: Scalar evidence (ids, counts, timestamps) that lets a
            reader verify the verdict against the raw artifact.
    """

    check: CheckName
    severity: Severity
    summary: str
    evidence: dict[str, str | int]


class RunAuditReportDict(TypedDict):
    """Full audit report over one run's artifacts.

    Attributes:
        events_path: The events JSONL artifact audited.
        capture_path: The sibling capture artifact audited (or the
            missing path, flagged by a ``capture_missing`` finding).
        findings: Verdicts sorted most severe first, then by check name.
        critical_count: Number of critical findings.
        warning_count: Number of warning findings.
        info_count: Number of info findings.
    """

    events_path: str
    capture_path: str
    findings: list[FindingDict]
    critical_count: int
    warning_count: int
    info_count: int


def make_finding(
    check: CheckName,
    severity: Severity,
    summary: str,
    **evidence: str | int,
) -> FindingDict:
    """Build one finding.

    Args:
        check: Which audit check produced the finding.
        severity: How loudly the finding should be treated.
        summary: One-sentence human statement of the verdict.
        **evidence: Scalar evidence keyed by name.

    Returns:
        The assembled finding.
    """
    return FindingDict(
        check=check,
        severity=severity,
        summary=summary,
        evidence=dict(evidence),
    )


def _finding_sort_key(finding: FindingDict) -> tuple[int, str, str]:
    """Sort key: severity first, then check name, then summary.

    Args:
        finding: Finding to key.

    Returns:
        Tuple ordering critical before warning before info.
    """
    return (
        _SEVERITY_ORDER[finding["severity"]],
        finding["check"],
        finding["summary"],
    )


def make_run_audit_report(
    events_path: str,
    capture_path: str,
    findings: list[FindingDict],
) -> RunAuditReportDict:
    """Assemble the report: sort findings and tally severities.

    Args:
        events_path: The events JSONL artifact audited.
        capture_path: The sibling capture artifact audited.
        findings: Verdicts in production order.

    Returns:
        The assembled report with findings sorted most severe first.
    """
    ordered = sorted(findings, key=_finding_sort_key)
    return RunAuditReportDict(
        events_path=events_path,
        capture_path=capture_path,
        findings=ordered,
        critical_count=sum(1 for f in ordered if f["severity"] == "critical"),
        warning_count=sum(1 for f in ordered if f["severity"] == "warning"),
        info_count=sum(1 for f in ordered if f["severity"] == "info"),
    )


def render_run_audit(report: RunAuditReportDict) -> str:
    """Render the audit report as a verdict list.

    Args:
        report: Report to render.

    Returns:
        Multi-line text: header, severity tally, one line per finding.
    """
    lines = [
        "TANKPIT RUN AUDIT",
        f"Events:  {report['events_path']}",
        f"Capture: {report['capture_path']}",
        f"Verdict: {report['critical_count']} critical, "
        f"{report['warning_count']} warnings, {report['info_count']} info",
        "-" * 60,
    ]
    for finding in report["findings"]:
        evidence = " ".join(f"{key}={value}" for key, value in sorted(finding["evidence"].items()))
        tag = _SEVERITY_TAG[finding["severity"]]
        suffix = f"  [{evidence}]" if evidence else ""
        lines.append(f"[{tag}] {finding['check']}: {finding['summary']}{suffix}")
    if not report["findings"]:
        lines.append("(no findings)")
    return "\n".join(lines)


__all__ = [
    "CheckName",
    "FindingDict",
    "RunAuditReportDict",
    "Severity",
    "make_finding",
    "make_run_audit_report",
    "render_run_audit",
]
