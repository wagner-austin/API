"""Scoring a runner audit's transcript: one ``CHECK`` line per expected check, in order.

Lifted from :func:`fleet.core.runner_audit.parse_audit_transcript` on
2026-10-20 (MCPs board task 1e9ab738), when loki's Windows audit came to
print the same lines; that audit left with loki (board task e705f08c), and
the bash audit is scored here.

A transcript is ``CHECK <id> OK`` or ``CHECK <id> DRIFT <detail>`` lines,
blank lines aside, whose ids are exactly the expected ones in order. A
transcript that stops early is a script that died midway, and the checks it
never reached must not read as passed.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import AppError, FleetErrorCode

from fleet.core.runner_audit_types import AuditFinding, ExpectedCheck


def score_transcript(
    host_name: str, expected: Sequence[ExpectedCheck], output: str
) -> list[AuditFinding]:
    """Validate a transcript against the checks it owes and score it.

    Args:
        host_name: The audited host, named in a refusal.
        expected: Every check the script promised, in order.
        output: The script's standard output.

    Returns:
        One finding per expected check, in declaration order.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` when any line is not a CHECK
            line, a DRIFT line carries no detail, or the sequence of check
            ids is not exactly the expected one.
    """
    rows: list[tuple[str, bool, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split(" ", 3)
        if len(parts) < 3 or parts[0] != "CHECK" or parts[2] not in ("OK", "DRIFT"):
            raise _unparsable(host_name, f"line is not a CHECK line: {line!r}")
        check_id, verdict = parts[1], parts[2]
        detail = parts[3] if len(parts) == 4 else ""
        if verdict == "DRIFT" and not detail.strip():
            raise _unparsable(host_name, f"DRIFT line for {check_id} carries no detail")
        rows.append((check_id, verdict == "OK", detail))
    reported_ids = [row[0] for row in rows]
    expected_ids = [check["check_id"] for check in expected]
    if reported_ids != expected_ids:
        raise _unparsable(
            host_name,
            f"expected checks {expected_ids} but the transcript reported {reported_ids}; "
            "a transcript that stops early is a script that died midway",
        )
    return [
        AuditFinding(check_id=check_id, ok=ok, detail=detail, reason=check["reason"])
        for (check_id, ok, detail), check in zip(rows, expected, strict=True)
    ]


def _unparsable(host_name: str, detail: str) -> AppError[FleetErrorCode]:
    """The error for a transcript that cannot be scored.

    Args:
        host_name: The host whose transcript failed.
        detail: What was wrong with it.

    Returns:
        The error to raise.
    """
    return AppError(
        FleetErrorCode.RUNNER_AUDIT_UNPARSABLE,
        f"the audit transcript from {host_name} cannot be scored: {detail}",
    )


__all__ = ["score_transcript"]
