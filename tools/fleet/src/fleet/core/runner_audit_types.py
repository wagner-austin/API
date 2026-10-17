"""The runner audit's records: what a check promises, what it found, and an audit's outcome.

Their own module because the audit (:mod:`fleet.core.runner_audit`) and the
rows it gathers beyond its script (:mod:`fleet.core.runner_dropped_check`)
both build them, and the audit imports those rows.
"""

from __future__ import annotations

from typing_extensions import TypedDict


class ExpectedCheck(TypedDict):
    """One check the audit script promises to report.

    Attributes:
        check_id: Stable identifier, derived from the roster entry so two
            audits of one host agree line for line.
        reason: Why the item matters -- printed beside a drift line so the
            reader learns what breaks without opening the roster.
    """

    check_id: str
    reason: str


class AuditFinding(TypedDict):
    """One check's verdict.

    Attributes:
        check_id: The check that reported.
        ok: Whether the host satisfies it.
        detail: What the host actually said, for the drift line.
        reason: The roster's reason for the check.
    """

    check_id: str
    ok: bool
    detail: str
    reason: str


class AuditOutcome(TypedDict):
    """What auditing a host produced, or why it produced nothing.

    Same shape as :class:`fleet.core.probe.ProbeOutcome` and for the same
    caller: an audit over several hosts must report an unreachable one as a
    line beside the hosts that answered, not as a crash that hides them.

    Attributes:
        findings: Every check's verdict, or ``None`` when the host could not
            be audited at all.
        reason: Why there are no findings. Empty when there are.
    """

    findings: list[AuditFinding] | None
    reason: str


__all__ = ["AuditFinding", "AuditOutcome", "ExpectedCheck"]
