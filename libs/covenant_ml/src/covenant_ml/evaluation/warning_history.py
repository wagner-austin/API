"""Shared first-incident evidence consistency for learning and alert evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from .warning_panel import WarningRow


def require_warning_history(rows: Sequence[WarningRow]) -> dict[str, int]:
    """Require one coherent first-breach history per borrower across supplied rows.

    Args:
        rows: Decoder-validated opportunities, including reused fold evidence.
            Repeated identical observations are allowed here; partition validators
            separately reject duplicate opportunities within a partition.

    Returns:
        dict[str, int]: Known first-breach timestamps indexed by borrower.

    Raises:
        ValueError: WARN-024 for conflicting first-breach dates or event-free
            surveillance extending through a known breach. Conflicting revisions
            require source adjudication before fitting, not automatic selection.
    """
    incidents: dict[str, int] = {}
    for row in rows:
        borrower = row["disclosure"]["borrower"]
        breach = row["followup"]["first_breach_ms"]
        if breach is not None:
            if borrower in incidents and incidents[borrower] != breach:
                raise ValueError("[WARN-024] a borrower has conflicting first-breach dates")
            incidents[borrower] = breach
    for row in rows:
        breach = incidents.get(row["disclosure"]["borrower"])
        if (
            breach is not None
            and row["followup"]["first_breach_ms"] is None
            and row["followup"]["observed_through_ms"] >= breach
        ):
            raise ValueError("[WARN-024] event-free surveillance contradicts a known breach")
    return incidents


__all__ = ["require_warning_history"]
