"""Verify that decoded CRSP evidence lies wholly inside one partition's bounds.

The writer runs this before sealing a partition and the evaluator runs it
after decrypting one, so neither relies on the SQL alone. A complete calendar
proves only that the calendar is complete; no table's date extent proves that
any individual security's history is complete.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_str

from .crsp_calendar import crsp_windows
from .crsp_extract import CrspSources
from .crsp_queries import CrspScope


def _within(rows: tuple[JSONObject, ...], field: str, first: str, last: str) -> None:
    """Require every present date in a column to lie in the inclusive interval."""
    for row in rows:
        day = row[field]
        if day is not None and not first <= require_str(row, field) <= last:
            raise ValueError(f"[CRSP-012] {field} lies outside its partition")


def check_crsp_partition(sources: CrspSources, scope: CrspScope, max_rows: int) -> None:
    """Refuse truncated, empty or out-of-partition evidence.

    Args:
        sources: Rows already verified by ``load_crsp_sources``.
        scope: The partition the rows were extracted for.
        max_rows: Per-relation limit; one more row means the query was truncated.

    Raises:
        ValueError: CRSP-012 for a truncated or empty relation, a row dated
            outside the partition or a calendar not exactly spanning it;
            CRSP-002 from ``crsp_windows`` for an incomplete calendar.
    """
    relations = {
        "membership": sources.membership,
        "returns": sources.returns,
        "calendar": sources.calendar,
        "delists": sources.delists,
    }
    for name, rows in relations.items():
        if len(rows) > max_rows:
            raise ValueError(f"[CRSP-012] {name} exceeds its row limit; the query was truncated")
        if name != "delists" and not rows:
            raise ValueError(f"[CRSP-012] {name} is empty")
    end = scope.observation_end
    _within(sources.returns, "dlycaldt", scope.history_start, end)
    _within(sources.membership, "namedt", "0001-01-01", end)
    _within(sources.membership, "nameendt", scope.history_start, end)
    _within(sources.delists, "delistingdt", scope.history_start, end)
    for field in ("deldlydt", "delamtdt", "delnextdt"):
        _within(sources.delists, field, "0001-01-01", end)
    calendar = tuple(
        (require_str(row, "caldt"), require_str(row, "tradingflg")) for row in sources.calendar
    )
    if calendar[0][0] != scope.calendar_start or calendar[-1][0] != scope.calendar_end:
        raise ValueError("[CRSP-012] the calendar does not span exactly its partition")
    crsp_windows(calendar, sources.protocol["train_start"], scope.last_decision)


__all__ = ["check_crsp_partition"]
