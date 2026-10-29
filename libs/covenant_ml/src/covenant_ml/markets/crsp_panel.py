"""Decision-time CRSP universe and momentum, independent of later outcomes."""

from __future__ import annotations

from math import isfinite, prod
from statistics import median
from typing import NamedTuple

from platform_core.json_utils import JSONObject, require_float, require_int, require_str

from .crsp_calendar import CrspWindow, crsp_member_on, crsp_windows
from .crsp_extract import CrspSources
from .crsp_protocol import CrspProtocol


class CrspFeature(NamedTuple):
    """A security's lagged eleven-month momentum and median dollar volume."""

    permno: int
    momentum: float
    liquidity: float


class CrspDecision(NamedTuple):
    """Eligible universe in liquidity order and mutually exclusive exclusion counts."""

    window: CrspWindow
    features: tuple[CrspFeature, ...]
    exclusions: tuple[tuple[str, int], ...]


def _month(day: str) -> int:
    """Map a canonical ISO date to a consecutive month number."""
    return int(day[:4]) * 12 + int(day[5:7]) - 1


def _ordinary(row: JSONObject, previous: str) -> bool:
    """Accept a one-session ordinary return only with independent predecessor evidence."""
    return (
        row["dlyret"] is not None
        and row["dlyprevdt"] == previous
        and row["dlyretmissflg"] == "NA"
        and row["dlydelflg"] == "N"
        and row["dlyretdurflg"] in ("D1", "D2", "D3", "D4", "DU")
    )


def _history_failure(
    history: dict[str, JSONObject], sessions: tuple[str, ...], required: set[str]
) -> str | None:
    """Check every needed return against the independent preceding session."""
    for index, day in enumerate(sessions):
        if day not in required:
            continue
        observed = history.get(day)
        if observed is None:
            return "daily_history_missing"
        if not _ordinary(observed, sessions[index - 1]):
            return "daily_history_invalid"
    return None


def _feature(
    permno: int,
    history: dict[str, JSONObject],
    sessions: tuple[str, ...],
    protocol: CrspProtocol,
) -> CrspFeature | str:
    """Compute one past-only feature row, or its first counted exclusion reason."""
    decision = sessions[-1]
    row = history.get(decision)
    if row is None or row["dlyprc"] is None:
        return "decision_price_missing"
    if abs(require_float(row, "dlyprc")) < protocol["min_price"]:
        return "decision_price_below_minimum"
    # For entry month t, t-12 through t-2 are the eleven months before
    # the decision month. The decision month's entire return is skipped.
    lower, upper = _month(decision) - 11, _month(decision) - 1
    momentum_days = tuple(day for day in sessions if lower <= _month(day) <= upper)
    if len(sessions) <= protocol["min_history"] or _month(sessions[0]) >= lower:
        return "calendar_history_insufficient"
    required = set(sessions[-protocol["min_history"] :]) | set(momentum_days)
    failure = _history_failure(history, sessions, required)
    if failure is not None:
        return failure
    dollars: list[float] = []
    for day in sessions[-protocol["liquidity_window"] :]:
        observed = history[day]
        if observed["dlyprc"] is None or observed["dlyvol"] is None:
            return "liquidity_missing"
        dollars.append(abs(require_float(observed, "dlyprc")) * require_float(observed, "dlyvol"))
    momentum = prod(1 + require_float(history[day], "dlyret") for day in momentum_days) - 1
    liquidity = median(dollars)
    if not isfinite(momentum) or any(not isfinite(value) for value in dollars):
        raise ValueError("[CRSP-008] feature arithmetic overflowed finite source values")
    return CrspFeature(permno, momentum, liquidity)


def _decision(
    window: CrspWindow,
    membership: dict[int, tuple[JSONObject, ...]],
    histories: dict[int, dict[str, JSONObject]],
    sessions: tuple[str, ...],
    protocol: CrspProtocol,
) -> CrspDecision:
    """Select by information available at the decision, never by fill availability."""
    counts: dict[str, int] = {}
    features: list[CrspFeature] = []
    past = tuple(day for day in sessions if day <= window.decision)
    for permno, rows in membership.items():
        if all(require_str(row, "namedt") > window.decision for row in rows):
            continue
        result: CrspFeature | str
        if not crsp_member_on(rows, permno, window.decision, protocol):
            result = "membership_ineligible"
        else:
            result = _feature(permno, histories.get(permno, {}), past, protocol)
        if isinstance(result, str):
            counts[result] = counts.get(result, 0) + 1
        else:
            features.append(result)
    ordered = sorted(features, key=lambda row: (-row.liquidity, row.permno))
    excess = len(ordered) - protocol["max_universe"]
    if excess > 0:
        counts["liquidity_rank_limit"] = excess
    return CrspDecision(
        window, tuple(ordered[: protocol["max_universe"]]), tuple(sorted(counts.items()))
    )


def build_crsp_panel(sources: CrspSources, first: str, last: str) -> tuple[CrspDecision, ...]:
    """Construct monthly decisions from verified sources and a complete CIZ calendar.

    A zero-sized eligible universe is preserved for the evaluator to refuse;
    it is never silently removed from the calendar. This function computes no
    labels, portfolio returns or final-period scores and grants no holdout access.

    Args:
        sources: Output of load_crsp_sources, including the frozen design.
        first: Inclusive first decision bound.
        last: Inclusive final decision bound.

    Returns:
        Decisions with liquidity-ranked features and one exclusion per rejected
        historical security. Ties always use ascending permanent identifier.

    Raises:
        ValueError: CRSP-008 for bounds outside the study, or non-session returns.
        ValueError: Calendar, membership and source validation errors propagate.
    """
    protocol = sources.protocol
    if first < protocol["train_start"] or last > protocol["holdout_end"]:
        raise ValueError("[CRSP-008] decision bounds lie outside the frozen study")
    calendar = tuple(
        (require_str(row, "caldt"), require_str(row, "tradingflg")) for row in sources.calendar
    )
    windows = crsp_windows(calendar, first, last)
    sessions = tuple(day for day, flag in calendar if flag == "Y")
    session_set = set(sessions)
    histories: dict[int, dict[str, JSONObject]] = {}
    for row in sources.returns:
        day = require_str(row, "dlycaldt")
        if day not in session_set:
            raise ValueError("[CRSP-008] return observation is outside declared sessions")
        histories.setdefault(require_int(row, "permno"), {})[day] = row
    memberships: dict[int, list[JSONObject]] = {}
    for row in sources.membership:
        memberships.setdefault(require_int(row, "permno"), []).append(row)
    grouped = {permno: tuple(rows) for permno, rows in memberships.items()}
    return tuple(_decision(window, grouped, histories, sessions, protocol) for window in windows)
