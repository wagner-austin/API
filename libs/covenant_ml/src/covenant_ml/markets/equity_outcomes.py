"""Exact-calendar outcomes retain every member of a frozen equity cohort."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from itertools import pairwise
from math import fsum, isfinite

from platform_core.json_utils import require_float

from .equity_history import adjusted_forward_return
from .equity_outcome_records import EquityWindowResult, decode_equity_window_result
from .equity_records import EquityPrice
from .validation import parse_timestamp, require_day, require_nonempty


def _require_window(
    cohort: tuple[str, ...],
    selected: tuple[str, ...],
    expected_days: tuple[str, ...],
    decision_at: str,
    as_of: str,
    round_trip_cost: float,
) -> tuple[float, bool]:
    """Validate the frozen design before deciding whether its calendar has elapsed.

    Args:
        cohort: Required unique ticker sequence.
        selected: Required unique subset of the cohort.
        expected_days: Ordered exact session dates.
        decision_at: Frozen aware decision timestamp.
        as_of: Actual aware evaluation timestamp.
        round_trip_cost: Fractional assumed cost in the closed interval zero to one.

    Returns:
        Validated cost and whether the exit calendar date has elapsed.

    Raises:
        ValueError: EQUITY-014 for invalid design or decision chronology.
        TypeError: Incorrect cost type; shared field validators also propagate.
    """
    cost = require_float({"cost": round_trip_cost}, "cost")
    if (
        not cohort
        or len(set(cohort)) != len(cohort)
        or not selected
        or len(set(selected)) != len(selected)
        or not set(selected).issubset(cohort)
        or not isfinite(cost)
        or not 0 <= cost <= 1
    ):
        raise ValueError("[EQUITY-014] require unique cohort, nonempty subset and valid cost")
    for symbol in cohort:
        require_nonempty({"symbol": symbol}, "symbol")
    for day in expected_days:
        require_day({"day": day}, "day")
    if len(expected_days) < 2 or any(left >= right for left, right in pairwise(expected_days)):
        raise ValueError("[EQUITY-014] declared calendar must contain ordered unique endpoints")
    decision, evaluated = parse_timestamp(decision_at), parse_timestamp(as_of)
    exit_midnight = parse_timestamp(expected_days[-1] + "T00:00:00Z")
    if decision >= exit_midnight + 86400000:
        raise ValueError("[EQUITY-014] the frozen decision must precede the end of the window")
    return cost, evaluated >= exit_midnight + 86400000


def evaluate_equity_window(
    histories: Mapping[str, tuple[EquityPrice, ...]],
    *,
    cohort: tuple[str, ...],
    selected: tuple[str, ...],
    expected_days: tuple[str, ...],
    decision_at: str,
    as_of: str,
    round_trip_cost: float,
) -> tuple[tuple[tuple[str, float], ...], EquityWindowResult] | None:
    """Score only an entire declared daily window, never surviving endpoints.

    Args:
        histories: Native-decoded completed bars for every cohort ticker.
            The caller must verify each source receipt is no later than as_of.
        cohort: Frozen nonempty unique ticker sequence.
        selected: Frozen nonempty unique subset receiving equal paper weights.
        expected_days: Exact ordered calendar dates, including entry and exit.
            This is the declared calendar, not a calendar inferred from prices.
        decision_at: Frozen aware timestamp strictly before the entry session.
        as_of: Actual aware evaluation timestamp, including before the decision.
        round_trip_cost: Declared nonnegative fractional cost, at most one.

    Returns:
        Every ticker's adjusted return in cohort order plus portfolio arithmetic.
        None until the UTC calendar day following the declared exit. This
        pending state proves no completed outcome; thereafter missing evidence
        fails. Native decoding separately requires completed-bar receipt times.

    Raises:
        ValueError: EQUITY-014 for invalid cohort, selection, calendar, cost or
            evaluation chronology; EQUITY-015 for incomplete or mismatched
            evidence, or an entry that does not follow the decision.
    """
    cost, ready = _require_window(
        cohort, selected, expected_days, decision_at, as_of, round_trip_cost
    )
    if not ready:
        return None
    if set(histories) != set(cohort):
        raise ValueError("[EQUITY-015] observed history keys must equal the entire frozen cohort")
    returns: list[tuple[str, float]] = []
    for symbol in cohort:
        history = histories[symbol]
        if any(row["symbol"] != symbol or row["currency"] != "USD" for row in history):
            raise ValueError("[EQUITY-015] history identity or currency differs from cohort")
        dated = tuple(
            (datetime.fromtimestamp(row["session_start_s"], UTC).date().isoformat(), row)
            for row in history
        )
        window = tuple(
            (day, row) for day, row in dated if expected_days[0] <= day <= expected_days[-1]
        )
        if tuple(day for day, _ in window) != expected_days:
            raise ValueError("[EQUITY-015] entry, exit or intervening declared dates are missing")
        outcome = adjusted_forward_return(
            tuple(row for _, row in window),
            available_at=decision_at,
            holding_bars=len(expected_days) - 1,
        )
        if outcome is None:
            raise ValueError("[EQUITY-015] the exact entry session must follow the frozen decision")
        returns.append((symbol, outcome[2]))
    gross_selected = fsum(value / len(selected) for symbol, value in returns if symbol in selected)
    gross_benchmark = fsum(value / len(cohort) for _, value in returns)
    net_selected = gross_selected - cost
    result = decode_equity_window_result(
        {
            "selected_return": gross_selected,
            "benchmark_return": gross_benchmark,
            "round_trip_cost": cost,
            "net_selected_return": net_selected,
            "net_excess_return": net_selected - gross_benchmark,
        }
    )
    return tuple(returns), result
