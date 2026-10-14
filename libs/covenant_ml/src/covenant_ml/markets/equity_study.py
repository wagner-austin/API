"""Immutable acquisition design bound to an independently frozen stock snapshot."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list,
)
from typing_extensions import ReadOnly, TypedDict

from .equity_outcomes import evaluate_equity_window
from .equity_signals import decode_strength_signal, rank_financial_strength
from .reporting import read_verified_market_report
from .validation import parse_timestamp, require_day, require_nonempty


class EquityStudy(TypedDict):
    """Exact calendar and paper weights derived from a verified snapshot."""

    snapshot_sha256: ReadOnly[str]
    decision_at: ReadOnly[str]
    cohort: ReadOnly[tuple[str, ...]]
    selected: ReadOnly[tuple[str, ...]]
    expected_days: ReadOnly[tuple[str, ...]]
    costs: ReadOnly[tuple[float, ...]]


def encode_equity_study(record: EquityStudy) -> JSONObject:
    """Encode the immutable acquisition contract.

    Args:
        record: Validated study.

    Returns:
        JSON identity, dates, membership and cost assumptions.
    """
    return {
        "snapshot_sha256": record["snapshot_sha256"],
        "decision_at": record["decision_at"],
        "cohort": list(record["cohort"]),
        "selected": list(record["selected"]),
        "expected_days": list(record["expected_days"]),
        "costs": list(record["costs"]),
    }


def decode_equity_study(document: JSONObject) -> EquityStudy:
    """Validate a complete design before performing network operations.

    Args:
        document: Untrusted study JSON.

    Returns:
        Immutable study with exact ordered membership and calendar.

    Raises:
        ValueError: EQUITY-016 for invalid digest or cost sequence; outcome
            calendar and membership errors propagate.
        TypeError: Required JSON fields have incorrect types.
    """
    digest = require_nonempty(document, "snapshot_sha256")
    costs = tuple(
        require_float({"cost": value}, "cost") for value in require_list(document, "costs")
    )
    if (
        len(digest) != 64
        or any(value not in "0123456789abcdef" for value in digest)
        or not costs
        or len(set(costs)) != len(costs)
        or any(not isfinite(cost) or not 0 <= cost <= 1 for cost in costs)
    ):
        raise ValueError("[EQUITY-016] require SHA256 identity and unique finite cost assumptions")
    cohort = tuple(
        require_nonempty({"symbol": value}, "symbol") for value in require_list(document, "cohort")
    )
    selected = tuple(
        require_nonempty({"symbol": value}, "symbol")
        for value in require_list(document, "selected")
    )
    days = tuple(
        require_day({"day": value}, "day") for value in require_list(document, "expected_days")
    )
    decision = require_nonempty(document, "decision_at")
    evaluate_equity_window(
        {},
        cohort=cohort,
        selected=selected,
        expected_days=days,
        decision_at=decision,
        as_of=decision,
        round_trip_cost=costs[0],
    )
    if parse_timestamp(decision) >= parse_timestamp(days[0] + "T23:59:59Z"):
        raise ValueError("[EQUITY-016] decision must precede the entry day's end")
    return EquityStudy(
        snapshot_sha256=digest,
        decision_at=decision,
        cohort=cohort,
        selected=selected,
        expected_days=days,
        costs=costs,
    )


def load_equity_study(snapshot: Path, calendar: JSONObject) -> tuple[EquityStudy, tuple[Path, ...]]:
    """Verify provenance, ranking, weights and the explicitly declared calendar.

    Args:
        snapshot: Complete frozen report with its original RunRecord and sources.
        calendar: Required snapshot_sha256 and expected_days, declared independently
            of observed prices. Holidays must be excluded by the study designer.

    Returns:
        Strict study and every verified snapshot evidence path.

    Raises:
        ValueError: EQUITY-017 for changed snapshot semantics or identity.
        OSError: Required original evidence is unavailable.
        TypeError: Required fields or row values are malformed.
    """
    report, sources, digest = read_verified_market_report(snapshot, "EQUITY-017")
    rows = tuple(narrow_json_to_dict(value) for value in require_list(report, "rows"))
    signals = tuple(decode_strength_signal(require_dict(row, "signal")) for row in rows)
    ranked = rank_financial_strength(signals)
    count = require_int(report, "selected_count")
    decision = require_nonempty(report, "decision_at")
    completed = require_nonempty(report, "completed_at")
    if (
        require_nonempty(calendar, "snapshot_sha256") != digest
        or require_int(report, "cohort_count") != len(rows)
        or not 1 <= count <= len(rows)
        or parse_timestamp(completed) >= parse_timestamp(decision)
        or require_int(report, "orders_submitted") != 0
        or require_bool(report, "advance_to_funded_trading")
    ):
        raise ValueError("[EQUITY-017] snapshot identity, chronology or paper cohort is invalid")
    for index, (row, (signal, liquidity, profit)) in enumerate(zip(rows, ranked, strict=True)):
        if (
            decode_strength_signal(require_dict(row, "signal")) != signal
            or require_int(row, "liquidity_rank") != liquidity
            or require_int(row, "profitability_rank") != profit
            or require_int(row, "score") != liquidity + profit
            or require_bool(row, "selected") != (index < count)
            or require_float(row, "portfolio_weight") != (1 / count if index < count else 0.0)
            or require_float(row, "benchmark_weight") != 1 / len(rows)
        ):
            raise ValueError("[EQUITY-017] ranking, selection or equal weights changed")
    study = decode_equity_study(
        {
            "snapshot_sha256": digest,
            "decision_at": decision,
            "cohort": [signal["symbol"] for signal in signals],
            "selected": [signal["symbol"] for signal in signals[:count]],
            "expected_days": require_list(calendar, "expected_days"),
            "costs": [
                require_float(report, "round_trip_cost"),
                require_float(report, "stress_round_trip_cost"),
            ],
        }
    )
    if (
        require_day(report, "entry_date") != study["expected_days"][0]
        or require_day(report, "exit_date") != study["expected_days"][-1]
        or require_int(report, "holding_bars") != len(study["expected_days"]) - 1
    ):
        raise ValueError("[EQUITY-017] declared calendar differs from frozen endpoints or horizon")
    return study, sources
