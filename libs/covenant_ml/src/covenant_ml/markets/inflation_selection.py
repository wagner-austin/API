"""Native inflation contracts selected under the frozen chronological protocol."""

from __future__ import annotations

import re
from calendar import month_name
from datetime import date
from decimal import Decimal

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_float,
    require_list,
    require_str,
)

from .inflation_model import earlier_errors, inflation_probability, rounded_inflation
from .inflation_records import InflationMonth, forecast_before
from .kalshi import MarketSpec, decode_spec
from .types import DOLLAR
from .validation import parse_fixed, parse_timestamp

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def inflation_target(event: str) -> str:
    """Validate a native inflation event identity and recover its target month.

    Args:
        event: Native event ticker from the requested inflation series.

    Returns:
        Canonical target year and month.

    Raises:
        ValueError: INFLATION-011 for unsupported event identity.
    """
    if re.fullmatch(r"(?:KX)?CPIYOY-[0-9]{2}[A-Z]{3}", event) is None:
        raise ValueError("[INFLATION-011] expected native yearly inflation event")
    code = event.rsplit("-", 1)[1]
    if code[2:] not in _MONTHS:
        raise ValueError("[INFLATION-011] event has invalid target month")
    return date(2000 + int(code[:2]), _MONTHS.index(code[2:]) + 1, 1).isoformat()[:7]


def inflation_inventory(documents: tuple[JSONObject, ...]) -> tuple[JSONObject, ...]:
    """Join exhausted inventories with explicit overlap consistency checks.

    Args:
        documents: Captured ordinary and historical terminal inventory pages.

    Returns:
        Unique native contracts ordered by ticker.

    Raises:
        ValueError: INFLATION-012 for nonterminal pages, identity or overlap conflicts.
        TypeError: Native required fields have incorrect types.
        KeyError: An overlapping record lacks a compared required field.
    """
    records: dict[str, JSONObject] = {}
    fields = (
        "event_ticker",
        "rules_primary",
        "floor_strike",
        "strike_type",
        "result",
        "close_time",
    )
    for document in documents:
        if require_str(document, "cursor") != "":
            raise ValueError("[INFLATION-012] inventory capture must exhaust pagination")
        page: set[str] = set()
        for value in require_list(document, "markets"):
            native = narrow_json_to_dict(value)
            ticker, event = require_str(native, "ticker"), require_str(native, "event_ticker")
            inflation_target(event)
            if ticker in page or not ticker.startswith(event + "-T"):
                raise ValueError("[INFLATION-012] duplicate or mismatched native ticker")
            page.add(ticker)
            if ticker in records:
                if any(native[field] != records[ticker][field] for field in fields):
                    raise ValueError("[INFLATION-012] overlapping inventories disagree")
            else:
                records[ticker] = native
    return tuple(records[ticker] for ticker in sorted(records))


def _contract_settlement(
    native: JSONObject, target: str, actual_value: float
) -> tuple[tuple[float, int] | None, str]:
    """Verify native rounded-threshold rules and first-release settlement agreement.

    Args:
        native: Validated native market identity.
        target: Canonical target month.
        actual_value: Official first-release inflation value.

    Returns:
        Strike and binary payout, or an explicit exclusion reason.

    Raises:
        ValueError: INFLATION-013 for inconsistent identity, strike or binary payout;
            malformed native values propagate their validation errors.
        TypeError: Native fields have incorrect types.
    """
    event, ticker = require_str(native, "event_ticker"), require_str(native, "ticker")
    strike = require_float(native, "floor_strike")
    boundary = rounded_inflation(strike)
    if boundary != Decimal(str(strike)) or ticker != event + "-T" + format(boundary, ".1f"):
        raise ValueError("[INFLATION-013] ticker and one-decimal strike must agree")
    target_date = date.fromisoformat(target + "-01")
    expected_rule = (
        f"If the Consumer Price Index (CPI) increases by more than {strike:.1f}% in the twelve "
        f"months ending {month_name[target_date.month]} {target_date.year} (as represented by "
        "the one-decimal place value reported by the Bureau of Labor Statistics), then the "
        "market resolves to Yes."
    )
    if (
        require_str(native, "strike_type") != "greater"
        or require_str(native, "rules_primary") != expected_rule
    ):
        return None, "unsupported_rules"
    result = require_str(native, "result")
    if require_str(native, "status") != "finalized" or result not in ("yes", "no"):
        return None, "not_binary_final"
    payout = parse_fixed(require_str(native, "settlement_value_dollars"), 4)
    if payout != (DOLLAR if result == "yes" else 0):
        raise ValueError("[INFLATION-013] binary result and payout conflict")
    if (rounded_inflation(actual_value) > boundary) != (result == "yes"):
        return None, "release_settlement_mismatch"
    return (strike, payout), "eligible"


def _selected_contract(
    native: JSONObject, month: InflationMonth, history: tuple[InflationMonth, ...]
) -> tuple[tuple[MarketSpec, float] | None, str]:
    """Validate release timing before deriving a strictly earlier prediction.

    Args:
        native: Validated native market identity.
        month: Matching validated forecast history.
        history: Complete histories filtered to earlier releases before fitting.

    Returns:
        Specification and probability pair, or an explicit exclusion reason.

    Raises:
        ValueError: Native settlement and chronology validation failures propagate.
        TypeError: Required native fields have incorrect types.
    """
    actual = month["actual"]
    if actual is None:
        return None, "missing_release"
    settlement, reason = _contract_settlement(native, month["month"], actual["value"])
    if settlement is None:
        return None, reason
    strike, payout = settlement
    day = actual["day"]
    decision = parse_timestamp(day + "T00:00:00Z")
    close_text = require_str(native, "close_time")
    close = parse_timestamp(close_text)
    if close_text[:10] != day or not decision < close:
        return None, "release_close_mismatch"
    if parse_timestamp(require_str(native, "open_time")) > decision:
        return None, "not_open_at_cutoff"
    forecast = forecast_before(month, day, 4)
    if forecast is None:
        return None, "missing_forecast"
    errors = earlier_errors(history, month["month"], day)
    if len(errors) < 36:
        return None, "insufficient_prior_errors"
    ticker = require_str(native, "ticker")
    event = require_str(native, "event_ticker")
    spec = decode_spec(
        {
            "ticker": ticker,
            "series": event.split("-", 1)[0],
            "day": day,
            "decision_ms": decision,
            "close_ms": close,
            "settled_ms": parse_timestamp(require_str(native, "settlement_ts")),
            "settlement": payout,
        }
    )
    return (spec, inflation_probability(forecast["value"], errors, strike)), "eligible"


def inflation_selection(
    records: tuple[JSONObject, ...], history: tuple[InflationMonth, ...]
) -> tuple[tuple[tuple[MarketSpec, float], ...], list[JSONValue]]:
    """Account for every native contract under the frozen evaluation dates.

    Args:
        records: Unique native inventories.
        history: Validated chronological forecast history.

    Returns:
        Eligible specification/probability pairs and complete exclusion accounting.

    Raises:
        ValueError: Native identity or observation validation failures propagate.
        TypeError: Required native fields have incorrect types.
    """
    months = {month["month"]: month for month in history}
    selected: list[tuple[MarketSpec, float]] = []
    coverage: list[JSONValue] = []
    for native in records:
        ticker, event = require_str(native, "ticker"), require_str(native, "event_ticker")
        target = inflation_target(event)
        if not "2024-01" <= target <= "2026-08":
            reason = "outside_evaluation"
        elif target not in months:
            reason = "missing_target_history"
        else:
            row, reason = _selected_contract(native, months[target], history)
            if row is not None:
                selected.append(row)
        coverage.append({"ticker": ticker, "event": event, "target": target, "status": reason})
    return tuple(selected), coverage
