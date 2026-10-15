"""Validate the fixed comparison plan and join quotes to a complete native inventory."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_int,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)

from .codecs import decode_quote
from .interval_quotes import _contracts, require_interval_plan
from .interval_records import IntervalRow, decode_interval_row
from .validation import parse_timestamp


def require_interval_comparison_plan(plan: JSONObject, quotes: JSONObject) -> None:
    """Refuse drift from the single declared development experiment.

    Args:
        plan: Persisted declaration made before scoring.
        quotes: Verified quote report and its acquisition plan.

    Raises:
        ValueError: IFORE-001 for undeclared settings or an invalid split boundary.
        TypeError: Required fields are missing or malformed.
    """
    window = require_dict(quotes, "plan")
    for field in ("schema_version", "fee_coefficient", "edge_hurdle", "serial_lags"):
        require_int(plan, field)
    require_float(plan, "inverse_regularization")
    for value in require_list(plan, "extra_friction"):
        narrow_json_to_int(value)
    expected: JSONObject = {
        "schema_version": 1,
        "study_kind": "exploratory_chronological_development",
        "fee_coefficient": 700,
        "edge_hurdle": 200,
        "extra_friction": [0, 100, 200],
        "serial_lags": 7,
        "inverse_regularization": 1.0,
        "models": ["raw_market", "earlier_asset_base_rate", "logistic_recalibration"],
        "features": ["prior_market_logit", "ether_indicator"],
    }
    if (
        any(plan.get(key) != value for key, value in expected.items())
        or not (
            require_int(window, "start_ts")
            < require_int(plan, "train_end_ts")
            < require_int(window, "end_ts")
        )
        or (
            require_int(window, "decision_offset_seconds"),
            require_int(window, "min_quote_age_seconds"),
            require_int(window, "max_quote_age_seconds"),
        )
        != (300, 1, 120)
    ):
        raise ValueError("[IFORE-001] comparison differs from the declared fixed experiment")
    parse_timestamp(require_str(plan, "declared_at"))


def interval_observations(
    cohort: JSONObject,
    quotes: JSONObject,
) -> tuple[tuple[IntervalRow, ...], tuple[JSONObject, ...]]:
    """Join complete identities and preserve every explicit exclusion reason.

    Args:
        cohort: Source-verified native contract inventory.
        quotes: Source-verified prior quote report.

    Returns:
        Valid binary observations and excluded identities with reasons.

    Raises:
        ValueError: IFORE-002 for identity, timestamp or quote inconsistencies.
        TypeError: Required native evidence is malformed.
    """
    quote_plan = require_dict(quotes, "plan")
    require_interval_plan(quote_plan, cohort)
    markets = _contracts(cohort, quote_plan)
    documents = [narrow_json_to_dict(value) for value in require_list(quotes, "contracts")]
    attached = {require_str(row, "ticker"): row for row in documents}
    if len(attached) != len(documents) or set(attached) != {
        require_str(row, "ticker") for row in markets
    }:
        raise ValueError("[IFORE-002] quote identities must exactly match the complete cohort")
    rows: list[IntervalRow] = []
    excluded: list[JSONObject] = []
    for market in markets:
        ticker = require_str(market, "ticker")
        attachment = attached[ticker]
        if any(attachment[key] != market[key] for key in ("series", "event_ticker", "decision_ms")):
            raise ValueError("[IFORE-002] quote metadata disagrees with the native contract")
        status = require_str(attachment, "status")
        if status != "two_sided":
            excluded.append({"ticker": ticker, "reason": "quote:" + status})
            continue
        quote = decode_quote(require_dict(attachment, "quote"))
        bid, ask = quote["bid"], quote["ask"]
        age = require_int(market, "decision_ms") - quote["timestamp_ms"]
        if (
            bid is None
            or ask is None
            or not (
                require_int(quote_plan, "min_quote_age_seconds") * 1000
                <= age
                <= require_int(quote_plan, "max_quote_age_seconds") * 1000
            )
            or age != require_int(attachment, "quote_age_ms")
        ):
            raise ValueError("[IFORE-002] two-sided quote must have valid sides and prior age")
        if bid == 0 or ask == 10000:
            excluded.append({"ticker": ticker, "reason": "boundary_quote"})
            continue
        result = require_str(market, "result")
        if result not in ("yes", "no"):
            excluded.append({"ticker": ticker, "reason": "nonbinary_or_unresolved_label"})
            continue
        values: JSONObject = {
            "ticker": ticker,
            "series": require_str(market, "series"),
            "close_ms": parse_timestamp(require_str(market, "close_time")),
            "decision_ms": require_int(market, "decision_ms"),
            "settled_ms": parse_timestamp(require_str(market, "settlement_ts")),
            "bid": bid,
            "ask": ask,
            "yes": int(result == "yes"),
        }
        rows.append(decode_interval_row(values))
    return tuple(rows), tuple(excluded)


def split_interval_rows(
    rows: tuple[IntervalRow, ...],
    boundary: int,
) -> tuple[tuple[IntervalRow, ...], tuple[IntervalRow, ...], list[JSONValue]]:
    """Partition whole close dates and purge labels unavailable at the fixed cutoff.

    Args:
        rows: Validated unique comparison observations.
        boundary: Exclusive training close-time boundary, Unix seconds at midnight.

    Returns:
        Earlier fitting rows, later evaluation rows and every purged identity.

    Raises:
        ValueError: IFORE-003 for duplicate identities, non-midnight or empty partitions.
    """
    if boundary % 86400 or len({row["ticker"] for row in rows}) != len(rows):
        raise ValueError("[IFORE-003] require unique rows and a midnight partition")
    test = tuple(row for row in rows if row["close_ms"] >= boundary * 1000)
    if not test:
        raise ValueError("[IFORE-003] later evaluation partition is empty")
    # The earliest nominal close is the boundary even when that slot is absent.
    # Its open-plus-five-minute decision is ten minutes before the boundary.
    first_decision = boundary * 1000 - 600000
    train = tuple(
        row
        for row in rows
        if row["close_ms"] < boundary * 1000 and row["settled_ms"] < first_decision
    )
    purged: list[JSONValue] = [
        row["ticker"]
        for row in rows
        if row["close_ms"] < boundary * 1000 and row["settled_ms"] >= first_decision
    ]
    if not train:
        raise ValueError("[IFORE-003] no earlier labels were available at first evaluation")
    return train, test, purged
