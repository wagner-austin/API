"""Prospective public evidence capture and outcome-free frozen-model predictions."""

from __future__ import annotations

from datetime import date, timedelta

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)

from .archive import ResponseArchive
from .collection import MarketCollector
from .forecast_inputs import bracket_probability, forecast_maximum, strike_bounds
from .forecast_model import MARKET_FEATURES, WEATHER_FEATURES, decode_fitted_logit, predict_logit
from .forecast_records import (
    CITY_COORDINATES,
    decode_comparison_plan,
    decode_forecast_features,
    decode_forecast_row,
    encode_forecast_features,
)
from .paper_records import PaperPlan, decode_native_book, encode_book_top, encode_paper_plan
from .validation import parse_timestamp, require_nonempty


class PaperReader:
    """Public archive reader enforcing real receipt times inside a frozen window."""

    def __init__(self, archive: ResponseArchive, plan: PaperPlan, capture: bool) -> None:
        """Bind the archive, explicit acquisition mode and accepted receipt window.

        Args:
            archive: Existing exact-response archive service.
            plan: Validated acquisition constraints.
            capture: True for fresh capture, False for strictly offline replay.
        """
        self.archive = archive
        self.capture = capture
        self.start = parse_timestamp(plan["window_start"])
        self.end = parse_timestamp(plan["window_end"])
        self.received_ms = self.start

    def read(self, path: str, params: dict[str, str | int]) -> JSONObject:
        """Acquire or replay a response and validate its original receipt timestamp.

        Args:
            path: Public API path.
            params: Canonical request arguments.

        Returns:
            Verified response object; received_ms identifies this response's receipt.

        Raises:
            ValueError: PAPER-004 for late, early or regressing receipt times.
            HTTPStatusError: Public request failed; errors propagate without recovery.
            OSError: Archive read or exclusive write failed.
        """
        if self.capture:
            self.archive.capture(path, params)
        envelope = self.archive.load_capture(path, params)
        received = parse_timestamp(envelope["captured_at"])
        if not self.received_ms <= received <= self.end:
            raise ValueError("[PAPER-004] response outside acquisition window or clock regressed")
        self.received_ms = received
        return narrow_json_to_dict(load_json_str(envelope["body"]))


def collect_paper_predictions(
    plan: PaperPlan,
    model_report: JSONObject,
    market: PaperReader,
    forecast: PaperReader,
) -> JSONObject:
    """Capture all event contracts, one explicit forecast run and visible book depth.

    Args:
        plan: Frozen single-city prospective specification.
        model_report: Hash-verified historical report containing both saved calibrators.
        market: Public exchange reader with enforced receipt times.
        forecast: Explicit-run forecast reader with enforced receipt times.

    Returns:
        Deterministic pre-outcome predictions, exclusions and acquisition evidence.

    Raises:
        ValueError: PAPER-005 for incompatible or unavailable training evidence;
            PAPER-006 for invalid live contracts; PAPER-007 for stale inputs.
        HTTPStatusError: A required public request failed.
        TypeError: Required native or model fields are malformed.
    """
    historical = decode_comparison_plan(require_dict(model_report, "plan"))
    models = require_dict(model_report, "models")
    baseline = decode_fitted_logit(require_dict(models, "market_calibrated"))
    augmented = decode_fitted_logit(require_dict(models, "market_plus_forecast"))
    training = tuple(
        decode_forecast_row(narrow_json_to_dict(row))
        for row in require_list(model_report, "training_rows")
    )
    if (
        historical["forecast_hour"] != plan["forecast_hour"]
        or historical["end"] >= plan["event_day"]
        or not training
        or any(row["settled_ms"] >= market.start for row in training)
        or baseline["features"] != MARKET_FEATURES
        or augmented["features"] != WEATHER_FEATURES
    ):
        raise ValueError(
            "[PAPER-005] models must use the declared cycle and already-settled training"
        )
    day = date.fromisoformat(plan["event_day"])
    months = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    event = f"{plan['series']}-{day.year % 100:02d}{months[day.month - 1]}{day.day:02d}"
    collector = MarketCollector(market.read)
    collector.verify_fee(plan["series"])
    native = tuple(collector.pages("/markets", {"event_ticker": event}, "markets"))
    tickers = tuple(require_nonempty(row, "ticker") for row in native)
    if not native or len(set(tickers)) != len(tickers):
        raise ValueError("[PAPER-006] event inventory must be nonempty and unique")
    latitude, longitude = CITY_COORDINATES[plan["series"]]
    initialized = (day - timedelta(days=1)).isoformat() + f"T{plan['forecast_hour']:02d}:00"
    document = forecast.read(
        "/v1/forecast",
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "models": "gfs_global",
            "run": initialized,
            "forecast_days": 3,
        },
    )
    maximum = forecast_maximum(document, day)
    predictions: list[JSONValue] = []
    excluded: list[JSONValue] = []
    for row in native:
        ticker = require_nonempty(row, "ticker")
        if (
            require_str(row, "event_ticker") != event
            or not ticker.startswith(event + "-")
            or require_str(row, "status") != "active"
            or require_str(row, "result") != ""
            or parse_timestamp(require_str(row, "open_time")) > market.received_ms
            or parse_timestamp(require_str(row, "close_time")) <= market.end
        ):
            raise ValueError("[PAPER-006] contract is not the requested active unresolved event")
        rules = require_nonempty(row, "rules_primary")
        bounds = strike_bounds(row)
        if bounds is None:
            excluded.append({"ticker": ticker, "reason": "unsupported_strike"})
            continue
        book = decode_native_book(market.read("/markets/" + ticker + "/orderbook", {"depth": 0}))
        if not 0 <= market.received_ms - forecast.received_ms <= 600000:
            raise ValueError(
                "[PAPER-007] forecast receipt must precede book by at most ten minutes"
            )
        if book is None:
            excluded.append({"ticker": ticker, "reason": "missing_book_side"})
            continue
        inputs = decode_forecast_features(
            {
                "series": plan["series"],
                "bid": book["bid"],
                "ask": book["ask"],
                "weather_p": bracket_probability(maximum, bounds),
            }
        )
        predictions.append(
            {
                "ticker": ticker,
                "inputs": encode_forecast_features(inputs),
                "book": encode_book_top(book),
                "book_received_ms": market.received_ms,
                "rules_primary": rules,
                "raw_market": (book["bid"] + book["ask"]) / 20000,
                "raw_forecast": inputs["weather_p"],
                "market_calibrated": predict_logit(baseline, (inputs,))[0],
                "market_plus_forecast": predict_logit(augmented, (inputs,))[0],
            }
        )
    return {
        "schema_version": 1,
        "plan": encode_paper_plan(plan),
        "forecast_init_ms": parse_timestamp(initialized + "Z"),
        "forecast_received_ms": forecast.received_ms,
        "forecast_f": maximum,
        "last_received_ms": max(market.received_ms, forecast.received_ms),
        "models": models,
        "predictions": predictions,
        "exclusions": excluded,
        "selected_contracts": len(native),
        "orders_submitted": 0,
        "interpretation": (
            "Prospective probabilities and visible size; no fills or returns measured."
        ),
    }


__all__ = ["PaperReader", "collect_paper_predictions"]
