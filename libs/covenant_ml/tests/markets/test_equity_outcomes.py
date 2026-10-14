"""Actual price archives prove whole-cohort accounting and exact-date failures."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.json_utils import JSONObject

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.equity_history import decode_equity_chart
from covenant_ml.markets.equity_outcome_records import EquityWindowResult
from covenant_ml.markets.equity_outcomes import evaluate_equity_window
from covenant_ml.markets.equity_records import EquityPrice, decode_equity_price, encode_equity_price
from covenant_ml.markets.validation import read_document

from ._server import ApiServer, api_server

__all__ = ["api_server"]

DAYS = ("2020-08-28", "2020-08-31")
COHORT = ("AAPL", "MSFT")
DECISION = "2020-08-28T00:00:00Z"
AS_OF = "2026-10-15T00:00:00Z"


def captured(symbol: str) -> JSONObject:
    """Read native source evidence for an explicit historical test symbol."""
    names = {"AAPL": "equity_apple_2020.json", "MSFT": "equity_msft_2020_window.json"}
    return read_document(Path(__file__).parent / "fixtures" / names[symbol])


def histories() -> dict[str, tuple[EquityPrice, ...]]:
    """Decode both actual histories using the production boundary."""
    return {
        symbol: decode_equity_chart(captured(symbol), symbol, "USD", received_at=AS_OF)
        for symbol in COHORT
    }


def test_complete_actual_portfolio_and_cost_sensitivity() -> None:
    apple = 125.05754089355469 / 120.9556655883789 - 1
    microsoft = 214.47921752929688 / 217.6935577392578 - 1
    benchmark = apple / 2 + microsoft / 2
    for cost in (0.002, 0.005):
        expected: EquityWindowResult = {
            "selected_return": apple,
            "benchmark_return": benchmark,
            "round_trip_cost": cost,
            "net_selected_return": apple - cost,
            "net_excess_return": apple - cost - benchmark,
        }
        assert evaluate_equity_window(
            histories(),
            cohort=COHORT,
            selected=("AAPL",),
            expected_days=DAYS,
            decision_at=DECISION,
            as_of=AS_OF,
            round_trip_cost=cost,
        ) == ((("AAPL", apple), ("MSFT", microsoft)), expected)


def test_prospective_window_remains_pending_without_prices() -> None:
    for observed in ("2026-10-14T06:00:00Z", "2026-11-11T23:59:59Z"):
        assert (
            evaluate_equity_window(
                {},
                cohort=COHORT,
                selected=("AAPL",),
                expected_days=("2026-10-14", "2026-11-11"),
                decision_at="2026-10-14T13:00:00Z",
                as_of=observed,
                round_trip_cost=0.002,
            )
            is None
        )


@pytest.mark.parametrize(
    "cohort,selected",
    [
        ((), ("AAPL",)),
        (("AAPL", "AAPL"), ("AAPL",)),
        (COHORT, ()),
        (COHORT, ("AAPL", "AAPL")),
        (COHORT, ("UNKNOWN",)),
    ],
)
def test_invalid_cohort_configuration(cohort: tuple[str, ...], selected: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="EQUITY-014"):
        evaluate_equity_window(
            {},
            cohort=cohort,
            selected=selected,
            expected_days=DAYS,
            decision_at=DECISION,
            as_of=AS_OF,
            round_trip_cost=0.002,
        )


@pytest.mark.parametrize("cost", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_cost(cost: float) -> None:
    with pytest.raises(ValueError, match="EQUITY-014"):
        evaluate_equity_window(
            {},
            cohort=COHORT,
            selected=("AAPL",),
            expected_days=DAYS,
            decision_at=DECISION,
            as_of=AS_OF,
            round_trip_cost=cost,
        )


@pytest.mark.parametrize(
    "days", [(), ("2020-08-28",), ("2020-08-28", "2020-08-28"), tuple(reversed(DAYS))]
)
def test_invalid_calendar(days: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="EQUITY-014"):
        evaluate_equity_window(
            {},
            cohort=COHORT,
            selected=("AAPL",),
            expected_days=days,
            decision_at=DECISION,
            as_of=AS_OF,
            round_trip_cost=0.002,
        )


def test_decision_outside_window_or_at_entry() -> None:
    with pytest.raises(ValueError, match="EQUITY-014"):
        evaluate_equity_window(
            histories(),
            cohort=COHORT,
            selected=("AAPL",),
            expected_days=DAYS,
            decision_at="2020-09-01T00:00:00Z",
            as_of=AS_OF,
            round_trip_cost=0.002,
        )
    with pytest.raises(ValueError, match="EQUITY-015"):
        evaluate_equity_window(
            histories(),
            cohort=COHORT,
            selected=("AAPL",),
            expected_days=DAYS,
            decision_at="2020-08-28T13:30:00Z",
            as_of=AS_OF,
            round_trip_cost=0.002,
        )


def test_missing_members_and_dates_are_not_dropped() -> None:
    actual = histories()
    missing = dict(actual)
    del missing["MSFT"]
    with pytest.raises(ValueError, match=r"EQUITY-015.*entire"):
        evaluate_equity_window(
            missing,
            cohort=COHORT,
            selected=("AAPL",),
            expected_days=DAYS,
            decision_at=DECISION,
            as_of=AS_OF,
            round_trip_cost=0.002,
        )
    for rows in ((), actual["MSFT"][:1], actual["MSFT"][1:], tuple(reversed(actual["MSFT"]))):
        malformed = dict(actual, MSFT=rows)
        with pytest.raises(ValueError, match=r"EQUITY-015.*dates"):
            evaluate_equity_window(
                malformed,
                cohort=COHORT,
                selected=("AAPL",),
                expected_days=DAYS,
                decision_at=DECISION,
                as_of=AS_OF,
                round_trip_cost=0.002,
            )
    for declared_days in (("2020-08-27", *DAYS), ("2020-08-28", "2020-08-30", "2020-08-31")):
        with pytest.raises(ValueError, match=r"EQUITY-015.*dates"):
            evaluate_equity_window(
                actual,
                cohort=COHORT,
                selected=("AAPL",),
                expected_days=declared_days,
                decision_at="2020-08-27T00:00:00Z",
                as_of=AS_OF,
                round_trip_cost=0.002,
            )


@pytest.mark.parametrize("key,value", [("symbol", "AAPL"), ("currency", "EUR")])
def test_mismatched_price_identity(key: str, value: str) -> None:
    actual = histories()
    row = encode_equity_price(actual["MSFT"][0])
    row[key] = value
    actual["MSFT"] = (decode_equity_price(row), actual["MSFT"][1])
    with pytest.raises(ValueError, match=r"EQUITY-015.*identity"):
        evaluate_equity_window(
            actual,
            cohort=COHORT,
            selected=("AAPL",),
            expected_days=DAYS,
            decision_at=DECISION,
            as_of=AS_OF,
            round_trip_cost=0.002,
        )


def test_actual_http_portfolio_replays_and_http_failure_propagates(
    api_server: ApiServer, tmp_path: Path
) -> None:
    for symbol in COHORT:
        api_server.documents["/" + symbol] = [captured(symbol)]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        online = {
            symbol: decode_equity_chart(
                archive.capture("/" + symbol, {}), symbol, "USD", received_at=AS_OF
            )
            for symbol in COHORT
        }
        offline = {
            symbol: decode_equity_chart(
                archive.replay("/" + symbol, {}), symbol, "USD", received_at=AS_OF
            )
            for symbol in COHORT
        }
        expected = evaluate_equity_window(
            histories(),
            cohort=COHORT,
            selected=("AAPL",),
            expected_days=DAYS,
            decision_at=DECISION,
            as_of=AS_OF,
            round_trip_cost=0.002,
        )
        assert (
            evaluate_equity_window(
                online,
                cohort=COHORT,
                selected=("AAPL",),
                expected_days=DAYS,
                decision_at=DECISION,
                as_of=AS_OF,
                round_trip_cost=0.002,
            )
            == expected
        )
        assert (
            evaluate_equity_window(
                offline,
                cohort=COHORT,
                selected=("AAPL",),
                expected_days=DAYS,
                decision_at=DECISION,
                as_of=AS_OF,
                round_trip_cost=0.002,
            )
            == expected
        )
        with pytest.raises(HTTPStatusError, match="404"):
            archive.capture("/missing-member", {})
    assert len(api_server.requests) == 3
