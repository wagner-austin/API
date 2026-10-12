"""Execution boundaries checked against explicit economic outcomes."""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

import pytest
from platform_core.json_utils import JSONValue, narrow_json_to_dict, require_list

from covenant_ml.markets.codecs import decode_history
from covenant_ml.markets.replay import quadratic_fee, replay_maker, replay_markets
from covenant_ml.markets.types import ReplayStatus

from ._fixtures import history, history_document, rule


@pytest.mark.parametrize("price", [0, 100, 5000, 9900, 10_000])
@pytest.mark.parametrize("contracts", [1, 10, 73])
@pytest.mark.parametrize("bps", [0, 175, 700])
def test_integer_fee_agrees_with_independent_decimal_formula(
    price: int,
    contracts: int,
    bps: int,
) -> None:
    probability = Decimal(price) / 10_000
    dollars = Decimal(bps) / 10_000 * contracts * probability * (1 - probability)
    cents = (dollars * 100).to_integral_value(rounding=ROUND_CEILING)
    assert quadratic_fee(price, contracts, bps) == int(cents) * 100


@pytest.mark.parametrize(
    "price,contracts,bps",
    [(-1, 1, 0), (10001, 1, 0), (5000, 0, 0), (5000, 1, -1), (5000, 1, 10001)],
)
def test_invalid_fee_inputs_are_refused(price: int, contracts: int, bps: int) -> None:
    with pytest.raises(ValueError, match="MARKET-010"):
        quadratic_fee(price, contracts, bps)


def test_sell_yes_win_and_loss_reconcile_with_collateral_and_fee() -> None:
    win = replay_maker(history(), rule(maker_rate_bps=175))
    loss = replay_maker(history(settlement=10000), rule(maker_rate_bps=175))
    assert (win["offer"], win["fee"], win["pnl"], win["capital"]) == (900, 100, 800, 9100)
    assert loss["pnl"] == -9200
    assert win["pnl"] - loss["pnl"] == 10_000
    assert win["status"] is loss["status"] is ReplayStatus.FILLED


@pytest.mark.parametrize(
    "quotes,status",
    [
        ([], ReplayStatus.NO_QUOTE),
        ([{"timestamp_ms": 10_000_001, "bid": 400, "ask": 1000}], ReplayStatus.NO_QUOTE),
        ([{"timestamp_ms": 1, "bid": 400, "ask": 1000}], ReplayStatus.STALE),
        ([{"timestamp_ms": 10_000_000, "bid": None, "ask": 1000}], ReplayStatus.NO_SIDES),
        ([{"timestamp_ms": 10_000_000, "bid": 400, "ask": None}], ReplayStatus.NO_SIDES),
        ([{"timestamp_ms": 10_000_000, "bid": 0, "ask": 100}], ReplayStatus.OUTSIDE_BAND),
        ([{"timestamp_ms": 10_000_000, "bid": 0, "ask": 1700}], ReplayStatus.OUTSIDE_BAND),
        ([{"timestamp_ms": 10_000_000, "bid": 900, "ask": 1000}], ReplayStatus.CROSSES),
        ([{"timestamp_ms": 10_000_000, "bid": 1000, "ask": 1000}], ReplayStatus.CROSSES),
    ],
)
def test_quote_exclusions_never_generate_money(
    quotes: list[JSONValue], status: ReplayStatus
) -> None:
    result = replay_maker(history(quotes=quotes), rule())
    assert result["status"] is status
    assert (result["contracts"], result["fee"], result["pnl"], result["capital"]) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("timestamp_ms", 10_001_000),
        ("timestamp_ms", 20_000_000),
        ("timestamp_ms", 9_000_000),
        ("price", 1099),
        ("taker_yes", False),
        ("block", True),
        ("quantity", 99),
    ],
)
def test_ineligible_trade_evidence_cannot_fill(key: str, value: JSONValue) -> None:
    document = history_document()
    trade = narrow_json_to_dict(require_list(document, "trades")[0])
    trade[key] = value
    result = replay_maker(decode_history(document), rule())
    assert result["status"] is ReplayStatus.UNFILLED
    assert result["pnl"] == 0


def test_fractional_volume_must_accumulate_at_the_required_price() -> None:
    trades: list[JSONValue] = [
        {
            "trade_id": "a",
            "timestamp_ms": 10_002_000,
            "price": 1100,
            "quantity": 40,
            "taker_yes": True,
            "block": False,
        },
        {
            "trade_id": "b",
            "timestamp_ms": 10_003_000,
            "price": 1100,
            "quantity": 60,
            "taker_yes": True,
            "block": False,
        },
        {
            "trade_id": "c",
            "timestamp_ms": 10_004_000,
            "price": 1000,
            "quantity": 900,
            "taker_yes": True,
            "block": False,
        },
    ]
    one = replay_maker(history(trades=trades), rule())
    two = replay_maker(history(trades=trades), rule(contracts=2))
    assert one["matching_quantity"] == two["matching_quantity"] == 100
    assert one["status"] is ReplayStatus.FILLED
    assert two["status"] is ReplayStatus.UNFILLED


def test_latest_prior_quote_is_selected_independent_of_input_order() -> None:
    quotes: list[JSONValue] = [
        {"timestamp_ms": 10_000_001, "bid": 400, "ask": 200},
        {"timestamp_ms": 10_000_000, "bid": 400, "ask": 1000},
        {"timestamp_ms": 9_999_999, "bid": 400, "ask": 1500},
    ]
    result = replay_maker(history(quotes=quotes), rule())
    assert (result["offer"], result["pnl"]) == (900, 900)


def test_panel_refuses_duplicate_positions_and_empty_samples() -> None:
    item = history()
    with pytest.raises(ValueError, match="MARKET-011"):
        replay_markets([], rule())
    with pytest.raises(ValueError, match="MARKET-011"):
        replay_markets([item, item], rule())
    assert replay_markets([item], rule()) == (replay_maker(item, rule()),)


__all__: list[str] = []
