"""Real public HTTP boundary, replay, rule drift, state failures and gross arithmetic."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import httpx
import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
)

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.sports_collection import _validate_markets, collect_sports_pair
from covenant_ml.markets.validation import read_document

from ._server import ApiServer, api_server
from ._sports_fixtures import SLUG, TICKER, native_markets, serve_sports, sports_pair

__all__ = ["api_server"]


def test_actual_capture_replay_skew_and_absent_depth(api_server: ApiServer, tmp_path: Path) -> None:
    serve_sports(api_server, 1)
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        k = ResponseArchive(
            tmp_path / "k", api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        u = ResponseArchive(
            tmp_path / "u", api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        pair = sports_pair()
        first = collect_sports_pair(pair, k, u, True, 0, 9999999999999, 10000)
        assert collect_sports_pair(pair, k, u, False, 0, 9999999999999, 10000) == first
        assert len(api_server.requests) == 4
        assert first["exclusion"] == ""
        assert first["directions"] == [
            {
                "direction": "kalshi_yes_us_no",
                "pair_price": 9700,
                "ordinary_completion_gross_bound": 300,
                "visible_size": 600,
                "size_checks": [
                    {
                        "contracts": 1,
                        "gross_cost": 9700,
                        "sufficient_top_depth": True,
                        "within_gross_20_dollar_budget": True,
                    },
                    {
                        "contracts": 5,
                        "gross_cost": 48500,
                        "sufficient_top_depth": True,
                        "within_gross_20_dollar_budget": True,
                    },
                    {
                        "contracts": 20,
                        "gross_cost": 194000,
                        "sufficient_top_depth": False,
                        "within_gross_20_dollar_budget": True,
                    },
                ],
            },
            {
                "direction": "us_yes_kalshi_no",
                "pair_price": 10500,
                "ordinary_completion_gross_bound": -500,
                "visible_size": 800,
                "size_checks": [
                    {
                        "contracts": 1,
                        "gross_cost": 10500,
                        "sufficient_top_depth": True,
                        "within_gross_20_dollar_budget": True,
                    },
                    {
                        "contracts": 5,
                        "gross_cost": 52500,
                        "sufficient_top_depth": True,
                        "within_gross_20_dollar_budget": True,
                    },
                    {
                        "contracts": 20,
                        "gross_cost": 210000,
                        "sufficient_top_depth": False,
                        "within_gross_20_dollar_budget": False,
                    },
                ],
            },
        ]
        assert first["fees_verified"] is False
        assert first["settlement_equivalence_verified"] is False
        assert first["orders_submitted"] == 0
        assert first["us_fee_coefficient"] == 0.0695
        skewed = collect_sports_pair(pair, k, u, False, 0, 9999999999999, 1)
        assert skewed["exclusion"] == "book_receipt_skew"
        assert skewed["directions"] == []
        for start, end, skew in ((2, 1, 1), (0, 9, 0), (0, 1, 1)):
            with pytest.raises(ValueError, match="SPORTS-006"):
                collect_sports_pair(pair, k, u, False, start, end, skew)
        replacements: tuple[tuple[ResponseArchive, str, JSONObject], ...] = (
            (
                k,
                "/markets/" + TICKER + "/orderbook",
                {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}},
            ),
            (
                u,
                "/v1/markets/" + SLUG + "/book",
                {
                    "marketData": {
                        "marketSlug": SLUG,
                        "state": "MARKET_STATE_OPEN",
                        "bids": [],
                        "offers": [],
                    }
                },
            ),
        )
        for store, path, replacement in replacements:
            target = store.file_for(store.request_url(path, {}))
            original = target.read_text("utf-8")
            rewrite_capture(target, replacement)
            missing = collect_sports_pair(pair, k, u, False, 0, 9999999999999, 10000)
            assert missing["exclusion"] == "missing_book_side"
            assert missing["directions"] == []
            target.write_text(original, encoding="utf-8")
        target = u.file_for(u.request_url("/v1/market/slug/" + SLUG, {}))
        native = native_markets()[1]
        native["feeCoefficient"] = -0.1
        rewrite_capture(target, {"market": native})
        with pytest.raises(ValueError, match="SPORTS-008"):
            collect_sports_pair(pair, k, u, False, 0, 9999999999999, 10000)
        assert len(api_server.requests) == 4


def rewrite_capture(path: Path, body: JSONObject) -> None:
    from hashlib import sha256

    envelope = read_document(path)
    text = dump_json_str(body)
    envelope["body"] = text
    envelope["digest"] = sha256(text.encode()).hexdigest()
    path.write_text(dump_json_str(envelope), encoding="utf-8")


@pytest.mark.parametrize(
    "venue,key,value",
    [
        ("k", "ticker", "wrong"),
        ("k", "status", "closed"),
        ("k", "result", "yes"),
        ("k", "custom_strike", {"hockey_team": "other"}),
        ("u", "slug", "wrong"),
        ("u", "sportsMarketType", "other"),
        ("u", "active", False),
        ("u", "closed", True),
        ("u", "marketSides", []),
        ("u", "marketSides", [{"long": False}, {"long": False}]),
        ("u", "marketSides", [{"long": True, "teamId": 1}, {"long": False}]),
    ],
)
def test_native_identity_and_state_failures(venue: str, key: str, value: JSONValue) -> None:
    k, u = native_markets()
    target = k if venue == "k" else u
    target[key] = value
    with pytest.raises(ValueError, match="SPORTS-004"):
        _validate_markets(sports_pair(), k, u)


def test_rules_drift_rejected() -> None:
    k, u = native_markets()
    k["rules_secondary"] = "Different cancellation rule."
    with pytest.raises(ValueError, match="SPORTS-005"):
        _validate_markets(sports_pair(), k, u)


def test_actual_http_failure_propagates(api_server: ApiServer, tmp_path: Path) -> None:
    serve_sports(api_server, 1)
    api_server.statuses["/markets/" + TICKER] = 503
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(httpx.HTTPStatusError, match="503"):
            collect_sports_pair(sports_pair(), store, store, True, 0, 9999999999999, 1000)
    assert len(api_server.requests) == 1
    assert list(tmp_path.glob("*.json")) == []
