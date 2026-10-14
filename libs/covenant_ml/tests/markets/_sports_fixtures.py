"""Native sports payloads served through the real HTTP integration boundary."""

from __future__ import annotations

from hashlib import sha256

from platform_core.json_utils import JSONObject, dump_json_str

from covenant_ml.markets.sports_collection import sports_rules
from covenant_ml.markets.sports_records import SportsPair, decode_sports_pair

from ._server import ApiServer

TICKER = "KXNHLGAME-26OCT13NJDET-NJ"
SLUG = "aec-nhl-nj-det-2026-10-13"


def native_markets() -> tuple[JSONObject, JSONObject]:
    return (
        {
            "ticker": TICKER,
            "status": "active",
            "result": "",
            "custom_strike": {"hockey_team": "nj-team"},
            "rules_primary": "New Jersey wins the full game.",
            "rules_secondary": "Cancelled games resolve to fair price.",
        },
        {
            "slug": SLUG,
            "sportsMarketType": "hockey_team_full_game_winner",
            "active": True,
            "closed": False,
            "description": "New Jersey wins, including overtime. Cancellation uses fair price.",
            "marketSides": [{"long": True, "teamId": 1493}, {"long": False, "teamId": 1486}],
            "feeCoefficient": 0.0695,
        },
    )


def sports_pair() -> SportsPair:
    return decode_sports_pair(
        {
            "kalshi_ticker": TICKER,
            "us_slug": SLUG,
            "kalshi_team": "nj-team",
            "us_team": 1493,
            "rules_sha256": sha256(
                dump_json_str(sports_rules(*native_markets())).encode()
            ).hexdigest(),
        }
    )


def us_book() -> JSONObject:
    return {
        "marketData": {
            "marketSlug": SLUG,
            "state": "MARKET_STATE_OPEN",
            "bids": [{"px": {"value": "0.5400", "currency": "USD"}, "qty": "7.5000"}],
            "offers": [{"px": {"value": "0.5500", "currency": "USD"}, "qty": "9.0000"}],
        }
    }


def serve_sports(server: ApiServer, rounds: int) -> None:
    kalshi, us = native_markets()
    documents: dict[str, JSONObject] = {
        "/markets/" + TICKER: {"market": kalshi},
        "/v1/market/slug/" + SLUG: {"market": us},
        "/markets/" + TICKER + "/orderbook": {
            "orderbook_fp": {
                "yes_dollars": [["0.5000", "8.00"]],
                "no_dollars": [["0.4900", "6.00"]],
            }
        },
        "/v1/markets/" + SLUG + "/book": us_book(),
    }
    for path, document in documents.items():
        server.documents[path] = [document for _ in range(rounds)]
