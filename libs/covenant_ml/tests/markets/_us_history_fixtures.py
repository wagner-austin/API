"""Historical native shapes served by the real local HTTP integration origin."""

from __future__ import annotations

from platform_core.json_utils import JSONObject

from covenant_ml.markets.validation import parse_timestamp


def us_game(day: str = "2026-01-01") -> JSONObject:
    return {
        "id": day,
        "slug": "aec-nhl-chi-dal-" + day,
        "startDate": day + "T00:00:00Z",
        "gameStartTime": day + "T22:00:00Z",
        "createdAt": day + "T00:00:00Z",
        "sportsMarketType": "moneyline",
        "marketSides": [
            {"long": True, "teamId": 1500, "team": {"id": 1500, "league": "nhl"}},
            {"long": False, "teamId": 1501, "team": {"id": 1501, "league": "nhl"}},
        ],
    }


def history(day: str = "2026-01-01") -> JSONObject:
    return {
        "history": [
            {
                "timestamp": parse_timestamp(day + "T12:00:00Z") // 1000,
                "longPrice": 0.61,
                "shortPrice": 0.41,
            }
        ]
    }


def mapping() -> JSONObject:
    return {
        "teams": [
            {"kalshi_id": "chicago-id", "us_id": 1500, "name": "Blackhawks"},
            {"kalshi_id": "dallas-id", "us_id": 1501, "name": "Stars"},
        ]
    }
