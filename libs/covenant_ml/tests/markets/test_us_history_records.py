"""Historical identity codecs reject native drift and preserve immutable records."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.markets.us_history import team_links
from covenant_ml.markets.us_history_records import (
    USGame,
    decode_native_us_game,
    decode_team_link,
    decode_us_game,
    encode_team_link,
    encode_us_game,
)

from ._us_history_fixtures import mapping, us_game


def test_round_trips_and_map() -> None:
    native = us_game()
    record = decode_native_us_game(native)
    expected: USGame = {
        "slug": "aec-nhl-chi-dal-2026-01-01",
        "day": "2026-01-01",
        "long_team": 1500,
        "short_team": 1501,
        "start_ms": 1767225600000,
        "game_ms": 1767304800000,
        "created_ms": 1767225600000,
    }
    assert record == expected
    assert decode_us_game(encode_us_game(record)) == record
    document: JSONObject = {"kalshi_id": "chicago-id", "us_id": 1500, "name": "Blackhawks"}
    assert encode_team_link(decode_team_link(document)) == document
    assert team_links(mapping()) == {"chicago-id": 1500, "dallas-id": 1501}


@pytest.mark.parametrize(
    "field,value",
    [
        ("slug", "wrong"),
        ("day", "2026-01-02"),
        ("long_team", 0),
        ("short_team", 1500),
        ("start_ms", -1),
        ("game_ms", 0),
        ("created_ms", -1),
    ],
)
def test_bad_normalized_identity(field: str, value: JSONValue) -> None:
    document = encode_us_game(decode_native_us_game(us_game()))
    document[field] = value
    with pytest.raises(ValueError, match="CROSS-002"):
        decode_us_game(document)


@pytest.mark.parametrize("kind", ["type", "count", "orientation", "identity", "league"])
def test_bad_native_sides(kind: str) -> None:
    document = us_game()
    sides = require_list(document, "marketSides")
    side = narrow_json_to_dict(sides[0])
    if kind == "type":
        document["sportsMarketType"] = "hockey_team_full_game_winner"
    elif kind == "count":
        sides.pop()
    elif kind == "orientation":
        side["long"] = False
    elif kind == "identity":
        side["teamId"] = 99
    else:
        require_dict(side, "team")["league"] = "nba"
    with pytest.raises(ValueError, match="CROSS-003"):
        decode_native_us_game(document)


@pytest.mark.parametrize("kind", ["empty", "duplicate_kalshi", "duplicate_us", "zero"])
def test_bad_mapping(kind: str) -> None:
    document = mapping()
    teams = require_list(document, "teams")
    first, second = (narrow_json_to_dict(value) for value in teams)
    if kind == "empty":
        teams.clear()
    elif kind == "duplicate_kalshi":
        second["kalshi_id"] = first["kalshi_id"]
    elif kind == "duplicate_us":
        second["us_id"] = first["us_id"]
    else:
        first["us_id"] = 0
    with pytest.raises(ValueError, match="CROSS-001" if kind == "zero" else "CROSS-004"):
        team_links(document)


def test_missing_and_wrong_types_propagate() -> None:
    with pytest.raises(TypeError, match="us_id"):
        decode_team_link({"us_id": True})
    document = us_game()
    del document["marketSides"]
    with pytest.raises(TypeError, match="marketSides"):
        decode_native_us_game(document)
