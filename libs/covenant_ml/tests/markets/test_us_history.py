"""Historical reads and native joins exercised through real HTTP and hashed replay."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.json_utils import (
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.us_history import cross_predictions, history_probability, us_inventory
from covenant_ml.markets.us_history_records import decode_native_us_game

from ._rating_fixtures import game
from ._server import ApiServer, api_server
from ._strategy_fixtures import observation
from ._us_history_fixtures import history, us_game

__all__ = ["api_server"]


def test_pagination_through_short_pages(api_server: ApiServer, tmp_path: Path) -> None:
    api_server.documents["/v1/markets"] = [
        {"markets": [us_game(), {"id": "unrelated", "slug": "other-sport"}]},
        {"markets": [us_game("2026-03-01")]},
        {"markets": []},
    ]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        games, counts = us_inventory(store.capture)
        assert tuple(row["day"] for row in games) == ("2026-01-01", "2026-03-01")
        assert counts == {"pages": 3, "native_markets": 3, "nhl_games": 2}
        assert us_inventory(store.replay) == (games, counts)
    assert [request.rsplit("offset=", 1)[1] for request in api_server.requests] == [
        "0",
        "500",
        "1000",
    ]


@pytest.mark.parametrize("kind", ["id", "slug", "oversized", "http"])
def test_bad_inventory(api_server: ApiServer, tmp_path: Path, kind: str) -> None:
    first, second = us_game(), us_game("2026-03-01")
    if kind in ("id", "slug"):
        second[kind] = first[kind]
    values: list[JSONValue] = [first] * 501 if kind == "oversized" else [first, second]
    api_server.documents["/v1/markets"] = [{"markets": values}]
    if kind == "http":
        api_server.statuses["/v1/markets"] = 503
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        if kind == "http":
            with pytest.raises(HTTPStatusError) as error:
                us_inventory(store.capture)
            assert error.value.response.status_code == 503
            assert list(tmp_path.glob("*.json")) == []
        else:
            with pytest.raises(ValueError, match="CROSS-005"):
                us_inventory(store.capture)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("valid", "eligible"),
        ("short", "eligible"),
        ("empty", "missing_history"),
        ("stale", "stale_history"),
        ("boundary_bid", "boundary_display"),
        ("boundary_ask", "boundary_display"),
        ("wide", "wide_display"),
        ("after_start", "outside_pregame_window"),
        ("before_open", "outside_pregame_window"),
    ],
)
def test_history_exclusions_and_orientation(
    api_server: ApiServer,
    tmp_path: Path,
    kind: str,
    reason: str,
) -> None:
    payload = history()
    rows = require_list(payload, "history")
    point = narrow_json_to_dict(rows[0])
    spec, _ = observation()
    cutoff = spec["decision_ms"]
    if kind == "empty":
        rows.clear()
    elif kind == "stale":
        point["timestamp"] = cutoff // 1000 - 61
    elif kind == "boundary_bid":
        point["shortPrice"] = 1
    elif kind == "boundary_ask":
        point["longPrice"] = 1
    elif kind == "wide":
        point["longPrice"] = 0.8
    elif kind == "after_start":
        cutoff += 36000000
    elif kind == "before_open":
        cutoff -= 86400000
    api_server.documents["/v1/price-history"] = [payload]
    selected = 1501 if kind == "short" else 1500
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        native = decode_native_us_game(us_game())
        actual = history_probability(native, cutoff, selected, store.capture)
        expected = (0.4 if kind == "short" else 0.6) if reason == "eligible" else None
        assert actual == (expected, reason)
        assert history_probability(native, cutoff, selected, store.replay) == actual
    assert len(api_server.requests) == (0 if reason == "outside_pregame_window" else 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("longPrice", -0.1),
        ("longPrice", 1.1),
        ("shortPrice", -0.1),
        ("shortPrice", 1.1),
        ("longPrice", 0.1),
        ("timestamp", 0),
        ("timestamp", 1767268801),
        ("longPrice", float("nan")),
    ],
)
def test_invalid_observation(
    api_server: ApiServer,
    tmp_path: Path,
    field: str,
    value: JSONValue,
) -> None:
    payload = history()
    narrow_json_to_dict(require_list(payload, "history")[0])[field] = value
    api_server.documents["/v1/price-history"] = [payload]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(ValueError, match="CROSS-007"):
            history_probability(
                decode_native_us_game(us_game()),
                observation()[0]["decision_ms"],
                1500,
                store.capture,
            )


@pytest.mark.parametrize("kind", ["duplicate", "reversed", "cutoff", "fraction", "team"])
def test_invalid_sequence_or_query(api_server: ApiServer, tmp_path: Path, kind: str) -> None:
    payload = history()
    points = require_list(payload, "history")
    point = dict(narrow_json_to_dict(points[0]))
    if kind == "reversed":
        point["timestamp"] = observation()[0]["decision_ms"] // 1000 - 1
    points.append(point)
    api_server.documents["/v1/price-history"] = [payload]
    cutoff = observation()[0]["decision_ms"]
    if kind == "cutoff":
        cutoff = 0
    elif kind == "fraction":
        cutoff += 1
    selected = 999 if kind == "team" else 1500
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(
            ValueError, match="CROSS-007" if kind in ("duplicate", "reversed") else "CROSS-006"
        ):
            history_probability(decode_native_us_game(us_game()), cutoff, selected, store.capture)


def test_join_complete_coverage_and_replay(api_server: ApiServer, tmp_path: Path) -> None:
    stamps = ("26JAN01", "26MAR01", "26MAR02")
    specs = [observation(stamp)[0] for stamp in stamps]
    records = [row for stamp in stamps for row in game(stamp)]
    native = us_game("2026-03-01")
    native["createdAt"] = "2026-03-07T00:00:00Z"
    games = (decode_native_us_game(us_game()), decode_native_us_game(native))
    api_server.documents["/v1/price-history"] = [history(), {"history": []}]
    links = {"chicago-id": 1500, "dallas-id": 1501}
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        result = cross_predictions(records, specs, games, links, store.capture)
        assert result == (
            {specs[0]["ticker"]: 0.6},
            [
                {
                    "ticker": specs[0]["ticker"],
                    "slug": games[0]["slug"],
                    "status": "eligible",
                    "late_created_metadata": False,
                },
                {
                    "ticker": specs[1]["ticker"],
                    "slug": games[1]["slug"],
                    "status": "missing_history",
                    "late_created_metadata": True,
                },
                {"ticker": specs[2]["ticker"], "status": "unmatched_game"},
            ],
        )
        assert cross_predictions(records, specs, games, links, store.replay) == result
    assert len(api_server.requests) == 2


@pytest.mark.parametrize(
    "kind",
    [
        "ambiguous",
        "incomplete",
        "duplicate_decision",
        "unmapped",
        "ticker",
        "duplicate_ticker",
        "same_team",
        "missing_selected",
    ],
)
def test_invalid_join(api_server: ApiServer, tmp_path: Path, kind: str) -> None:
    records = list(game("26JAN01"))
    spec, _ = observation()
    specs = [spec, spec] if kind == "duplicate_decision" else [spec]
    native = decode_native_us_game(us_game())
    games = [native, native] if kind == "ambiguous" else [native]
    links = {"chicago-id": 1500, "dallas-id": 1501}
    if kind == "incomplete":
        records.pop()
    elif kind == "unmapped":
        del links["chicago-id"]
    elif kind == "ticker":
        records[0]["ticker"] = "wrong"
    elif kind == "duplicate_ticker":
        records[1]["ticker"] = records[0]["ticker"]
    elif kind == "same_team":
        require_dict(records[1], "custom_strike")["hockey_team"] = "chicago-id"
    elif kind == "missing_selected":
        records[0]["ticker"] = spec["ticker"].rsplit("-", 1)[0] + "-ZZZ"
    api_server.documents["/v1/price-history"] = [history()]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        store = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(ValueError, match="CROSS-008"):
            cross_predictions(records, specs, games, links, store.capture)
