"""Complete native inventory through real HTTP, immutable disk and offline replay."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.json_utils import JSONObject, JSONValue, narrow_json_to_dict, require_list

from covenant_ml.markets.collection import MarketCollector
from covenant_ml.markets.event_cohort import collect_event_cohort, require_cohort_window
from covenant_ml.markets.validation import parse_timestamp

from ._server import ApiServer, api_server
from .test_acquisition import archive

__all__ = ["api_server"]

START = parse_timestamp("2026-01-01T00:00:00Z") // 1000
END = START + 86400


def cohort_market(ticker: str = "KXTEST-EVENT-A") -> JSONObject:
    return {
        "ticker": ticker,
        "event_ticker": "KXTEST-EVENT",
        "open_time": "2025-12-31T00:00:00Z",
        "close_time": "2026-01-01T12:00:00Z",
        "status": "finalized",
        "result": "yes",
        "settlement_value_dollars": "1.0000",
        "volume_fp": "12.34",
    }


def serve_cohort(server: ApiServer) -> None:
    server.documents["/markets"] = [{"markets": [cohort_market()], "cursor": ""}]
    server.documents["/historical/markets"] = [{"markets": [], "cursor": ""}]


def test_complete_inventory_retains_untraded_unresolved_and_both_tiers(
    tmp_path: Path, api_server: ApiServer
) -> None:
    first = cohort_market()
    zero = dict(cohort_market("KXTEST-EVENT-B"), volume_fp="0.00", result="", status="closed")
    historical = dict(
        cohort_market("KXTEST-OTHER-C"),
        event_ticker="KXTEST-OTHER",
        result="scalar",
        settlement_value_dollars="0.5000",
    )
    outside = dict(cohort_market("KXTEST-EVENT-D"), close_time="2026-01-02T00:00:00Z")
    before = dict(cohort_market("KXTEST-EVENT-E"), close_time="2025-12-31T23:59:59.999Z")
    inclusive = dict(cohort_market("KXTEST-EVENT-F"), close_time="2026-01-01T00:00:00Z")
    last = dict(cohort_market("KXTEST-EVENT-G"), close_time="2026-01-01T23:59:59.999Z")
    api_server.documents["/markets"] = [
        {"markets": [first, outside, before], "cursor": "next"},
        {"markets": [zero, inclusive, last], "cursor": ""},
    ]
    api_server.documents["/historical/markets"] = [
        {"markets": [first, historical], "cursor": ""},
    ]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_event_cohort(MarketCollector(store.capture), ("KXTEST",), START, END)
        assert (
            collect_event_cohort(MarketCollector(store.replay), ("KXTEST",), START, END) == report
        )
    assert report["families"] == [
        {
            "series": "KXTEST",
            "tiers": [
                {"path": "/markets", "rows_received": 6, "outside_window": 2},
                {"path": "/historical/markets", "rows_received": 2, "outside_window": 0},
            ],
            "identical_cross_tier_overlap": 1,
            "open_interest_differences": [],
            "contract_count": 5,
            "event_count": 2,
            "events": ["KXTEST-EVENT", "KXTEST-OTHER"],
            "zero_volume_contract_count": 1,
            "declared_result_counts": {"": 1, "yes": 3, "no": 0, "scalar": 1},
            "markets": [first, zero, inclusive, last, historical],
        }
    ]
    assert api_server.requests == [
        f"/markets?limit=1000&max_close_ts={END}&min_close_ts={START - 1}&series_ticker=KXTEST",
        f"/markets?cursor=next&limit=1000&max_close_ts={END}"
        f"&min_close_ts={START - 1}&series_ticker=KXTEST",
        "/historical/markets?limit=1000&series_ticker=KXTEST",
    ]
    assert report["selection_clock"] == "capture_time_close_time"
    assert report["orders_submitted"] == 0


@pytest.mark.parametrize(
    "series,start,end",
    [
        ((), 0, 1),
        (("KX", "KX"), 0, 1),
        (("KX-?",), 0, 1),
        (("KXÉ",), 0, 1),
        (("KX",), -1, 1),
        (("KX",), 1, 1),
        (("KX",), 2, 1),
        (("KX",), 0, 366 * 86400 + 1),
    ],
)
def test_invalid_plan(series: tuple[str, ...], start: int, end: int) -> None:
    with pytest.raises(ValueError, match="COHORT-001"):
        require_cohort_window(series, start, end)


@pytest.mark.parametrize(
    "key,value,code",
    [
        ("ticker", "OTHER-EVENT-A", "COHORT-002"),
        ("event_ticker", "OTHER-EVENT", "COHORT-002"),
        ("event_ticker", "KXTEST-OTHER", "COHORT-002"),
        ("open_time", "2026-01-01T12:00:00Z", "COHORT-004"),
        ("open_time", "1960-01-01T00:00:00Z", "COHORT-004"),
        ("volume_fp", "-1.00", "MARKET-005"),
        ("result", "unsupported", "TAPE-005"),
    ],
)
def test_bad_native_evidence_propagates(
    tmp_path: Path, api_server: ApiServer, key: str, value: JSONValue, code: str
) -> None:
    serve_cohort(api_server)
    row = cohort_market()
    row[key] = value
    api_server.documents["/markets"] = [{"markets": [row], "cursor": ""}]
    store = archive(tmp_path, api_server)
    with closing(store.client), pytest.raises(ValueError, match=code):
        collect_event_cohort(MarketCollector(store.capture), ("KXTEST",), START, END)
    assert len(api_server.requests) == 1


@pytest.mark.parametrize("conflict", [False, True])
def test_duplicate_and_conflicting_identity_fail(
    tmp_path: Path, api_server: ApiServer, conflict: bool
) -> None:
    serve_cohort(api_server)
    if conflict:
        api_server.documents["/historical/markets"] = [
            {"markets": [dict(cohort_market(), volume_fp="99.00")], "cursor": ""}
        ]
    else:
        api_server.documents["/markets"] = [
            {"markets": [cohort_market(), cohort_market()], "cursor": ""}
        ]
    store = archive(tmp_path, api_server)
    with (
        closing(store.client),
        pytest.raises(ValueError, match="COHORT-003" if conflict else "COHORT-002"),
    ):
        collect_event_cohort(MarketCollector(store.capture), ("KXTEST",), START, END)
    assert len(api_server.requests) == (2 if conflict else 1)


def test_empty_families_preserved_and_missing_tier_fails(
    tmp_path: Path, api_server: ApiServer
) -> None:
    api_server.documents["/markets"] = [{"markets": [], "cursor": ""}]
    api_server.documents["/historical/markets"] = [{"markets": [], "cursor": ""}]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_event_cohort(MarketCollector(store.capture), ("KXTEST",), START, END)
    assert report["families"] == [
        {
            "series": "KXTEST",
            "tiers": [
                {"path": p, "rows_received": 0, "outside_window": 0}
                for p in ("/markets", "/historical/markets")
            ],
            "identical_cross_tier_overlap": 0,
            "open_interest_differences": [],
            "contract_count": 0,
            "event_count": 0,
            "events": [],
            "zero_volume_contract_count": 0,
            "declared_result_counts": {"": 0, "yes": 0, "no": 0, "scalar": 0},
            "markets": [],
        }
    ]
    serve_cohort(api_server)
    del api_server.documents["/historical/markets"]
    failed = archive(tmp_path / "failed", api_server)
    with closing(failed.client), pytest.raises(HTTPStatusError, match="404"):
        collect_event_cohort(MarketCollector(failed.capture), ("KXTEST",), START, END)
    assert len(tuple((tmp_path / "failed").glob("*.json"))) == 1


def test_mutable_open_interest_is_preserved_without_changing_cohort(
    tmp_path: Path, api_server: ApiServer
) -> None:
    current = dict(cohort_market(), open_interest_fp="123.45")
    historical = dict(current, open_interest_fp="0.00")
    api_server.documents["/markets"] = [{"markets": [current], "cursor": ""}]
    api_server.documents["/historical/markets"] = [{"markets": [historical], "cursor": ""}]
    store = archive(tmp_path, api_server)
    with closing(store.client):
        report = collect_event_cohort(MarketCollector(store.capture), ("KXTEST",), START, END)
        assert (
            collect_event_cohort(MarketCollector(store.replay), ("KXTEST",), START, END) == report
        )
    family = narrow_json_to_dict(require_list(report, "families")[0])
    assert family["markets"] == [current]
    assert family["identical_cross_tier_overlap"] == 0
    assert family["contract_count"] == 1
    assert family["open_interest_differences"] == [
        {
            "ticker": "KXTEST-EVENT-A",
            "current_open_interest_fp": "123.45",
            "historical_open_interest_fp": "0.00",
        }
    ]


def test_outcome_and_volume_changes_do_not_select_the_inventory(
    tmp_path: Path, api_server: ApiServer
) -> None:
    families: list[JSONObject] = []
    for index, result in enumerate(("yes", "no")):
        row = dict(
            cohort_market(),
            result=result,
            settlement_value_dollars="1.0000" if result == "yes" else "0.0000",
            volume_fp="0.00" if result == "yes" else "123.45",
        )
        api_server.documents["/markets"] = [{"markets": [row], "cursor": ""}]
        api_server.documents["/historical/markets"] = [{"markets": [], "cursor": ""}]
        store = archive(tmp_path / str(index), api_server)
        with closing(store.client):
            report = collect_event_cohort(MarketCollector(store.capture), ("KXTEST",), START, END)
        families.append(narrow_json_to_dict(require_list(report, "families")[0]))
    assert [row["events"] for row in families] == [["KXTEST-EVENT"], ["KXTEST-EVENT"]]
    assert [row["contract_count"] for row in families] == [1, 1]
    assert [row["zero_volume_contract_count"] for row in families] == [1, 0]
    assert [row["declared_result_counts"] for row in families] == [
        {"": 0, "yes": 1, "no": 0, "scalar": 0},
        {"": 0, "yes": 0, "no": 1, "scalar": 0},
    ]
