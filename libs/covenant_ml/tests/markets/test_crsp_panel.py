"""Hand-calculated momentum, liquidity ties and future-mutation invariance."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_str

from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_panel import build_crsp_panel

from ._crsp_fixtures import source_rows, write_sources


def test_decision_features_have_independent_arithmetic_and_deterministic_ties(
    tmp_path: Path,
) -> None:
    rows = source_rows()
    path, _ = write_sources(tmp_path, rows)
    panel = build_crsp_panel(load_crsp_sources(path), "2019-12-01", "2019-12-31")
    decision = panel[0]
    assert decision.window == ("2019-12-31", "2020-01-01", "2020-02-03")
    assert [feature.permno for feature in decision.features] == [1, 2]
    assert [feature.liquidity for feature in decision.features] == [1000.0, 1000.0]
    assert decision.features[0].momentum == pytest.approx(-0.12)
    assert decision.features[1].momentum == 0.0
    assert decision.exclusions == (("liquidity_rank_limit", 1),)
    # A later crash, price/volume changes, or future listing cannot alter selection.
    for row in rows["returns"]:
        if require_str(row, "dlycaldt") > "2019-12-31":
            row["dlyret"], row["dlyprc"], row["dlyvol"] = -1.0, 0.01, 0.0
    rows["membership"].append(
        {"permno": 99, "namedt": "2020-01-01", "nameendt": "2024-12-31", "shrcd": 10, "exchcd": 1}
    )
    rows["membership"].reverse()
    rows["returns"].reverse()
    path, _ = write_sources(tmp_path, rows)
    assert build_crsp_panel(load_crsp_sources(path), "2019-12-01", "2019-12-31") == panel


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("dlyprc", None, "decision_price_missing"),
        ("dlyprc", 4.99, "decision_price_below_minimum"),
        ("dlyret", None, "daily_history_invalid"),
        ("dlyretmissflg", "MP", "daily_history_invalid"),
        ("dlydelflg", "Y", "daily_history_invalid"),
        ("dlyretdurflg", "DD", "daily_history_invalid"),
        ("dlyprevdt", "2019-12-27", "daily_history_invalid"),
        ("dlyvol", None, "liquidity_missing"),
    ],
)
def test_each_ineligible_security_has_one_counted_reason(
    tmp_path: Path, field: str, value: JSONValue, reason: str
) -> None:
    rows = source_rows()
    for row in rows["returns"]:
        if row["permno"] == 1 and row["dlycaldt"] == "2019-12-31":
            row[field] = value
    path, _ = write_sources(tmp_path, rows)
    decision = build_crsp_panel(load_crsp_sources(path), "2019-12-01", "2019-12-31")[0]
    assert decision.exclusions == ((reason, 1),)
    assert [feature.permno for feature in decision.features] == [2, 3]


def test_missing_history_and_membership_are_counted_without_silently_dropping_dates(
    tmp_path: Path,
) -> None:
    rows = source_rows()
    rows["returns"] = [row for row in rows["returns"] if row["permno"] != 1]
    rows["membership"][1]["shrcd"] = 12
    rows["returns"] = [
        row
        for row in rows["returns"]
        if not (row["permno"] == 3 and row["dlycaldt"] == "2019-06-03")
    ]
    path, _ = write_sources(tmp_path, rows)
    decision = build_crsp_panel(load_crsp_sources(path), "2019-12-01", "2019-12-31")[0]
    assert decision.features == ()
    assert decision.exclusions == (
        ("daily_history_missing", 1),
        ("decision_price_missing", 1),
        ("membership_ineligible", 1),
    )


def test_history_calendar_and_liquidity_evidence_are_not_imputed(tmp_path: Path) -> None:
    rows = source_rows()
    path, _ = write_sources(tmp_path, rows)
    short = build_crsp_panel(load_crsp_sources(path), "2019-02-01", "2019-02-28")[0]
    assert short.features == ()
    assert short.exclusions == (("calendar_history_insufficient", 3),)
    for row in rows["returns"]:
        if row["permno"] == 1 and row["dlycaldt"] == "2019-12-30":
            row["dlyprc"] = None
    path, _ = write_sources(tmp_path, rows)
    result = build_crsp_panel(load_crsp_sources(path), "2019-12-01", "2019-12-31")[0]
    assert result.exclusions == (("liquidity_missing", 1),)


@pytest.mark.parametrize("mutation", ["non_session", "overlap", "overflow", "before", "after"])
def test_panel_refuses_ambiguous_or_impossible_sources(tmp_path: Path, mutation: str) -> None:
    rows = source_rows()
    first, last = "2019-12-01", "2019-12-31"
    if mutation == "non_session":
        rows["returns"][0]["dlycaldt"] = "2018-12-08"
    elif mutation == "overlap":
        duplicate: JSONObject = {**rows["membership"][0], "namedt": "2019-01-01"}
        rows["membership"].append(duplicate)
    elif mutation == "overflow":
        for row in rows["returns"]:
            row["dlyprc"], row["dlyvol"] = 1e308, 1e308
    elif mutation == "before":
        first = "2018-12-01"
    else:
        last = "2025-01-01"
    path, _ = write_sources(tmp_path, rows)
    with pytest.raises(ValueError, match=r"CRSP-003|CRSP-008"):
        build_crsp_panel(load_crsp_sources(path), first, last)
