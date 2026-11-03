"""Independent interval arithmetic and refusals at trading and terminal boundaries."""

from __future__ import annotations

import re
from datetime import date, timedelta
from math import prod
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_float, require_str

from covenant_ml.markets.crsp_calendar import CrspWindow, crsp_return_span
from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_holdings import (
    crsp_window_path,
    evaluate_crsp_holdings,
    index_crsp_outcomes,
)
from covenant_ml.markets.crsp_panel import build_crsp_panel
from covenant_ml.markets.crsp_return_path import (
    HeldRefusal,
    crsp_compound_held,
    crsp_held_spans,
    crsp_resolve_held,
)
from covenant_ml.markets.crsp_study import crsp_study_rows

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import missing_price_rows, pre_entry_rows, study_protocol
from .test_crsp_execution_boundaries import event
from .test_crsp_holdings import W1, W2, daily, outcomes


def missing(day: str, **changes: JSONValue) -> JSONObject:
    """An explicit missing-price record, not an absent source row."""
    row = daily(1, W1.entry, 0.0)
    row.update(
        {
            "dlycaldt": day,
            "dlyret": None,
            "dlyprevdt": None,
            "dlyretmissflg": "MP",
            "dlyretdurflg": "MR",
            "dlyprc": None,
            "dlyprcflg": "MP",
        }
    )
    row.update(changes)
    return row


def bridged_rows() -> list[JSONObject]:
    """One two-session +10% return, with a missing intermediate close."""
    return [
        daily(1, W1.entry, 0.0),
        missing("2020-02-04"),
        daily(1, W1.exit, 0.1, dlyprevdt=W1.entry, dlyretdurflg="P1"),
    ]


@pytest.mark.parametrize("price_flag", ["MP", "SU"])
def test_complete_span_is_counted_once_in_monthly_wealth_and_actual_costs(
    price_flag: str,
) -> None:
    rows = bridged_rows()
    rows[1]["dlyprcflg"] = price_flag
    evidence = outcomes(rows, [])
    path = crsp_window_path(evidence, 1, W1)
    assert (path.filled, path.growth, path.terminal) == (True, 1.1, None)
    (period,) = evaluate_crsp_holdings(evidence, (W1,), ((1,),), 10)
    assert period.selected == (1,)
    assert period.unfilled == ()
    assert period.net_return == pytest.approx(1.1 * 0.999 / 1.001 - 1)
    assert period.turnover == pytest.approx(2.1 / 1.001)
    assert period.cost == pytest.approx(0.001 * 2.1 / 1.001)
    assert period.held == ((1, 1.0),)
    assert period.cash == 0


@pytest.mark.parametrize("gap", range(1, 10))
@pytest.mark.parametrize("price_flag", ["MP", "SU"])
def test_vendor_duration_must_cover_exactly_two_through_ten_sessions(
    gap: int, price_flag: str
) -> None:
    dates = [date(2020, 6, 1) + timedelta(days=n) for n in range(20)]
    sessions = tuple(day.isoformat() for day in dates if day.weekday() < 5)
    entry, end = sessions[0], sessions[gap + 1]
    records = {(1, day): missing(day, dlyprcflg=price_flag) for day in sessions[1 : gap + 1]}
    close = daily(1, W1.entry, -0.2, dlycaldt=end, dlyprevdt=entry, dlyretdurflg=f"P{gap}")
    records[(1, end)] = close
    assert not crsp_return_span(close, sessions[gap], 1)
    assert crsp_compound_held(records, 1, entry, sessions[1 : gap + 2]) == 0.8


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dlyret", 0.0),
        ("dlyprevdt", W1.entry),
        ("dlyretmissflg", "MV"),
        ("dlyretdurflg", "D1"),
        ("dlydelflg", "Y"),
        ("dlyprc", 10.0),
        ("dlyprcflg", "NT"),
        ("dlyprcflg", "HA"),
        ("dlyretmissflg", "NT"),
        ("dlyretmissflg", "DG"),
    ],
)
@pytest.mark.parametrize("price_flag", ["MP", "SU"])
def test_only_consistent_missing_price_evidence_can_defer_measurement(
    key: str,
    value: JSONValue,
    price_flag: str,
) -> None:
    rows = bridged_rows()
    rows[1]["dlyprcflg"] = price_flag
    rows[1][key] = value
    with pytest.raises(ValueError, match="CRSP-009"):
        crsp_window_path(outcomes(rows, []), 1, W1)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dlyret", None),
        ("dlyprevdt", "2020-01-31"),
        ("dlyprevdt", "2020-02-04"),
        ("dlyretmissflg", "RA"),
        ("dlydelflg", "Y"),
        ("dlyretdurflg", "D1"),
        ("dlyretdurflg", "P2"),
        ("dlyretdurflg", "DD"),
    ],
)
@pytest.mark.parametrize("price_flag", ["MP", "SU"])
def test_gap_closure_rejects_unknown_values_overlaps_and_wrong_spans(
    key: str,
    value: JSONValue,
    price_flag: str,
) -> None:
    rows = bridged_rows()
    rows[1]["dlyprcflg"] = price_flag
    rows[2][key] = value
    with pytest.raises(ValueError, match="CRSP-009"):
        crsp_window_path(outcomes(rows, []), 1, W1)


def test_absent_row_is_not_a_vendor_missing_price_observation() -> None:
    rows = bridged_rows()
    with pytest.raises(ValueError, match=r"CRSP-009.*absent"):
        crsp_window_path(outcomes([rows[0], rows[2]], []), 1, W1)


@pytest.mark.parametrize("price_flag", ["MP", "SU"])
def test_gap_cannot_consume_a_return_beyond_a_terminal_boundary(price_flag: str) -> None:
    rows = bridged_rows()
    rows[1]["dlyprcflg"] = price_flag
    terminal = event("2020-02-04")
    terminal.update({"deldlydt": "2020-02-05", "delamtdt": "2020-02-05"})
    with pytest.raises(ValueError, match="CRSP-011"):
        crsp_window_path(outcomes(rows, [terminal]), 1, W1)


def test_span_closed_on_terminal_date_precedes_one_terminal_multiplier() -> None:
    window = CrspWindow(W1.decision, W1.entry, W2.exit)
    evidence = outcomes(bridged_rows(), [event(W1.exit)])
    path = crsp_window_path(evidence, 1, window)
    assert path.growth == 1.1
    assert path.terminal == ("2020-03-03", 0.6)
    (period,) = evaluate_crsp_holdings(evidence, (window,), ((1,),), 10)
    assert period.net_return == pytest.approx(1.1 * 0.6 / 1.001 - 1)
    assert period.held == ()
    assert len(period.receivables) == 1


@pytest.mark.parametrize("price_flag", ["MP", "SU"])
def test_missing_exit_cannot_borrow_a_future_return_or_supply_a_fill(price_flag: str) -> None:
    rows = [
        daily(1, W1.entry, 0.0),
        daily(1, "2020-02-04", 0.1),
        missing(W1.exit, dlyprcflg=price_flag),
        daily(1, "2020-03-03", 0.2, dlyprevdt="2020-02-04", dlyretdurflg="P1"),
    ]
    with pytest.raises(ValueError, match="CRSP-011"):
        crsp_window_path(outcomes(rows, []), 1, W1)


@pytest.mark.parametrize("price_flag", ["MP", "SU"])
def test_missing_entry_is_not_a_filled_position_even_when_later_span_exists(
    price_flag: str,
) -> None:
    rows = bridged_rows()
    rows[0] = missing(W1.entry, dlyprcflg=price_flag)
    with pytest.raises(ValueError, match="CRSP-004"):
        crsp_window_path(outcomes(rows, []), 1, W1)


@pytest.mark.parametrize("price_flag", ["MP", "SU"])
def test_unsupported_long_gap_is_not_accepted_as_an_arbitrary_vendor_span(price_flag: str) -> None:
    days = tuple(f"2020-06-{day:02}" for day in range(2, 13))
    records = {(1, day): missing(day, dlyprcflg=price_flag) for day in days[:-1]}
    records[(1, days[-1])] = daily(1, W1.entry, 0.2, dlyretdurflg="P10")
    with pytest.raises(ValueError, match="CRSP-009"):
        crsp_compound_held(records, 1, "2020-06-01", days)


FLAGGED_ENTRY = "2020-06-01"
FLAGGED_DAYS = ("2020-06-02", "2020-06-03")


@pytest.mark.parametrize(
    ("records", "refusal"),
    [
        (
            {
                (1, FLAGGED_DAYS[0]): daily(
                    1, W1.entry, 0.1, dlyprevdt=FLAGGED_ENTRY, dlyretmissflg="MV"
                )
            },
            HeldRefusal(
                "span MV/DU", "[CRSP-009] held return for 1 on 2020-06-02 has an invalid span"
            ),
        ),
        (
            {
                (1, FLAGGED_DAYS[0]): missing(FLAGGED_DAYS[0]),
                (1, FLAGGED_DAYS[1]): daily(1, W1.entry, 0.1, dlyprevdt=FLAGGED_ENTRY),
            },
            HeldRefusal(
                "span NA/DU after a price gap",
                "[CRSP-009] held return for 1 on 2020-06-03 has an invalid span",
            ),
        ),
        (
            {},
            HeldRefusal("absent row", "[CRSP-009] held return for 1 on 2020-06-02 is absent"),
        ),
        (
            {(1, FLAGGED_DAYS[0]): missing(FLAGGED_DAYS[0])},
            HeldRefusal(
                "price gap tail", "[CRSP-011] held return for 1 ends with an unresolved price gap"
            ),
        ),
    ],
)
def test_resolution_names_its_first_refusal_and_keeps_no_partial_span(
    records: dict[tuple[int, str], JSONObject], refusal: HeldRefusal
) -> None:
    days = FLAGGED_DAYS[: max(1, len(records))]
    assert crsp_resolve_held(records, 1, FLAGGED_ENTRY, days) == ((), refusal)
    with pytest.raises(ValueError, match=re.escape(refusal.message)):
        crsp_held_spans(records, 1, FLAGGED_ENTRY, days)


def test_two_separate_gaps_do_not_accumulate_or_double_count_their_returns() -> None:
    days = ("2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05")
    records = {(1, days[0]): missing(days[0]), (1, days[2]): missing(days[2])}
    records[(1, days[1])] = daily(1, W1.entry, 0.1, dlyprevdt="2020-06-01", dlyretdurflg="P1")
    records[(1, days[3])] = daily(1, W1.entry, -0.2, dlyprevdt=days[1], dlyretdurflg="P1")
    assert crsp_compound_held(records, 1, "2020-06-01", days) == pytest.approx(0.88)


def test_mixed_suspension_and_missing_price_span_has_only_one_observed_value() -> None:
    days = ("2020-06-02", "2020-06-03", "2020-06-04")
    records = {
        (1, days[0]): missing(days[0], dlyprcflg="SU"),
        (1, days[1]): missing(days[1]),
        (1, days[2]): daily(1, W1.entry, -0.2, dlyprevdt="2020-06-01", dlyretdurflg="P2"),
    }
    assert crsp_resolve_held(records, 1, "2020-06-01", days) == (((2, 0.8),), None)
    assert crsp_resolve_held(records, 1, "2020-06-01", days[:1]) == (
        (),
        HeldRefusal(
            "price gap tail", "[CRSP-011] held return for 1 ends with an unresolved price gap"
        ),
    )


@pytest.mark.parametrize(("price_flag", "gap"), [("MP", 1), ("SU", 9)])
def test_real_extract_labels_use_the_same_span_without_future_universe_filtering(
    tmp_path: Path,
    price_flag: str,
    gap: int,
) -> None:
    manifest, _ = write_sources(
        tmp_path, missing_price_rows(price_flag=price_flag, gap=gap), study_protocol()
    )
    sources = load_crsp_sources(manifest)
    panel = build_crsp_panel(sources, "2019-02-01", "2019-02-28")
    assert tuple(feature.permno for feature in panel[0].features) == (8, 7, 6, 5, 4, 3)
    labels = crsp_study_rows(index_crsp_outcomes(sources), panel, ("momentum", "liquidity"))
    original = pre_entry_rows()["returns"]
    values = [1.0] + [
        prod(
            1 + require_float(row, "dlyret")
            for row in original
            if row["permno"] == permno
            and "2019-03-01" < require_str(row, "dlycaldt") <= "2019-04-01"
        )
        for permno in (7, 6, 5, 4, 3)
    ]
    average = sum(values) / 6
    assert [int(labels.labels.flat[index]) for index in range(len(values))] == [
        int(value > average) for value in values
    ]
    path = crsp_window_path(index_crsp_outcomes(sources), 7, panel[0].window)
    assert path.growth == pytest.approx(values[1])
