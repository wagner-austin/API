"""Hand-calculated monthly holdings, dated terminal values and incomplete-evidence refusals."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_float, require_int, require_str

from covenant_ml.markets.crsp_calendar import CrspWindow
from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_holdings import (
    CrspOutcomes,
    crsp_window_path,
    evaluate_crsp_holdings,
    index_crsp_outcomes,
)
from covenant_ml.markets.crsp_protocol import decode_crsp_protocol

from ._crsp_fixtures import panel_protocol, source_rows, write_sources
from .test_crsp_protocol import protocol_document

SESSIONS = (
    "2020-02-03",
    "2020-02-04",
    "2020-03-02",
    "2020-03-03",
    "2020-04-01",
    "2020-04-02",
    "2020-05-01",
)
W1 = CrspWindow("2020-01-31", "2020-02-03", "2020-03-02")
W2 = CrspWindow("2020-02-28", "2020-03-02", "2020-04-01")
W3 = CrspWindow("2020-03-31", "2020-04-01", "2020-05-01")


def daily(permno: int, day: str, ret: float, **changes: JSONValue) -> JSONObject:
    """A tradable ordinary daily record whose predecessor is the previous declared session."""
    index = SESSIONS.index(day)
    row: JSONObject = {
        "permno": permno,
        "dlycaldt": day,
        "dlyprevdt": SESSIONS[index - 1] if index else None,
        "dlyret": ret,
        "dlyretmissflg": "NA",
        "dlyretdurflg": "DU",
        "dlydelflg": "N",
        "dlyprc": 10.0,
        "dlyprcflg": "TR",
        "dlyvol": 100.0,
    }
    row.update(changes)
    return row


def outcomes(rows: list[JSONObject], delists: list[JSONObject]) -> CrspOutcomes:
    """Index explicit evidence under the committed frozen design."""
    returns = {(require_int(row, "permno"), require_str(row, "dlycaldt")): row for row in rows}
    grouped: dict[int, list[JSONObject]] = {}
    for row in delists:
        grouped.setdefault(require_int(row, "permno"), []).append(row)
    return CrspOutcomes(
        decode_crsp_protocol(protocol_document()),
        SESSIONS,
        returns,
        {permno: tuple(items) for permno, items in grouped.items()},
    )


def drift_rows() -> list[JSONObject]:
    """Stock 1 gains 10%, stock 2 later loses 5% and stock 3 gains 2%."""
    return [
        daily(1, "2020-02-03", 0.0),
        daily(1, "2020-02-04", 0.10),
        daily(1, "2020-03-02", 0.0),
        daily(2, "2020-02-03", 0.0),
        daily(2, "2020-02-04", 0.0),
        daily(2, "2020-03-02", 0.0),
        daily(2, "2020-03-03", -0.05),
        daily(2, "2020-04-01", 0.0),
        daily(3, "2020-03-02", 0.0),
        daily(3, "2020-03-03", 0.02),
        daily(3, "2020-04-01", 0.0),
    ]


def terminal_rows() -> tuple[list[JSONObject], list[JSONObject]]:
    """Stock 4 delists after one session; its daily terminal copy must never compound."""
    rows = [
        daily(4, "2020-02-03", 0.0),
        daily(4, "2020-02-04", 0.0),
        daily(4, "2020-03-02", 0.5, dlydelflg="Y", dlyretdurflg="DD"),
    ]
    for day in SESSIONS:
        rows.append(daily(5, day, 0.10 if day == "2020-05-01" else 0.0))
    event: JSONObject = {
        "permno": 4,
        "delistingdt": "2020-02-04",
        "deldlydt": "2020-02-05",
        "delamtdt": "2020-03-02",
        "delnextdt": None,
        "delret": -0.4,
        "delretmisstype": "NA",
    }
    return rows, [event]


def test_drifted_weights_turnover_and_final_liquidation_match_hand_arithmetic() -> None:
    evidence = outcomes(drift_rows(), [])
    first, second = evaluate_crsp_holdings(evidence, (W1, W2), ((1, 2), (2, 3)), 10)
    # Entry: one unit of cash bought equally, 1.0 of notional at 10 bp.
    assert first.turnover == pytest.approx(1.0)
    assert first.cost == pytest.approx(0.001)
    assert first.net_return == pytest.approx(0.4995 * 1.1 + 0.4995 - 1)
    assert [permno for permno, _ in first.held] == [1, 2]
    assert first.held[0][1] == pytest.approx(0.54945 / 1.04895)
    assert first.held[1][1] == pytest.approx(0.4995 / 1.04895)
    assert first.receivables == ()
    # Rebalance from drifted 0.54945/0.4995 to 0.524475 each in stocks 2 and 3.
    traded = 0.54945 + (0.524475 - 0.4995) + 0.524475
    each = (1.04895 - 0.001 * traded) / 2
    before_exit = each * 0.95 + each * 1.02
    assert second.selected == (2, 3)
    assert [permno for permno, _ in second.held] == [2, 3]
    assert second.held[0][1] == pytest.approx(each * 0.95 / before_exit)
    assert second.turnover == pytest.approx((traded + before_exit) / 1.04895)
    assert second.cost == pytest.approx(0.001 * (traded + before_exit) / 1.04895)
    assert second.net_return == pytest.approx(before_exit * 0.999 / 1.04895 - 1)


def test_terminal_value_counts_once_and_waits_for_a_strictly_later_rebalance() -> None:
    rows, delists = terminal_rows()
    evidence = outcomes(rows, delists)
    path = crsp_window_path(evidence, 4, W1)
    assert path.growth == 1.0
    # 1 + (-0.4) is exactly the double 0.6, valued at the latest recorded date.
    assert path.terminal == ("2020-03-02", 0.6)
    selections = ((4, 5), (5,), (5,))
    first, second, third = evaluate_crsp_holdings(evidence, (W1, W2, W3), selections, 10)
    # The daily +50% delisting copy is ignored; -40% applies once.
    assert first.net_return == pytest.approx(0.4995 * 0.6 + 0.4995 - 1)
    ((held, weight),) = first.held
    assert held == 5
    assert weight == pytest.approx(0.4995 / 0.7992)
    (owed,) = first.receivables
    assert (owed.permno, owed.valued) == (4, "2020-03-02")
    assert owed.amount == pytest.approx(0.2997 / 0.7992)
    # Valued at 2020-03-02, which is not strictly before this entry: still locked cash.
    assert second.turnover == 0.0
    assert second.cost == 0.0
    assert second.net_return == pytest.approx(0.0)
    assert second.receivables[0].valued == "2020-03-02"
    # Released at 2020-04-01 and reinvested in stock 5, which then gains 10%.
    invested = 0.7992 - 0.001 * 0.2997
    assert third.turnover == pytest.approx((0.2997 + invested * 1.1) / 0.7992)
    assert third.receivables == ()
    assert third.net_return == pytest.approx(invested * 1.1 * 0.999 / 0.7992 - 1)


def test_index_reads_declared_sessions_and_keys_from_verified_files(tmp_path: Path) -> None:
    path, _ = write_sources(tmp_path, source_rows(), panel_protocol())
    indexed = index_crsp_outcomes(load_crsp_sources(path))
    assert indexed.sessions[:2] == ("2018-12-03", "2018-12-04")
    assert all(earlier < later for earlier, later in pairwise(indexed.sessions))
    assert indexed.returns[(1, "2019-01-02")]["dlyret"] == 0.1
    assert indexed.delists == {}


def test_index_groups_terminal_events_by_security(tmp_path: Path) -> None:
    rows = source_rows()
    _, events = terminal_rows()
    events[0]["permno"] = 1
    events[0]["delistingdt"] = "2020-03-02"
    rows["delists"] = events
    path, _ = write_sources(tmp_path, rows, panel_protocol())
    indexed = index_crsp_outcomes(load_crsp_sources(path))
    assert list(indexed.delists) == [1]
    assert indexed.delists[1][0]["delret"] == -0.4


def test_index_refuses_unordered_sessions(tmp_path: Path) -> None:
    rows = source_rows()
    calendar = rows["calendar"]
    calendar[2], calendar[3] = calendar[3], calendar[2]
    path, _ = write_sources(tmp_path, rows, panel_protocol())
    with pytest.raises(ValueError, match="CRSP-009"):
        index_crsp_outcomes(load_crsp_sources(path))


def _drop(rows: list[JSONObject], permno: int, day: str) -> list[JSONObject]:
    """Remove one security-session record."""
    return [row for row in rows if (row["permno"], row["dlycaldt"]) != (permno, day)]


def _change(
    rows: list[JSONObject], permno: int, day: str, **changes: JSONValue
) -> list[JSONObject]:
    """Replace one security-session record's named fields."""
    return [
        daily(permno, day, require_float(row, "dlyret"), **changes)
        if (row["permno"], row["dlycaldt"]) == (permno, day)
        else row
        for row in rows
    ]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (_drop(drift_rows(), 2, "2020-02-04"), "held return for 2 on 2020-02-04"),
        (_change(drift_rows(), 2, "2020-02-04", dlydelflg="Y"), "held return for 2"),
        (_change(drift_rows(), 2, "2020-02-04", dlyprevdt="2020-01-31"), "held return for 2"),
        (_change(drift_rows(), 2, "2020-02-04", dlyretmissflg="MR"), "held return for 2"),
        (_drop(drift_rows(), 3, "2020-03-02"), "no execution record for 3 on 2020-03-02"),
        (_change(drift_rows(), 3, "2020-03-02", dlyprcflg="BA"), "CRSP-004"),
        (_change(drift_rows(), 1, "2020-03-02", dlyprcflg="BA"), "CRSP-004"),
        (_change(drift_rows(), 3, "2020-04-01", dlyprcflg="BA"), "CRSP-004"),
        (
            _change(
                _change(drift_rows(), 1, "2020-02-04", dlyret=1e308), 1, "2020-03-02", dlyret=1e308
            ),
            "not finite",
        ),
    ],
)
def test_incomplete_held_evidence_publishes_no_outcome(
    rows: list[JSONObject], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_crsp_holdings(outcomes(rows, []), (W1, W2), ((1, 2), (2, 3)), 10)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("delamtdt", "2020-03-03", "dated after the exit"),
        ("delnextdt", "2020-03-04", "dated after the exit"),
        ("delretmisstype", "DM", "CRSP-005"),
        ("delret", -1.0, "portfolio value was exhausted"),
    ],
)
def test_unresolved_or_late_terminal_values_are_refused(
    key: str, value: JSONValue, message: str
) -> None:
    rows, delists = terminal_rows()
    delists[0][key] = value
    with pytest.raises(ValueError, match=message):
        evaluate_crsp_holdings(outcomes(rows, delists), (W1, W2, W3), ((4,), (5,), (5,)), 10)


def test_two_terminal_events_in_one_window_are_ambiguous() -> None:
    rows, delists = terminal_rows()
    second = dict(delists[0])
    second["delistingdt"] = "2020-02-03"
    with pytest.raises(ValueError, match="two terminal events"):
        evaluate_crsp_holdings(outcomes(rows, [*delists, second]), (W1,), ((4, 5),), 10)


@pytest.mark.parametrize(
    ("windows", "selections", "cost", "message"),
    [
        ((), (), 10, "one selection for every window"),
        ((W1, W2), ((1, 2),), 10, "one selection for every window"),
        ((W1, W3), ((1, 2), (2, 3)), 10, "previous exit"),
        ((W1,), ((),), 10, "nonempty and distinct"),
        ((W1,), ((1, 1),), 10, "nonempty and distinct"),
        ((W1,), ((1, 2),), 7, "frozen design declares"),
        ((CrspWindow("2020-01-31", "2020-02-05", "2020-03-02"),), ((1,),), 10, "sessions"),
    ],
)
def test_requests_outside_the_frozen_design_are_refused(
    windows: tuple[CrspWindow, ...],
    selections: tuple[tuple[int, ...], ...],
    cost: int,
    message: str,
) -> None:
    off_calendar = daily(1, "2020-02-04", 0.0)
    off_calendar["dlycaldt"] = "2020-02-05"
    rows = [*drift_rows(), off_calendar]
    with pytest.raises(ValueError, match=message):
        evaluate_crsp_holdings(outcomes(rows, []), windows, selections, cost)
