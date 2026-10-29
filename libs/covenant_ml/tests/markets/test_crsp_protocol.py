"""Real frozen protocol codecs and independent calendar/valuation examples."""

from __future__ import annotations

from datetime import date, timedelta
from math import inf, nan
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.crsp_calendar import (
    CrspWindow,
    crsp_member_on,
    crsp_terminal_value,
    crsp_windows,
    require_crsp_execution_price,
)
from covenant_ml.markets.crsp_protocol import decode_crsp_protocol, encode_crsp_protocol
from covenant_ml.markets.validation import read_document


def protocol_document() -> JSONObject:
    """Read the committed design so tests cannot silently use a different protocol."""
    return read_document(Path(__file__).parents[2] / "docs" / "crsp-monthly-v1.json")


def calendar_rows() -> tuple[tuple[str, str], ...]:
    """Enumerate an explicit four-month calendar with a holiday and Saturday session."""
    start = date(2023, 12, 1)
    end = date(2024, 3, 31)
    days = tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))
    holidays = {"2023-12-25", "2024-01-01"}
    return tuple(
        (
            day.isoformat(),
            "Y"
            if (day.weekday() < 5 and day.isoformat() not in holidays)
            or day.isoformat() == "2024-01-06"
            else "N",
        )
        for day in days
    )


def terminal_document() -> JSONObject:
    """A dated valuation received after its conventional daily return row."""
    return {
        "delistingdt": "2024-01-05",
        "deldlydt": "2024-01-08",
        "delamtdt": "2024-01-15",
        "delnextdt": None,
        "delret": -0.4,
        "delretmisstype": "NA",
    }


def test_frozen_protocol_roundtrip_preserves_dates_and_cost_order() -> None:
    document = protocol_document()
    protocol = decode_crsp_protocol(document)
    assert encode_crsp_protocol(protocol) == document
    assert protocol["holdout_start"] == "2022-01-01"
    assert protocol["holdout_end"] == "2024-12-31"
    assert protocol["coverage_end"] == "2025-02-28"
    assert protocol["cost_bps"] == (10, 5, 25)
    assert protocol["share_codes"] == (10, 11)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("extra", 1),
        ("membership", "stocknames"),
        ("share_codes", [11, 10]),
        ("train_start", "2015-01-01"),
        ("min_price", nan),
        ("min_price", -1.0),
        ("min_price", inf),
        ("selection_fraction", 0.0),
        ("selection_fraction", nan),
        ("selection_fraction", 1.1),
        ("version", 2),
        ("liquidity_window", 253),
    ],
)
def test_invalid_protocol_is_refused(key: str, value: JSONValue) -> None:
    document = protocol_document()
    document[key] = value
    with pytest.raises(ValueError):
        decode_crsp_protocol(document)


def test_calendar_uses_declared_sessions_and_first_close_of_following_months() -> None:
    windows = crsp_windows(calendar_rows(), "2023-12-01", "2024-01-31")
    assert windows == (
        CrspWindow("2023-12-29", "2024-01-02", "2024-02-01"),
        CrspWindow("2024-01-31", "2024-02-01", "2024-03-01"),
    )
    # Trading flags, not weekdays, decide the endpoints.
    rows = tuple(
        (day, "N" if day < "2024-01-06" and day[:7] == "2024-01" else flag)
        for day, flag in calendar_rows()
    )
    assert crsp_windows(rows, "2023-12-01", "2023-12-31")[0].entry == "2024-01-06"


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "reversed",
        "first",
        "last",
        "gap",
        "duplicate",
        "flag",
        "no_sessions",
    ],
)
def test_incomplete_calendar_never_infers_a_missing_session(case: str) -> None:
    rows = calendar_rows()
    start, end = "2023-12-01", "2024-01-31"
    if case == "empty":
        rows = ()
    elif case == "reversed":
        start, end = end, start
    elif case == "first":
        rows = rows[1:]
    elif case == "last":
        rows = rows[:-1]
    elif case == "gap":
        rows = rows[:10] + rows[11:]
    elif case == "duplicate":
        rows = rows[:10] + rows[9:]
    elif case == "flag":
        rows = ((rows[0][0], "X"), *rows[1:])
    else:
        rows = tuple((day, "N" if day[:7] == "2024-01" else flag) for day, flag in rows)
    with pytest.raises(ValueError, match="CRSP-002"):
        crsp_windows(rows, start, end)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2023-11-30", "2024-01-31"),
        ("2023-12-01", "2024-04-01"),
        ("2023-12-01", "2024-02-29"),
        ("2023-12-01", "2023-12-05"),
    ],
)
def test_calendar_requires_decisions_and_every_exit(start: str, end: str) -> None:
    with pytest.raises(ValueError, match="CRSP-002"):
        crsp_windows(calendar_rows(), start, end)


def test_membership_uses_inclusive_dates_and_rejects_ambiguous_intervals() -> None:
    protocol = decode_crsp_protocol(protocol_document())
    row: JSONObject = {
        "permno": 1,
        "namedt": "2024-01-01",
        "nameendt": "2024-01-31",
        "shrcd": 10,
        "exchcd": 1,
    }
    assert crsp_member_on((row,), 1, "2024-01-01", protocol)
    assert crsp_member_on((row,), 1, "2024-01-31", protocol)
    assert not crsp_member_on((row,), 1, "2024-02-01", protocol)
    assert not crsp_member_on((row,), 2, "2024-01-31", protocol)
    with pytest.raises(ValueError, match="CRSP-003"):
        crsp_member_on((row, row), 1, "2024-01-31", protocol)
    row["shrcd"] = 12
    assert not crsp_member_on((row,), 1, "2024-01-31", protocol)
    row["shrcd"], row["exchcd"] = 11, 4
    assert not crsp_member_on((row,), 1, "2024-01-31", protocol)
    row["nameendt"] = "2023-12-31"
    with pytest.raises(ValueError, match="CRSP-003"):
        crsp_member_on((row,), 1, "2024-01-31", protocol)


def test_execution_price_is_a_trade_not_absolute_quote_or_terminal_amount() -> None:
    protocol = decode_crsp_protocol(protocol_document())
    assert require_crsp_execution_price({"dlyprc": 25.0, "dlyprcflg": "TR"}, protocol) == 25.0
    for price, flag in ((-25.0, "BA"), (25.0, "DP"), (nan, "TR"), (0.0, "TR"), (-1.0, "TR")):
        with pytest.raises(ValueError, match="CRSP-004"):
            require_crsp_execution_price({"dlyprc": price, "dlyprcflg": flag}, protocol)


def test_terminal_value_uses_dated_amount_and_returns_one_multiplier() -> None:
    row = terminal_document()
    assert crsp_terminal_value(row) == ("2024-01-15", 0.6)
    row["delnextdt"] = "2024-01-20"
    assert crsp_terminal_value(row) == ("2024-01-20", 0.6)
    row["delret"] = -1.0
    assert crsp_terminal_value(row) == ("2024-01-20", 0.0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("delretmisstype", "DM"),
        ("delret", nan),
        ("delret", -1.1),
        ("deldlydt", "2024-01-05"),
        ("delamtdt", "2024-01-04"),
    ],
)
def test_unresolved_terminal_value_fails(key: str, value: JSONValue) -> None:
    row = terminal_document()
    row[key] = value
    with pytest.raises(ValueError, match="CRSP-005"):
        crsp_terminal_value(row)


def test_terminal_dates_are_required_not_imputed() -> None:
    row = terminal_document()
    del row["delnextdt"]
    with pytest.raises(ValueError, match="CRSP-005"):
        crsp_terminal_value(row)
    row = terminal_document()
    row["delamtdt"] = None
    with pytest.raises((ValueError, TypeError)):
        crsp_terminal_value(row)
