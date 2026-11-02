"""Realized contracts from verified CIZ sources: spans, gaps and delistings."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, require_float, require_str

from covenant_ml.markets.crsp_calendar import crsp_month
from covenant_ml.markets.crsp_extract import CrspSources, load_crsp_sources
from covenant_ml.markets.crsp_holdings import index_crsp_outcomes
from covenant_ml.markets.crsp_return_path import crsp_compound_held, crsp_held_spans
from covenant_ml.options.crsp_paths import (
    ContractPath,
    PathCounts,
    contract_path,
    decode_path_counts,
    encode_path_counts,
    realized_contracts,
)
from covenant_ml.options.rates import RateObservation
from covenant_ml.options.types import ERR_PATH_SHAPE, ERR_RATE, AsianPutContract

from ..markets._crsp_fixtures import write_sources
from ..markets._crsp_study_fixtures import (
    DELISTED,
    missing_price_rows,
    study_protocol,
    study_rows,
)
from ._fixtures import listed, listed_ints

#: Four dates five sessions apart: twenty sessions, about four weeks.
MONTHLY = AsianPutContract(
    spot=100.0, strike=100.0, years=0.08, exercise_dates=4, sessions_per_date=5
)
FIRST, LAST = "2019-01-01", "2019-06-30"


def _sources(directory: Path, rows: dict[str, list[JSONObject]]) -> CrspSources:
    """Write the rows with the study protocol and load them through the verified reader."""
    path, _ = write_sources(directory, rows, study_protocol())
    return load_crsp_sources(path)


def _return(rows: dict[str, list[JSONObject]], permno: int, day: str) -> float:
    """The fixture's own return for one security and session."""
    (row,) = [
        row
        for row in rows["returns"]
        if row["permno"] == permno and require_str(row, "dlycaldt") == day
    ]
    return require_float(row, "dlyret")


def _rates() -> tuple[RateObservation, ...]:
    """A month-start yield of ``month`` percent through 2019."""
    return tuple(RateObservation(f"2019-{month:02d}-01", float(month)) for month in range(1, 13))


def test_ordinary_sessions_carry_their_own_returns(tmp_path: Path) -> None:
    rows = study_rows()
    outcomes = index_crsp_outcomes(_sources(tmp_path, rows))
    days = ("2019-01-02", "2019-01-03", "2019-01-04")
    path = contract_path(outcomes, 1, "2019-01-01", days)
    expected: list[float] = [_return(rows, 1, day) for day in days]
    assert listed(path.returns) == pytest.approx(expected)
    assert (path.gap_sessions, path.terminal, path.exclusion) == (0, False, None)


def test_a_missing_price_span_is_carried_flat_and_closed_once(tmp_path: Path) -> None:
    rows = missing_price_rows()
    outcomes = index_crsp_outcomes(_sources(tmp_path, rows))
    days = ("2019-03-11", "2019-03-12", "2019-03-13")
    spans = crsp_held_spans(outcomes.returns, 7, "2019-03-08", days)
    assert [index for index, _ in spans] == [1, 2]
    path = contract_path(outcomes, 7, "2019-03-08", days)
    closed = _return(rows, 7, "2019-03-12")
    expected: list[float] = [0.0, closed, _return(rows, 7, "2019-03-13")]
    assert listed(path.returns) == pytest.approx(expected)
    assert (path.gap_sessions, path.terminal, path.exclusion) == (1, False, None)
    growth = crsp_compound_held(outcomes.returns, 7, "2019-03-08", days)
    assert growth == pytest.approx((1 + closed) * (1 + _return(rows, 7, "2019-03-13")))


def test_a_delisting_applies_its_terminal_value_once_and_holds_it(tmp_path: Path) -> None:
    rows = study_rows()
    outcomes = index_crsp_outcomes(_sources(tmp_path, rows))
    days = ("2019-03-13", "2019-03-14", "2019-03-15", "2019-03-18", "2019-03-19")
    path = contract_path(outcomes, DELISTED, "2019-03-12", days)
    held = [_return(rows, DELISTED, "2019-03-13"), _return(rows, DELISTED, "2019-03-14")]
    # The daily copy on 2019-03-15 carries +30% and is never read.
    expected: list[float] = [*held, -0.25, 0.0, 0.0]
    assert listed(path.returns) == pytest.approx(expected)
    assert (path.gap_sessions, path.terminal, path.exclusion) == (0, True, None)


def test_a_delisting_on_the_last_session_ends_nothing_early(tmp_path: Path) -> None:
    rows = study_rows()
    outcomes = index_crsp_outcomes(_sources(tmp_path, rows))
    days = ("2019-03-13", "2019-03-14")
    path = contract_path(outcomes, DELISTED, "2019-03-12", days)
    assert (path.terminal, path.exclusion) == (False, None)
    expected: list[float] = [_return(rows, DELISTED, day) for day in days]
    assert listed(path.returns) == pytest.approx(expected)


LIFE = ("2019-03-13", "2019-03-14", "2019-03-15")


def _exclusion(path: ContractPath) -> tuple[str | None, int, int, bool]:
    """The fields an excluded path must carry: its reason and nothing measured."""
    return path.exclusion, int(path.returns.size), path.gap_sessions, path.terminal


def test_two_delistings_inside_one_life_exclude_the_contract(tmp_path: Path) -> None:
    rows = study_rows()
    second = dict(rows["delists"][0])
    second.update({"delistingdt": "2019-03-13", "deldlydt": "2019-03-14", "delamtdt": "2019-03-14"})
    rows["delists"].append(second)
    outcomes = index_crsp_outcomes(_sources(tmp_path, rows))
    path = contract_path(outcomes, DELISTED, "2019-03-12", LIFE)
    assert _exclusion(path) == ("two delistings", 0, 0, False)


def test_a_delisting_the_vendor_cannot_value_excludes_the_contract(tmp_path: Path) -> None:
    rows = study_rows()
    rows["delists"][0].update({"delretmisstype": "DM", "delret": None})
    outcomes = index_crsp_outcomes(_sources(tmp_path, rows))
    path = contract_path(outcomes, DELISTED, "2019-03-12", LIFE)
    assert _exclusion(path) == ("delisting DM", 0, 0, False)


def _flag(rows: dict[str, list[JSONObject]], permno: int, day: str) -> None:
    """Mark one fixture return as the vendor's missing corporate action value."""
    (row,) = [
        row
        for row in rows["returns"]
        if row["permno"] == permno and require_str(row, "dlycaldt") == day
    ]
    row["dlyretmissflg"] = "MV"


def test_a_return_the_vendor_flags_missing_excludes_the_contract(tmp_path: Path) -> None:
    rows = study_rows()
    _flag(rows, 1, "2019-01-03")
    outcomes = index_crsp_outcomes(_sources(tmp_path, rows))
    days = ("2019-01-02", "2019-01-03", "2019-01-04")
    path = contract_path(outcomes, 1, "2019-01-01", days)
    assert _exclusion(path) == ("span MV/DU", 0, 0, False)


def test_contracts_follow_the_panel_and_count_their_assumptions(tmp_path: Path) -> None:
    contracts, counts = realized_contracts(
        _sources(tmp_path, study_rows()), FIRST, LAST, MONTHLY, _rates()
    )
    assert counts["decisions"] == 6
    assert counts["contracts"] == len(contracts.securities) == len(contracts.returns)
    assert int(contracts.returns.size) == 20 * counts["contracts"]
    assert counts["terminal_contracts"] == 1
    assert (counts["gap_contracts"], counts["gap_sessions"]) == (0, 0)
    assert int(contracts.months.flat[0]) == crsp_month("2019-01-31")
    assert float(contracts.carry_rates.flat[0]) == pytest.approx(math.log(1.01))
    assert DELISTED in {int(value) for value in contracts.securities.flat}
    assert counts["excluded"] == {}


def test_an_excluded_contract_is_counted_and_leaves_the_rest_unchanged(tmp_path: Path) -> None:
    (tmp_path / "whole").mkdir()
    (tmp_path / "flagged").mkdir()
    whole, whole_counts = realized_contracts(
        _sources(tmp_path / "whole", study_rows()), FIRST, LAST, MONTHLY, _rates()
    )
    rows = study_rows()
    # Only the last decision's contract on security 3 lives through this session, and
    # no later decision's panel can read it, so nothing else may change.
    _flag(rows, 3, "2019-07-10")
    kept, counts = realized_contracts(
        _sources(tmp_path / "flagged", rows), FIRST, LAST, MONTHLY, _rates()
    )
    assert counts["excluded"] == {"span MV/DU": 1}
    assert counts["contracts"] == whole_counts["contracts"] - 1
    assert counts["decisions"] == whole_counts["decisions"]
    pairs = list(zip(listed_ints(whole.securities), listed_ints(whole.months), strict=True))
    flagged = (3, crsp_month("2019-06-28"))
    (dropped,) = [index for index, pair in enumerate(pairs) if pair == flagged]
    survivors = [pair for index, pair in enumerate(pairs) if index != dropped]
    kept_pairs = zip(listed_ints(kept.securities), listed_ints(kept.months), strict=True)
    assert list(kept_pairs) == survivors
    steps = int(whole.returns.shape[1])
    whole_values = listed(whole.returns)
    expected = [
        value
        for index, value in enumerate(whole_values)
        if not dropped * steps <= index < (dropped + 1) * steps
    ]
    assert listed(kept.returns) == expected


def test_a_gap_inside_a_contract_is_counted(tmp_path: Path) -> None:
    _, counts = realized_contracts(
        _sources(tmp_path, missing_price_rows()), "2019-02-01", "2019-02-28", MONTHLY, _rates()
    )
    assert (counts["gap_contracts"], counts["gap_sessions"]) == (1, 1)


def test_a_decision_before_any_rate_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=ERR_RATE):
        realized_contracts(_sources(tmp_path, study_rows()), FIRST, LAST, MONTHLY, _rates()[2:])


def test_a_contract_that_outlives_the_sources_is_refused(tmp_path: Path) -> None:
    # Forty-three sessions follow the December 2019 decision before the extract ends.
    long = AsianPutContract(
        spot=100.0, strike=100.0, years=0.2, exercise_dates=10, sessions_per_date=5
    )
    with pytest.raises(ValueError, match=ERR_PATH_SHAPE):
        realized_contracts(
            _sources(tmp_path, study_rows()), "2019-12-01", "2019-12-31", long, _rates()
        )


def test_path_counts_roundtrip() -> None:
    counts = PathCounts(
        decisions=6,
        contracts=36,
        gap_contracts=1,
        gap_sessions=2,
        terminal_contracts=1,
        excluded={"absent row": 2, "span MV/D1": 1},
    )
    assert decode_path_counts(encode_path_counts(counts)) == counts
