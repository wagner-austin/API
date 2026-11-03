"""A delisting on an exit close with no closing trade, and its value dated after exit."""

from __future__ import annotations

from datetime import date
from math import prod
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_list,
    require_str,
)

from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_holdings import (
    CrspPath,
    crsp_window_path,
    evaluate_crsp_holdings,
    index_crsp_outcomes,
)
from covenant_ml.markets.crsp_panel import build_crsp_panel
from covenant_ml.markets.crsp_study import crsp_study_rows

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import (
    DELISTED,
    LATE_VALUE,
    UNSOLD_EXIT,
    study_protocol,
    unsold_exit_rows,
)
from ._crsp_worker_fixtures import run_monthly_worker
from .test_crsp_holdings import SESSIONS, W1, W2, W3, daily, outcomes


def unsold_rows() -> tuple[list[JSONObject], list[JSONObject]]:
    """Stock 1 delists on W1's quote-only exit; stock 2 trades every session."""
    rows = [
        daily(1, W1.entry, 0.0),
        daily(1, "2020-02-04", 0.1),
        daily(1, W1.exit, 0.0, dlyprcflg="BA"),
        daily(1, "2020-03-03", 0.5, dlydelflg="Y", dlyretdurflg="DD", dlyprcflg="DA"),
    ]
    rows.extend(daily(2, day, 0.1 if day == "2020-04-02" else 0.0) for day in SESSIONS)
    event: JSONObject = {
        "permno": 1,
        "delistingdt": W1.exit,
        "deldlydt": "2020-03-03",
        "delamtdt": "2020-03-05",
        "delnextdt": None,
        "delret": -0.4,
        "delretmisstype": "NA",
    }
    return rows, [event]


def test_delisting_on_an_unsold_exit_is_counted_once_and_locked_past_the_exit() -> None:
    rows, delists = unsold_rows()
    evidence = outcomes(rows, delists)
    old = crsp_window_path(evidence, 1, W1)
    assert old.filled
    assert old.growth == pytest.approx(1.1)
    # The quote-day return and the delisting return chain once; the +50%
    # daily delisting copy after the exit is never read.
    assert old.terminal == ("2020-03-05", 0.6)
    # A new order at that same quote is unfilled: no second claim.
    assert crsp_window_path(evidence, 1, W2) == CrspPath(False, 1.0, None)
    first, second, third = evaluate_crsp_holdings(evidence, (W1, W2, W3), ((1,), (1, 2), (2,)), 10)
    entry = 1 / 1.001
    assert first.held == ()
    (owed,) = first.receivables
    assert (owed.permno, owed.valued, owed.amount) == (1, "2020-03-05", 1.0)
    assert first.net_return == pytest.approx(entry * 0.66 - 1)
    # Valued after W2's entry close: nothing is reinvestable there.
    assert second.unfilled == (1,)
    assert (second.turnover, second.cost, second.net_return) == (0.0, 0.0, 0.0)
    assert second.receivables[0].valued == "2020-03-05"
    # Released at W3's entry and reinvested in stock 2, which gains 10%.
    assert third.receivables == ()
    assert third.turnover == pytest.approx((1 + 1.1) / 1.001)
    assert third.net_return == pytest.approx(1.1 * 0.999 / 1.001 - 1)


def test_events_inside_and_on_an_unsold_exit_are_ambiguous() -> None:
    rows, delists = unsold_rows()
    earlier = dict(delists[0])
    earlier["delistingdt"] = "2020-02-04"
    with pytest.raises(ValueError, match="two terminal events"):
        crsp_window_path(outcomes(rows, [*delists, earlier]), 1, W1)


def test_unsold_exit_terminal_value_must_itself_be_resolved() -> None:
    rows, delists = unsold_rows()
    delists[0]["delretmisstype"] = "DG"
    delists[0]["delret"] = None
    with pytest.raises(ValueError, match=r"CRSP-005.*delisting DG"):
        crsp_window_path(outcomes(rows, delists), 1, W1)


def _multiplier(rows: dict[str, list[JSONObject]], permno: int, entry: str, end: str) -> float:
    """Independently compound one security's daily returns over ``(entry, end]``."""
    return prod(
        1 + require_float(row, "dlyret")
        for row in rows["returns"]
        if row["permno"] == permno and entry < require_str(row, "dlycaldt") <= end
    )


def test_real_extract_dates_every_label_of_the_decision_by_its_latest_value(
    tmp_path: Path,
) -> None:
    rows = unsold_exit_rows()
    manifest, _ = write_sources(tmp_path, rows, study_protocol())
    sources = load_crsp_sources(manifest)
    panel = build_crsp_panel(sources, "2019-02-01", "2019-03-31")
    assert [decision.window.exit for decision in panel] == [UNSOLD_EXIT, "2019-05-01"]
    labels = crsp_study_rows(index_crsp_outcomes(sources), panel, ("momentum", "liquidity"))
    expected: list[int] = []
    for decision in panel:
        values: list[float] = []
        entry, close = decision.window.entry, decision.window.exit
        for feature in decision.features:
            if feature.permno != DELISTED:
                values.append(_multiplier(rows, feature.permno, entry, close))
            elif close == UNSOLD_EXIT:
                # Held through the quote-only exit, then the -25% delisting return.
                values.append(_multiplier(rows, DELISTED, entry, close) * 0.75)
            else:
                # Ordered at that same quote: unfilled, kept as cash.
                values.append(1.0)
        assert DELISTED in [feature.permno for feature in decision.features]
        benchmark = sum(values) / len(values)
        expected.extend(int(value > benchmark) for value in values)
    assert [int(labels.labels.flat[index]) for index in range(len(expected))] == expected
    # The universe benchmark depends on stock 8's late value, so no label of
    # the February decision is known before that value's date.
    first = int((labels.decisions == 0).sum())
    assert set(labels.available[:first].tolist()) == {date.fromisoformat(LATE_VALUE).toordinal()}
    assert set(labels.available[first:].tolist()) == {date(2019, 5, 1).toordinal()}


def test_real_worker_publishes_the_late_receivable_and_the_unfilled_reentry(
    tmp_path: Path,
) -> None:
    result, output = run_monthly_worker(tmp_path, unsold_exit_rows())
    assert list(tmp_path.glob("research-plaintext-*")) == []
    assert not (tmp_path / "keys" / "final.agekey").exists()
    assert (result.returncode, result.stderr) == (0, "")
    payload = narrow_json_to_dict(load_json_str(output.read_text()))
    (arm,) = [
        narrow_json_to_dict(item)
        for item in require_list(payload, "results")
        if narrow_json_to_dict(item)["arm"] == "equal_weight"
        and narrow_json_to_dict(item)["cost_bps"] == 10
    ]
    months = {
        require_str(month, "decision"): month
        for month in (narrow_json_to_dict(item) for item in require_list(arm, "months"))
    }
    february, march, april = months["2019-02-28"], months["2019-03-29"], months["2019-04-30"]
    # Six selected, five still held: stock 8 became the claim below.
    assert require_list(february, "selected") == [3, 4, 5, 6, 7, 8]
    assert len(require_list(february, "held")) == 5
    (claim,) = [narrow_json_to_dict(item) for item in require_list(february, "receivables")]
    assert (claim["permno"], claim["valued"]) == (DELISTED, LATE_VALUE)
    assert require_list(march, "unfilled") == [DELISTED]
    (locked,) = [narrow_json_to_dict(item) for item in require_list(march, "receivables")]
    assert (locked["permno"], locked["valued"]) == (DELISTED, LATE_VALUE)
    assert require_list(april, "receivables") == []
