"""Entry and exit proofs through the same path that constructs study labels."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, require_str

from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_holdings import (
    crsp_window_path,
    evaluate_crsp_holdings,
    index_crsp_outcomes,
)
from covenant_ml.markets.crsp_panel import build_crsp_panel
from covenant_ml.markets.crsp_study import crsp_study_rows

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import DELISTED, study_protocol, study_rows
from .test_crsp_holdings import W1, daily, outcomes


def event(day: str) -> JSONObject:
    """A dated forty-percent loss for the requested boundary event."""
    return {
        "permno": 1,
        "delistingdt": day,
        "deldlydt": "2020-03-03" if day == W1.exit else "2020-02-04",
        "delamtdt": "2020-03-03" if day == W1.exit else "2020-02-04",
        "delnextdt": None,
        "delret": -0.4,
        "delretmisstype": "NA",
    }


def test_delisting_on_entry_counts_terminal_value_once_without_later_daily_rows() -> None:
    evidence = outcomes([daily(1, W1.entry, 0.0)], [event(W1.entry)])
    path = crsp_window_path(evidence, 1, W1)
    assert path.growth == 1.0
    assert path.terminal == ("2020-02-04", 0.6)
    (period,) = evaluate_crsp_holdings(evidence, (W1,), ((1,),), 10)
    assert period.turnover == 1.0
    assert period.cost == 0.001
    assert period.net_return == pytest.approx(0.999 * 0.6 - 1)
    assert period.held == ()
    (owed,) = period.receivables
    assert (owed.permno, owed.valued, owed.amount) == (1, "2020-02-04", 1.0)


def test_delisting_on_exit_allows_last_trade_sale_without_post_exit_terminal_loss() -> None:
    evidence = outcomes(
        [daily(1, W1.entry, 0.0), daily(1, "2020-02-04", 0.1), daily(1, W1.exit, 0.0)],
        [event(W1.exit)],
    )
    path = crsp_window_path(evidence, 1, W1)
    assert path.growth == pytest.approx(1.1)
    assert path.terminal is None
    (period,) = evaluate_crsp_holdings(evidence, (W1,), ((1,),), 10)
    assert period.net_return == pytest.approx(0.999 * 1.1 * 0.999 - 1)
    assert period.receivables == ()


@pytest.mark.parametrize("flag", ["DP", "TR"])
def test_terminal_record_before_entry_is_never_an_execution_even_with_trade_flag(
    flag: str,
) -> None:
    evidence = outcomes(
        [daily(1, W1.entry, 0.5, dlydelflg="Y", dlyretdurflg="DD", dlyprcflg=flag)],
        [event(W1.decision)],
    )
    with pytest.raises(ValueError, match=r"CRSP-004.*terminal record.*is not a fill"):
        crsp_window_path(evidence, 1, W1)


@pytest.mark.parametrize("day", [W1.entry, W1.exit])
def test_label_path_requires_trade_prices_at_both_ordinary_endpoints(day: str) -> None:
    rows = [daily(1, W1.entry, 0.0), daily(1, "2020-02-04", 0.1), daily(1, W1.exit, 0.0)]
    for row in rows:
        if row["dlycaldt"] == day:
            row["dlyprcflg"] = "BA"
    with pytest.raises(ValueError, match=r"CRSP-004.*positive closing trade"):
        crsp_window_path(outcomes(rows, []), 1, W1)


def test_missing_entry_fails_before_later_ordinary_returns_are_examined() -> None:
    with pytest.raises(ValueError, match="no execution record for 1 on 2020-02-03"):
        crsp_window_path(outcomes([], []), 1, W1)


def test_verified_extract_preserves_selection_and_refuses_pre_entry_delisting_label(
    tmp_path: Path,
) -> None:
    rows = study_rows()
    rows["returns"] = [
        row
        for row in rows["returns"]
        if row["permno"] != DELISTED or require_str(row, "dlycaldt") <= "2019-03-01"
    ]
    for row in rows["returns"]:
        if row["permno"] == DELISTED and row["dlycaldt"] == "2019-03-01":
            row.update({"dlydelflg": "Y", "dlyretdurflg": "DD", "dlyprcflg": "DP"})
    rows["delists"][0].update(
        {"delistingdt": "2019-02-28", "deldlydt": "2019-03-01", "delamtdt": "2019-03-01"}
    )
    manifest, _ = write_sources(tmp_path, rows, study_protocol())
    sources = load_crsp_sources(manifest)
    panel = build_crsp_panel(sources, "2019-02-01", "2019-02-28")
    assert panel[0].window.entry == "2019-03-01"
    assert tuple(feature.permno for feature in panel[0].features) == (8, 7, 6, 5, 4, 3)
    with pytest.raises(ValueError, match=r"CRSP-004.*terminal record for 8 on 2019-03-01"):
        crsp_study_rows(index_crsp_outcomes(sources), panel, ("momentum", "liquidity"))
