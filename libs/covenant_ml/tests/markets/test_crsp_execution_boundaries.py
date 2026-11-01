"""Entry and exit proofs through the same path that constructs study labels."""

from __future__ import annotations

from math import prod
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, require_float, require_str

from covenant_ml.markets.crsp_extract import load_crsp_sources
from covenant_ml.markets.crsp_holdings import (
    crsp_window_path,
    evaluate_crsp_holdings,
    index_crsp_outcomes,
)
from covenant_ml.markets.crsp_panel import build_crsp_panel
from covenant_ml.markets.crsp_study import crsp_study_rows

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import pre_entry_rows, study_protocol
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
    assert path.filled
    assert period.turnover == pytest.approx(1 / 1.001)
    assert period.cost == pytest.approx(0.001 / 1.001)
    assert period.net_return == pytest.approx(0.6 / 1.001 - 1)
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
    assert period.net_return == pytest.approx(1.1 * 0.999 / 1.001 - 1)
    assert period.receivables == ()


def test_terminal_record_with_a_contradictory_trade_flag_refuses_evaluation() -> None:
    evidence = outcomes(
        [daily(1, W1.entry, 0.5, dlydelflg="Y", dlyretdurflg="DD", dlyprcflg="TR")],
        [event(W1.decision)],
    )
    with pytest.raises(ValueError, match="CRSP-010"):
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


def test_verified_extract_preserves_selection_and_labels_rejected_order_cash(
    tmp_path: Path,
) -> None:
    rows = pre_entry_rows()
    manifest, _ = write_sources(tmp_path, rows, study_protocol())
    sources = load_crsp_sources(manifest)
    panel = build_crsp_panel(sources, "2019-02-01", "2019-02-28")
    assert panel[0].window.entry == "2019-03-01"
    assert tuple(feature.permno for feature in panel[0].features) == (8, 7, 6, 5, 4, 3)
    labels = crsp_study_rows(index_crsp_outcomes(sources), panel, ("momentum", "liquidity"))
    # Five ordinary stock paths and one dollar kept in cash; no security is removed.
    multipliers = [1.0] + [
        prod(
            1 + require_float(row, "dlyret")
            for row in rows["returns"]
            if row["permno"] == permno
            and "2019-03-01" < require_str(row, "dlycaldt") <= "2019-04-01"
        )
        for permno in (7, 6, 5, 4, 3)
    ]
    benchmark = sum(multipliers) / 6
    assert [int(label) for label in labels.labels] == [
        int(value > benchmark) for value in multipliers
    ]
    assert labels.features.shape == (6, 2)
