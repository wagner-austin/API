"""Rejected purchases retain cash without changing selection or erasing held losses."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_list,
)
from platform_core.testing import sample_run_fingerprint

from covenant_ml.markets.crsp_calendar import CrspWindow
from covenant_ml.markets.crsp_holdings import crsp_window_path, evaluate_crsp_holdings
from covenant_ml.markets.research_trial import execute_trial
from covenant_ml.markets.research_trial_records import decode_research_trial

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import pre_entry_rows, study_protocol
from ._encrypted_fixtures import create_access, evaluator_keys, seal_extract
from ._trial_fixtures import frozen_request, trial_document
from .test_crsp_execution_boundaries import event
from .test_crsp_holdings import W1, W2, daily, drift_rows, outcomes


def rejected_entry() -> JSONObject:
    """One terminal record, explicitly unsuitable for execution."""
    return daily(1, W1.entry, -0.4, dlydelflg="Y", dlyretdurflg="DD", dlyprcflg="DP")


@pytest.mark.parametrize("cost_bps", [5, 10, 25])
def test_all_rejected_orders_retain_cash_without_fees_or_terminal_losses(cost_bps: int) -> None:
    terminal = event(W1.decision)
    terminal.update({"delret": None, "delretmisstype": "DM", "delamtdt": "2030-01-01"})
    evidence = outcomes([rejected_entry()], [terminal])
    path = crsp_window_path(evidence, 1, W1)
    assert (path.filled, path.growth, path.terminal) == (False, 1.0, None)
    (period,) = evaluate_crsp_holdings(evidence, (W1,), ((1,),), cost_bps)
    assert period.selected == period.unfilled == (1,)
    assert (period.cash, period.turnover, period.cost, period.net_return) == (1.0, 0.0, 0.0, 0.0)
    assert period.held == ()
    assert period.receivables == ()


@pytest.mark.parametrize("cost_bps", [5, 10, 25])
def test_mixed_rejection_preserves_equal_cash_slot_and_charges_only_actual_trades(
    cost_bps: int,
) -> None:
    rows = [rejected_entry()]
    rows.extend([daily(2, W1.entry, 0.0), daily(2, "2020-02-04", 0.1), daily(2, W1.exit, 0.0)])
    (period,) = evaluate_crsp_holdings(
        outcomes(rows, [event(W1.decision)]), (W1,), ((1, 2),), cost_bps
    )
    rate = cost_bps / 10000
    # Cash x, stock purchase x and purchase fee rate*x must sum to one.
    each = 1 / (2 + rate)
    gross_end = 2.1 * each
    actual_trades = each + 1.1 * each
    assert period.selected == (1, 2)
    assert period.unfilled == (1,)
    assert period.cash == pytest.approx(1 / 2.1)
    ((permno, weight),) = period.held
    assert permno == 2
    assert weight == pytest.approx(1.1 / 2.1)
    assert period.turnover == pytest.approx(actual_trades)
    assert period.cost == pytest.approx(rate * actual_trades)
    assert period.net_return == pytest.approx(gross_end - rate * 1.1 * each - 1)


def test_retained_cash_rebalances_next_month_without_replacing_the_rejected_order() -> None:
    rows = [rejected_entry(), *[row for row in drift_rows() if row["permno"] != 1]]
    first, second = evaluate_crsp_holdings(
        outcomes(rows, [event(W1.decision)]), (W1, W2), ((1, 2), (2, 3)), 10
    )
    first_each = 1 / 2.001
    # First stock was flat: cash and stock have equal values. At the next close,
    # sell some stock 2 and buy stock 3, so actual turnover is first_each.
    start = 2 * first_each
    each = (start - 0.001 * first_each) / 2
    end_before_sale = each * (0.95 + 1.02)
    assert first.unfilled == (1,)
    assert second.unfilled == ()
    assert second.cash == 0.0
    assert second.turnover == pytest.approx((first_each + end_before_sale) / start)
    assert second.cost == pytest.approx(0.001 * second.turnover)
    assert second.net_return == pytest.approx(end_before_sale * 0.999 / start - 1)


def test_all_rejected_next_selection_sells_old_positions_and_retains_net_cash() -> None:
    rows = [row for row in drift_rows() if row["permno"] == 2]
    rows.append(daily(1, W2.entry, -0.4, dlydelflg="Y", dlyretdurflg="DD", dlyprcflg="DP"))
    _, final = evaluate_crsp_holdings(
        outcomes(rows, [event(W2.decision)]), (W1, W2), ((2,), (1,)), 10
    )
    assert (final.selected, final.unfilled, final.held) == ((1,), (1,), ())
    assert (final.cash, final.turnover, final.cost) == (1.0, 1.0, 0.001)
    assert final.net_return == pytest.approx(-0.001)


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "different_date", "trade", "duration", "order"]
)
def test_terminal_entry_requires_consistent_evidence(fault: str) -> None:
    row = rejected_entry()
    events = [event(W1.decision)]
    window = W1
    if fault == "missing":
        events = []
    elif fault == "duplicate":
        events.append(dict(events[0]))
    elif fault == "different_date":
        events[0]["delistingdt"] = "2020-01-30"
    elif fault == "trade":
        row["dlyprcflg"] = "TR"
    elif fault == "duration":
        row["dlyretdurflg"] = "DU"
    else:
        window = CrspWindow(W1.entry, W1.entry, W1.exit)
        events[0]["delistingdt"] = W1.entry
    with pytest.raises(ValueError, match="CRSP-010"):
        crsp_window_path(outcomes([row], events), 1, window)


def test_confirmed_rejection_does_not_hide_missing_returns_of_another_held_security() -> None:
    evidence = outcomes([rejected_entry(), daily(2, W1.entry, 0.0)], [event(W1.decision)])
    with pytest.raises(ValueError, match="held return for 2"):
        evaluate_crsp_holdings(evidence, (W1,), ((1, 2),), 10)


def test_existing_position_cannot_be_reclassified_as_unfilled_cash() -> None:
    rows = drift_rows()
    for row in rows:
        if row["permno"] == 1 and row["dlycaldt"] == W2.entry:
            row.update({"dlydelflg": "Y", "dlyretdurflg": "DD", "dlyprcflg": "DP"})
    # Without an event inside its first holding window, the held loss is unknown.
    with pytest.raises(ValueError, match="held return for 1"):
        evaluate_crsp_holdings(outcomes(rows, []), (W1, W2), ((1, 2), (2, 3)), 10)


def test_encrypted_monthly_study_publishes_rejected_orders_and_cleans_plaintext(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest, _ = write_sources(source, pre_entry_rows(), study_protocol())
    access = create_access(tmp_path)
    sealed = seal_extract(manifest, tmp_path / "encrypted", access, "train-validation")
    evaluation: JSONObject = {
        "design": study_protocol(),
        "stopping_months": 6,
        "fold_months": 12,
        "confidence": 0.9,
    }
    trial = decode_research_trial(frozen_request(evaluation, sealed, ["momentum", "liquidity"]))
    output = tmp_path / "result" / "evaluation.json"
    record = execute_trial(
        trial,
        sample_run_fingerprint(image_digest="d" * 64),
        "c" * 64,
        output,
        keys=evaluator_keys(tmp_path),
        scratch=tmp_path,
    )
    assert record.is_file()
    payload = narrow_json_to_dict(load_json_str(output.read_text()))
    assert payload["kind"] == "research-monthly-validation-v2"
    universe = [
        narrow_json_to_dict(item)
        for item in require_list(payload, "results")
        if narrow_json_to_dict(item)["arm"] == "equal_weight"
        and narrow_json_to_dict(item)["cost_bps"] == 10
    ]
    assert len(universe) == 1
    months = [narrow_json_to_dict(item) for item in require_list(universe[0], "months")]
    (period,) = [item for item in months if item["decision"] == "2019-02-28"]
    assert require_list(period, "selected") == [3, 4, 5, 6, 7, 8]
    assert require_list(period, "unfilled") == [8]
    assert require_float(period, "cash") > 0
    assert len(require_list(period, "held")) == 5
    # Stock 8 was owned in the preceding month. Its terminal receivable stays
    # locked at this entry close; rejecting a new order must not erase it or
    # create a second claim. Convert normalized weights back across the return.
    (previous,) = [item for item in months if item["decision"] == "2019-01-31"]
    (prior_claim,) = require_list(previous, "receivables")
    (current_claim,) = require_list(period, "receivables")
    prior = narrow_json_to_dict(prior_claim)
    current = narrow_json_to_dict(current_claim)
    assert (current["permno"], current["valued"]) == (8, "2019-03-01")
    assert require_float(current, "amount") == pytest.approx(
        require_float(prior, "amount") / (1 + require_float(period, "net_return"))
    )
    (following,) = [item for item in months if item["decision"] == "2019-03-29"]
    assert require_list(following, "receivables") == []
    assert list(tmp_path.glob("research-plaintext-*")) == []


def test_old_rule_identity_cannot_admit_the_new_evaluator(tmp_path: Path) -> None:
    document = trial_document(tmp_path)
    evaluation = require_dict(document, "evaluation")
    old_design = dict(require_dict(evaluation, "design"))
    old_design.pop("entry_rule")
    old_design.pop("cost_rule")
    old_design["version"] = 1
    old_evaluation = dict(evaluation)
    old_evaluation["design"] = old_design
    old_digest = sha256(
        dump_json_str(
            {
                "kind": "crsp-monthly-holdings-validation-v1",
                "settings": old_evaluation,
            }
        ).encode()
    ).hexdigest()
    assert old_digest != require_dict(document, "policy")["protocol_digest"]
    require_dict(document, "policy")["protocol_digest"] = old_digest
    require_dict(document, "proposal")["protocol_digest"] = old_digest
    with pytest.raises(ValueError, match="RESEARCH_TRIAL_LINEAGE"):
        execute_trial(
            decode_research_trial(document),
            sample_run_fingerprint(image_digest="d" * 64),
            "c" * 64,
            tmp_path / "result" / "out.json",
            keys=evaluator_keys(tmp_path),
            scratch=tmp_path,
        )
    assert not (tmp_path / "result").exists()
    assert list(tmp_path.glob("research-plaintext-*")) == []


def test_old_encrypted_protocol_is_refused_without_reinterpreting_its_data(tmp_path: Path) -> None:
    old_design = study_protocol()
    old_design.pop("entry_rule")
    old_design.pop("cost_rule")
    old_design["version"] = 1
    source = tmp_path / "source"
    source.mkdir()
    manifest, _ = write_sources(source, pre_entry_rows(), old_design)
    access = create_access(tmp_path)
    sealed = seal_extract(manifest, tmp_path / "encrypted", access, "train-validation")
    evaluation: JSONObject = {
        "design": study_protocol(),
        "stopping_months": 6,
        "fold_months": 12,
        "confidence": 0.9,
    }
    trial = decode_research_trial(frozen_request(evaluation, sealed, ["momentum"]))
    with pytest.raises(ValueError, match="CRSP-001"):
        execute_trial(
            trial,
            sample_run_fingerprint(image_digest="d" * 64),
            "c" * 64,
            tmp_path / "result" / "out.json",
            keys=evaluator_keys(tmp_path),
            scratch=tmp_path,
        )
    assert not (tmp_path / "result").exists()
    assert list(tmp_path.glob("research-plaintext-*")) == []
