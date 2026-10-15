"""Malformed evidence and unavailable-label rejection without recovery paths."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from platform_core.json_utils import (
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.markets.interval_comparison import compare_intervals
from covenant_ml.markets.interval_inputs import (
    interval_observations,
    require_interval_comparison_plan,
    split_interval_rows,
)
from covenant_ml.markets.interval_scoring import interval_daily_estimate, interval_quoted_policy

from .test_interval_comparison import comparison_documents
from .test_interval_quotes import START


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("train_end_ts", START),
        ("train_end_ts", START + 16 * 86400),
        ("extra_friction", [0, 50]),
        ("inverse_regularization", 10.0),
        ("features", ["future_return"]),
    ],
)
def test_plan_drift_fails(field: str, value: JSONValue) -> None:
    _, quotes, plan = comparison_documents()
    plan[field] = value
    with pytest.raises(ValueError, match="IFORE-001"):
        require_interval_comparison_plan(plan, quotes)


@pytest.mark.parametrize(
    "mode", ["duplicate", "missing", "identity", "age", "future", "null_bid", "null_ask"]
)
def test_invalid_attachment_fails(mode: str) -> None:
    cohort, quotes, _ = comparison_documents()
    attachments = require_list(quotes, "contracts")
    first = narrow_json_to_dict(attachments[0])
    if mode == "duplicate":
        attachments.append(deepcopy(first))
    elif mode == "missing":
        attachments.pop()
    elif mode == "identity":
        first["event_ticker"] = "OTHER"
    elif mode == "age":
        first["quote_age_ms"] = 1
    elif mode == "future":
        require_dict(first, "quote")["timestamp_ms"] = (START + 360) * 1000
    else:
        require_dict(first, "quote")["bid" if mode == "null_bid" else "ask"] = None
    with pytest.raises(ValueError, match="IFORE-002"):
        interval_observations(cohort, quotes)


def test_all_exclusions_are_retained_with_identities() -> None:
    cohort, quotes, _ = comparison_documents()
    attached = require_list(quotes, "contracts")
    narrow_json_to_dict(attached[0])["status"] = "omitted_by_endpoint"
    require_dict(narrow_json_to_dict(attached[1]), "quote")["bid"] = 0
    require_dict(narrow_json_to_dict(attached[2]), "quote")["ask"] = 10000
    family = narrow_json_to_dict(require_list(cohort, "families")[0])
    narrow_json_to_dict(require_list(family, "markets")[3])["result"] = ""
    rows, exclusions = interval_observations(cohort, quotes)
    assert len(rows) == 28
    assert exclusions == (
        {"ticker": "KXBTC15M-D0-A", "reason": "quote:omitted_by_endpoint"},
        {"ticker": "KXBTC15M-D1-A", "reason": "boundary_quote"},
        {"ticker": "KXBTC15M-D2-A", "reason": "boundary_quote"},
        {"ticker": "KXBTC15M-D3-A", "reason": "nonbinary_or_unresolved_label"},
    )


def test_labels_at_or_after_first_decision_are_purged() -> None:
    cohort, quotes, plan = comparison_documents()
    family = narrow_json_to_dict(require_list(cohort, "families")[0])
    first = narrow_json_to_dict(require_list(family, "markets")[0])
    first["settlement_ts"] = datetime.fromtimestamp(START + 6 * 86400 - 600, UTC).isoformat()
    rows, _ = interval_observations(cohort, quotes)
    train, test, purged = split_interval_rows(rows, START + 6 * 86400)
    assert (len(train), len(test), purged) == (11, 20, ["KXBTC15M-D0-A"])
    changed = deepcopy(cohort)
    changed_family = narrow_json_to_dict(require_list(changed, "families")[0])
    narrow_json_to_dict(require_list(changed_family, "markets")[0])["result"] = "yes"
    assert compare_intervals(changed, quotes, plan) == compare_intervals(cohort, quotes, plan)
    with pytest.raises(ValueError, match="IFORE-003"):
        split_interval_rows(rows, START + 1)
    with pytest.raises(ValueError, match="IFORE-003"):
        split_interval_rows((*rows, rows[0]), START + 6 * 86400)
    with pytest.raises(ValueError, match="IFORE-003"):
        split_interval_rows(rows, START + 20 * 86400)
    with pytest.raises(ValueError, match="IFORE-003"):
        split_interval_rows(rows, START)


def test_invalid_daily_sample_and_policy_inputs_fail() -> None:
    cohort, quotes, _ = comparison_documents()
    rows, _ = interval_observations(cohort, quotes)
    for subset in (
        rows[:2],
        tuple(row for row in rows if row["close_ms"] // 86400000 != START // 86400 + 2),
    ):
        with pytest.raises(ValueError, match="IFORE-006"):
            interval_daily_estimate(tuple(0.0 for _ in subset), subset)
    for probabilities, friction in (
        ((), 0),
        (tuple(0.5 for _ in rows), 50),
        (tuple(float("nan") for _ in rows), 0),
    ):
        with pytest.raises(ValueError, match="IFORE-007"):
            interval_quoted_policy(rows, probabilities, friction)


def test_evaluation_missingness_cannot_move_training_availability_cutoff() -> None:
    cohort, quotes, plan = comparison_documents()
    family = narrow_json_to_dict(require_list(cohort, "families")[0])
    earlier = narrow_json_to_dict(require_list(family, "markets")[0])
    earlier["settlement_ts"] = datetime.fromtimestamp(START + 6 * 86400, UTC).isoformat()
    baseline = compare_intervals(cohort, quotes, plan)
    first_evaluation = narrow_json_to_dict(require_list(family, "markets")[6])
    first_evaluation["result"] = ""
    missing = compare_intervals(cohort, quotes, plan)
    assert missing["evaluation_count"] == 19
    assert missing["model"] == baseline["model"]
    assert missing["training_ids"] == baseline["training_ids"]
    assert (
        missing["purged_training_labels"] == baseline["purged_training_labels"] == ["KXBTC15M-D0-A"]
    )


def test_quote_clock_and_boolean_plan_drift_are_rejected() -> None:
    _, quotes, plan = comparison_documents()
    require_dict(quotes, "plan")["decision_offset_seconds"] = 600
    with pytest.raises(ValueError, match="IFORE-001"):
        require_interval_comparison_plan(plan, quotes)
    _, quotes, plan = comparison_documents()
    plan["schema_version"] = True
    with pytest.raises(TypeError, match="schema_version"):
        require_interval_comparison_plan(plan, quotes)
