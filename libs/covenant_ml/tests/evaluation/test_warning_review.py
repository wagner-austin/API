"""Real ranking and paired review evaluation, with hand-computed batch outcomes."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
)

from covenant_ml.evaluation.warning_panel import WarningRow, decode_warning_row
from covenant_ml.evaluation.warning_review import (
    compare_warning_reviews,
    decode_warning_review,
    encode_warning_review,
    select_warning_reviews,
    warning_review_rates,
)

from .test_warning_panel import row_document


def evidence(borrower: str, breach: int | None, shift: int = 0) -> WarningRow:
    """Build a validated test opportunity with explicit future evidence."""
    document = row_document(borrower, shift)
    require_dict(document, "followup")["first_breach_ms"] = breach
    return decode_warning_row(document)


def review_document() -> JSONObject:
    """Return a batch with two alerts, one detected breach and one missed breach."""
    return {
        "eligible": 4,
        "budget": 2,
        "breaches": 2,
        "selected": ["C", "A"],
        "lead_times_ms": [40],
    }


def test_ranking_is_order_invariant_and_budget_is_capacity() -> None:
    scores = {"C": 4.0, "B": 2.0, "A": 2.0, "D": -1.0}
    assert select_warning_reviews(scores, 2) == ("C", "A")
    assert select_warning_reviews(dict(reversed(tuple(scores.items()))), 2) == ("C", "A")
    assert select_warning_reviews(scores, 10) == ("C", "A", "B", "D")
    assert select_warning_reviews(scores, 1) == ("C",)


@pytest.mark.parametrize("budget", [0, -1, True])
def test_invalid_budget(budget: int) -> None:
    with pytest.raises(ValueError, match="WARN-010"):
        select_warning_reviews({"A": 0.5}, budget)


@pytest.mark.parametrize(
    "scores",
    [{}, {"": 1.0}, {" A": 1.0}, {"A": float("nan")}, {"A": float("inf")}],
)
def test_invalid_scores(scores: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="WARN-011"):
        select_warning_reviews(scores, 1)


def test_paired_batch_replay_and_future_outcomes_cannot_change_selection(tmp_path: Path) -> None:
    rows = (evidence("A", 70), evidence("B", 90), evidence("C", None), evidence("D", None))
    models = {
        "ratio": {"A": 2.0, "B": 2.0, "C": 4.0, "D": -1.0},
        "logistic": {"A": 0.8, "B": 0.9, "C": 0.1, "D": 0.2},
    }
    results = compare_warning_reviews(rows, models, 2, 110)
    ratio = results["ratio"]
    assert encode_warning_review(ratio) == review_document()
    assert warning_review_rates(ratio) == (0.5, 0.5, 1)
    assert results["logistic"]["selected"] == ("B", "A")
    assert results["logistic"]["lead_times_ms"] == (60, 40)
    assert warning_review_rates(results["logistic"]) == (1.0, 1.0, 0)
    path = tmp_path / "reviews.json"
    path.write_text(dump_json_str(encode_warning_review(ratio)), encoding="utf-8")
    assert (
        decode_warning_review(narrow_json_to_dict(load_json_str(path.read_text("utf-8")))) == ratio
    )
    changed = tuple(evidence(row["disclosure"]["borrower"], None) for row in rows)
    changed_results = compare_warning_reviews(changed, models, 2, 110)
    for model in models:
        assert changed_results[model]["selected"] == results[model]["selected"]
        assert warning_review_rates(changed_results[model]) == (0.0, None, 2)
    assert compare_warning_reviews(tuple(reversed(rows)), models, 2, 110) == results
    assert models["ratio"] == {"A": 2.0, "B": 2.0, "C": 4.0, "D": -1.0}


def test_undersized_batch_reports_actual_denominator_and_boundary_lead() -> None:
    results = compare_warning_reviews(
        (evidence("A", 100), evidence("B", None)),
        {"ratio": {"A": 4.0, "B": 2.0}, "model": {"A": 0.3, "B": 0.9}},
        10,
        110,
    )
    assert results["model"]["lead_times_ms"] == (70,)
    assert results["model"]["budget"] == 10
    assert warning_review_rates(results["model"]) == (0.5, 1.0, 1)
    assert decode_warning_review(encode_warning_review(results["model"])) == results["model"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("eligible", 0),
        ("budget", 0),
        ("breaches", -1),
        ("breaches", 5),
        ("selected", ["A"]),
        ("selected", ["A", "A"]),
        ("selected", ["", "B"]),
        ("selected", [" A", "B"]),
        ("lead_times_ms", [1, 2, 3]),
        ("breaches", 4),
        ("lead_times_ms", [0]),
        ("lead_times_ms", [2**53]),
    ],
)
def test_impossible_result_rejected(key: str, value: JSONValue) -> None:
    document = review_document()
    document[key] = value
    with pytest.raises(ValueError, match="WARN-012"):
        decode_warning_review(document)


@pytest.mark.parametrize(
    "key,value",
    [("eligible", True), ("selected", [1]), ("lead_times_ms", [True]), ("budget", None)],
)
def test_wrong_result_types_rejected(key: str, value: JSONValue) -> None:
    document = review_document()
    document[key] = value
    with pytest.raises(TypeError):
        decode_warning_review(document)


@pytest.mark.parametrize("case", ["empty", "one-model", "duplicate", "windows", "keys", "name"])
def test_comparison_requires_identical_batches(case: str) -> None:
    rows: tuple[WarningRow, ...] = (evidence("A", None), evidence("B", 70))
    models = {"ratio": {"A": 1.0, "B": 2.0}, "model": {"A": 0.2, "B": 0.5}}
    if case == "empty":
        rows = ()
    elif case == "one-model":
        del models["ratio"]
    elif case == "duplicate":
        rows = (rows[0], rows[0])
    elif case == "windows":
        rows = (rows[0], evidence("B", None, 1))
    elif case == "keys":
        models["ratio"] = {"A": 1.0, "C": 2.0}
    else:
        models[" "] = models.pop("ratio")
    with pytest.raises(ValueError, match="WARN-013"):
        compare_warning_reviews(rows, models, 1, 111)


@pytest.mark.parametrize("case", ["censored", "horizon", "availability"])
def test_comparison_refuses_immature_evidence(case: str) -> None:
    document = row_document()
    followup = require_dict(document, "followup")
    as_of = 110
    if case == "censored":
        followup.update(first_breach_ms=None, observed_through_ms=80)
    elif case == "horizon":
        followup.update(observed_through_ms=80, available_ms=80)
        as_of = 80
    else:
        as_of = 109
    rows = (decode_warning_row(document),)
    with pytest.raises(ValueError, match="WARN-014"):
        compare_warning_reviews(rows, {"ratio": {"A": 1.0}, "model": {"A": 0.5}}, 1, as_of)
