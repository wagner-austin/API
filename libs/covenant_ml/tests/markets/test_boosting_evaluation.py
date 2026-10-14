"""Actual expanding-window models obey label availability and future-data isolation."""

from pathlib import Path

import pytest
from platform_core.json_utils import (
    narrow_json_to_dict,
    require_dict,
    require_int,
    require_list,
    require_str,
)

from covenant_ml.markets.boosting_evaluation import evaluate_boosting
from covenant_ml.markets.forecast_records import (
    decode_comparison_plan,
    decode_forecast_row,
    encode_comparison_plan,
    encode_forecast_row,
)

from ._forecast_fixtures import comparison_plan
from .test_boosting_model import training_rows


@pytest.mark.parametrize("mutation", ["labels", "features"])
def test_future_changes_preserve_past_models_and_predictions(tmp_path: Path, mutation: str) -> None:
    document = encode_comparison_plan(comparison_plan())
    document.update(start="2026-01-01", train_end="2026-03-01", end="2026-03-21")
    plan = decode_comparison_plan(document)
    rows = training_rows()
    original = evaluate_boosting(rows, plan, ("2026-03-11", "2026-03-21"), tmp_path / "first")
    documents = [encode_forecast_row(row) for row in rows]
    for row, raw in zip(rows, documents, strict=True):
        if row["event_day"] > "2026-03-11":
            if mutation == "labels":
                raw["yes"] = 1 - row["yes"]
            else:
                raw["weather_p"] = 1 - row["weather_p"]
    mutated = tuple(decode_forecast_row(raw) for raw in documents)
    second = evaluate_boosting(mutated, plan, ("2026-03-11", "2026-03-21"), tmp_path / "second")
    first_folds, second_folds = require_list(original, "folds"), require_list(second, "folds")
    assert first_folds[0] == second_folds[0]
    assert (tmp_path / "first/fold-1.txt").read_bytes() == (
        tmp_path / "second/fold-1.txt"
    ).read_bytes()
    assert (tmp_path / "first/fold-2.txt").read_bytes() == (
        tmp_path / "second/fold-2.txt"
    ).read_bytes()
    last = narrow_json_to_dict(first_folds[1])
    changed = narrow_json_to_dict(second_folds[1])
    assert last["logistic_model"] == changed["logistic_model"]
    assert last["scores"] != changed["scores"]
    old_arms = require_dict(require_dict(last, "scores"), "arms")
    new_arms = require_dict(require_dict(changed, "scores"), "arms")
    if mutation == "labels":
        for name in old_arms:
            assert (
                require_dict(old_arms, name)["predictions"]
                == require_dict(new_arms, name)["predictions"]
            )
    else:
        assert (
            require_dict(old_arms, "boosting")["predictions"]
            != require_dict(new_arms, "boosting")["predictions"]
        )
    selected = []
    for raw_fold in first_folds:
        fold = narrow_json_to_dict(raw_fold)
        assert require_int(fold, "latest_training_settlement_ms") < require_int(
            fold, "earliest_evaluation_decision_ms"
        )
        assert fold["unavailable_training_labels"] == 2
        assert require_dict(fold, "scores")["primary_reference"] == "logistic"
        selected.extend(require_list(fold, "evaluation_rows"))
    assert len(selected) == 40
    assert {require_str(narrow_json_to_dict(row), "ticker") for row in selected} == {
        row["ticker"] for row in rows if row["event_day"] > plan["train_end"]
    }
    with pytest.raises(FileExistsError):
        evaluate_boosting(rows, plan, ("2026-03-11", "2026-03-21"), tmp_path / "first")


@pytest.mark.parametrize("ends", [("2026-02-01", "2026-03-21"), ("2026-03-11", "2026-03-20")])
def test_invalid_folds_fail_before_creating_artifacts(
    tmp_path: Path, ends: tuple[str, str]
) -> None:
    document = encode_comparison_plan(comparison_plan())
    document.update(start="2026-01-01", train_end="2026-03-01", end="2026-03-21")
    with pytest.raises(ValueError, match="BOOST-004"):
        evaluate_boosting(training_rows(), decode_comparison_plan(document), ends, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
