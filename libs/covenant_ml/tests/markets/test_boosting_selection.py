"""Real candidate fits, deterministic selection, exact accounting and availability."""

from pathlib import Path

import pytest
from platform_core.json_utils import (
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_int,
    require_list,
)

from covenant_ml.markets.boosting_selection import prediction_diagnostic, select_boosting
from covenant_ml.markets.forecast_records import decode_forecast_row, encode_forecast_row

from .test_boosting_model import training_rows


def test_selection_records_every_candidate_and_actual_argmin(tmp_path: Path) -> None:
    rows = training_rows()
    _, weather, evidence = select_boosting(rows, tmp_path / "selected.txt")
    candidates = [narrow_json_to_dict(v) for v in require_list(evidence, "candidates")]
    assert len(candidates) == 12
    losses = [require_float(require_dict(v, "validation"), "date_mean_brier") for v in candidates]
    winner = candidates[losses.index(min(losses))]
    assert evidence["selected_parameters"] == winner["parameters"]
    assert weather == winner["weather"]
    assert len(tuple(tmp_path.glob("*.txt"))) == 13
    assert evidence["unavailable_training_labels"] == 2
    assert len(require_list(evidence, "training_tickers")) == 138
    assert len(require_list(evidence, "validation_tickers")) == 20
    assert require_int(evidence, "latest_training_settlement_ms") < require_int(
        evidence, "earliest_validation_decision_ms"
    )
    for candidate in candidates:
        validation = require_dict(candidate, "validation")
        probabilities = tuple(
            require_float({"p": p}, "p") for p in require_list(validation, "predictions")
        )
        expected = (
            sum((p - r["yes"]) ** 2 for p, r in zip(probabilities, rows[-20:], strict=True)) / 20
        )
        assert require_float(validation, "date_mean_brier") == pytest.approx(expected, abs=1e-15)
    # More rounds can learn the real fixture's separation; this is not market evidence.
    assert losses[9] < losses[1]
    assert losses[0] == 0.25
    assert set(require_dict(evidence, "logistic_validation")) == {"market", "weather"}


def test_selection_rejects_short_history_and_unavailable_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"BOOST-008.*twenty"):
        select_boosting(training_rows()[:20], tmp_path / "short.txt")
    changed = []
    for row in training_rows():
        document = encode_forecast_row(row)
        document["settled_ms"] = training_rows()[-1]["settled_ms"]
        changed.append(decode_forecast_row(document))
    with pytest.raises(ValueError, match=r"BOOST-008.*no inner"):
        select_boosting(changed, tmp_path / "late.txt")
    assert tuple(tmp_path.iterdir()) == ()


def test_diagnostic_rejects_prediction_misalignment() -> None:
    with pytest.raises(ValueError, match="zip"):
        prediction_diagnostic(training_rows(), (0.5,))
