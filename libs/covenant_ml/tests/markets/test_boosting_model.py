"""Real boosted models respond to past data without reading future observations."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from covenant_ml.markets.boosting_model import feature_matrix, fit_boosting
from covenant_ml.markets.forecast_model import float_values
from covenant_ml.markets.forecast_records import (
    ForecastRow,
    decode_forecast_row,
    encode_forecast_row,
)

from ._forecast_fixtures import forecast_row


def training_rows() -> tuple[ForecastRow, ...]:
    return tuple(
        forecast_row(date(2026, 1, 1) + timedelta(days=day), yes)
        for day in range(80)
        for yes in (0, 1)
    )


def test_real_boosting_past_control_and_native_roundtrip(tmp_path: Path) -> None:
    rows = training_rows()
    original = fit_boosting(rows, tmp_path / "original.txt", 20, 4, True)
    predicted = original.booster_.predict(feature_matrix(rows, True))
    assert float_values(predicted)[0] < 0.5 < float_values(predicted)[1]
    documents = [encode_forecast_row(row) for row in rows]
    for document, row in zip(documents, rows, strict=True):
        document["yes"] = 1 - row["yes"]
    changed = tuple(decode_forecast_row(document) for document in documents)
    control = fit_boosting(changed, tmp_path / "control.txt", 20, 4, True)
    opposite = control.booster_.predict(feature_matrix(rows, True))
    assert float_values(opposite)[1] < 0.5 < float_values(opposite)[0]
    assert not np.array_equal(predicted, opposite)
    from covenant_ml.backends.lightgbm.backend import _BoosterFactory

    constructor: _BoosterFactory = __import__("lightgbm").Booster
    loaded = constructor(model_file=str(tmp_path / "original.txt"))
    assert np.array_equal(loaded.predict(feature_matrix(rows, True)), predicted)
    with pytest.raises(FileExistsError, match="BOOST-002"):
        fit_boosting(rows, tmp_path / "original.txt", 20, 4, True)
    assert np.array_equal(loaded.predict(feature_matrix(rows, True)), predicted)


def test_boosting_refuses_single_class_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BOOST-001"):
        fit_boosting((forecast_row(),), tmp_path / "invalid.txt", 20, 4, True)
    assert not (tmp_path / "invalid.txt").exists()


@pytest.mark.parametrize("rounds,leaves", [(0, 4), (20, 3)])
def test_undeclared_capacity_fails(tmp_path: Path, rounds: int, leaves: int) -> None:
    with pytest.raises(ValueError, match="BOOST-007"):
        fit_boosting(training_rows(), tmp_path / "bad.txt", rounds, leaves, False)
    assert not (tmp_path / "bad.txt").exists()
