"""Equal-date scoring, serial uncertainty and real fee accounting."""

from datetime import date, timedelta
from math import sqrt

import pytest
from platform_core.json_utils import require_dict, require_float

from covenant_ml.markets.boosting_scores import daily_means, score_predictions

from ._forecast_fixtures import forecast_row


def test_dates_have_equal_weight_and_alignment_is_required() -> None:
    rows = (forecast_row(), forecast_row(), forecast_row(date(2026, 4, 4)))
    assert daily_means((0.0, 1.0, 0.0), rows) == (0.5, 0.0)
    with pytest.raises(ValueError, match="zip"):
        daily_means((0.0,), rows)


@pytest.mark.parametrize("mode", ["identical", "perfect", "serial"])
def test_real_uncertainty_intervals_do_not_hide_serial_failure(mode: str) -> None:
    rows = tuple(forecast_row(date(2026, 4, 1) + timedelta(days=i), 0) for i in range(20))
    boosted = tuple(0.5 for _ in rows)
    if mode == "perfect":
        boosted = tuple(0.0 for _ in rows)
    elif mode == "serial":
        boosted = tuple(0.0 if i < 10 else sqrt(0.3) for i in range(20))
    report = score_predictions(
        rows,
        {
            "raw_market": (0.5,) * 20,
            "logistic": (0.5,) * 20,
            "boosting": boosted,
        },
    )
    comparison = require_dict(require_dict(report, "comparisons"), "logistic")
    assert comparison["both_upper_bounds_negative"] is (mode == "perfect")
    if mode == "serial":
        assert require_float(require_dict(comparison, "date_clustered"), "ci_high") < 0
        assert require_float(require_dict(comparison, "daily_newey_west"), "ci_high") > 0
    if mode == "perfect":
        policy = require_dict(require_dict(require_dict(report, "arms"), "boosting"), "policy")
        assert policy["trades"] == 20
        assert policy["pnl"] == 76000
        assert policy["extra_cent_pnl"] == 74000


def test_scoring_refuses_short_samples_and_wrong_arms() -> None:
    with pytest.raises(ValueError, match="BOOST-003"):
        score_predictions((forecast_row(),), {})
    rows = tuple(forecast_row(date(2026, 4, day)) for day in range(1, 10))
    with pytest.raises(ValueError, match="BOOST-003"):
        score_predictions(rows, {})
