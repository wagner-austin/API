"""The doom model's deployment surface: decode, watch, score, latch.

The arithmetic under test is the training arithmetic -- the exporter fits
on features from this same class -- so these tests pin what the model both
learned and reads (log 2026-08-09, the replication verdict).
"""

from __future__ import annotations

import math

import pytest

from rw_bot.policy.doom import COLUMNS, DoomError, DoomLatch, DoomWatch
from rw_bot.policy.head import HeadError, decode_head_model

_HEAD = '{"window": 4, "threshold": 0.7, "intercept": -1.0}'


def _row(**overrides: int) -> list[int]:
    values = dict.fromkeys(COLUMNS, 0)
    values.update(overrides)
    return [values[name] for name in COLUMNS]


def _model_lines(*features: str) -> list[str]:
    return [_HEAD, *features]


def test_the_watch_computes_the_training_features_exactly() -> None:
    """Golden arithmetic: means, last, extremes, the half-split slope, the
    ratios and the naval timing trio, on a four-sample window computed by
    hand."""
    watch = DoomWatch(4)
    watch.feed(_row(army=2, income=10, rival_income=30, worth=100, rival=200))
    watch.feed(_row(army=4, income=10, rival_income=30, worth=100, rival=200))
    watch.feed(_row(army=6, income=10, rival_income=30, worth=100, rival=200, navy_seen=1))
    watch.feed(_row(army=8, income=10, rival_income=30, worth=100, rival=200, navy_blood=2))
    feats = watch.features()
    assert feats["army_mean"] == 5.0
    assert feats["army_last"] == 8.0
    assert feats["army_max"] == 8.0
    assert feats["army_min"] == 2.0
    # slope: mean(6, 8) - mean(2, 4) = 7 - 3
    assert feats["army_slope"] == 4.0
    assert feats["rival_income_ratio"] == 3.0
    assert feats["rival_worth_ratio"] == 2.0
    assert feats["first_navy_contact"] == 2.0
    assert feats["first_navy_blood"] == 3.0
    assert feats["navy_pressure"] == 0.25


def test_a_navyless_window_reports_the_window_as_never() -> None:
    watch = DoomWatch(2)
    watch.feed(_row())
    watch.feed(_row())
    feats = watch.features()
    assert feats["first_navy_contact"] == 2.0
    assert feats["first_navy_blood"] == 2.0


def test_samples_past_the_window_are_ignored() -> None:
    """The features are a photograph; the model's moment does not move."""
    watch = DoomWatch(2)
    watch.feed(_row(army=2))
    watch.feed(_row(army=4))
    watch.feed(_row(army=1000))
    assert watch.features()["army_mean"] == 3.0


def test_a_wrong_figure_count_and_an_early_read_stop_loudly() -> None:
    watch = DoomWatch(2)
    with pytest.raises(DoomError) as caught:
        watch.feed([1, 2, 3])
    assert caught.value.code == "RW-DOOM-002"
    watch.feed(_row())
    with pytest.raises(DoomError) as caught:
        watch.features()
    assert caught.value.code == "RW-DOOM-002"


def test_the_score_is_the_standardized_logistic() -> None:
    """One feature, hand arithmetic: z = intercept + coef * (x - mean) / std."""
    model = decode_head_model(
        _model_lines('{"name": "army_mean", "mean": 3.0, "std": 2.0, "coef": 2.0}')
    )
    watch = DoomWatch(4)
    for army in (2, 4, 6, 8):
        watch.feed(_row(army=army))
    # army_mean = 5 -> z = -1 + 2 * (5 - 3) / 2 = 1
    assert watch.score(model) == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_a_model_naming_an_unknown_feature_stops_loudly() -> None:
    """Train/serve drift surfaces through the shared scorer's own code."""
    model = decode_head_model(
        _model_lines('{"name": "no_such_feature", "mean": 0.0, "std": 1.0, "coef": 1.0}')
    )
    watch = DoomWatch(2)
    watch.feed(_row())
    watch.feed(_row())
    with pytest.raises(HeadError) as caught:
        watch.score(model)
    assert caught.value.code == "RW-HEAD-002"


def test_the_latch_scores_once_at_the_window_and_holds() -> None:
    """Law eight in miniature: one decision for a match-reshaping response.
    The latch arms at the window when the score clears the threshold and
    ignores everything after."""
    model = decode_head_model(
        _model_lines('{"name": "navy_seen_mean", "mean": 0.0, "std": 1.0, "coef": 5.0}')
    )
    latch = DoomLatch(model)
    assert latch.armed is False
    latch.feed(_row(navy_seen=3))
    assert latch.armed is False  # window of 4 still open
    for _ in range(3):
        latch.feed(_row(navy_seen=3))
    assert latch.armed is True
    # Later peaceful samples cannot disarm it.
    latch.feed(_row())
    assert latch.armed is True


def test_a_quiet_window_leaves_the_latch_unarmed() -> None:
    model = decode_head_model(
        _model_lines('{"name": "navy_seen_mean", "mean": 0.0, "std": 1.0, "coef": 5.0}')
    )
    latch = DoomLatch(model)
    for _ in range(5):
        latch.feed(_row())
    assert latch.armed is False
