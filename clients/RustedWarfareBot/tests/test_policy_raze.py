"""The raze watch and the brace latch: a sliding photograph, one arming.

What is pinned: the window slides (the oldest sample falls out), the
feature arithmetic matches the doom vocabulary computed by hand, the drop
pair reads the momentum figure exactly, and the latch arms once and holds
(law eight on a moving photograph).
"""

from __future__ import annotations

import math

import pytest

from rw_bot.policy.head import decode_head_model
from rw_bot.policy.raze import COLUMNS, BraceLatch, RazeError, RazeWatch


def _row(**overrides: int) -> tuple[int, ...]:
    values = dict.fromkeys(COLUMNS, 0)
    values.update(overrides)
    return tuple(values[name] for name in COLUMNS)


def _model_lines(*features: str) -> list[str]:
    return ['{"window": 4, "threshold": 0.7, "intercept": -1.0}', *features]


def test_a_window_below_two_is_refused_at_construction() -> None:
    """The half-split slope needs both halves; a one-sample window would
    divide by zero inside the feature arithmetic instead of here."""
    with pytest.raises(RazeError) as caught:
        RazeWatch(1)
    assert caught.value.code == "RW-RAZE-001"


def test_a_wrong_figure_count_is_a_wiring_bug() -> None:
    with pytest.raises(RazeError) as caught:
        RazeWatch(4).feed((1, 2, 3))
    assert caught.value.code == "RW-RAZE-001"


def test_features_before_the_window_fills_are_refused() -> None:
    watch = RazeWatch(4)
    watch.feed(_row())
    with pytest.raises(RazeError) as caught:
        watch.features()
    assert caught.value.code == "RW-RAZE-001"


def test_the_window_slides_and_the_stats_follow() -> None:
    """Hand arithmetic on army over a 4-window, then one more sample: the
    oldest value leaves every statistic."""
    watch = RazeWatch(4)
    for army in (2, 4, 6, 8):
        watch.feed(_row(army=army))
    feats = watch.features()
    assert feats["army_mean"] == pytest.approx(5.0)
    assert feats["army_last"] == 8.0
    assert feats["army_max"] == 8.0
    assert feats["army_min"] == 2.0
    # slope = mean(6, 8) - mean(2, 4) = 4
    assert feats["army_slope"] == pytest.approx(4.0)
    watch.feed(_row(army=10))
    slid = watch.features()
    # The 2 fell out: window is (4, 6, 8, 10).
    assert slid["army_mean"] == pytest.approx(7.0)
    assert slid["army_min"] == 4.0
    assert slid["army_slope"] == pytest.approx(4.0)


def test_the_drop_pair_reads_the_momentum_figure() -> None:
    """rival_army_drop is window max minus latest -- the exact figure the
    calibration probe measured ([[impossible-economy-problem]])."""
    watch = RazeWatch(4)
    for rival_army, extractors in ((10_000, 5), (14_000, 5), (12_000, 4), (9_500, 2)):
        watch.feed(_row(rival_army=rival_army, extractors=extractors))
    feats = watch.features()
    assert feats["rival_army_drop"] == pytest.approx(4_500.0)
    assert feats["extractors_drop"] == pytest.approx(3.0)


def test_the_score_is_the_standardized_logistic_through_the_shared_head() -> None:
    model = decode_head_model(
        _model_lines('{"name": "army_mean", "mean": 3.0, "std": 2.0, "coef": 2.0}')
    )
    watch = RazeWatch(4)
    for army in (2, 4, 6, 8):
        watch.feed(_row(army=army))
    # army_mean = 5 -> z = -1 + 2 * (5 - 3) / 2 = 1
    assert watch.score(model) == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_the_latch_arms_once_and_holds_through_quiet() -> None:
    """Law eight on a moving photograph: the first clearing score arms it,
    and later peaceful windows cannot disarm it."""
    model = decode_head_model(
        _model_lines('{"name": "extractors_drop", "mean": 0.0, "std": 1.0, "coef": 5.0}')
    )
    latch = BraceLatch(model)
    for extractors in (5, 5, 5):
        latch.feed(_row(extractors=extractors))
    assert latch.armed is False  # window of 4 still open
    latch.feed(_row(extractors=2))
    assert latch.armed is True
    for _ in range(6):
        latch.feed(_row(extractors=5))
    assert latch.armed is True


def test_a_calm_match_never_arms() -> None:
    model = decode_head_model(
        _model_lines('{"name": "extractors_drop", "mean": 0.0, "std": 1.0, "coef": 5.0}')
    )
    latch = BraceLatch(model)
    for _ in range(8):
        latch.feed(_row(extractors=5))
    assert latch.armed is False
