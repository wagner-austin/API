"""The shared head-model shape: one decoder, one scoring rule.

Extracted from the doom deployment when the second head arrived; what is
pinned here is the file contract every corpus-trained head rides in and
the standardized-logistic arithmetic both sides of train/serve parity
score through.
"""

from __future__ import annotations

import math

import pytest

from rw_bot.policy.head import HeadError, decode_head_model, score_features

_HEAD = '{"window": 4, "threshold": 0.7, "intercept": -1.0}'


def test_a_model_decodes_with_its_scalars_and_features() -> None:
    model = decode_head_model(
        [_HEAD, '{"name": "army_mean", "mean": 5.0, "std": 2.0, "coef": 1.5}']
    )
    assert model["window"] == 4
    assert model["threshold"] == 0.7
    assert model["intercept"] == -1.0
    assert model["features"]["army_mean"] == (5.0, 2.0, 1.5)


def test_an_empty_file_a_bad_window_and_a_bad_std_stop_loudly() -> None:
    with pytest.raises(HeadError) as caught:
        decode_head_model([])
    assert caught.value.code == "RW-HEAD-001"
    with pytest.raises(HeadError) as caught:
        decode_head_model(['{"window": 0, "threshold": 0.7, "intercept": 0.0}'])
    assert caught.value.code == "RW-HEAD-001"
    with pytest.raises(HeadError) as caught:
        decode_head_model([_HEAD, '{"name": "a", "mean": 0.0, "std": 0.0, "coef": 1.0}'])
    assert caught.value.code == "RW-HEAD-001"
    with pytest.raises(HeadError) as caught:
        decode_head_model([_HEAD])
    assert caught.value.code == "RW-HEAD-001"


def test_the_score_is_the_standardized_logistic() -> None:
    """Hand arithmetic: z = intercept + coef * (x - mean) / std."""
    model = decode_head_model([_HEAD, '{"name": "x", "mean": 3.0, "std": 2.0, "coef": 2.0}'])
    # x = 5 -> z = -1 + 2 * (5 - 3) / 2 = 1
    assert score_features(model, {"x": 5.0}) == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_an_unknown_feature_is_a_loud_train_serve_drift() -> None:
    model = decode_head_model([_HEAD, '{"name": "missing", "mean": 0.0, "std": 1.0, "coef": 1.0}'])
    with pytest.raises(HeadError) as caught:
        score_features(model, {"present": 1.0})
    assert caught.value.code == "RW-HEAD-002"
