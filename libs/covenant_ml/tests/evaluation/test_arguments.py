"""Tests for parsing the evaluation entry points' structured flags."""

from __future__ import annotations

import pytest

from covenant_ml.benchmarking.factory import make_baseline_trainers
from covenant_ml.evaluation.arguments import (
    parse_fractions,
    parse_groups,
    parse_unit,
    resampling_settings,
    select_trainer,
)
from covenant_ml.evaluation.types import (
    ERR_INVALID_ARGUMENT,
    ERR_INVALID_CONFIDENCE,
    ERR_INVALID_FRACTION,
    ERR_INVALID_RESAMPLES,
    ResamplingUnit,
)

from ._fixtures import tiny_config


@pytest.mark.parametrize("unit", list(ResamplingUnit))
def test_every_unit_parses_from_its_own_word(unit: ResamplingUnit) -> None:
    assert parse_unit(unit.value) is unit


def test_an_unknown_unit_is_refused_naming_the_admitted_ones() -> None:
    with pytest.raises(ValueError, match="row, cluster, block"):
        parse_unit("company")


def test_settings_bind_every_flag() -> None:
    bound = resampling_settings(ResamplingUnit.BLOCK, 5, 100, 9, 0.9)
    assert bound == {
        "unit": ResamplingUnit.BLOCK,
        "block_length": 5,
        "n_resamples": 100,
        "resample_seed": 9,
        "confidence": 0.9,
    }


def test_settings_refuse_a_bad_confidence_before_any_fit() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_CONFIDENCE):
        resampling_settings(ResamplingUnit.ROW, 1, 100, 0, 1.0)


def test_settings_refuse_one_resample() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_RESAMPLES):
        resampling_settings(ResamplingUnit.ROW, 1, 1, 0, 0.95)


def test_groups_parse_named_and_bare_forms() -> None:
    assert parse_groups("X1;liquidity=X3, X4") == [
        {"name": "X1", "features": ["X1"]},
        {"name": "liquidity", "features": ["X3", "X4"]},
    ]


@pytest.mark.parametrize("spec", ["", "X1;", "=X1", "name=", "name=X1,,X2"])
def test_an_empty_group_or_feature_is_refused(spec: str) -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        parse_groups(spec)


def test_fractions_parse_in_order() -> None:
    assert parse_fractions("0.1, 0.5,1") == [0.1, 0.5, 1.0]


def test_a_fraction_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        parse_fractions("0.1,half")


def test_fractions_out_of_order_are_refused() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_FRACTION):
        parse_fractions("0.5,0.25")


def test_the_named_model_is_selected() -> None:
    trainers = make_baseline_trainers(tiny_config())
    assert select_trainer(trainers, "xgboost").model_name.value == "xgboost"


def test_an_unknown_model_is_refused_naming_the_available_ones() -> None:
    trainers = make_baseline_trainers(tiny_config())
    with pytest.raises(ValueError, match="cleargbm, lightgbm, xgboost"):
        select_trainer(trainers, "catboost")
