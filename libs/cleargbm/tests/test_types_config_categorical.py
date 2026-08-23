"""Tests for the categorical_features config decode validation."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    JSONDict,
    JSONTypeError,
    JSONValue,
    decode_gradient_boosting_config,
)

# =============================================================================
# GradientBoostingModel Tests
# =============================================================================


class TestCategoricalFeaturesDecode:
    """Validation of the categorical_features index list."""

    @staticmethod
    def _raw(value: JSONValue) -> JSONDict:
        return {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": value,
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }

    def test_rejects_a_non_list(self) -> None:
        """A dict is neither a list nor null."""
        with pytest.raises(JSONTypeError, match="categorical_features must be list or None"):
            decode_gradient_boosting_config(self._raw({"a": 1}))

    def test_rejects_an_empty_list(self) -> None:
        """Null is the only spelling of all-numeric."""
        with pytest.raises(ValueError, match="non-empty"):
            decode_gradient_boosting_config(self._raw([]))

    def test_rejects_a_non_int_element(self) -> None:
        """Strings and bools are not feature indices."""
        with pytest.raises(JSONTypeError, match=r"categorical_features\[1\] must be int"):
            decode_gradient_boosting_config(self._raw([0, "one"]))
        with pytest.raises(JSONTypeError, match=r"categorical_features\[0\] must be int"):
            decode_gradient_boosting_config(self._raw([True]))

    def test_rejects_a_negative_index(self) -> None:
        """Feature indices are non-negative."""
        with pytest.raises(ValueError, match=r"must be >= 0, got -1"):
            decode_gradient_boosting_config(self._raw([-1]))

    def test_rejects_an_unsorted_list(self) -> None:
        """Strictly ascending is the one canonical spelling of a set."""
        with pytest.raises(ValueError, match="strictly ascending"):
            decode_gradient_boosting_config(self._raw([2, 1]))
        with pytest.raises(ValueError, match="strictly ascending"):
            decode_gradient_boosting_config(self._raw([1, 1]))
