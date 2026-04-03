"""Tests for Optuna parameter sampling functions.

Tests sample_param_int, sample_param_float, and sample_param_str.
"""

from __future__ import annotations

from covenant_ml.optimizer.optuna_backend._sampling import (
    sample_param_float,
    sample_param_int,
    sample_param_str,
)
from covenant_ml.optimizer.types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    CategoricalStringSpec,
    FloatRangeSpec,
    IntRangeSpec,
)

from .conftest import FakeTrial

# =============================================================================
# Tests: Integer Parameter Sampling
# =============================================================================


def test_sample_param_int_range_spec() -> None:
    """sample_param_int handles IntRangeSpec correctly."""
    trial = FakeTrial(0)
    spec: IntRangeSpec = {"param_type": "int", "low": 3, "high": 10, "log_scale": False}
    result = sample_param_int(trial, "max_depth", spec)
    assert 3 <= result <= 10


def test_sample_param_int_categorical_spec() -> None:
    """sample_param_int handles CategoricalIntSpec correctly."""
    trial = FakeTrial(0)
    spec: CategoricalIntSpec = {"param_type": "categorical_int", "choices": (3, 5, 7, 10)}
    result = sample_param_int(trial, "max_depth", spec)
    assert result in (3, 5, 7, 10)


def test_sample_param_int_varies_by_trial() -> None:
    """sample_param_int returns different values for different trials."""
    spec: IntRangeSpec = {"param_type": "int", "low": 1, "high": 100, "log_scale": False}
    values = [sample_param_int(FakeTrial(i), "x", spec) for i in range(10)]
    assert len(set(values)) > 1


# =============================================================================
# Tests: Float Parameter Sampling
# =============================================================================


def test_sample_param_float_range_spec() -> None:
    """sample_param_float handles FloatRangeSpec correctly."""
    trial = FakeTrial(0)
    spec: FloatRangeSpec = {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True}
    result = sample_param_float(trial, "learning_rate", spec)
    assert 0.01 <= result <= 0.3


def test_sample_param_float_categorical_spec() -> None:
    """sample_param_float handles CategoricalFloatSpec correctly."""
    trial = FakeTrial(0)
    spec: CategoricalFloatSpec = {"param_type": "categorical_float", "choices": (0.01, 0.1, 0.3)}
    result = sample_param_float(trial, "learning_rate", spec)
    assert result in (0.01, 0.1, 0.3)


def test_sample_param_float_varies_by_trial() -> None:
    """sample_param_float returns different values for different trials."""
    spec: FloatRangeSpec = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    values = [sample_param_float(FakeTrial(i), "x", spec) for i in range(10)]
    assert len(set(values)) > 1


# =============================================================================
# Tests: String Parameter Sampling
# =============================================================================


def test_sample_param_str_returns_string() -> None:
    """sample_param_str returns string value from choices."""
    spec: CategoricalStringSpec = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    trial = FakeTrial(0)
    result = sample_param_str(trial, "boosting_type", spec)
    assert result in ("gbdt", "dart")


def test_sample_param_str_varies_by_trial() -> None:
    """sample_param_str returns different values for different trials."""
    spec: CategoricalStringSpec = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    trial0 = FakeTrial(0)
    trial1 = FakeTrial(1)
    result0 = sample_param_str(trial0, "boosting_type", spec)
    result1 = sample_param_str(trial1, "boosting_type", spec)
    # trial 0 selects index 0 (gbdt), trial 1 selects index 1 (dart)
    assert result0 == "gbdt"
    assert result1 == "dart"
