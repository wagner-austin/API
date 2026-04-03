"""Tests for Rust native training adapters.

Verifies that the full Rust training loop produces a functional model,
and that hook wiring/unwiring works correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cleargbm import _hooks_native
from cleargbm._rust_native_adapters import (
    _config_to_rust_dict,
    _load_native_functions,
    _rust_predict_proba_model,
    _rust_predict_raw_model,
    _rust_train_gradient_boosting,
    unwire_native_hooks,
    wire_native_hooks,
)
from cleargbm.ensemble import (
    predict_proba_native,
    predict_raw_native,
    train_gradient_boosting_native,
)
from cleargbm.types import GradientBoostingConfig

# Load native functions so adapter functions can reference Rust bindings.
# Must run after imports but before any test calls adapter functions.
_load_native_functions()

# =============================================================================
# Fixtures
# =============================================================================


def _make_config() -> GradientBoostingConfig:
    """Build a minimal training config for testing.

    Returns:
        GradientBoostingConfig with small values for fast training.
    """
    return GradientBoostingConfig(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        max_bins=8,
        subsample=1.0,
        random_state=42,
        track_contributions=False,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=1,
        early_stopping_rounds=None,
    )


def _make_binary_data() -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    tuple[str, ...],
]:
    """Create small binary classification dataset.

    Returns:
        Tuple of (features, labels, feature_names).
    """
    x: NDArray[np.float64] = np.array(
        (
            (1.0, 2.0),
            (2.0, 3.0),
            (3.0, 1.0),
            (4.0, 4.0),
            (5.0, 0.5),
            (0.5, 5.0),
            (1.5, 1.5),
            (3.5, 3.5),
        ),
        dtype=np.float64,
    )
    y: NDArray[np.int64] = np.array((0, 0, 1, 1, 1, 0, 0, 1), dtype=np.int64)
    return x, y, ("feat_a", "feat_b")


# =============================================================================
# Config conversion tests
# =============================================================================


class TestConfigToRustDict:
    """Tests for _config_to_rust_dict conversion."""

    def test_extracts_12_keys(self) -> None:
        """Output dict has exactly the 12 keys Rust expects."""
        config = _make_config()
        rust_dict = _config_to_rust_dict(config)
        expected_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_samples_split",
            "min_samples_leaf",
            "max_bins",
            "subsample",
            "random_state",
            "reg_alpha",
            "reg_lambda",
            "monotonic_constraints",
            "early_stopping_rounds",
        }
        assert set(rust_dict.keys()) == expected_keys

    def test_excludes_python_only_fields(self) -> None:
        """Python-only fields are excluded from Rust dict."""
        config = _make_config()
        rust_dict = _config_to_rust_dict(config)
        assert "max_features" not in rust_dict
        assert "track_contributions" not in rust_dict
        assert "n_jobs" not in rust_dict

    def test_preserves_values(self) -> None:
        """Values are copied correctly from config."""
        config = _make_config()
        rust_dict = _config_to_rust_dict(config)
        assert rust_dict["n_estimators"] == 5
        assert rust_dict["max_depth"] == 2
        assert rust_dict["learning_rate"] == 0.1
        assert rust_dict["random_state"] == 42

    def test_converts_monotonic_constraints_tuple_to_list(self) -> None:
        """Monotonic constraints tuple is converted to list for Rust."""
        config = _make_config()
        config_with_mc = GradientBoostingConfig(
            **{**config, "monotonic_constraints": (0, 1)},
        )
        rust_dict = _config_to_rust_dict(config_with_mc)
        assert rust_dict["monotonic_constraints"] == [0, 1]

    def test_none_monotonic_constraints_stays_none(self) -> None:
        """None monotonic constraints stays None."""
        config = _make_config()
        rust_dict = _config_to_rust_dict(config)
        assert rust_dict["monotonic_constraints"] is None

    def test_none_early_stopping_stays_none(self) -> None:
        """None early stopping rounds stays None."""
        config = _make_config()
        rust_dict = _config_to_rust_dict(config)
        assert rust_dict["early_stopping_rounds"] is None

    def test_early_stopping_value_preserved(self) -> None:
        """Early stopping rounds value is preserved."""
        config = _make_config()
        config_with_es = GradientBoostingConfig(
            **{**config, "early_stopping_rounds": 10},
        )
        rust_dict = _config_to_rust_dict(config_with_es)
        assert rust_dict["early_stopping_rounds"] == 10


# =============================================================================
# Native training adapter tests
# =============================================================================


class TestRustTrainGradientBoosting:
    """Tests for _rust_train_gradient_boosting adapter."""

    def test_returns_native_model(self) -> None:
        """Training produces a functional model that can predict."""
        x, y, names = _make_binary_data()
        config = _make_config()
        model = _rust_train_gradient_boosting(x, y, None, None, config, names)
        proba = _rust_predict_proba_model(model, x)
        assert len(proba) == 8

    def test_with_validation_data(self) -> None:
        """Training with validation data produces a functional model."""
        x, y, names = _make_binary_data()
        config = _make_config()
        model = _rust_train_gradient_boosting(x, y, x, y, config, names)
        proba = _rust_predict_proba_model(model, x)
        assert len(proba) == 8

    def test_with_early_stopping(self) -> None:
        """Training with early stopping produces a functional model."""
        x, y, names = _make_binary_data()
        config = GradientBoostingConfig(
            **{**_make_config(), "early_stopping_rounds": 3},
        )
        model = _rust_train_gradient_boosting(x, y, x, y, config, names)
        proba = _rust_predict_proba_model(model, x)
        assert len(proba) == 8


# =============================================================================
# Native prediction adapter tests
# =============================================================================


class TestRustPredictProbaModel:
    """Tests for _rust_predict_proba_model adapter."""

    def test_returns_tuple_of_pairs(self) -> None:
        """Predictions are tuple of (p0, p1) pairs."""
        x, y, names = _make_binary_data()
        config = _make_config()
        model = _rust_train_gradient_boosting(x, y, None, None, config, names)
        proba = _rust_predict_proba_model(model, x)
        assert len(proba) == 8
        for p0, p1 in proba:
            assert abs(p0 + p1 - 1.0) < 1e-10

    def test_probabilities_in_valid_range(self) -> None:
        """All probabilities are in [0, 1]."""
        x, y, names = _make_binary_data()
        config = _make_config()
        model = _rust_train_gradient_boosting(x, y, None, None, config, names)
        proba = _rust_predict_proba_model(model, x)
        for p0, p1 in proba:
            assert 0.0 <= p0 <= 1.0
            assert 0.0 <= p1 <= 1.0


class TestRustPredictRawModel:
    """Tests for _rust_predict_raw_model adapter."""

    def test_returns_1d_array(self) -> None:
        """Raw predictions are a 1D numpy array."""
        x, y, names = _make_binary_data()
        config = _make_config()
        model = _rust_train_gradient_boosting(x, y, None, None, config, names)
        raw = _rust_predict_raw_model(model, x)
        assert raw.shape == (8,)
        assert raw.dtype == np.float64

    def test_consistent_with_proba(self) -> None:
        """Raw predictions are consistent with probabilities."""
        x, y, names = _make_binary_data()
        config = _make_config()
        model = _rust_train_gradient_boosting(x, y, None, None, config, names)
        raw = _rust_predict_raw_model(model, x)
        proba = _rust_predict_proba_model(model, x)
        for i in range(8):
            # sigmoid(raw) should equal prob_class_1
            raw_val: float = raw.item(i)
            import math

            expected_p1: float = 1.0 / (1.0 + math.exp(-raw_val))
            actual_p1: float = proba[i][1]
            assert abs(expected_p1 - actual_p1) < 1e-6


# =============================================================================
# Hook wiring tests
# =============================================================================


class TestNativeHookWiring:
    """Tests for wire_native_hooks() and unwire_native_hooks()."""

    def test_wire_sets_all_hooks(self) -> None:
        """wire_native_hooks() sets all 3 native hooks."""
        unwire_native_hooks()

        assert _hooks_native._train_native_backend is None
        assert _hooks_native._predict_raw_native_backend is None
        assert _hooks_native._predict_proba_native_backend is None

        wire_native_hooks()

        assert _hooks_native._train_native_backend is _rust_train_gradient_boosting
        assert _hooks_native._predict_raw_native_backend is _rust_predict_raw_model
        assert _hooks_native._predict_proba_native_backend is _rust_predict_proba_model

        unwire_native_hooks()

    def test_unwire_clears_all_hooks(self) -> None:
        """unwire_native_hooks() resets all 3 hooks to None."""
        wire_native_hooks()
        unwire_native_hooks()

        assert _hooks_native._train_native_backend is None
        assert _hooks_native._predict_raw_native_backend is None
        assert _hooks_native._predict_proba_native_backend is None

    def test_train_native_raises_when_unwired(self) -> None:
        """train_native raises RuntimeError when hooks not set."""
        unwire_native_hooks()
        x, y, names = _make_binary_data()
        config = _make_config()
        with pytest.raises(RuntimeError, match="Rust backend"):
            _hooks_native.train_native(x, y, None, None, config, names)

    def test_predict_raw_native_raises_when_unwired(self) -> None:
        """predict_raw_native raises RuntimeError when hooks not set."""
        unwire_native_hooks()
        x: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)

        class _FakeModel: ...

        with pytest.raises(RuntimeError, match="Rust backend"):
            _hooks_native.predict_raw_native(_FakeModel(), x)

    def test_predict_proba_native_raises_when_unwired(self) -> None:
        """predict_proba_native raises RuntimeError when hooks not set."""
        unwire_native_hooks()
        x: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)

        class _FakeModel: ...

        with pytest.raises(RuntimeError, match="Rust backend"):
            _hooks_native.predict_proba_native(_FakeModel(), x)


# =============================================================================
# Ensemble public API tests
# =============================================================================


class TestEnsembleNativeAPI:
    """Tests for ensemble.py native training public API."""

    def test_train_gradient_boosting_native_end_to_end(self) -> None:
        """Full native pipeline: train → predict_raw → predict_proba."""
        wire_native_hooks()
        x, y, names = _make_binary_data()
        config = _make_config()

        model = train_gradient_boosting_native(x, y, None, None, config, names)
        raw = predict_raw_native(model, x)
        proba = predict_proba_native(model, x)

        assert raw.shape == (8,)
        assert len(proba) == 8
        for p0, p1 in proba:
            assert abs(p0 + p1 - 1.0) < 1e-10

        unwire_native_hooks()

    def test_train_validates_inputs(self) -> None:
        """train_gradient_boosting_native validates training inputs."""
        wire_native_hooks()
        x_empty: NDArray[np.float64] = np.zeros((0, 2), dtype=np.float64)
        y_empty: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        config = _make_config()
        with pytest.raises(ValueError, match="must not be empty"):
            train_gradient_boosting_native(
                x_empty,
                y_empty,
                None,
                None,
                config,
                ("a", "b"),
            )
        unwire_native_hooks()

    def test_predict_raw_native_validates_empty(self) -> None:
        """predict_raw_native rejects empty feature matrix."""
        wire_native_hooks()
        x, y, names = _make_binary_data()
        config = _make_config()
        model = train_gradient_boosting_native(x, y, None, None, config, names)
        x_empty: NDArray[np.float64] = np.zeros((0, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="must not be empty"):
            predict_raw_native(model, x_empty)
        unwire_native_hooks()

    def test_predict_proba_native_validates_empty(self) -> None:
        """predict_proba_native rejects empty feature matrix."""
        wire_native_hooks()
        x, y, names = _make_binary_data()
        config = _make_config()
        model = train_gradient_boosting_native(x, y, None, None, config, names)
        x_empty: NDArray[np.float64] = np.zeros((0, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="must not be empty"):
            predict_proba_native(model, x_empty)
        unwire_native_hooks()

    def test_raises_without_rust_backend(self) -> None:
        """Native API raises RuntimeError when Rust not active."""
        unwire_native_hooks()
        x, y, names = _make_binary_data()
        config = _make_config()
        with pytest.raises(RuntimeError, match="Rust backend"):
            train_gradient_boosting_native(x, y, None, None, config, names)
