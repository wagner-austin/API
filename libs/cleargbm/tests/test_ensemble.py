"""Tests for the ``cleargbm.ensemble`` public API.

These exercise the strict Python boundary in front of the Rust core:
input-shape validation, config translation, and end-to-end training +
prediction on a small linearly-separable dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cleargbm.ensemble import (
    _config_to_rust_dict,
    _validate_training_inputs,
    export_model_json,
    predict_proba,
    predict_raw,
    train_gradient_boosting,
)
from cleargbm.types import GradientBoostingConfig, GrowthStrategy


def _make_config(
    n_estimators: int = 3,
    max_depth: int = 2,
    reg_lambda: float = 0.0,
    monotonic_constraints: tuple[int, ...] | None = None,
    early_stopping_rounds: int | None = None,
    n_jobs: int = 1,
    growth_strategy: GrowthStrategy = "depth_wise",
    num_leaves: int | None = None,
) -> GradientBoostingConfig:
    """Return a minimal valid training config."""
    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.3,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=None,
        max_bins=8,
        subsample=1.0,
        random_state=42,
        monotonic_constraints=monotonic_constraints,
        reg_alpha=0.0,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        growth_strategy=growth_strategy,
        num_leaves=num_leaves,
        scale_pos_weight=1.0,
    )


def _make_binary_dataset(
    n_samples: int = 40,
    n_features: int = 2,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    """Return a small linearly-separable binary classification dataset."""
    rng = np.random.default_rng(seed)
    x: NDArray[np.float64] = rng.random((n_samples, n_features), dtype=np.float64)
    score: NDArray[np.float64] = np.sum(x[:, : n_features // 2 + 1], axis=1)
    y: NDArray[np.int64] = (score > (n_features // 2 + 1) / 2.0).astype(np.int64)
    names = tuple(f"f{i}" for i in range(n_features))
    return x, y, names


# =============================================================================
# _validate_training_inputs
# =============================================================================


class TestValidateTrainingInputs:
    """The boundary validator rejects malformed inputs before Rust dispatch."""

    def test_empty_x_train_raises(self) -> None:
        """An empty training matrix is a boundary error."""
        x = np.zeros((0, 3), dtype=np.float64)
        y = np.zeros(0, dtype=np.int64)
        with pytest.raises(ValueError, match="x_train must not be empty"):
            _validate_training_inputs(x, y, ("a", "b", "c"))

    def test_y_length_mismatch_raises(self) -> None:
        """y_train row count must match x_train row count."""
        x = np.zeros((5, 3), dtype=np.float64)
        y = np.zeros(3, dtype=np.int64)
        with pytest.raises(ValueError, match="same length"):
            _validate_training_inputs(x, y, ("a", "b", "c"))

    def test_feature_name_count_mismatch_raises(self) -> None:
        """feature_names length must match x_train column count."""
        x = np.zeros((5, 3), dtype=np.float64)
        y = np.zeros(5, dtype=np.int64)
        with pytest.raises(ValueError, match="feature names"):
            _validate_training_inputs(x, y, ("a", "b"))


# =============================================================================
# _config_to_rust_dict
# =============================================================================


class TestConfigToRustDict:
    """Config translation: every field crosses; monotonic list is passed through."""

    def test_carries_the_sixteen_hyperparameters_plus_n_jobs(self) -> None:
        """The Rust-side dict has exactly the 17 keys the Rust trainer reads."""
        result = _config_to_rust_dict(_make_config())
        expected = {
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
            "n_jobs",
            "growth_strategy",
            "num_leaves",
            "scale_pos_weight",
            "max_features",
        }
        assert set(result.keys()) == expected

    def test_carries_the_growth_policy_and_its_budget(self) -> None:
        """Both halves of the growth axis must reach Rust.

        Dropping either one is silent: the trainer would fall back to its own
        reading of the config and report an arm it did not run.
        """
        result = _config_to_rust_dict(_make_config(growth_strategy="leaf_wise", num_leaves=31))
        assert result["growth_strategy"] == "leaf_wise"
        assert result["num_leaves"] == 31

    def test_forwards_n_jobs_to_the_rust_core(self) -> None:
        """n_jobs must reach Rust, where it bounds the worker pool.

        Regression guard: n_jobs was previously dropped here, so the Rust core
        fell back to rayon's global pool and used every core regardless of what
        the caller asked for.
        """
        result = _config_to_rust_dict(_make_config(n_jobs=3))
        assert result["n_jobs"] == 3

    def test_forwards_max_features_to_the_rust_core(self) -> None:
        """max_features must reach Rust, where it budgets each split search.

        Regression guard: this field was dropped here for as long as the
        core lacked the capability, leaving a config knob the trainer
        silently ignored.
        """
        result = _config_to_rust_dict(_make_config())
        assert result["max_features"] is None

    def test_monotonic_constraints_none_stays_none(self) -> None:
        """None constraint stays None in the dict."""
        result = _config_to_rust_dict(_make_config(monotonic_constraints=None))
        assert result["monotonic_constraints"] is None

    def test_monotonic_constraints_tuple_becomes_list_of_ints(self) -> None:
        """Tuple of ints becomes a list of ints for Rust consumption."""
        cfg = _make_config(monotonic_constraints=(1, 0, -1))
        result = _config_to_rust_dict(cfg)
        assert result["monotonic_constraints"] == [1, 0, -1]


# =============================================================================
# train_gradient_boosting + predict_proba + predict_raw
# =============================================================================


class TestTrainAndPredict:
    """End-to-end path: train a small model, then predict with it."""

    def test_train_returns_native_model_handle(self) -> None:
        """Training yields an opaque native handle usable by predict_*."""
        x, y, names = _make_binary_dataset()
        model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=_make_config(), feature_names=names
        )
        # The handle must accept predict_proba without error and return a
        # per-sample pair (2-tuple of floats).
        result = predict_proba(model, x)
        n_rows: int = int(x.shape[0])
        assert len(result) == n_rows
        for p in result:
            assert len(p) == 2
            p0, p1 = p
            # Value-level assertion: probabilities are floats summing to 1.
            assert abs((p0 + p1) - 1.0) < 1e-12

    def test_predict_proba_returns_valid_probabilities(self) -> None:
        """Probabilities sum to 1 per sample and lie in [0, 1]."""
        x, y, names = _make_binary_dataset()
        model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=_make_config(), feature_names=names
        )
        result = predict_proba(model, x)
        for p0, p1 in result:
            assert 0.0 <= p0 <= 1.0
            assert 0.0 <= p1 <= 1.0
            assert abs((p0 + p1) - 1.0) < 1e-12

    def test_predict_raw_returns_one_score_per_sample(self) -> None:
        """predict_raw output has the same length as the input row count."""
        x, y, names = _make_binary_dataset()
        model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=_make_config(), feature_names=names
        )
        raw = predict_raw(model, x)
        n_rows: int = int(x.shape[0])
        assert raw.shape == (n_rows,)

    def test_train_with_validation_set_runs(self) -> None:
        """A validation split is accepted (used internally by early stopping)."""
        x, y, names = _make_binary_dataset(n_samples=60)
        x_val, y_val, _ = _make_binary_dataset(n_samples=20, seed=7)
        model = train_gradient_boosting(
            x_train=x,
            y_train=y,
            x_val=x_val,
            y_val=y_val,
            config=_make_config(early_stopping_rounds=2),
            feature_names=names,
        )
        # Successful training returns a live handle.
        n_val_rows: int = int(x_val.shape[0])
        assert predict_raw(model, x_val).shape == (n_val_rows,)

    def test_predict_proba_empty_x_raises(self) -> None:
        """predict_proba on an empty feature matrix rejects at the boundary."""
        x, y, names = _make_binary_dataset()
        model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=_make_config(), feature_names=names
        )
        empty = np.zeros((0, x.shape[1]), dtype=np.float64)
        with pytest.raises(ValueError, match="x must not be empty"):
            predict_proba(model, empty)

    def test_predict_raw_empty_x_raises(self) -> None:
        """predict_raw on an empty feature matrix rejects at the boundary."""
        x, y, names = _make_binary_dataset()
        model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=_make_config(), feature_names=names
        )
        empty = np.zeros((0, x.shape[1]), dtype=np.float64)
        with pytest.raises(ValueError, match="x must not be empty"):
            predict_raw(model, empty)

    def test_train_empty_x_raises_at_boundary(self) -> None:
        """train_gradient_boosting rejects an empty training matrix."""
        empty_x = np.zeros((0, 2), dtype=np.float64)
        empty_y = np.zeros(0, dtype=np.int64)
        with pytest.raises(ValueError, match="x_train must not be empty"):
            train_gradient_boosting(
                x_train=empty_x,
                y_train=empty_y,
                x_val=None,
                y_val=None,
                config=_make_config(),
                feature_names=("a", "b"),
            )


class TestExportModelJson:
    """Model introspection through the public JSON surface."""

    def test_export_returns_the_tree_structure(self) -> None:
        """The export names every tree and its nodes."""
        x, y, names = _make_binary_dataset()
        model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=_make_config(), feature_names=names
        )

        document = export_model_json(model)

        assert '"trees"' in document
        assert '"nodes"' in document
        assert '"is_leaf"' in document

    def test_export_is_stable_for_one_model(self) -> None:
        """Serializing the same model twice yields the same document."""
        x, y, names = _make_binary_dataset()
        model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=_make_config(), feature_names=names
        )

        assert export_model_json(model) == export_model_json(model)
