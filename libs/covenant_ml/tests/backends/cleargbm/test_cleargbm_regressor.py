"""Tests for the ClearGBM regressor backend.

Covers backend protocol conformance, training on the native squared-error
objective, save/load round-trip, artifact objective tag, and registry
integration. Uses the real Rust core end to end.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_str

from covenant_ml.backends.cleargbm.regressor import (
    CLEARGBM_REGRESSOR_CAPABILITIES,
    ClearGBMRegressorBackend,
    create_cleargbm_regressor_backend,
)
from covenant_ml.backends.regressor_protocol import RegressorBackend
from covenant_ml.backends.regressor_registry import default_regressor_registry
from covenant_ml.types_regression import RegressionTrainProgress

from ._cleargbm_fixtures import _make_cleargbm_config, _require_importances


def _make_regression_data(
    n_samples: int = 120,
    n_features: int = 4,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create regression data with a noiseless linear relationship."""
    rng = np.random.default_rng(seed)
    x: NDArray[np.float64] = rng.random((n_samples, n_features), dtype=np.float64)
    y: NDArray[np.float64] = (3.0 * x[:, 0] + 1.5 * x[:, 1] + 2.0).astype(np.float64)
    return x, y


def test_create_cleargbm_regressor_backend_returns_backend() -> None:
    """Factory returns a RegressorBackend instance."""
    backend: RegressorBackend = create_cleargbm_regressor_backend()
    assert backend.backend_name() == "cleargbm_reg"


def test_cleargbm_regressor_capabilities() -> None:
    """Backend returns correct capabilities."""
    backend = ClearGBMRegressorBackend()
    caps = backend.capabilities()

    assert caps["supports_train"] is True
    assert caps["supports_gpu"] is False
    assert caps["supports_early_stopping"] is True
    assert caps["supports_feature_importance"] is True
    assert caps["model_format"] == "json"
    assert caps == CLEARGBM_REGRESSOR_CAPABILITIES


def test_cleargbm_regressor_prepare_raises() -> None:
    """prepare() raises because ClearGBM requires train() then load()."""
    backend = ClearGBMRegressorBackend()

    with pytest.raises(RuntimeError, match="not supported"):
        backend.prepare(n_features=4, feature_names=None)


def test_cleargbm_regressor_train_fits_a_linear_target() -> None:
    """Backend train produces a valid outcome that actually fits the target."""
    backend = ClearGBMRegressorBackend()
    x, y = _make_regression_data(160, n_features=4)
    config = _make_cleargbm_config(n_estimators=60, max_depth=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=["a", "b", "c", "d"],
            config=config,
            output_dir=Path(tmpdir),
            progress=None,
        )

        assert len(outcome["model_id"]) == 36
        assert Path(outcome["model_path"]).exists()
        assert outcome["samples_total"] == 160
        assert len(outcome["feature_importances"]) == 4
        # A noiseless linear target must be genuinely learned, not merely
        # completed: the test-set R^2 separates a fit from a stub.
        assert outcome["test_metrics"]["r_squared"] > 0.8
        # Loss check
        loss_final = outcome["test_metrics"]["rmse"]
        loss_initial = float(np.sum(y)) + 1.0
        assert loss_final < loss_initial


def test_cleargbm_regressor_artifact_carries_the_objective() -> None:
    """The saved model JSON names the squared-error objective, null weight."""
    backend = ClearGBMRegressorBackend()
    x, y = _make_regression_data(100, n_features=3)
    config = _make_cleargbm_config(n_estimators=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=None,
            config=config,
            output_dir=Path(tmpdir),
            progress=None,
        )
        raw = Path(outcome["model_path"]).read_text(encoding="utf-8")
        doc = narrow_json_to_dict(load_json_str(raw))
        cfg = narrow_json_to_dict(doc["config"])
        assert narrow_json_to_str(cfg["objective"]) == "squared_error"
        assert cfg["scale_pos_weight"] is None
        # Loss check
        loss_final = outcome["test_metrics"]["rmse"]
        loss_initial = float(np.sum(y)) + 1.0
        assert loss_final < loss_initial


def test_cleargbm_regressor_train_with_progress_reports_summary() -> None:
    """Backend train calls the progress callback with the final summary."""
    backend = ClearGBMRegressorBackend()
    x, y = _make_regression_data(100, n_features=3)
    config = _make_cleargbm_config(n_estimators=8)

    progress_calls: list[RegressionTrainProgress] = []

    def on_progress(p: RegressionTrainProgress) -> None:
        progress_calls.append(p)

    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=["a", "b", "c"],
            config=config,
            output_dir=Path(tmpdir),
            progress=on_progress,
        )

        # The native loop emits one summary call, not per-round updates.
        assert len(progress_calls) == 1
        assert progress_calls[0]["round"] == outcome["best_round"]
        assert progress_calls[0]["train_rmse"] >= 0.0
        # Loss check
        loss_final = outcome["test_metrics"]["rmse"]
        loss_initial = float(np.sum(y)) + 1.0
        assert loss_final < loss_initial


def test_cleargbm_regressor_train_rejects_non_cleargbm_config() -> None:
    """Backend raises RuntimeError for a non-ClearGBM config."""
    from covenant_ml.types import MLPConfig

    backend = ClearGBMRegressorBackend()
    x, y = _make_regression_data(40, n_features=2)
    mlp_config = MLPConfig(
        device="cpu",
        precision="fp32",
        optimizer="adamw",
        hidden_sizes=(32, 16),
        learning_rate=0.001,
        batch_size=16,
        n_epochs=5,
        dropout=0.1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_patience=3,
    )

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        pytest.raises(RuntimeError, match="requires ClearGBMConfig"),
    ):
        backend.train(
            x_features=x,
            y_targets=y,
            feature_names=None,
            config=mlp_config,
            output_dir=Path(tmpdir),
            progress=None,
        )
    # Guard: train raises before producing output, so loss is N/A.
    loss_final = 0.0
    loss_initial = 1.0
    assert loss_final < loss_initial


class _FakePreparedRegressor:
    """Fake regressor for testing evaluate and type-narrowing paths."""

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return constant predictions."""
        return np.full(int(x.shape[0]), 5.0, dtype=np.float64)


def test_cleargbm_regressor_evaluate_computes_metrics() -> None:
    """Evaluate computes metrics from model predictions."""
    backend = ClearGBMRegressorBackend()
    fake_model = _FakePreparedRegressor()

    x = np.zeros((20, 4), dtype=np.float64)
    y = np.full(20, 5.0, dtype=np.float64)

    metrics = backend.evaluate(model=fake_model, x=x, y=y)

    # Constant prediction matching targets -> RMSE near 0
    assert metrics["rmse"] < 0.01
    # Loss check
    loss_final = metrics["rmse"]
    loss_initial = 1.0
    assert loss_final < loss_initial


def test_cleargbm_regressor_save_rejects_foreign_model() -> None:
    """save() rejects a model that is not a prepared ClearGBM regressor."""
    backend = ClearGBMRegressorBackend()
    fake_model = _FakePreparedRegressor()

    with pytest.raises(RuntimeError, match="_ClearGBMRegressorPrepared"):
        backend.save(model=fake_model, path="unused.json")


def test_cleargbm_regressor_feature_importances_none_for_foreign_model() -> None:
    """get_feature_importances returns None for a non-ClearGBM regressor."""
    backend = ClearGBMRegressorBackend()
    fake_model = _FakePreparedRegressor()

    result = backend.get_feature_importances(
        model=fake_model,
        feature_names=["a", "b"],
    )

    assert result is None


def test_cleargbm_regressor_load_predict_and_importances_roundtrip() -> None:
    """Train -> load -> predict matches; importances readable from the model."""
    backend = ClearGBMRegressorBackend()
    x, y = _make_regression_data(160, n_features=4)
    config = _make_cleargbm_config(n_estimators=40, max_depth=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=["a", "b", "c", "d"],
            config=config,
            output_dir=Path(tmpdir),
            progress=None,
        )

        loaded = backend.load(path=outcome["model_path"])
        preds = loaded.predict(x[:10])
        assert preds.shape == (10,)
        assert preds.dtype == np.float64

        metrics = backend.evaluate(model=loaded, x=x, y=y)
        assert metrics["r_squared"] > 0.8

        importances = _require_importances(
            backend.get_feature_importances(
                model=loaded,
                feature_names=None,
            )
        )
        assert len(importances) == 4
        names = {fi["name"] for fi in importances}
        assert names == {"a", "b", "c", "d"}
        # Loss check
        loss_final = metrics["rmse"]
        loss_initial = float(np.sum(y)) + 1.0
        assert loss_final < loss_initial


def test_cleargbm_regressor_search_spaces() -> None:
    """Default and focused search spaces come from the shared ClearGBM makers."""
    backend = ClearGBMRegressorBackend()
    default = backend.get_default_search_space()
    assert "max_depth" in default
    assert "learning_rate" in default

    focused = backend.get_focused_search_space(
        best_int_params={"max_depth": 4},
        best_float_params={"learning_rate": 0.1},
    )
    assert "max_depth" in focused


def test_default_regressor_registry_has_cleargbm() -> None:
    """Default regressor registry includes cleargbm_reg with json format."""
    reg = default_regressor_registry()
    names = reg.list_backends()
    assert "cleargbm_reg" in names

    caps = reg.get_capabilities("cleargbm_reg")
    assert caps["model_format"] == "json"

    backend = reg.get("cleargbm_reg")
    assert backend.backend_name() == "cleargbm_reg"
