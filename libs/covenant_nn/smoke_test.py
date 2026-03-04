"""End-to-end smoke test for all regression backends."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from covenant_ml.backends import (
    create_lightgbm_regressor_backend,
    create_xgboost_regressor_backend,
)
from covenant_ml.datasets.testing import create_fake_regression_dataset_loader
from covenant_ml.datasets.types import DatasetConfig, TargetColumnSpec
from covenant_ml.testing import (
    make_lightgbm_regressor_config,
    make_lstm_regressor_config,
    make_mlp_regressor_config,
    make_xgboost_regressor_config,
    set_cuda_hook,
)
from numpy.typing import NDArray

from covenant_nn.backends import (
    create_lstm_regressor_backend,
    create_mlp_regressor_backend,
)


def _make_config() -> DatasetConfig:
    return DatasetConfig(
        name="smoke",
        display_name="Smoke",
        folder="fake",
        file_name="d.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="t",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=200,
        n_features_expected=10,
        positive_class_ratio_expected=0.0,
    )


def _load_data() -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    loader = create_fake_regression_dataset_loader(n_samples=200, n_features=10, random_state=42)
    dataset = loader.load(_make_config(), Path("/fake"))
    return dataset["x"], dataset["y"], list(dataset["meta"]["feature_names"])


def test_xgboost() -> None:
    """XGBoost regressor: train -> save -> load -> predict."""
    print("--- XGBoost Regressor ---")
    set_cuda_hook(lambda: False)
    x, y, names = _load_data()

    backend = create_xgboost_regressor_backend()
    config = make_xgboost_regressor_config(n_estimators=20, learning_rate=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=names,
            config=config,
            output_dir=Path(tmpdir),
            progress=None,
        )
        rmse = outcome["test_metrics"]["rmse"]
        r2 = outcome["test_metrics"]["r_squared"]
        print(f"  Test RMSE: {rmse:.4f}, R2: {r2:.4f}")

        loaded = backend.load(path=outcome["model_path"])
        preds = loaded.predict(x[:5])
        print(f"  Predictions: {[round(float(p), 3) for p in preds]}")
        print(f"  Actuals:     {[round(float(a), 3) for a in y[:5]]}")
    print("  OK\n")


def test_lightgbm() -> None:
    """LightGBM regressor: train -> save -> load -> predict."""
    print("--- LightGBM Regressor ---")
    set_cuda_hook(lambda: False)
    x, y, names = _load_data()

    backend = create_lightgbm_regressor_backend()
    config = make_lightgbm_regressor_config(n_estimators=20, learning_rate=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=names,
            config=config,
            output_dir=Path(tmpdir),
            progress=None,
        )
        rmse = outcome["test_metrics"]["rmse"]
        r2 = outcome["test_metrics"]["r_squared"]
        print(f"  Test RMSE: {rmse:.4f}, R2: {r2:.4f}")

        loaded = backend.load(path=outcome["model_path"])
        preds = loaded.predict(x[:5])
        print(f"  Predictions: {[round(float(p), 3) for p in preds]}")
        print(f"  Actuals:     {[round(float(a), 3) for a in y[:5]]}")
    print("  OK\n")


def test_mlp() -> None:
    """MLP regressor: train -> save -> load -> predict."""
    print("--- MLP Regressor ---")
    set_cuda_hook(lambda: False)
    x, y, names = _load_data()

    backend = create_mlp_regressor_backend()
    config = make_mlp_regressor_config(
        hidden_sizes=(32, 16),
        n_epochs=10,
        learning_rate=0.01,
        batch_size=64,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=names,
            config=config,
            output_dir=Path(tmpdir),
            progress=None,
        )
        rmse = outcome["test_metrics"]["rmse"]
        r2 = outcome["test_metrics"]["r_squared"]
        print(f"  Test RMSE: {rmse:.4f}, R2: {r2:.4f}")

        loaded = backend.load(path=outcome["model_path"])
        preds = loaded.predict(x[:5])
        print(f"  Predictions: {[round(float(p), 3) for p in preds]}")
        print(f"  Actuals:     {[round(float(a), 3) for a in y[:5]]}")
    print("  OK\n")


def test_lstm() -> None:
    """LSTM regressor: train -> save -> load -> predict."""
    print("--- LSTM Regressor ---")
    set_cuda_hook(lambda: False)
    x, y, names = _load_data()

    backend = create_lstm_regressor_backend()
    config = make_lstm_regressor_config(
        hidden_size=16,
        num_layers=1,
        n_epochs=10,
        learning_rate=0.01,
        batch_size=64,
        sequence_length=2,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = backend.train(
            x_features=x,
            y_targets=y,
            feature_names=names,
            config=config,
            output_dir=Path(tmpdir),
            progress=None,
        )
        rmse = outcome["test_metrics"]["rmse"]
        r2 = outcome["test_metrics"]["r_squared"]
        print(f"  Test RMSE: {rmse:.4f}, R2: {r2:.4f}")

        loaded = backend.load(path=outcome["model_path"])
        preds = loaded.predict(x[:5])
        print(f"  Predictions: {[round(float(p), 3) for p in preds]}")
        print(f"  Actuals:     {[round(float(a), 3) for a in y[:5]]}")
    print("  OK\n")


if __name__ == "__main__":
    print("=" * 60)
    print("All 4 Regression Backends - Smoke Test")
    print("=" * 60)
    print()
    test_xgboost()
    test_lightgbm()
    test_mlp()
    test_lstm()
    print("All 4 backends passed!")
