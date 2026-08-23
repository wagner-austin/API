"""ClearGBM backend integration tests with actual training.

Tests the full training loop, prediction, save/load, and error paths.
Uses real US bankruptcy data for integration tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.backends.cleargbm import (
    create_cleargbm_backend,
)
from covenant_ml.backends.cleargbm.config_resolution import (
    _is_cleargbm_config,
)
from covenant_ml.backends.protocol import PreparedClassifier
from covenant_ml.types import (
    ClearGBMConfig,
)
from tests.backends.cleargbm._cleargbm_fixtures import (
    _invoke_cleargbm_train,
    _make_cleargbm_config,
    _make_synthetic_dataset,
    _require_importances,
)

from ...conftest import load_us_bankruptcy_data


def test_cleargbm_backend_evaluate_computes_metrics(tmp_path: Path) -> None:
    """ClearGBMBackend.evaluate computes metrics using loaded model."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Load the trained model and evaluate
    loaded_model = backend.load(path=outcome["model_path"])
    metrics = backend.evaluate(model=loaded_model, x=x, y=y)

    # Metrics should be computed correctly
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0  # Log loss is always positive


def test_cleargbm_backend_evaluate_on_subset(tmp_path: Path) -> None:
    """ClearGBMBackend.evaluate works on data subset."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_samples=100)

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Evaluate on first 50 samples only
    loaded_model = backend.load(path=outcome["model_path"])
    metrics = backend.evaluate(model=loaded_model, x=x[:50], y=y[:50])

    assert 0.0 <= metrics["auc"] <= 1.0


def test_cleargbm_backend_save_and_load(tmp_path: Path) -> None:
    """ClearGBMBackend.save and load round-trip correctly."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Load the saved model
    loaded = backend.load(path=outcome["model_path"])

    # Verify it can predict
    proba = loaded.predict_proba(x)
    assert proba.shape == (100, 2)


def test_cleargbm_backend_save_creates_json_file(tmp_path: Path) -> None:
    """ClearGBMBackend.save creates a JSON file."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=3)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    model_path = Path(outcome["model_path"])
    assert model_path.exists()
    assert model_path.suffix == ".json"

    # Verify it's valid JSON by reading
    content = model_path.read_text(encoding="utf-8")
    assert content.startswith("{")


def test_cleargbm_backend_save_raises_for_wrong_model_type(tmp_path: Path) -> None:
    """ClearGBMBackend.save raises for non-ClearGBMPrepared model."""
    backend = create_cleargbm_backend()

    class _FakeModel:
        """Fake model that's not _ClearGBMPrepared."""

        def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            n = int(x.shape[0])
            return np.full((n, 2), 0.5, dtype=np.float64)

    fake_model: PreparedClassifier = _FakeModel()
    save_path = str(tmp_path / "fake.json")

    with pytest.raises(RuntimeError, match="must be _ClearGBMPrepared"):
        backend.save(model=fake_model, path=save_path)


def test_cleargbm_backend_load_and_predict(tmp_path: Path) -> None:
    """ClearGBMBackend.load loads a trained model that can predict."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Load the trained model
    loaded_model = backend.load(path=outcome["model_path"])

    # Predict probabilities
    proba = loaded_model.predict_proba(x)

    # Verify shape and values
    assert proba.shape == (100, 2)
    min_val = float(np.min(proba))
    max_val = float(np.max(proba))
    assert min_val >= 0.0
    assert max_val <= 1.0


def test_cleargbm_backend_get_feature_importances(tmp_path: Path) -> None:
    """ClearGBMBackend.get_feature_importances returns importance list."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_features=4)

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Load model and get importances
    loaded_model = backend.load(path=outcome["model_path"])
    importances = _require_importances(
        backend.get_feature_importances(model=loaded_model, feature_names=names)
    )

    # Verify count matches features
    assert len(importances) == 4
    # Check structure via first element
    first = importances[0]
    assert first["name"] == "f0" or first["name"] == "f1" or first["name"] in names
    assert first["importance"] >= 0.0
    assert first["rank"] == 1


def test_cleargbm_backend_get_feature_importances_sorted_by_rank(tmp_path: Path) -> None:
    """ClearGBMBackend.get_feature_importances returns sorted by rank."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset(n_features=6)

    config = _make_cleargbm_config(n_estimators=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    loaded_model = backend.load(path=outcome["model_path"])
    importances = _require_importances(
        backend.get_feature_importances(model=loaded_model, feature_names=names)
    )

    # Verify ranks are sequential starting at 1
    ranks = [imp["rank"] for imp in importances]
    expected_ranks = list(range(1, len(ranks) + 1))
    assert ranks == expected_ranks


def test_cleargbm_backend_get_feature_importances_returns_none_for_wrong_type(
    tmp_path: Path,
) -> None:
    """ClearGBMBackend.get_feature_importances returns None for wrong model type."""
    backend = create_cleargbm_backend()

    class _FakeModel:
        """Fake model that's not _ClearGBMPrepared."""

        def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            n = int(x.shape[0])
            return np.full((n, 2), 0.5, dtype=np.float64)

    fake_model: PreparedClassifier = _FakeModel()
    result = backend.get_feature_importances(model=fake_model, feature_names=["a", "b"])

    assert result is None


def test_cleargbm_backend_us_bankruptcy_full_pipeline(tmp_path: Path) -> None:
    """Full pipeline test with US bankruptcy dataset."""
    backend = create_cleargbm_backend()
    dataset = load_us_bankruptcy_data()
    x, y, names = dataset["x"], dataset["y"], dataset["feature_names"]

    config: ClearGBMConfig = {
        "n_estimators": 15,
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": None,
        "colsample_bytree": None,
        "categorical_features": None,
        "max_bins": 64,
        "subsample": 0.8,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_state": 42,
        "early_stopping_rounds": 5,
        "monotonic_constraints": None,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "growth_strategy": "depth_wise",
        "num_leaves": None,
    }

    # Train
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify training completed
    assert outcome["samples_total"] == dataset["n_samples"]
    assert Path(outcome["model_path"]).exists()

    # Load and evaluate
    loaded_model = backend.load(path=outcome["model_path"])
    test_metrics = backend.evaluate(model=loaded_model, x=x, y=y)

    # Should produce reasonable metrics
    assert test_metrics["auc"] > 0.0
    assert test_metrics["loss"] < 10.0  # Sanity check

    # Verify feature importances
    importances = _require_importances(
        backend.get_feature_importances(model=loaded_model, feature_names=names)
    )
    assert len(importances) == len(names)


def test_cleargbm_backend_model_id_is_uuid(tmp_path: Path) -> None:
    """Model ID is a valid UUID string."""
    import uuid

    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=3)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify model_id is a valid UUID
    model_id = outcome["model_id"]
    parsed_uuid = uuid.UUID(model_id)
    assert str(parsed_uuid) == model_id


def test_cleargbm_backend_config_preserved_in_outcome(tmp_path: Path) -> None:
    """Training config is preserved in TrainOutcome."""
    backend = create_cleargbm_backend()
    x, y, names = _make_synthetic_dataset()

    config = _make_cleargbm_config(n_estimators=7, max_depth=5)
    outcome = _invoke_cleargbm_train(backend, x, y, names, config, tmp_path)

    # Verify config is preserved by narrowing to ClearGBMConfig
    saved_config = outcome["config"]
    # ClearGBMConfig has min_samples_split which other configs don't
    assert _is_cleargbm_config(saved_config)
    # Now we can access ClearGBM-specific keys
    cleargbm_config: ClearGBMConfig = saved_config
    assert cleargbm_config["n_estimators"] == 7
    assert cleargbm_config["max_depth"] == 5
