"""Tests for worker _test_hooks module.

Tests the real loader implementations for LogReg and RandomForest models.
These are thin wrappers around the model loader functions that provide
dependency injection points for testing.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_radar_api.worker._test_hooks import (
    _real_logreg_loader,
    _real_random_forest_loader,
    logreg_loader,
    random_forest_loader,
)


class TestLogRegLoaderHook:
    """Tests for logreg_loader hook and _real_logreg_loader implementation."""

    def test_logreg_loader_is_real_implementation(self) -> None:
        """Verify logreg_loader defaults to real implementation."""
        assert logreg_loader is _real_logreg_loader

    def test_real_logreg_loader_loads_model(self, tmp_path: Path) -> None:
        """_real_logreg_loader loads a valid LogReg model file.

        Creates a LogReg model, saves it to a joblib file, then loads it
        via the hook and verifies prediction works correctly.
        """
        from covenant_ml.backends.logreg.backend import _get_sklearn_imports

        # Get sklearn imports via typed accessor
        logreg_ctor, dump_fn, _ = _get_sklearn_imports()

        # Create and fit a model
        x_data: NDArray[np.float64] = np.random.randn(100, 10).astype(np.float64)
        y_data: NDArray[np.int64] = np.random.randint(0, 2, size=100).astype(np.int64)

        model = logreg_ctor(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=200,
            tol=1e-4,
            random_state=42,
            class_weight=None,
            l1_ratio=None,
            n_jobs=-1,
        )
        model.fit(x_data, y_data)

        # Save model
        model_path = tmp_path / "logreg_model.joblib"
        dump_fn(model, str(model_path))

        # Load via hook and verify
        loaded = _real_logreg_loader(model_path)

        # Test prediction
        x: NDArray[np.float64] = np.random.randn(3, 10).astype(np.float64)
        proba = loaded.predict_proba(x)
        assert proba.shape == (3, 2)
        # Probabilities should sum to ~1
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))

    def test_real_logreg_loader_file_not_found(self, tmp_path: Path) -> None:
        """_real_logreg_loader raises FileNotFoundError for missing file."""
        model_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(FileNotFoundError):
            _real_logreg_loader(model_path)


class TestRandomForestLoaderHook:
    """Tests for random_forest_loader hook and _real_random_forest_loader implementation."""

    def test_random_forest_loader_is_real_implementation(self) -> None:
        """Verify random_forest_loader defaults to real implementation."""
        assert random_forest_loader is _real_random_forest_loader

    def test_real_random_forest_loader_loads_model(self, tmp_path: Path) -> None:
        """_real_random_forest_loader loads a valid RandomForest model file.

        Creates a RandomForest model, saves it to a joblib file, then loads it
        via the hook and verifies prediction works correctly.
        """
        from covenant_ml.backends.random_forest.backend import _get_sklearn_imports

        # Get sklearn imports via typed accessor
        rf_ctor, dump_fn, _ = _get_sklearn_imports()

        # Create and fit a model
        x_data: NDArray[np.float64] = np.random.randn(100, 10).astype(np.float64)
        y_data: NDArray[np.int64] = np.random.randint(0, 2, size=100).astype(np.int64)

        model = rf_ctor(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight=None,
            n_jobs=-1,
            random_state=42,
            oob_score=False,
        )
        model.fit(x_data, y_data)

        # Save model
        model_path = tmp_path / "rf_model.joblib"
        dump_fn(model, str(model_path))

        # Load via hook and verify
        loaded = _real_random_forest_loader(model_path)

        # Test prediction
        x: NDArray[np.float64] = np.random.randn(3, 10).astype(np.float64)
        proba = loaded.predict_proba(x)
        assert proba.shape == (3, 2)
        # Probabilities should sum to ~1
        sums: NDArray[np.float64] = proba.sum(axis=1)
        assert bool(np.allclose(sums, 1.0))

    def test_real_random_forest_loader_file_not_found(self, tmp_path: Path) -> None:
        """_real_random_forest_loader raises FileNotFoundError for missing file."""
        model_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(FileNotFoundError):
            _real_random_forest_loader(model_path)
