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

from covenant_radar_api.worker._hook_defaults import (
    _real_logreg_loader,
    _real_random_forest_loader,
)
from covenant_radar_api.worker._test_hooks import (
    logreg_loader,
    random_forest_loader,
)
from covenant_radar_api.worker.optimize_types import UnifiedOptimizeParseResult


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
        from covenant_ml.backends.logreg.backend import (
            _create_logreg_model,
            _get_joblib_imports,
        )

        # Create and fit a model
        x_data: NDArray[np.float64] = np.random.randn(100, 10).astype(np.float64)
        y_data: NDArray[np.int64] = np.random.randint(0, 2, size=100).astype(np.int64)

        model = _create_logreg_model(
            penalty="l2",
            inverse_reg_strength=1.0,
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
        dump_fn, _ = _get_joblib_imports()
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


# =============================================================================
# Tests for _real_timeseries_registry and _real_optimizer_registry
# =============================================================================


class TestRealTimeseriesRegistry:
    """Tests for _real_timeseries_registry function."""

    def test_returns_registry_with_kaggle_amex(self) -> None:
        """_real_timeseries_registry returns registry containing kaggle_amex_default."""
        from covenant_radar_api.worker._hook_defaults import _real_timeseries_registry

        registry = _real_timeseries_registry()
        names = registry.list_names()
        assert "kaggle_amex_default" in names


class TestRealOptimizerRegistry:
    """Tests for _real_optimizer_registry function."""

    def test_returns_registry_with_optuna_tpe(self) -> None:
        """_real_optimizer_registry returns registry containing optuna_tpe."""
        from covenant_radar_api.worker._hook_defaults import _real_optimizer_registry

        registry = _real_optimizer_registry()
        # Registry supports .get() — optuna_tpe should be registered
        strategy = registry.get("optuna_tpe")
        assert callable(strategy.optimize)


# =============================================================================
# Tests for _real_objective_factory
# =============================================================================


class TestRealObjectiveFactory:
    """Tests for _real_objective_factory function with all 7 backends."""

    def _make_config(self) -> UnifiedOptimizeParseResult:
        """Create minimal config for objective factory tests.

        Returns:
            UnifiedOptimizeParseResult with default values.
        """
        return UnifiedOptimizeParseResult(
            backend="xgboost",
            dataset="taiwan",
            n_trials=1,
            timeout_seconds=None,
            device="cpu",
            feature_preset="none",
            random_state=42,
            early_stopping_rounds=2,
            n_jobs=1,
            precision="fp32",
            nn_optimizer="adamw",
            n_epochs=2,
            early_stopping_patience=2,
            sequence_length=3,
            bidirectional=False,
        )

    def _make_data(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
        """Create small test data for objective factory.

        Returns:
            Tuple of (features, labels, feature_names).
        """
        rng = np.random.RandomState(42)
        x: NDArray[np.float64] = rng.randn(50, 5).astype(np.float64)
        y: NDArray[np.int64] = rng.randint(0, 2, size=50).astype(np.int64)
        names = [f"f{i}" for i in range(5)]
        return x, y, names

    def test_xgboost_objective(self) -> None:
        """_real_objective_factory creates XGBoost objective with n_features."""
        from covenant_radar_api.worker._hook_defaults import _real_objective_factory

        x, y, names = self._make_data()
        config = self._make_config()
        obj = _real_objective_factory("xgboost", x, y, names, config)
        assert obj.n_features == 5

    def test_lightgbm_objective(self) -> None:
        """_real_objective_factory creates LightGBM objective with n_features."""
        from covenant_radar_api.worker._hook_defaults import _real_objective_factory

        x, y, names = self._make_data()
        config = self._make_config()
        obj = _real_objective_factory("lightgbm", x, y, names, config)
        assert obj.n_features == 5

    def test_cleargbm_objective(self) -> None:
        """_real_objective_factory creates ClearGBM objective with n_features."""
        from covenant_radar_api.worker._hook_defaults import _real_objective_factory

        x, y, names = self._make_data()
        config = self._make_config()
        obj = _real_objective_factory("cleargbm", x, y, names, config)
        assert obj.n_features == 5

    def test_logreg_objective(self) -> None:
        """_real_objective_factory creates LogReg objective with n_features."""
        from covenant_radar_api.worker._hook_defaults import _real_objective_factory

        x, y, names = self._make_data()
        config = self._make_config()
        obj = _real_objective_factory("logreg", x, y, names, config)
        assert obj.n_features == 5

    def test_random_forest_objective(self) -> None:
        """_real_objective_factory creates RandomForest objective with n_features."""
        from covenant_radar_api.worker._hook_defaults import _real_objective_factory

        x, y, names = self._make_data()
        config = self._make_config()
        obj = _real_objective_factory("random_forest", x, y, names, config)
        assert obj.n_features == 5

    def test_mlp_objective(self) -> None:
        """_real_objective_factory creates MLP objective with n_features."""
        from covenant_radar_api.worker._hook_defaults import _real_objective_factory

        x, y, names = self._make_data()
        config = self._make_config()
        obj = _real_objective_factory("mlp", x, y, names, config)
        assert obj.n_features == 5

    def test_lstm_objective(self) -> None:
        """_real_objective_factory creates LSTM objective with n_features."""
        from covenant_radar_api.worker._hook_defaults import _real_objective_factory

        x, y, names = self._make_data()
        config = self._make_config()
        obj = _real_objective_factory("lstm", x, y, names, config)
        assert obj.n_features == 5


# =============================================================================
# Tests for _real_data_bank_uploader
# =============================================================================


class TestRealDataBankUploader:
    """Tests for _real_data_bank_uploader function."""

    def test_uploads_model_file_and_returns_file_id(self, tmp_path: Path) -> None:
        """Test _real_data_bank_uploader uploads file via DataBankClient."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from platform_core.json_utils import dump_json_str

        from covenant_radar_api.worker._hook_defaults import _real_data_bank_uploader

        # Create a dummy model file
        model_path = tmp_path / "test_model.ubj"
        model_path.write_bytes(b"\x00\x01\x02\x03")

        # Start a local HTTP server that returns valid upload response
        received_requests: list[str] = []

        class UploadHandler(BaseHTTPRequestHandler):
            """Handler that accepts POST /files and returns upload response."""

            def do_POST(self) -> None:
                """Handle POST request."""
                received_requests.append(self.path)
                # Consume request body to avoid connection reset
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length > 0:
                    self.rfile.read(content_length)
                body = dump_json_str(
                    {
                        "file_id": "test_model.ubj",
                        "size": 4,
                        "sha256": "abc123",
                        "content_type": "application/octet-stream",
                        "created_at": "2025-01-01T00:00:00Z",
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, format: str, *args: str) -> None:
                """Suppress log output."""

        server = HTTPServer(("127.0.0.1", 0), UploadHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        base_url = f"http://127.0.0.1:{port}"
        try:
            file_id = _real_data_bank_uploader(model_path, base_url, "test-api-key")
        finally:
            server.shutdown()
            thread.join(timeout=5.0)
            server.server_close()

        assert file_id == "test_model.ubj"
        assert len(received_requests) == 1
        assert received_requests[0] == "/files"
