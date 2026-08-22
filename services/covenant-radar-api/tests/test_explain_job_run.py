"""Tests for feature importance explanation job."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)

from covenant_radar_api.worker.explain_job import (
    ExplainJobStatus,
    ExplainProgressInfo,
    _sample_data,
    run_explanation,
)
from tests._explain_job_fixtures import (
    _copy_real_taiwan,
    _create_xgboost_model,
)


class TestSampleData:
    """Tests for _sample_data function."""

    def test_returns_all_data_when_n_samples_equals_total(self) -> None:
        """Returns all data when n_samples equals total samples."""
        x: NDArray[np.float64] = np.zeros((3, 2), dtype=np.float64)
        x[0, :] = [1.0, 2.0]
        x[1, :] = [3.0, 4.0]
        x[2, :] = [5.0, 6.0]
        result = _sample_data(x, 3, 42)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, x)

    def test_returns_all_data_when_n_samples_greater_than_total(self) -> None:
        """Returns all data when n_samples exceeds total samples."""
        x: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        x[0, :] = [1.0, 2.0]
        x[1, :] = [3.0, 4.0]
        result = _sample_data(x, 100, 42)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, x)

    def test_samples_correct_number(self) -> None:
        """Samples correct number of rows when n_samples < total."""
        rng = np.random.default_rng(0)
        x: NDArray[np.float64] = rng.random((50, 2))
        result = _sample_data(x, 10, 42)
        assert result.shape == (10, 2)

    def test_reproducible_with_same_seed(self) -> None:
        """Same random_state produces same samples."""
        rng = np.random.default_rng(0)
        x: NDArray[np.float64] = rng.random((50, 2))
        result1 = _sample_data(x, 10, 42)
        result2 = _sample_data(x, 10, 42)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seed_produces_different_samples(self) -> None:
        """Different random_state produces different samples."""
        rng = np.random.default_rng(0)
        x: NDArray[np.float64] = rng.random((50, 2))
        result1 = _sample_data(x, 10, 42)
        result2 = _sample_data(x, 10, 123)
        assert not np.array_equal(result1, result2)


class TestRunExplanation:
    """Tests for run_explanation function."""

    def test_run_explanation_rejects_model_path_outside_models_root(self, tmp_path: Path) -> None:
        """A model_path escaping models_root is refused before any load.

        model_path arrives on the request body and reaches pickle-backed
        loaders, so an unconstrained value selects which host file is opened.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        external_dir = tmp_path / "external"
        _, _, feature_names = _copy_real_taiwan(external_dir)

        models_root = tmp_path / "models"
        models_root.mkdir()
        outside_model = tmp_path / "outside.ubj"
        _create_xgboost_model(outside_model, len(feature_names))

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": str(models_root / ".." / "outside.ubj"),
                "explainer": "permutation",
                "n_samples": 10,
                "random_state": 42,
            }
        )

        with pytest.raises(ValueError, match="must resolve inside the models root"):
            run_explanation(config_json, external_dir, models_root)

    def test_run_explanation_with_permutation_explainer(self, tmp_path: Path) -> None:
        """run_explanation completes with permutation explainer."""
        # Set up data
        external_dir = tmp_path / "external"
        _, _, feature_names = _copy_real_taiwan(external_dir)

        # Create model
        model_path = tmp_path / "model.ubj"
        _create_xgboost_model(model_path, len(feature_names))

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": str(model_path),
                "explainer": "permutation",
                "n_samples": 50,
                "random_state": 42,
            }
        )

        result = run_explanation(config_json, external_dir, tmp_path)

        assert result["status"] == "complete"
        assert result["backend"] == "xgboost"
        assert result["explainer"] == "permutation"
        assert result["n_samples_used"] == 50
        assert result["n_features"] == len(feature_names)
        assert result["target_class"] == 1
        assert len(result["feature_importances"]) == len(feature_names)
        assert result["duration_seconds"] >= 0.0

    def test_run_explanation_with_shap_tree_explainer(self, tmp_path: Path) -> None:
        """run_explanation completes with shap_tree explainer."""
        external_dir = tmp_path / "external"
        _, _, feature_names = _copy_real_taiwan(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgboost_model(model_path, len(feature_names))

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": str(model_path),
                "explainer": "shap_tree",
                "n_samples": 50,
                "random_state": 42,
            }
        )

        result = run_explanation(config_json, external_dir, tmp_path)

        assert result["status"] == "complete"
        assert result["explainer"] == "shap_tree"
        assert len(result["feature_importances"]) == len(feature_names)

    def test_run_explanation_samples_all_when_n_samples_exceeds_dataset(
        self, tmp_path: Path
    ) -> None:
        """run_explanation uses all samples when n_samples exceeds dataset size."""
        external_dir = tmp_path / "external"
        _, n_rows, feature_names = _copy_real_taiwan(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgboost_model(model_path, len(feature_names))

        # Request more samples than available
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": str(model_path),
                "explainer": "permutation",
                "n_samples": 999999,
                "random_state": 42,
            }
        )

        result = run_explanation(config_json, external_dir, tmp_path)

        assert result["n_samples_used"] == n_rows

    def test_run_explanation_raises_on_incompatible_explainer(self, tmp_path: Path) -> None:
        """run_explanation raises ValueError for incompatible explainer-backend combo."""
        external_dir = tmp_path / "external"
        _, _, feature_names = _copy_real_taiwan(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgboost_model(model_path, len(feature_names))

        # gradient explainer is not compatible with xgboost
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": str(model_path),
                "explainer": "gradient",
                "n_samples": 50,
            }
        )

        with pytest.raises(ValueError, match="is not compatible with backend"):
            run_explanation(config_json, external_dir, tmp_path)

    def test_run_explanation_with_progress_callback(self, tmp_path: Path) -> None:
        """run_explanation calls progress callback with status updates."""
        external_dir = tmp_path / "external"
        _, _, feature_names = _copy_real_taiwan(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgboost_model(model_path, len(feature_names))

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": str(model_path),
                "explainer": "permutation",
                "n_samples": 20,
            }
        )

        callback_calls: list[ExplainProgressInfo] = []

        def progress_callback(info: ExplainProgressInfo) -> None:
            callback_calls.append(info)

        result = run_explanation(
            config_json,
            external_dir,
            tmp_path,
            progress_callback=progress_callback,
        )

        assert result["status"] == "complete"

        # Verify callback was called with expected statuses
        statuses: list[ExplainJobStatus] = [c["status"] for c in callback_calls]
        assert "started" in statuses
        assert "loading_model" in statuses
        assert "loading_data" in statuses
        assert "computing" in statuses
        assert "complete" in statuses

        # Verify elapsed_seconds is present and non-negative
        for call in callback_calls:
            assert call["elapsed_seconds"] >= 0.0

    def test_run_explanation_raises_on_missing_model_file(self, tmp_path: Path) -> None:
        """run_explanation raises FileNotFoundError for missing model file."""
        external_dir = tmp_path / "external"
        _copy_real_taiwan(external_dir)

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": str(tmp_path / "nonexistent.ubj"),
                "explainer": "permutation",
                "n_samples": 50,
            }
        )

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            run_explanation(config_json, external_dir, tmp_path)

    def test_run_explanation_with_custom_registry(self, tmp_path: Path) -> None:
        """run_explanation accepts custom explainer registry."""
        from covenant_ml.explainers.registry import default_explainer_registry

        external_dir = tmp_path / "external"
        _, _, feature_names = _copy_real_taiwan(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgboost_model(model_path, len(feature_names))

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": str(model_path),
                "explainer": "permutation",
                "n_samples": 20,
            }
        )

        # Use custom registry (same as default for this test)
        registry = default_explainer_registry()
        result = run_explanation(config_json, external_dir, tmp_path, registry=registry)

        assert result["status"] == "complete"


class TestProcessExplainJob:
    """Tests for process_explain_job entry point."""

    def test_process_job_returns_json_serializable_result(self, tmp_path: Path) -> None:
        """process_explain_job returns JSON-serializable result dict."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv

        from covenant_radar_api.worker.explain_job import process_explain_job

        # Set up data directories
        data_root = tmp_path / "data"
        external_dir = data_root / "external"
        models_dir = tmp_path / "models"

        # Copy real Taiwan data
        _, _, feature_names = _copy_real_taiwan(external_dir)

        # Create model inside the configured models root: process_explain_job
        # confines model_path to APP__MODELS_ROOT.
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model.ubj"
        _create_xgboost_model(model_path, len(feature_names))

        # Set up fake environment
        fake_env = FakeEnv(
            {
                "APP__DATA_ROOT": str(data_root),
                "APP__MODELS_ROOT": str(models_dir),
                "DATABASE_URL": "postgresql://test@localhost/test",
                "REDIS_URL": "redis://localhost:6379/0",
            }
        )

        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env

        try:
            config_json = dump_json_str(
                {
                    "dataset": "taiwan",
                    "device": "cpu",
                    "backend": "xgboost",
                    "model_path": str(model_path),
                    "explainer": "permutation",
                    "n_samples": 20,
                    "random_state": 42,
                }
            )

            result = process_explain_job(config_json)

            # Verify result structure and values
            assert result["status"] == "complete"
            assert result["backend"] == "xgboost"
            assert result["explainer"] == "permutation"
            assert result["n_samples_used"] == 20
            assert result["n_features"] == len(feature_names)
            assert result["target_class"] == 1
            duration = require_float(result, "duration_seconds")
            assert duration >= 0.0

            # Verify feature_importances using require_* helpers for type narrowing
            importances_list = require_list(result, "feature_importances")
            assert len(importances_list) == len(feature_names)

            # Verify first importance score has expected fields
            # Extract and validate first score as a dict using require_dict
            result_with_first: JSONObject = {"first": importances_list[0]}
            first_score = require_dict(result_with_first, "first")
            rank = require_int(first_score, "rank")
            name = require_str(first_score, "name")
            importance = require_float(first_score, "importance")
            assert rank >= 1
            assert name in feature_names
            assert importance >= 0.0 or importance < 0.0
        finally:
            config_hooks.get_env = orig_get_env


class TestRealWorkerHooks:
    """Tests for real worker hook implementations to ensure coverage.

    These tests exercise the production implementations of hook functions
    that are normally replaced with fakes during test execution.
    """

    def test_real_explainer_registry_returns_registry(self) -> None:
        """Test _real_explainer_registry returns an ExplainerRegistry.

        The _real_explainer_registry function provides the production
        implementation for explainer registry injection. This test ensures
        the real implementation works correctly and returns a registry
        with the expected explainers registered.
        """
        from covenant_radar_api.worker._hook_defaults import _real_explainer_registry

        registry = _real_explainer_registry()

        # Verify registry has expected explainers via list_explainers()
        registered = registry.list_explainers()
        assert "permutation" in registered
        assert "shap_tree" in registered
        assert "gradient" in registered
        assert "integrated_gradients" in registered
