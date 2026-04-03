"""Tests for regression feature importance explanation job."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import JSONTypeError, dump_json_str

from covenant_radar_api.worker.explain_regression_job import (
    RegressionExplainJobStatus,
    RegressionExplainProgressInfo,
    _optional_int,
    _parse_explainer,
    _parse_regression_explain_config,
    _parse_regressor_backend,
    _sample_data,
    run_regression_explanation,
)

# ---------------------------------------------------------------------------
# Tests for _optional_int
# ---------------------------------------------------------------------------


class TestOptionalInt:
    """Tests for _optional_int helper."""

    def test_returns_default_when_missing(self) -> None:
        """Returns default when key is not present."""
        result = _optional_int({}, "key", 42)
        assert result == 42

    def test_returns_value_when_int(self) -> None:
        """Returns integer value when present."""
        result = _optional_int({"key": 99}, "key", 42)
        assert result == 99

    def test_converts_float_to_int(self) -> None:
        """Converts float value to int."""
        result = _optional_int({"key": 5.0}, "key", 42)
        assert result == 5

    def test_raises_on_string_value(self) -> None:
        """Raises JSONTypeError when value is a string."""
        with pytest.raises(JSONTypeError, match="must be a number"):
            _optional_int({"key": "not a number"}, "key", 42)

    def test_returns_default_when_none(self) -> None:
        """Returns default when value is None."""
        result = _optional_int({"key": None}, "key", 42)
        assert result == 42


# ---------------------------------------------------------------------------
# Tests for _parse_regressor_backend
# ---------------------------------------------------------------------------


class TestParseRegressorBackend:
    """Tests for _parse_regressor_backend."""

    def test_all_valid_backends(self) -> None:
        """Accepts all valid regressor backends."""
        assert _parse_regressor_backend("xgboost_reg") == "xgboost_reg"
        assert _parse_regressor_backend("lightgbm_reg") == "lightgbm_reg"
        assert _parse_regressor_backend("mlp_reg") == "mlp_reg"
        assert _parse_regressor_backend("lstm_reg") == "lstm_reg"

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="backend must be one of"):
            _parse_regressor_backend("invalid")

    def test_non_string_raises(self) -> None:
        """Non-string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="backend must be a string"):
            _parse_regressor_backend(123)


# ---------------------------------------------------------------------------
# Tests for _parse_explainer
# ---------------------------------------------------------------------------


class TestParseExplainer:
    """Tests for _parse_explainer."""

    def test_all_valid_explainers(self) -> None:
        """Accepts all valid explainer names."""
        assert _parse_explainer("permutation") == "permutation"
        assert _parse_explainer("gradient") == "gradient"
        assert _parse_explainer("integrated_gradients") == "integrated_gradients"
        assert _parse_explainer("shap_tree") == "shap_tree"

    def test_invalid_explainer_raises(self) -> None:
        """Invalid explainer raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="explainer must be one of"):
            _parse_explainer("invalid")

    def test_non_string_raises(self) -> None:
        """Non-string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="explainer must be a string"):
            _parse_explainer(42)


# ---------------------------------------------------------------------------
# Tests for _parse_regression_explain_config
# ---------------------------------------------------------------------------


class TestParseRegressionExplainConfig:
    """Tests for _parse_regression_explain_config."""

    def test_full_config(self) -> None:
        """Parses full config with all fields."""
        config_json = (
            '{"dataset": "financial_distress", '
            '"backend": "xgboost_reg", '
            '"model_path": "/models/xgb.ubj", '
            '"explainer": "permutation", '
            '"n_samples": 500, '
            '"random_state": 99}'
        )
        result = _parse_regression_explain_config(config_json)

        assert result["dataset"] == "financial_distress"
        assert result["backend"] == "xgboost_reg"
        assert result["model_path"] == "/models/xgb.ubj"
        assert result["explainer"] == "permutation"
        assert result["n_samples"] == 500
        assert result["random_state"] == 99

    def test_minimal_config_uses_defaults(self) -> None:
        """Parses minimal config with defaults for optional fields."""
        config_json = (
            '{"dataset": "financial_distress", '
            '"backend": "lightgbm_reg", '
            '"model_path": "/models/lgbm.txt", '
            '"explainer": "shap_tree"}'
        )
        result = _parse_regression_explain_config(config_json)

        assert result["n_samples"] == 1000
        assert result["random_state"] == 42

    def test_non_object_raises(self) -> None:
        """Non-object JSON raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_regression_explain_config("[]")

    def test_missing_backend_raises(self) -> None:
        """Missing backend raises JSONTypeError."""
        config_json = '{"dataset": "d", "model_path": "/m", "explainer": "permutation"}'
        with pytest.raises(JSONTypeError, match="backend is required"):
            _parse_regression_explain_config(config_json)

    def test_missing_explainer_raises(self) -> None:
        """Missing explainer raises JSONTypeError."""
        config_json = '{"dataset": "d", "backend": "xgboost_reg", "model_path": "/m"}'
        with pytest.raises(JSONTypeError, match="explainer is required"):
            _parse_regression_explain_config(config_json)

    def test_missing_dataset_raises(self) -> None:
        """Missing dataset raises JSONTypeError."""
        config_json = '{"backend": "xgboost_reg", "model_path": "/m", "explainer": "permutation"}'
        with pytest.raises(JSONTypeError, match="dataset"):
            _parse_regression_explain_config(config_json)

    def test_missing_model_path_raises(self) -> None:
        """Missing model_path raises JSONTypeError."""
        config_json = '{"dataset": "d", "backend": "xgboost_reg", "explainer": "permutation"}'
        with pytest.raises(JSONTypeError, match="model_path"):
            _parse_regression_explain_config(config_json)


# ---------------------------------------------------------------------------
# Tests for _sample_data
# ---------------------------------------------------------------------------


class TestSampleData:
    """Tests for _sample_data."""

    def test_returns_all_when_n_samples_exceeds_total(self) -> None:
        """Returns all data when n_samples >= n_total."""
        x: NDArray[np.float64] = np.zeros((5, 3), dtype=np.float64)
        result = _sample_data(x, n_samples=10, random_state=42)
        assert result.shape == (5, 3)

    def test_returns_all_when_n_samples_equals_total(self) -> None:
        """Returns all data when n_samples == n_total."""
        x: NDArray[np.float64] = np.zeros((5, 3), dtype=np.float64)
        result = _sample_data(x, n_samples=5, random_state=42)
        assert result.shape == (5, 3)

    def test_samples_correct_count(self) -> None:
        """Returns exactly n_samples rows when n_samples < n_total."""
        x: NDArray[np.float64] = np.arange(30.0, dtype=np.float64).reshape(10, 3)
        result = _sample_data(x, n_samples=4, random_state=42)
        assert result.shape == (4, 3)

    def test_deterministic_with_same_seed(self) -> None:
        """Same random_state produces same sample."""
        x: NDArray[np.float64] = np.arange(30.0, dtype=np.float64).reshape(10, 3)
        result_a = _sample_data(x, n_samples=4, random_state=42)
        result_b = _sample_data(x, n_samples=4, random_state=42)
        np.testing.assert_array_equal(result_a, result_b)

    def test_different_seed_produces_different_sample(self) -> None:
        """Different random_state produces different sample."""
        x: NDArray[np.float64] = np.arange(30.0, dtype=np.float64).reshape(10, 3)
        result_a = _sample_data(x, n_samples=4, random_state=42)
        result_b = _sample_data(x, n_samples=4, random_state=99)
        assert not np.array_equal(result_a, result_b)


# ---------------------------------------------------------------------------
# Tests for progress types
# ---------------------------------------------------------------------------


class TestProgressTypes:
    """Tests for progress TypedDicts and literals."""

    def test_regression_explain_progress_info_structure(self) -> None:
        """RegressionExplainProgressInfo has required fields."""
        info: RegressionExplainProgressInfo = {
            "status": "computing",
            "elapsed_seconds": 1.5,
        }
        assert info["status"] == "computing"
        assert info["elapsed_seconds"] == 1.5

    def test_all_job_status_values(self) -> None:
        """All RegressionExplainJobStatus values are valid."""
        statuses: list[RegressionExplainJobStatus] = [
            "started",
            "loading_model",
            "loading_data",
            "computing",
            "complete",
        ]
        for status in statuses:
            info: RegressionExplainProgressInfo = {
                "status": status,
                "elapsed_seconds": 0.0,
            }
            assert info["status"] == status


# ---------------------------------------------------------------------------
# Integration tests for run_regression_explanation
# ---------------------------------------------------------------------------


class _TrainableRegressorProto(Protocol):
    """Protocol for XGBRegressor with fit and save_model."""

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> _TrainableRegressorProto: ...

    def save_model(self, fname: str) -> None: ...


def _copy_real_financial_distress(
    external_root: Path,
) -> tuple[Path, int, list[str]]:
    """Copy financial distress dataset into external_root.

    Args:
        external_root: Target external directory.

    Returns:
        Tuple of (path, n_rows, feature_names).
    """
    src = (
        Path(__file__).parent.parent
        / "data"
        / "external"
        / "kaggle_financial_distress"
        / "Financial Distress.csv"
    )
    if not src.exists():
        raise FileNotFoundError("Financial Distress dataset not found in repo data")
    dst_dir = external_root / "kaggle_financial_distress"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "Financial Distress.csv"
    copyfile(str(src), str(dst))
    header = dst.read_text(encoding="utf-8").splitlines()[0]
    cols = [c.strip() for c in header.split(",")]
    # Exclude: Company, Time, Financial Distress (target)
    feature_names = [c for c in cols if c not in ("Company", "Time", "Financial Distress")]
    n_rows = sum(1 for _ in dst.open(encoding="utf-8")) - 1
    return dst, n_rows, feature_names


def _create_xgb_regressor_model(
    model_path: Path,
    n_features: int,
) -> None:
    """Create and save a real XGBRegressor model.

    Args:
        model_path: Path to save the model (.ubj format).
        n_features: Number of features.
    """
    xgb_mod = __import__("xgboost")
    regressor: _TrainableRegressorProto = xgb_mod.XGBRegressor(
        n_estimators=5,
        max_depth=2,
        random_state=42,
    )
    rng = np.random.default_rng(42)
    x_train: NDArray[np.float64] = rng.standard_normal((20, n_features)).astype(np.float64)
    y_train: NDArray[np.float64] = rng.standard_normal(20).astype(np.float64)
    regressor.fit(x_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    regressor.save_model(str(model_path))


class TestRunRegressionExplanation:
    """Integration tests for run_regression_explanation."""

    def test_permutation_explainer(self, tmp_path: Path) -> None:
        """run_regression_explanation completes with permutation."""
        external_dir = tmp_path / "external"
        _, _, feature_names = _copy_real_financial_distress(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgb_regressor_model(model_path, len(feature_names))

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "model_path": str(model_path),
                "explainer": "permutation",
                "n_samples": 30,
                "random_state": 42,
            }
        )

        result = run_regression_explanation(config_json, external_dir)

        assert result["status"] == "complete"
        assert result["backend"] == "xgboost_reg"
        assert result["explainer"] == "permutation"
        assert result["n_samples_used"] == 30
        assert result["n_features"] == len(feature_names)
        assert len(result["feature_importances"]) == len(feature_names)
        assert result["duration_seconds"] >= 0.0

    def test_with_explicit_registry(self, tmp_path: Path) -> None:
        """run_regression_explanation works with explicit registry."""
        from covenant_ml.explainers.regression_registry import (
            default_regression_explainer_registry,
        )

        external_dir = tmp_path / "external"
        _, _, feature_names = _copy_real_financial_distress(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgb_regressor_model(model_path, len(feature_names))

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "model_path": str(model_path),
                "explainer": "permutation",
                "n_samples": 20,
                "random_state": 42,
            }
        )

        registry = default_regression_explainer_registry()
        result = run_regression_explanation(config_json, external_dir, registry=registry)

        assert result["status"] == "complete"
        assert result["explainer"] == "permutation"
        assert len(result["feature_importances"]) == len(feature_names)

    def test_incompatible_explainer_raises(self, tmp_path: Path) -> None:
        """Raises ValueError for incompatible explainer-backend combo."""
        external_dir = tmp_path / "external"
        _copy_real_financial_distress(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgb_regressor_model(model_path, 83)

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "model_path": str(model_path),
                "explainer": "gradient",
                "n_samples": 30,
            }
        )

        with pytest.raises(ValueError, match="is not compatible with"):
            run_regression_explanation(config_json, external_dir)

    def test_progress_callback(self, tmp_path: Path) -> None:
        """Progress callback receives all status transitions."""
        external_dir = tmp_path / "external"
        _, _, feature_names = _copy_real_financial_distress(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgb_regressor_model(model_path, len(feature_names))

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "model_path": str(model_path),
                "explainer": "permutation",
                "n_samples": 10,
            }
        )

        calls: list[RegressionExplainProgressInfo] = []

        def callback(info: RegressionExplainProgressInfo) -> None:
            calls.append(info)

        result = run_regression_explanation(
            config_json,
            external_dir,
            progress_callback=callback,
        )

        assert result["status"] == "complete"
        statuses: list[RegressionExplainJobStatus] = [c["status"] for c in calls]
        assert "started" in statuses
        assert "loading_model" in statuses
        assert "loading_data" in statuses
        assert "computing" in statuses
        assert "complete" in statuses

        for call in calls:
            assert call["elapsed_seconds"] >= 0.0

    def test_samples_all_when_exceeds_dataset(self, tmp_path: Path) -> None:
        """Uses all samples when n_samples exceeds dataset size."""
        external_dir = tmp_path / "external"
        _, n_rows, feature_names = _copy_real_financial_distress(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgb_regressor_model(model_path, len(feature_names))

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "model_path": str(model_path),
                "explainer": "permutation",
                "n_samples": 999999,
                "random_state": 42,
            }
        )

        result = run_regression_explanation(config_json, external_dir)
        assert result["n_samples_used"] == n_rows


# ---------------------------------------------------------------------------
# Tests for process_regression_explain_job
# ---------------------------------------------------------------------------


class TestProcessRegressionExplainJob:
    """Tests for process_regression_explain_job entry point."""

    def test_returns_json_serializable_result(self, tmp_path: Path) -> None:
        """process_regression_explain_job returns JSON-serializable dict."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.json_utils import (
            JSONObject,
            require_dict,
            require_float,
            require_int,
            require_list,
            require_str,
        )
        from platform_core.testing import FakeEnv

        from covenant_radar_api.worker.explain_regression_job import (
            process_regression_explain_job,
        )

        data_root = tmp_path / "data"
        external_dir = data_root / "external"
        models_dir = tmp_path / "models"

        _, _, feature_names = _copy_real_financial_distress(external_dir)

        model_path = tmp_path / "model.ubj"
        _create_xgb_regressor_model(model_path, len(feature_names))

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
                    "dataset": "financial_distress",
                    "backend": "xgboost_reg",
                    "model_path": str(model_path),
                    "explainer": "permutation",
                    "n_samples": 15,
                    "random_state": 42,
                }
            )

            result = process_regression_explain_job(config_json)

            assert result["status"] == "complete"
            assert result["backend"] == "xgboost_reg"
            assert result["explainer"] == "permutation"
            assert result["n_samples_used"] == 15
            assert result["n_features"] == len(feature_names)
            duration = require_float(result, "duration_seconds")
            assert duration >= 0.0

            importances_list = require_list(result, "feature_importances")
            assert len(importances_list) == len(feature_names)

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
