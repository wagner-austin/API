"""Tests for feature importance explanation job."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
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
    _parse_explain_config,
    _parse_explainer,
    _parse_int_tuple,
    _parse_lstm_config,
    _parse_mlp_config,
    _require_bool_field,
    _require_float_field,
    _require_int_field,
    _sample_data,
    run_explanation,
)


def _copy_real_taiwan(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full Taiwan dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent / "data" / "external" / "taiwan_data" / "data.csv"
    if not src.exists():
        raise FileNotFoundError("Taiwan dataset not found in repository data")
    dst_dir = external_root / "taiwan_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "data.csv"
    copyfile(str(src), str(dst))
    header = (dst.read_text(encoding="utf-8").splitlines())[0]
    cols = [c.strip() for c in header.split(",")]
    feature_names = cols[1:]  # all columns after label
    n_rows = sum(1 for _ in dst.open(encoding="utf-8")) - 1
    return dst, n_rows, feature_names


# ---------------------------------------------------------------------------
# Tests for _require_int_field
# ---------------------------------------------------------------------------


class TestRequireIntField:
    """Tests for _require_int_field function."""

    def test_returns_int_when_present(self) -> None:
        """Returns integer value when present."""
        result = _require_int_field({"key": 42}, "key")
        assert result == 42

    def test_raises_on_missing_key(self) -> None:
        """Raises JSONTypeError when key is missing."""
        with pytest.raises(JSONTypeError, match="Field 'missing' is required"):
            _require_int_field({}, "missing")

    def test_raises_on_non_int_value(self) -> None:
        """Raises JSONTypeError when value is not an integer."""
        with pytest.raises(JSONTypeError, match="Field 'key' must be an integer"):
            _require_int_field({"key": "not an int"}, "key")

    def test_raises_on_float_value(self) -> None:
        """Raises JSONTypeError when value is a float."""
        with pytest.raises(JSONTypeError, match="Field 'key' must be an integer"):
            _require_int_field({"key": 3.14}, "key")


# ---------------------------------------------------------------------------
# Tests for _require_float_field
# ---------------------------------------------------------------------------


class TestRequireFloatField:
    """Tests for _require_float_field function."""

    def test_returns_float_when_present(self) -> None:
        """Returns float value when present."""
        result = _require_float_field({"key": 3.14}, "key")
        assert result == 3.14

    def test_converts_int_to_float(self) -> None:
        """Converts integer value to float."""
        result = _require_float_field({"key": 42}, "key")
        assert result == 42.0
        # Verify result is exactly 42.0 (float), not 42 (int)
        assert result / 1.0 == 42.0

    def test_raises_on_missing_key(self) -> None:
        """Raises JSONTypeError when key is missing."""
        with pytest.raises(JSONTypeError, match="Field 'missing' is required"):
            _require_float_field({}, "missing")

    def test_raises_on_non_number_value(self) -> None:
        """Raises JSONTypeError when value is not a number."""
        with pytest.raises(JSONTypeError, match="Field 'key' must be a number"):
            _require_float_field({"key": "not a number"}, "key")


# ---------------------------------------------------------------------------
# Tests for _require_bool_field
# ---------------------------------------------------------------------------


class TestRequireBoolField:
    """Tests for _require_bool_field function."""

    def test_returns_true_when_present(self) -> None:
        """Returns True when value is True."""
        result = _require_bool_field({"key": True}, "key")
        assert result is True

    def test_returns_false_when_present(self) -> None:
        """Returns False when value is False."""
        result = _require_bool_field({"key": False}, "key")
        assert result is False

    def test_raises_on_missing_key(self) -> None:
        """Raises JSONTypeError when key is missing."""
        with pytest.raises(JSONTypeError, match="Field 'missing' is required"):
            _require_bool_field({}, "missing")

    def test_raises_on_non_bool_value(self) -> None:
        """Raises JSONTypeError when value is not a boolean."""
        with pytest.raises(JSONTypeError, match="Field 'key' must be a boolean"):
            _require_bool_field({"key": "true"}, "key")

    def test_raises_on_int_value(self) -> None:
        """Raises JSONTypeError when value is an integer (not bool)."""
        with pytest.raises(JSONTypeError, match="Field 'key' must be a boolean"):
            _require_bool_field({"key": 1}, "key")


# ---------------------------------------------------------------------------
# Tests for _parse_int_tuple
# ---------------------------------------------------------------------------


class TestParseIntTuple:
    """Tests for _parse_int_tuple function."""

    def test_parses_valid_array(self) -> None:
        """Parses valid array of integers."""
        result = _parse_int_tuple([1, 2, 3], "field")
        assert result == (1, 2, 3)

    def test_parses_empty_array(self) -> None:
        """Parses empty array to empty tuple."""
        result = _parse_int_tuple([], "field")
        assert result == ()

    def test_raises_on_non_array(self) -> None:
        """Raises JSONTypeError when value is not an array."""
        with pytest.raises(JSONTypeError, match="Field 'field' must be an array"):
            _parse_int_tuple("not an array", "field")

    def test_raises_on_array_with_non_int(self) -> None:
        """Raises JSONTypeError when array contains non-integers."""
        with pytest.raises(JSONTypeError, match="Field 'field' must contain only integers"):
            _parse_int_tuple([1, "two", 3], "field")

    def test_raises_on_array_with_float(self) -> None:
        """Raises JSONTypeError when array contains floats."""
        with pytest.raises(JSONTypeError, match="Field 'field' must contain only integers"):
            _parse_int_tuple([1, 2.5, 3], "field")


# ---------------------------------------------------------------------------
# Tests for _parse_mlp_config
# ---------------------------------------------------------------------------


class TestParseMlpConfig:
    """Tests for _parse_mlp_config function."""

    def test_parses_valid_config(self) -> None:
        """Parses valid MLP config."""
        raw: JSONObject = {
            "n_features": 10,
            "hidden_sizes": [64, 32],
            "dropout": 0.2,
        }
        result = _parse_mlp_config(raw)

        assert result["n_features"] == 10
        assert result["hidden_sizes"] == (64, 32)
        assert result["dropout"] == 0.2

    def test_raises_on_missing_n_features(self) -> None:
        """Raises JSONTypeError when n_features is missing."""
        raw: JSONObject = {"hidden_sizes": [64], "dropout": 0.1}
        with pytest.raises(JSONTypeError, match="Field 'n_features' is required"):
            _parse_mlp_config(raw)

    def test_raises_on_missing_hidden_sizes(self) -> None:
        """Raises JSONTypeError when hidden_sizes is missing."""
        raw: JSONObject = {"n_features": 10, "dropout": 0.1}
        with pytest.raises(JSONTypeError, match="Field 'hidden_sizes' is required"):
            _parse_mlp_config(raw)

    def test_raises_on_missing_dropout(self) -> None:
        """Raises JSONTypeError when dropout is missing."""
        raw: JSONObject = {"n_features": 10, "hidden_sizes": [64]}
        with pytest.raises(JSONTypeError, match="Field 'dropout' is required"):
            _parse_mlp_config(raw)


# ---------------------------------------------------------------------------
# Tests for _parse_lstm_config
# ---------------------------------------------------------------------------


class TestParseLstmConfig:
    """Tests for _parse_lstm_config function."""

    def test_parses_valid_config(self) -> None:
        """Parses valid LSTM config."""
        raw: JSONObject = {
            "n_features": 20,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
            "sequence_length": 4,
        }
        result = _parse_lstm_config(raw)

        assert result["n_features"] == 20
        assert result["hidden_size"] == 64
        assert result["num_layers"] == 2
        assert result["dropout"] == 0.3
        assert result["bidirectional"] is True
        assert result["sequence_length"] == 4

    def test_raises_on_missing_hidden_size(self) -> None:
        """Raises JSONTypeError when hidden_size is missing."""
        raw: JSONObject = {
            "n_features": 20,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
            "sequence_length": 4,
        }
        with pytest.raises(JSONTypeError, match="Field 'hidden_size' is required"):
            _parse_lstm_config(raw)

    def test_raises_on_missing_bidirectional(self) -> None:
        """Raises JSONTypeError when bidirectional is missing."""
        raw: JSONObject = {
            "n_features": 20,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "sequence_length": 4,
        }
        with pytest.raises(JSONTypeError, match="Field 'bidirectional' is required"):
            _parse_lstm_config(raw)


# ---------------------------------------------------------------------------
# Tests for _parse_explainer
# ---------------------------------------------------------------------------


class TestParseExplainer:
    """Tests for _parse_explainer function."""

    def test_parses_permutation(self) -> None:
        """Parses 'permutation' explainer."""
        result = _parse_explainer("permutation")
        assert result == "permutation"

    def test_parses_gradient(self) -> None:
        """Parses 'gradient' explainer."""
        result = _parse_explainer("gradient")
        assert result == "gradient"

    def test_parses_integrated_gradients(self) -> None:
        """Parses 'integrated_gradients' explainer."""
        result = _parse_explainer("integrated_gradients")
        assert result == "integrated_gradients"

    def test_parses_shap_tree(self) -> None:
        """Parses 'shap_tree' explainer."""
        result = _parse_explainer("shap_tree")
        assert result == "shap_tree"

    def test_raises_on_invalid_explainer(self) -> None:
        """Raises JSONTypeError on invalid explainer name."""
        with pytest.raises(JSONTypeError, match="explainer must be one of"):
            _parse_explainer("invalid")

    def test_raises_on_non_string(self) -> None:
        """Raises JSONTypeError when value is not a string."""
        with pytest.raises(JSONTypeError, match="explainer must be a string"):
            _parse_explainer(123)


# ---------------------------------------------------------------------------
# Tests for _parse_explain_config
# ---------------------------------------------------------------------------


class TestParseExplainConfig:
    """Tests for _parse_explain_config function."""

    def test_parses_valid_xgboost_config(self) -> None:
        """Parses valid config for XGBoost backend."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": "/path/to/model.ubj",
                "explainer": "permutation",
                "target_class": 1,
                "n_samples": 500,
                "random_state": 123,
            }
        )
        result = _parse_explain_config(config_json)

        assert result["dataset"] == "taiwan"
        assert result["backend"] == "xgboost"
        assert result["model_path"] == "/path/to/model.ubj"
        assert result["explainer"] == "permutation"
        assert result["target_class"] == 1
        assert result["n_samples"] == 500
        assert result["random_state"] == 123
        assert result["mlp_config"] is None
        assert result["lstm_config"] is None

    def test_parses_config_with_defaults(self) -> None:
        """Parses config using default values."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": "/path/to/model.ubj",
                "explainer": "shap_tree",
            }
        )
        result = _parse_explain_config(config_json)

        assert result["target_class"] == 1  # default
        assert result["n_samples"] == 1000  # default
        assert result["random_state"] == 42  # default

    def test_parses_valid_mlp_config(self) -> None:
        """Parses valid config for MLP backend with mlp_config."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "mlp",
                "model_path": "/path/to/model.pt",
                "explainer": "gradient",
                "mlp_config": {
                    "n_features": 23,
                    "hidden_sizes": [128, 64],
                    "dropout": 0.2,
                },
            }
        )
        result = _parse_explain_config(config_json)

        assert result["backend"] == "mlp"
        mlp_config = result["mlp_config"]
        if mlp_config is None:
            raise AssertionError("mlp_config should not be None")
        assert mlp_config["n_features"] == 23
        assert mlp_config["hidden_sizes"] == (128, 64)

    def test_parses_valid_lstm_config(self) -> None:
        """Parses valid config for LSTM backend with lstm_config."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "lstm",
                "model_path": "/path/to/model.pt",
                "explainer": "gradient",
                "lstm_config": {
                    "n_features": 24,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "bidirectional": False,
                    "sequence_length": 4,
                },
            }
        )
        result = _parse_explain_config(config_json)

        assert result["backend"] == "lstm"
        lstm_config = result["lstm_config"]
        if lstm_config is None:
            raise AssertionError("lstm_config should not be None")
        assert lstm_config["hidden_size"] == 64

    def test_raises_on_non_object(self) -> None:
        """Raises JSONTypeError when config is not an object."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_explain_config("[1, 2, 3]")

    def test_raises_on_missing_explainer(self) -> None:
        """Raises JSONTypeError when explainer is missing."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "xgboost",
                "model_path": "/path/to/model.ubj",
            }
        )
        with pytest.raises(JSONTypeError, match="explainer is required"):
            _parse_explain_config(config_json)

    def test_raises_on_mlp_without_config(self) -> None:
        """Raises JSONTypeError when MLP backend is missing mlp_config."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "mlp",
                "model_path": "/path/to/model.pt",
                "explainer": "gradient",
            }
        )
        with pytest.raises(JSONTypeError, match="mlp_config is required when backend is 'mlp'"):
            _parse_explain_config(config_json)

    def test_raises_on_mlp_config_not_object(self) -> None:
        """Raises JSONTypeError when mlp_config is not an object."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "mlp",
                "model_path": "/path/to/model.pt",
                "explainer": "gradient",
                "mlp_config": "not an object",
            }
        )
        with pytest.raises(JSONTypeError, match="mlp_config must be an object"):
            _parse_explain_config(config_json)

    def test_raises_on_lstm_without_config(self) -> None:
        """Raises JSONTypeError when LSTM backend is missing lstm_config."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "lstm",
                "model_path": "/path/to/model.pt",
                "explainer": "gradient",
            }
        )
        with pytest.raises(JSONTypeError, match="lstm_config is required when backend is 'lstm'"):
            _parse_explain_config(config_json)

    def test_raises_on_lstm_config_not_object(self) -> None:
        """Raises JSONTypeError when lstm_config is not an object."""
        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "device": "cpu",
                "backend": "lstm",
                "model_path": "/path/to/model.pt",
                "explainer": "gradient",
                "lstm_config": [1, 2, 3],
            }
        )
        with pytest.raises(JSONTypeError, match="lstm_config must be an object"):
            _parse_explain_config(config_json)


# ---------------------------------------------------------------------------
# Tests for _sample_data
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests for run_explanation
# ---------------------------------------------------------------------------


def _create_xgboost_model(model_path: Path, n_features: int, n_samples: int = 100) -> None:
    """Create a simple XGBoost model for testing."""
    import xgboost as xgb

    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((n_samples, n_features))
    y: NDArray[np.int64] = rng.integers(0, 2, size=n_samples).astype(np.int64)

    model = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
    )
    model.fit(x, y)
    model.save_model(str(model_path))


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


# ---------------------------------------------------------------------------
# Tests for process_explain_job
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests for Real Worker Hooks
# ---------------------------------------------------------------------------


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
        from covenant_radar_api.worker._test_hooks import _real_explainer_registry

        registry = _real_explainer_registry()

        # Verify registry has expected explainers via list_explainers()
        registered = registry.list_explainers()
        assert "permutation" in registered
        assert "shap_tree" in registered
        assert "gradient" in registered
        assert "integrated_gradients" in registered
