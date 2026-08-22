"""Tests for feature importance explanation job."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
)

from covenant_radar_api.worker.explain_job import (
    _parse_explain_config,
    _parse_explainer,
    _parse_int_tuple,
    _parse_lstm_config,
    _parse_mlp_config,
    _require_bool_field,
    _require_float_field,
    _require_int_field,
)


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
