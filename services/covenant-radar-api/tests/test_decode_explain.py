"""Tests for HTTP request body parsing."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.api.decode_ml import (
    parse_explain_request,
)
from covenant_radar_api.api.decode_regression import (
    parse_regression_explain_request,
    parse_regression_predict_request,
)


class TestParseExplainRequest:
    """Tests for parse_explain_request."""

    def test_valid_explain_request_full(self) -> None:
        """Test parsing valid explain request with all fields."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "xgboost",
            "model_path": "/models/xgboost.ubj",
            "explainer": "permutation",
            "target_class": 1,
            "n_samples": 500,
            "random_state": 42
        }"""
        result = parse_explain_request(body)

        assert result["dataset"] == "taiwan"
        assert result["backend"] == "xgboost"
        assert result["model_path"] == "/models/xgboost.ubj"
        assert result["explainer"] == "permutation"
        assert result["target_class"] == 1
        assert result["n_samples"] == 500
        assert result["random_state"] == 42

    def test_valid_explain_request_minimal(self) -> None:
        """Test parsing valid explain request with defaults for optional fields."""
        body = b"""{
            "dataset": "us",
            "backend": "mlp",
            "model_path": "/models/mlp.pt",
            "explainer": "gradient"
        }"""
        result = parse_explain_request(body)

        assert result["dataset"] == "us"
        assert result["backend"] == "mlp"
        assert result["model_path"] == "/models/mlp.pt"
        assert result["explainer"] == "gradient"
        assert result["target_class"] == 1
        assert result["n_samples"] == 1000
        assert result["random_state"] == 42

    def test_valid_explainer_integrated_gradients(self) -> None:
        """Test parsing with integrated_gradients explainer."""
        body = b"""{
            "dataset": "polish",
            "backend": "lstm",
            "model_path": "/models/lstm.pt",
            "explainer": "integrated_gradients"
        }"""
        result = parse_explain_request(body)

        assert result["explainer"] == "integrated_gradients"
        assert result["backend"] == "lstm"

    def test_valid_explainer_shap_tree(self) -> None:
        """Test parsing with shap_tree explainer."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lightgbm",
            "model_path": "/models/lgbm.txt",
            "explainer": "shap_tree"
        }"""
        result = parse_explain_request(body)

        assert result["explainer"] == "shap_tree"
        assert result["backend"] == "lightgbm"

    def test_all_valid_backends(self) -> None:
        """Test parsing with all valid backend types."""
        backends = [
            "xgboost",
            "mlp",
            "lstm",
            "lightgbm",
            "cleargbm",
            "logreg",
            "random_forest",
        ]
        for backend in backends:
            body = f'''{{
                "dataset": "taiwan",
                "backend": "{backend}",
                "model_path": "/models/model",
                "explainer": "permutation"
            }}'''.encode()
            result = parse_explain_request(body)
            assert result["backend"] == backend

    def test_all_valid_datasets(self) -> None:
        """Test parsing with all valid dataset types."""
        datasets = ["taiwan", "us", "polish"]
        for dataset in datasets:
            body = f'''{{
                "dataset": "{dataset}",
                "backend": "xgboost",
                "model_path": "/models/model.ubj",
                "explainer": "permutation"
            }}'''.encode()
            result = parse_explain_request(body)
            assert result["dataset"] == dataset

    def test_missing_dataset_raises_json_type_error(self) -> None:
        """Test that missing dataset raises JSONTypeError."""
        body = b"""{
            "backend": "xgboost",
            "model_path": "/models/model.ubj",
            "explainer": "permutation"
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'dataset'"):
            parse_explain_request(body)

    def test_missing_backend_raises_json_type_error(self) -> None:
        """Test that missing backend raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "model_path": "/models/model.ubj",
            "explainer": "permutation"
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'backend'"):
            parse_explain_request(body)

    def test_missing_model_path_raises_json_type_error(self) -> None:
        """Test that missing model_path raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "xgboost",
            "explainer": "permutation"
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'model_path'"):
            parse_explain_request(body)

    def test_missing_explainer_raises_json_type_error(self) -> None:
        """Test that missing explainer raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "xgboost",
            "model_path": "/models/model.ubj"
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'explainer'"):
            parse_explain_request(body)

    def test_invalid_dataset_raises_value_error(self) -> None:
        """Test that invalid dataset raises ValueError."""
        body = b"""{
            "dataset": "invalid_dataset",
            "backend": "xgboost",
            "model_path": "/models/model.ubj",
            "explainer": "permutation"
        }"""
        with pytest.raises(ValueError, match="dataset must be one of"):
            parse_explain_request(body)

    def test_invalid_backend_raises_json_type_error(self) -> None:
        """Test that invalid backend raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "invalid_backend",
            "model_path": "/models/model.ubj",
            "explainer": "permutation"
        }"""
        with pytest.raises(JSONTypeError, match="backend must be one of"):
            parse_explain_request(body)

    def test_invalid_explainer_raises_json_type_error(self) -> None:
        """Test that invalid explainer raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "xgboost",
            "model_path": "/models/model.ubj",
            "explainer": "invalid_explainer"
        }"""
        with pytest.raises(JSONTypeError, match="explainer must be one of"):
            parse_explain_request(body)

    def test_non_string_backend_raises_json_type_error(self) -> None:
        """Test that non-string backend raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": 123,
            "model_path": "/models/model.ubj",
            "explainer": "permutation"
        }"""
        with pytest.raises(JSONTypeError, match="backend must be a string"):
            parse_explain_request(body)

    def test_non_string_explainer_raises_json_type_error(self) -> None:
        """Test that non-string explainer raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "xgboost",
            "model_path": "/models/model.ubj",
            "explainer": 123
        }"""
        with pytest.raises(JSONTypeError, match="explainer must be a string"):
            parse_explain_request(body)

    def test_invalid_target_class_type_raises_json_type_error(self) -> None:
        """Test that non-numeric target_class raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "xgboost",
            "model_path": "/models/model.ubj",
            "explainer": "permutation",
            "target_class": "one"
        }"""
        with pytest.raises(JSONTypeError, match="Field 'target_class' must be a number"):
            parse_explain_request(body)

    def test_invalid_n_samples_type_raises_json_type_error(self) -> None:
        """Test that non-numeric n_samples raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "xgboost",
            "model_path": "/models/model.ubj",
            "explainer": "permutation",
            "n_samples": "many"
        }"""
        with pytest.raises(JSONTypeError, match="Field 'n_samples' must be a number"):
            parse_explain_request(body)

    def test_target_class_zero(self) -> None:
        """Test parsing with target_class set to 0."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "xgboost",
            "model_path": "/models/model.ubj",
            "explainer": "permutation",
            "target_class": 0
        }"""
        result = parse_explain_request(body)

        assert result["target_class"] == 0

    def test_custom_n_samples_and_random_state(self) -> None:
        """Test parsing with custom n_samples and random_state."""
        body = b"""{
            "dataset": "polish",
            "backend": "mlp",
            "model_path": "/models/mlp.pt",
            "explainer": "gradient",
            "n_samples": 2000,
            "random_state": 123
        }"""
        result = parse_explain_request(body)

        assert result["n_samples"] == 2000
        assert result["random_state"] == 123


class TestParseRegressionPredictRequest:
    """Tests for parse_regression_predict_request."""

    def test_valid_request(self) -> None:
        """Test parsing valid regression predict request."""
        body = b"""{
            "backend": "xgboost_reg",
            "model_path": "/models/regressor.ubj",
            "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        }"""
        result = parse_regression_predict_request(body)

        assert result["backend"] == "xgboost_reg"
        assert result["model_path"] == "/models/regressor.ubj"
        assert result["features"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_all_backends_accepted(self) -> None:
        """All 4 regressor backends are accepted."""
        for backend in ("xgboost_reg", "lightgbm_reg", "mlp_reg", "lstm_reg"):
            body = f'{{"backend": "{backend}", "model_path": "/m", "features": [[1.0]]}}'.encode()
            result = parse_regression_predict_request(body)
            assert result["backend"] == backend

    def test_default_backend_is_xgboost_reg(self) -> None:
        """Missing backend defaults to xgboost_reg."""
        body = b'{"model_path": "/m", "features": [[1.0]]}'
        result = parse_regression_predict_request(body)
        assert result["backend"] == "xgboost_reg"

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        body = b'{"backend": "invalid", "model_path": "/m", "features": [[1.0]]}'
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_regression_predict_request(body)

    def test_missing_model_path_raises(self) -> None:
        """Missing model_path raises JSONTypeError."""
        body = b'{"backend": "xgboost_reg", "features": [[1.0]]}'
        with pytest.raises(JSONTypeError, match="model_path"):
            parse_regression_predict_request(body)

    def test_missing_features_raises(self) -> None:
        """Missing features raises JSONTypeError."""
        body = b'{"backend": "xgboost_reg", "model_path": "/m"}'
        with pytest.raises(JSONTypeError, match="features"):
            parse_regression_predict_request(body)

    def test_empty_features_raises(self) -> None:
        """Empty features list raises JSONTypeError."""
        body = b'{"backend": "xgboost_reg", "model_path": "/m", "features": []}'
        with pytest.raises(JSONTypeError, match="non-empty"):
            parse_regression_predict_request(body)

    def test_non_list_feature_row_raises(self) -> None:
        """Non-list feature row raises JSONTypeError."""
        body = b'{"backend": "xgboost_reg", "model_path": "/m", "features": ["bad"]}'
        with pytest.raises(JSONTypeError, match=r"features\[0\] must be a list"):
            parse_regression_predict_request(body)

    def test_non_number_feature_value_raises(self) -> None:
        """Non-number feature value raises JSONTypeError."""
        body = b'{"backend": "xgboost_reg", "model_path": "/m", "features": [["bad"]]}'
        with pytest.raises(JSONTypeError, match=r"features\[0\]\[0\] must be a number"):
            parse_regression_predict_request(body)

    def test_integer_features_converted_to_float(self) -> None:
        """Integer feature values are converted to float."""
        body = b'{"backend": "xgboost_reg", "model_path": "/m", "features": [[1, 2, 3]]}'
        result = parse_regression_predict_request(body)
        assert result["features"] == [[1.0, 2.0, 3.0]]

    def test_non_object_body_raises(self) -> None:
        """Non-object JSON body raises JSONTypeError."""
        body = b'"just a string"'
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            parse_regression_predict_request(body)

    def test_multiple_samples(self) -> None:
        """Multiple samples are parsed correctly."""
        body = b"""{
            "backend": "lightgbm_reg",
            "model_path": "/m.txt",
            "features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        }"""
        result = parse_regression_predict_request(body)
        assert len(result["features"]) == 3
        assert result["features"][2] == [5.0, 6.0]


class TestParseRegressionExplainRequest:
    """Tests for parse_regression_explain_request."""

    def test_valid_request_full(self) -> None:
        """Parse valid regression explain request with all fields."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "xgboost_reg",
            "model_path": "/models/xgb_reg.ubj",
            "explainer": "permutation",
            "n_samples": 500,
            "random_state": 99
        }"""
        result = parse_regression_explain_request(body)

        assert result["dataset"] == "financial_distress"
        assert result["backend"] == "xgboost_reg"
        assert result["model_path"] == "/models/xgb_reg.ubj"
        assert result["explainer"] == "permutation"
        assert result["n_samples"] == 500
        assert result["random_state"] == 99

    def test_valid_request_minimal(self) -> None:
        """Parse regression explain request with defaults."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "lightgbm_reg",
            "model_path": "/models/lgbm_reg.txt",
            "explainer": "shap_tree"
        }"""
        result = parse_regression_explain_request(body)

        assert result["dataset"] == "financial_distress"
        assert result["backend"] == "lightgbm_reg"
        assert result["explainer"] == "shap_tree"
        assert result["n_samples"] == 1000
        assert result["random_state"] == 42

    def test_all_backends_accepted(self) -> None:
        """All 4 regressor backends are accepted."""
        for backend in (
            "xgboost_reg",
            "lightgbm_reg",
            "mlp_reg",
            "lstm_reg",
        ):
            body = (
                f'{{"dataset": "financial_distress", '
                f'"backend": "{backend}", '
                f'"model_path": "/m", '
                f'"explainer": "permutation"}}'
            ).encode()
            result = parse_regression_explain_request(body)
            assert result["backend"] == backend

    def test_all_explainers_accepted(self) -> None:
        """All 4 explainer types are accepted."""
        for explainer in (
            "permutation",
            "gradient",
            "integrated_gradients",
            "shap_tree",
        ):
            body = (
                f'{{"dataset": "financial_distress", '
                f'"backend": "xgboost_reg", '
                f'"model_path": "/m", '
                f'"explainer": "{explainer}"}}'
            ).encode()
            result = parse_regression_explain_request(body)
            assert result["explainer"] == explainer

    def test_missing_dataset_raises(self) -> None:
        """Missing dataset raises JSONTypeError."""
        body = b"""{
            "backend": "xgboost_reg",
            "model_path": "/m",
            "explainer": "permutation"
        }"""
        with pytest.raises(JSONTypeError, match="dataset"):
            parse_regression_explain_request(body)

    def test_missing_backend_raises(self) -> None:
        """Missing backend raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "model_path": "/m",
            "explainer": "permutation"
        }"""
        with pytest.raises(
            JSONTypeError,
            match="Missing required field 'backend'",
        ):
            parse_regression_explain_request(body)

    def test_missing_model_path_raises(self) -> None:
        """Missing model_path raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "xgboost_reg",
            "explainer": "permutation"
        }"""
        with pytest.raises(JSONTypeError, match="model_path"):
            parse_regression_explain_request(body)

    def test_missing_explainer_raises(self) -> None:
        """Missing explainer raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "xgboost_reg",
            "model_path": "/m"
        }"""
        with pytest.raises(
            JSONTypeError,
            match="Missing required field 'explainer'",
        ):
            parse_regression_explain_request(body)

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "invalid",
            "model_path": "/m",
            "explainer": "permutation"
        }"""
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_regression_explain_request(body)

    def test_invalid_explainer_raises(self) -> None:
        """Invalid explainer raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "xgboost_reg",
            "model_path": "/m",
            "explainer": "invalid"
        }"""
        with pytest.raises(JSONTypeError, match="explainer must be one of"):
            parse_regression_explain_request(body)

    def test_non_string_backend_raises(self) -> None:
        """Non-string backend raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": 123,
            "model_path": "/m",
            "explainer": "permutation"
        }"""
        with pytest.raises(JSONTypeError, match="backend must be a string"):
            parse_regression_explain_request(body)

    def test_non_string_explainer_raises(self) -> None:
        """Non-string explainer raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "xgboost_reg",
            "model_path": "/m",
            "explainer": 123
        }"""
        with pytest.raises(JSONTypeError, match="explainer must be a string"):
            parse_regression_explain_request(body)

    def test_invalid_n_samples_type_raises(self) -> None:
        """Non-numeric n_samples raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "xgboost_reg",
            "model_path": "/m",
            "explainer": "permutation",
            "n_samples": "many"
        }"""
        with pytest.raises(
            JSONTypeError,
            match="Field 'n_samples' must be a number",
        ):
            parse_regression_explain_request(body)

    def test_non_object_body_raises(self) -> None:
        """Non-object JSON body raises JSONTypeError."""
        body = b"[]"
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            parse_regression_explain_request(body)
