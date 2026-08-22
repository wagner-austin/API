"""Tests for HTTP request body parsing."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.api.decode_ml import (
    parse_optimize_request,
)
from covenant_radar_api.api.decode_regression import (
    parse_regression_optimize_request,
)


class TestParseOptimizeRequest:
    """Tests for parse_optimize_request with unified API parsing.

    The API edge only validates common fields. Backend-specific fields
    (precision, optimizer, n_epochs, etc.) are parsed by the worker job.
    """

    def test_valid_optimize_request_minimal(self) -> None:
        """Test parsing valid optimize request with minimal fields."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50
        }"""
        result = parse_optimize_request(body)

        assert result["backend"] == "xgboost"
        assert result["dataset"] == "taiwan"
        assert result["n_trials"] == 50
        assert result["timeout_seconds"] is None
        assert result["device"] == "auto"
        assert result["feature_preset"] == "none"
        assert result["random_state"] == 42

    def test_valid_optimize_request_full(self) -> None:
        """Test parsing valid optimize request with all common fields."""
        body = b"""{
            "dataset": "us",
            "backend": "xgboost",
            "n_trials": 100,
            "timeout_seconds": 3600,
            "device": "cuda",
            "feature_preset": "full",
            "random_state": 123
        }"""
        result = parse_optimize_request(body)

        assert result["backend"] == "xgboost"
        assert result["dataset"] == "us"
        assert result["n_trials"] == 100
        assert result["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["feature_preset"] == "full"
        assert result["random_state"] == 123

    def test_valid_optimize_request_polish_dataset(self) -> None:
        """Test parsing optimize request for polish dataset."""
        body = b"""{
            "dataset": "polish",
            "n_trials": 25
        }"""
        result = parse_optimize_request(body)

        assert result["dataset"] == "polish"
        assert result["n_trials"] == 25

    def test_valid_optimize_request_cpu_device(self) -> None:
        """Test parsing optimize request with CPU device."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "device": "cpu"
        }"""
        result = parse_optimize_request(body)

        assert result["device"] == "cpu"

    def test_invalid_dataset_raises_value_error(self) -> None:
        """Test that invalid dataset raises ValueError."""
        body = b"""{
            "dataset": "invalid_dataset",
            "n_trials": 50
        }"""
        with pytest.raises(ValueError, match="dataset must be one of"):
            parse_optimize_request(body)

    def test_missing_dataset_raises_json_type_error(self) -> None:
        """Test that missing dataset raises JSONTypeError."""
        body = b"""{
            "n_trials": 50
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'dataset'"):
            parse_optimize_request(body)

    def test_missing_n_trials_raises_json_type_error(self) -> None:
        """Test that missing n_trials raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan"
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'n_trials'"):
            parse_optimize_request(body)

    def test_invalid_device_raises_json_type_error(self) -> None:
        """Test that invalid device raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "device": "tpu"
        }"""
        with pytest.raises(JSONTypeError, match="device must be one of: cpu, cuda, auto"):
            parse_optimize_request(body)

    def test_invalid_timeout_type_raises_json_type_error(self) -> None:
        """Test that non-integer timeout raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "timeout_seconds": "fast"
        }"""
        with pytest.raises(JSONTypeError, match="timeout_seconds must be an integer"):
            parse_optimize_request(body)

    def test_non_string_device_raises_json_type_error(self) -> None:
        """Test that non-string device raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "device": 123
        }"""
        with pytest.raises(JSONTypeError, match="device must be a string"):
            parse_optimize_request(body)

    def test_null_timeout_allowed(self) -> None:
        """Test that null timeout is allowed and results in None."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "timeout_seconds": null
        }"""
        result = parse_optimize_request(body)

        assert result["timeout_seconds"] is None

    def test_valid_feature_preset_log_only(self) -> None:
        """Test parsing optimize request with log_only feature preset."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "feature_preset": "log_only"
        }"""
        result = parse_optimize_request(body)

        assert result["feature_preset"] == "log_only"

    def test_valid_feature_preset_ratios_only(self) -> None:
        """Test parsing optimize request with ratios_only feature preset."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "feature_preset": "ratios_only"
        }"""
        result = parse_optimize_request(body)

        assert result["feature_preset"] == "ratios_only"

    def test_valid_feature_preset_none(self) -> None:
        """Test parsing optimize request with explicit none feature preset."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "feature_preset": "none"
        }"""
        result = parse_optimize_request(body)

        assert result["feature_preset"] == "none"

    def test_invalid_feature_preset_raises_json_type_error(self) -> None:
        """Test that invalid feature_preset raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "feature_preset": "invalid"
        }"""
        with pytest.raises(JSONTypeError, match="feature_preset must be one of"):
            parse_optimize_request(body)

    def test_non_string_feature_preset_raises_json_type_error(self) -> None:
        """Test that non-string feature_preset raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "feature_preset": 123
        }"""
        with pytest.raises(JSONTypeError, match="feature_preset must be a string"):
            parse_optimize_request(body)

    def test_explicit_xgboost_backend(self) -> None:
        """Test parsing optimize request with explicit xgboost backend."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "xgboost",
            "n_trials": 50
        }"""
        result = parse_optimize_request(body)

        assert result["backend"] == "xgboost"
        assert result["dataset"] == "taiwan"
        assert result["n_trials"] == 50

    def test_all_seven_backends_accepted(self) -> None:
        """Test all 7 backends are accepted by the API edge."""
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
            body = f'{{"dataset": "taiwan", "backend": "{backend}", "n_trials": 10}}'.encode()
            result = parse_optimize_request(body)
            assert result["backend"] == backend

    def test_invalid_backend_raises_value_error(self) -> None:
        """Test that invalid backend raises ValueError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "invalid_backend",
            "n_trials": 50
        }"""
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_optimize_request(body)

    def test_non_string_backend_raises_json_type_error(self) -> None:
        """Test that non-string backend raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": 123,
            "n_trials": 50
        }"""
        with pytest.raises(JSONTypeError, match="backend must be a string"):
            parse_optimize_request(body)

    def test_mlp_backend_ignores_backend_specific_fields(self) -> None:
        """Test MLP backend only returns common fields at API edge."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "n_trials": 50,
            "precision": "fp16",
            "optimizer": "adam"
        }"""
        result = parse_optimize_request(body)

        assert result["backend"] == "mlp"
        assert result["dataset"] == "taiwan"
        assert result["n_trials"] == 50
        # Backend-specific fields are NOT in the API parse result
        assert "precision" not in result
        assert "optimizer" not in result

    def test_lstm_backend_common_fields_only(self) -> None:
        """Test LSTM backend returns only common fields at API edge."""
        body = b"""{
            "dataset": "us",
            "backend": "lstm",
            "n_trials": 50,
            "sequence_length": 10,
            "bidirectional": true
        }"""
        result = parse_optimize_request(body)

        assert result["backend"] == "lstm"
        assert result["dataset"] == "us"
        assert result["n_trials"] == 50
        assert "sequence_length" not in result
        assert "bidirectional" not in result

    def test_lightgbm_backend_common_fields_only(self) -> None:
        """Test LightGBM backend returns only common fields at API edge."""
        body = b"""{
            "dataset": "polish",
            "backend": "lightgbm",
            "n_trials": 50,
            "early_stopping_rounds": 20
        }"""
        result = parse_optimize_request(body)

        assert result["backend"] == "lightgbm"
        assert result["dataset"] == "polish"
        assert "early_stopping_rounds" not in result


class TestParseRegressionOptimizeRequest:
    """Tests for parse_regression_optimize_request.

    The API edge validates common fields. Backend-specific fields
    (early_stopping_rounds, n_jobs) are parsed by the worker job.
    """

    def test_valid_request_minimal(self) -> None:
        """Test parsing valid regression optimize request with minimal fields."""
        body = b"""{
            "dataset": "financial_distress",
            "n_trials": 50
        }"""
        result = parse_regression_optimize_request(body)

        assert result["backend"] == "xgboost_reg"
        assert result["dataset"] == "financial_distress"
        assert result["n_trials"] == 50
        assert result["timeout_seconds"] is None
        assert result["device"] == "auto"
        assert result["feature_preset"] == "none"
        assert result["random_state"] == 42

    def test_valid_request_full(self) -> None:
        """Test parsing valid regression optimize request with all common fields."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "lightgbm_reg",
            "n_trials": 100,
            "timeout_seconds": 3600,
            "device": "cuda",
            "feature_preset": "full",
            "random_state": 123
        }"""
        result = parse_regression_optimize_request(body)

        assert result["backend"] == "lightgbm_reg"
        assert result["dataset"] == "financial_distress"
        assert result["n_trials"] == 100
        assert result["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["feature_preset"] == "full"
        assert result["random_state"] == 123

    def test_xgboost_reg_backend(self) -> None:
        """Test parsing with explicit xgboost_reg backend."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "xgboost_reg",
            "n_trials": 25
        }"""
        result = parse_regression_optimize_request(body)

        assert result["backend"] == "xgboost_reg"

    def test_all_four_regressor_backends(self) -> None:
        """Test all 4 regressor backends are accepted."""
        backends = ["xgboost_reg", "lightgbm_reg", "mlp_reg", "lstm_reg"]
        for backend in backends:
            body = (
                f'{{"dataset": "financial_distress", "backend": "{backend}", "n_trials": 10}}'
            ).encode()
            result = parse_regression_optimize_request(body)
            assert result["backend"] == backend

    def test_invalid_dataset_raises_value_error(self) -> None:
        """Test that invalid regression dataset raises ValueError."""
        body = b"""{
            "dataset": "invalid_dataset",
            "n_trials": 50
        }"""
        with pytest.raises(ValueError, match="dataset must be one of"):
            parse_regression_optimize_request(body)

    def test_missing_dataset_raises_json_type_error(self) -> None:
        """Test that missing dataset raises JSONTypeError."""
        body = b"""{
            "n_trials": 50
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'dataset'"):
            parse_regression_optimize_request(body)

    def test_missing_n_trials_raises_json_type_error(self) -> None:
        """Test that missing n_trials raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress"
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'n_trials'"):
            parse_regression_optimize_request(body)

    def test_invalid_backend_raises_value_error(self) -> None:
        """Test that invalid regressor backend raises ValueError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "xgboost",
            "n_trials": 50
        }"""
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_regression_optimize_request(body)

    def test_non_string_backend_raises_json_type_error(self) -> None:
        """Test that non-string backend raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": 123,
            "n_trials": 50
        }"""
        with pytest.raises(JSONTypeError, match="backend must be a string"):
            parse_regression_optimize_request(body)

    def test_invalid_device_raises_json_type_error(self) -> None:
        """Test that invalid device raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "n_trials": 50,
            "device": "tpu"
        }"""
        with pytest.raises(JSONTypeError, match="device must be one of: cpu, cuda, auto"):
            parse_regression_optimize_request(body)

    def test_invalid_timeout_type_raises_json_type_error(self) -> None:
        """Test that non-integer timeout raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "n_trials": 50,
            "timeout_seconds": "fast"
        }"""
        with pytest.raises(JSONTypeError, match="timeout_seconds must be an integer"):
            parse_regression_optimize_request(body)

    def test_null_timeout_allowed(self) -> None:
        """Test that null timeout results in None."""
        body = b"""{
            "dataset": "financial_distress",
            "n_trials": 50,
            "timeout_seconds": null
        }"""
        result = parse_regression_optimize_request(body)
        assert result["timeout_seconds"] is None

    def test_cpu_device(self) -> None:
        """Test parsing with CPU device."""
        body = b"""{
            "dataset": "financial_distress",
            "n_trials": 50,
            "device": "cpu"
        }"""
        result = parse_regression_optimize_request(body)
        assert result["device"] == "cpu"

    def test_feature_preset_log_only(self) -> None:
        """Test parsing with log_only feature preset."""
        body = b"""{
            "dataset": "financial_distress",
            "n_trials": 50,
            "feature_preset": "log_only"
        }"""
        result = parse_regression_optimize_request(body)
        assert result["feature_preset"] == "log_only"

    def test_feature_preset_ratios_only(self) -> None:
        """Test parsing with ratios_only feature preset."""
        body = b"""{
            "dataset": "financial_distress",
            "n_trials": 50,
            "feature_preset": "ratios_only"
        }"""
        result = parse_regression_optimize_request(body)
        assert result["feature_preset"] == "ratios_only"

    def test_invalid_feature_preset_raises(self) -> None:
        """Test that invalid feature_preset raises JSONTypeError."""
        body = b"""{
            "dataset": "financial_distress",
            "n_trials": 50,
            "feature_preset": "invalid"
        }"""
        with pytest.raises(JSONTypeError, match="feature_preset must be one of"):
            parse_regression_optimize_request(body)

    def test_backend_specific_fields_not_in_result(self) -> None:
        """Backend-specific fields are NOT in the API parse result."""
        body = b"""{
            "dataset": "financial_distress",
            "backend": "lightgbm_reg",
            "n_trials": 50,
            "early_stopping_rounds": 20,
            "n_jobs": 4
        }"""
        result = parse_regression_optimize_request(body)
        assert result["backend"] == "lightgbm_reg"
        assert "early_stopping_rounds" not in result
        assert "n_jobs" not in result
