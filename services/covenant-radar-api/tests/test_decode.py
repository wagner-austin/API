"""Tests for HTTP request body parsing."""

from __future__ import annotations

import pytest
from covenant_domain import DealId
from platform_core.json_utils import InvalidJsonError, JSONTypeError

from covenant_radar_api.api.decode import (
    parse_covenant_id_request,
    parse_covenant_request,
    parse_deal_id_request,
    parse_deal_request,
    parse_evaluate_request,
    parse_explain_request,
    parse_external_train_request,
    parse_measurements_request,
    parse_optimize_request,
    parse_predict_request,
    parse_train_request,
    parse_update_deal_request,
)


class TestParseDealRequest:
    """Tests for parse_deal_request."""

    def test_valid_deal_request(self) -> None:
        """Test parsing a valid deal request."""
        body = b"""{
            "id": {"value": "deal-123"},
            "name": "Test Deal",
            "borrower": "Acme Corp",
            "sector": "Technology",
            "region": "North America",
            "commitment_amount_cents": 1000000,
            "currency": "USD",
            "maturity_date_iso": "2025-12-31"
        }"""
        result = parse_deal_request(body)

        assert result["id"]["value"] == "deal-123"
        assert result["name"] == "Test Deal"
        assert result["borrower"] == "Acme Corp"
        assert result["sector"] == "Technology"
        assert result["region"] == "North America"
        assert result["commitment_amount_cents"] == 1000000
        assert result["currency"] == "USD"
        assert result["maturity_date_iso"] == "2025-12-31"

    def test_missing_field_raises_json_type_error(self) -> None:
        """Test that missing required field raises JSONTypeError."""
        body = b"""{"id": {"value": "deal-123"}}"""
        with pytest.raises(JSONTypeError, match="Missing required field"):
            parse_deal_request(body)

    def test_invalid_json_raises(self) -> None:
        """Test that invalid JSON raises InvalidJsonError."""
        body = b"not valid json"
        with pytest.raises(InvalidJsonError):
            parse_deal_request(body)

    def test_non_object_raises_type_error(self) -> None:
        """Test that non-object JSON raises TypeError."""
        body = b"[]"
        with pytest.raises(TypeError, match="Request body must be a JSON object"):
            parse_deal_request(body)


class TestParseUpdateDealRequest:
    """Tests for parse_update_deal_request."""

    def test_valid_update_request(self) -> None:
        """Test parsing a valid update deal request."""
        body = b"""{
            "name": "Updated Deal",
            "borrower": "New Corp",
            "sector": "Finance",
            "region": "Europe",
            "commitment_amount_cents": 2000000,
            "currency": "EUR",
            "maturity_date_iso": "2026-06-30"
        }"""
        deal_id = DealId(value="existing-deal-id")
        result = parse_update_deal_request(body, deal_id)

        assert result["id"]["value"] == "existing-deal-id"
        assert result["name"] == "Updated Deal"
        assert result["borrower"] == "New Corp"
        assert result["commitment_amount_cents"] == 2000000

    def test_missing_field_raises_json_type_error(self) -> None:
        """Test that missing required field raises JSONTypeError."""
        body = b"""{"name": "Test"}"""
        deal_id = DealId(value="test-id")
        with pytest.raises(JSONTypeError, match="Missing required field"):
            parse_update_deal_request(body, deal_id)

    def test_wrong_type_raises_json_type_error(self) -> None:
        """Test that wrong field type raises JSONTypeError."""
        body = b"""{
            "name": 123,
            "borrower": "Corp",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": 1000,
            "currency": "USD",
            "maturity_date_iso": "2025-01-01"
        }"""
        deal_id = DealId(value="test-id")
        with pytest.raises(JSONTypeError, match="Field 'name' must be a string"):
            parse_update_deal_request(body, deal_id)

    def test_wrong_int_type_raises_json_type_error(self) -> None:
        """Test that wrong int field type raises JSONTypeError."""
        body = b"""{
            "name": "Deal",
            "borrower": "Corp",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": "not an int",
            "currency": "USD",
            "maturity_date_iso": "2025-01-01"
        }"""
        deal_id = DealId(value="test-id")
        match_msg = "Field 'commitment_amount_cents' must be an integer"
        with pytest.raises(JSONTypeError, match=match_msg):
            parse_update_deal_request(body, deal_id)


class TestParseDealIdRequest:
    """Tests for parse_deal_id_request."""

    def test_valid_deal_id_request(self) -> None:
        """Test parsing a valid deal ID request."""
        body = b"""{"value": "deal-123"}"""
        result = parse_deal_id_request(body)

        assert result["value"] == "deal-123"

    def test_missing_value_raises_json_type_error(self) -> None:
        """Test that missing value field raises JSONTypeError."""
        body = b"""{}"""
        with pytest.raises(JSONTypeError, match="Missing required field 'value'"):
            parse_deal_id_request(body)


class TestParseCovenantIdRequest:
    """Tests for parse_covenant_id_request."""

    def test_valid_covenant_id_request(self) -> None:
        """Test parsing a valid covenant ID request."""
        body = b"""{"value": "cov-456"}"""
        result = parse_covenant_id_request(body)

        assert result["value"] == "cov-456"

    def test_missing_value_raises_json_type_error(self) -> None:
        """Test that missing value field raises JSONTypeError."""
        body = b"""{}"""
        with pytest.raises(JSONTypeError, match="Missing required field 'value'"):
            parse_covenant_id_request(body)


class TestParseCovenantRequest:
    """Tests for parse_covenant_request."""

    def test_valid_covenant_request(self) -> None:
        """Test parsing a valid covenant request."""
        body = b"""{
            "id": {"value": "cov-123"},
            "deal_id": {"value": "deal-456"},
            "name": "Debt to EBITDA",
            "formula": "total_debt / ebitda",
            "threshold_value_scaled": 3500000,
            "threshold_direction": "<=",
            "frequency": "QUARTERLY"
        }"""
        result = parse_covenant_request(body)

        assert result["id"]["value"] == "cov-123"
        assert result["deal_id"]["value"] == "deal-456"
        assert result["name"] == "Debt to EBITDA"
        assert result["formula"] == "total_debt / ebitda"
        assert result["threshold_value_scaled"] == 3500000
        assert result["threshold_direction"] == "<="
        assert result["frequency"] == "QUARTERLY"

    def test_invalid_direction_raises_json_type_error(self) -> None:
        """Test that invalid threshold direction raises JSONTypeError."""
        body = b"""{
            "id": {"value": "cov-123"},
            "deal_id": {"value": "deal-456"},
            "name": "Test",
            "formula": "a / b",
            "threshold_value_scaled": 1000000,
            "threshold_direction": "==",
            "frequency": "QUARTERLY"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid ThresholdDirection"):
            parse_covenant_request(body)

    def test_invalid_frequency_raises_json_type_error(self) -> None:
        """Test that invalid frequency raises JSONTypeError."""
        body = b"""{
            "id": {"value": "cov-123"},
            "deal_id": {"value": "deal-456"},
            "name": "Test",
            "formula": "a / b",
            "threshold_value_scaled": 1000000,
            "threshold_direction": "<=",
            "frequency": "MONTHLY"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid CovenantFrequency"):
            parse_covenant_request(body)


class TestParseMeasurementsRequest:
    """Tests for parse_measurements_request."""

    def test_valid_measurements_request(self) -> None:
        """Test parsing a valid measurements request."""
        body = b"""{
            "measurements": [
                {
                    "deal_id": {"value": "deal-123"},
                    "period_start_iso": "2024-01-01",
                    "period_end_iso": "2024-03-31",
                    "metric_name": "total_debt",
                    "metric_value_scaled": 5000000000
                },
                {
                    "deal_id": {"value": "deal-123"},
                    "period_start_iso": "2024-01-01",
                    "period_end_iso": "2024-03-31",
                    "metric_name": "ebitda",
                    "metric_value_scaled": 1500000000
                }
            ]
        }"""
        result = parse_measurements_request(body)

        assert len(result) == 2
        assert result[0]["metric_name"] == "total_debt"
        assert result[1]["metric_name"] == "ebitda"

    def test_empty_measurements_list(self) -> None:
        """Test parsing empty measurements list."""
        body = b"""{"measurements": []}"""
        result = parse_measurements_request(body)

        assert len(result) == 0

    def test_missing_measurements_key_raises(self) -> None:
        """Test that missing measurements key raises JSONTypeError."""
        body = b"""{}"""
        with pytest.raises(JSONTypeError, match="Missing required field 'measurements'"):
            parse_measurements_request(body)

    def test_non_list_measurements_raises_json_type_error(self) -> None:
        """Test that non-list measurements raises JSONTypeError."""
        body = b"""{"measurements": "not a list"}"""
        with pytest.raises(JSONTypeError, match="Field 'measurements' must be an array"):
            parse_measurements_request(body)

    def test_non_object_measurement_raises_type_error(self) -> None:
        """Test that non-object measurement item raises TypeError."""
        body = b"""{"measurements": ["not an object"]}"""
        with pytest.raises(TypeError, match="Each measurement must be a JSON object"):
            parse_measurements_request(body)

    def test_invalid_measurement_field_raises(self) -> None:
        """Test that invalid measurement field raises appropriate error."""
        body = b"""{
            "measurements": [
                {
                    "deal_id": {"value": "deal-123"},
                    "period_start_iso": 12345,
                    "period_end_iso": "2024-03-31",
                    "metric_name": "test",
                    "metric_value_scaled": 1000
                }
            ]
        }"""
        with pytest.raises(JSONTypeError, match="Field 'period_start_iso' must be a string"):
            parse_measurements_request(body)


class TestParseEvaluateRequest:
    """Tests for parse_evaluate_request."""

    def test_valid_evaluate_request(self) -> None:
        """Test parsing a valid evaluate request."""
        body = b"""{
            "deal_id": "deal-123",
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "tolerance_ratio_scaled": 100000
        }"""
        result = parse_evaluate_request(body)

        assert result["deal_id"] == "deal-123"
        assert result["period_start_iso"] == "2024-01-01"
        assert result["period_end_iso"] == "2024-03-31"
        assert result["tolerance_ratio_scaled"] == 100000

    def test_missing_field_raises_json_type_error(self) -> None:
        """Test that missing field raises JSONTypeError."""
        body = b"""{"deal_id": "deal-123"}"""
        with pytest.raises(JSONTypeError, match="Missing required field"):
            parse_evaluate_request(body)


class TestParsePredictRequest:
    """Tests for parse_predict_request."""

    def test_valid_predict_request(self) -> None:
        """Test parsing a valid predict request."""
        body = b"""{"deal_id": "deal-456"}"""
        result = parse_predict_request(body)

        assert result["deal_id"] == "deal-456"

    def test_missing_deal_id_raises_json_type_error(self) -> None:
        """Test that missing deal_id raises JSONTypeError."""
        body = b"""{}"""
        with pytest.raises(JSONTypeError, match="Missing required field 'deal_id'"):
            parse_predict_request(body)


class TestParseTrainRequest:
    """Tests for parse_train_request."""

    def test_valid_train_request(self) -> None:
        """Test parsing a valid train request."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "early_stopping_rounds": 10
        }"""
        result = parse_train_request(body)

        assert result["learning_rate"] == 0.1
        assert result["max_depth"] == 6
        assert result["n_estimators"] == 100
        assert result["subsample"] == 0.8
        assert result["colsample_bytree"] == 0.8
        assert result["random_state"] == 42
        assert result["train_ratio"] == 0.7
        assert result["val_ratio"] == 0.15
        assert result["test_ratio"] == 0.15
        assert result["early_stopping_rounds"] == 10
        # reg_alpha/reg_lambda default when not provided
        assert result["reg_alpha"] == 0.0
        assert result["reg_lambda"] == 1.0
        assert result["device"] == "auto"

    def test_request_with_defaults(self) -> None:
        """Test parsing with optional fields defaulted."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        result = parse_train_request(body)

        assert result["learning_rate"] == 0.1
        # Default values for optional fields
        assert result["train_ratio"] == 0.7
        assert result["val_ratio"] == 0.15
        assert result["test_ratio"] == 0.15
        assert result["early_stopping_rounds"] == 10
        assert result["reg_alpha"] == 0.0
        assert result["reg_lambda"] == 1.0
        assert result["device"] == "auto"

    def test_train_request_with_regularization_and_scale(self) -> None:
        """Test parsing reg params, device, and scale_pos_weight."""
        body = b"""{
            "learning_rate": 0.2,
            "max_depth": 4,
            "n_estimators": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "random_state": 7,
            "device": "cuda",
            "reg_alpha": 2.5,
            "reg_lambda": 3.5,
            "scale_pos_weight": 1.2
        }"""
        result = parse_train_request(body)

        assert result["device"] == "cuda"
        assert result["reg_alpha"] == 2.5
        assert result["reg_lambda"] == 3.5
        assert result["scale_pos_weight"] == 1.2
        assert result["n_estimators"] == 50

    def test_train_request_invalid_scale_pos_weight(self) -> None:
        """Test parsing rejects invalid scale_pos_weight type."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "scale_pos_weight": "heavy"
        }"""
        with pytest.raises(JSONTypeError, match="scale_pos_weight must be a number"):
            parse_train_request(body)

    def test_train_request_invalid_ratio_type(self) -> None:
        """Test parsing rejects non-numeric ratio values."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": "big"
        }"""
        with pytest.raises(JSONTypeError, match="Field 'train_ratio' must be a number"):
            parse_train_request(body)

    def test_train_request_invalid_device(self) -> None:
        """Test parsing rejects unsupported device value."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "tpu"
        }"""
        with pytest.raises(JSONTypeError, match="device must be one of: cpu, cuda, auto"):
            parse_train_request(body)

    def test_train_request_device_cpu(self) -> None:
        """Test parsing accepts explicit CPU device."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "cpu"
        }"""
        result = parse_train_request(body)
        assert result["device"] == "cpu"

    def test_train_request_device_auto_string(self) -> None:
        """Test parsing accepts explicit auto device string."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "auto"
        }"""
        result = parse_train_request(body)
        assert result["device"] == "auto"

    def test_train_request_non_string_device(self) -> None:
        """Test parsing rejects non-string device types."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": 123
        }"""
        with pytest.raises(JSONTypeError, match="device must be a string"):
            parse_train_request(body)

    def test_early_stopping_as_float(self) -> None:
        """Test parsing early_stopping_rounds as float (converts to int)."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 15.0
        }"""
        result = parse_train_request(body)

        assert result["early_stopping_rounds"] == 15

    def test_early_stopping_invalid_type(self) -> None:
        """Test parsing rejects non-numeric early_stopping_rounds."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": "fast"
        }"""
        with pytest.raises(JSONTypeError, match="Field 'early_stopping_rounds' must be a number"):
            parse_train_request(body)

    def test_missing_field_raises_json_type_error(self) -> None:
        """Test that missing field raises JSONTypeError."""
        body = b"""{"learning_rate": 0.1}"""
        with pytest.raises(JSONTypeError, match="Missing required field"):
            parse_train_request(body)


class TestParseExternalTrainRequest:
    """Tests for parse_external_train_request."""

    def test_valid_xgboost_request_defaults_to_xgboost(self) -> None:
        """Test parsing valid XGBoost request with default backend."""
        body = b"""{
            "dataset": "taiwan",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "xgboost"
        assert result["dataset"] == "taiwan"
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["max_depth"] == 6
        assert result["config"]["n_estimators"] == 100
        assert result["config"]["device"] == "auto"
        assert result["config"]["train_ratio"] == 0.7
        assert result["config"]["val_ratio"] == 0.15
        assert result["config"]["test_ratio"] == 0.15
        assert result["config"]["early_stopping_rounds"] == 10
        assert result["config"]["reg_alpha"] == 0.0
        assert result["config"]["reg_lambda"] == 1.0

    def test_valid_xgboost_request_explicit_backend(self) -> None:
        """Test parsing valid XGBoost request with explicit backend."""
        body = b"""{
            "dataset": "us",
            "backend": "xgboost",
            "learning_rate": 0.2,
            "max_depth": 4,
            "n_estimators": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "random_state": 7,
            "device": "cpu",
            "scale_pos_weight": 2.5
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "xgboost"
        assert result["dataset"] == "us"
        assert result["config"]["device"] == "cpu"
        assert result["config"]["scale_pos_weight"] == 2.5

    def test_valid_mlp_request(self) -> None:
        """Test parsing valid MLP request."""
        body = b"""{
            "dataset": "polish",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, 32],
            "precision": "fp32",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "mlp"
        assert result["dataset"] == "polish"
        assert result["config"]["learning_rate"] == 0.001
        assert result["config"]["batch_size"] == 32
        assert result["config"]["n_epochs"] == 100
        assert result["config"]["dropout"] == 0.2
        assert result["config"]["hidden_sizes"] == (64, 32)
        assert result["config"]["precision"] == "fp32"
        assert result["config"]["optimizer"] == "adamw"
        assert result["config"]["random_state"] == 42
        assert result["config"]["early_stopping_patience"] == 10
        assert result["config"]["device"] == "auto"
        assert result["config"]["train_ratio"] == 0.7

    def test_mlp_request_with_cuda_and_fp16(self) -> None:
        """Test parsing MLP request with CUDA device and fp16 precision."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.01,
            "batch_size": 64,
            "n_epochs": 50,
            "dropout": 0.1,
            "hidden_sizes": [128, 64, 32],
            "precision": "fp16",
            "optimizer": "adam",
            "random_state": 123,
            "early_stopping_patience": 5,
            "device": "cuda"
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "mlp"
        assert result["config"]["device"] == "cuda"
        assert result["config"]["precision"] == "fp16"
        assert result["config"]["optimizer"] == "adam"
        assert result["config"]["hidden_sizes"] == (128, 64, 32)

    def test_mlp_request_with_bf16_and_sgd(self) -> None:
        """Test parsing MLP request with bf16 precision and SGD optimizer."""
        body = b"""{
            "dataset": "us",
            "backend": "mlp",
            "learning_rate": 0.1,
            "batch_size": 16,
            "n_epochs": 200,
            "dropout": 0.0,
            "hidden_sizes": [32],
            "precision": "bf16",
            "optimizer": "sgd",
            "random_state": 0,
            "early_stopping_patience": 20
        }"""
        result = parse_external_train_request(body)

        # Use if for type narrowing (discriminated union)
        if result["backend"] != "mlp":
            raise AssertionError("Expected mlp backend")
        assert result["config"]["precision"] == "bf16"
        assert result["config"]["optimizer"] == "sgd"

    def test_mlp_request_with_auto_precision(self) -> None:
        """Test parsing MLP request with auto precision."""
        body = b"""{
            "dataset": "polish",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, 32],
            "precision": "auto",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        result = parse_external_train_request(body)

        # Use if for type narrowing (discriminated union)
        if result["backend"] != "mlp":
            raise AssertionError("Expected mlp backend")
        assert result["config"]["precision"] == "auto"

    def test_request_with_custom_split_ratios(self) -> None:
        """Test parsing request with custom split ratios."""
        body = b"""{
            "dataset": "taiwan",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2
        }"""
        result = parse_external_train_request(body)

        assert result["config"]["train_ratio"] == 0.6
        assert result["config"]["val_ratio"] == 0.2
        assert result["config"]["test_ratio"] == 0.2

    def test_valid_lstm_request(self) -> None:
        """Test parsing valid LSTM request."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lstm",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": true,
            "sequence_length": 5,
            "precision": "fp32",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "lstm"
        assert result["dataset"] == "taiwan"
        # Use if for type narrowing (discriminated union)
        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["learning_rate"] == 0.001
        assert result["config"]["batch_size"] == 32
        assert result["config"]["n_epochs"] == 100
        assert result["config"]["dropout"] == 0.2
        assert result["config"]["hidden_size"] == 64
        assert result["config"]["num_layers"] == 2
        assert result["config"]["bidirectional"] is True
        assert result["config"]["sequence_length"] == 5
        assert result["config"]["precision"] == "fp32"
        assert result["config"]["random_state"] == 42
        assert result["config"]["early_stopping_patience"] == 10
        assert result["config"]["device"] == "auto"

    def test_lstm_request_with_cuda_and_fp16(self) -> None:
        """Test parsing LSTM request with CUDA device and fp16 precision."""
        body = b"""{
            "dataset": "us",
            "backend": "lstm",
            "learning_rate": 0.01,
            "batch_size": 64,
            "n_epochs": 50,
            "dropout": 0.1,
            "hidden_size": 128,
            "num_layers": 3,
            "bidirectional": false,
            "sequence_length": 10,
            "precision": "fp16",
            "random_state": 123,
            "early_stopping_patience": 5,
            "device": "cuda"
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["device"] == "cuda"
        assert result["config"]["precision"] == "fp16"
        assert result["config"]["bidirectional"] is False

    def test_lstm_request_with_bf16_and_auto_precision(self) -> None:
        """Test parsing LSTM request with bf16 and auto precision modes."""
        body = b"""{
            "dataset": "polish",
            "backend": "lstm",
            "learning_rate": 0.1,
            "batch_size": 16,
            "n_epochs": 200,
            "dropout": 0.0,
            "hidden_size": 32,
            "num_layers": 1,
            "bidirectional": true,
            "sequence_length": 3,
            "precision": "bf16",
            "random_state": 0,
            "early_stopping_patience": 20
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["precision"] == "bf16"

    def test_lstm_request_auto_precision(self) -> None:
        """Test parsing LSTM request with auto precision."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lstm",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": true,
            "sequence_length": 5,
            "precision": "auto",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "lstm":
            raise AssertionError("Expected lstm backend")
        assert result["config"]["precision"] == "auto"

    def test_lstm_missing_bidirectional_raises_error(self) -> None:
        """Test that missing bidirectional field raises error."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lstm",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_size": 64,
            "num_layers": 2,
            "sequence_length": 5,
            "precision": "fp32",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="bidirectional must be a boolean"):
            parse_external_train_request(body)

    def test_lstm_invalid_precision_raises_error(self) -> None:
        """Test that invalid precision for LSTM raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lstm",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": true,
            "sequence_length": 5,
            "precision": "invalid_precision",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="precision must be fp32, fp16, bf16, or auto"):
            parse_external_train_request(body)

    def test_valid_lightgbm_request(self) -> None:
        """Test parsing valid LightGBM request."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "lightgbm",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        result = parse_external_train_request(body)

        assert result["backend"] == "lightgbm"
        assert result["dataset"] == "taiwan"
        if result["backend"] != "lightgbm":
            raise AssertionError("Expected lightgbm backend")
        assert result["config"]["learning_rate"] == 0.1
        assert result["config"]["max_depth"] == 6
        assert result["config"]["n_estimators"] == 100
        assert result["config"]["num_leaves"] == 31
        assert result["config"]["min_child_samples"] == 20
        assert result["config"]["subsample"] == 0.8
        assert result["config"]["colsample_bytree"] == 0.8
        assert result["config"]["random_state"] == 42
        assert result["config"]["device"] == "auto"
        assert result["config"]["early_stopping_rounds"] == 10
        assert result["config"]["reg_alpha"] == 0.0
        assert result["config"]["reg_lambda"] == 1.0

    def test_lightgbm_request_with_custom_regularization(self) -> None:
        """Test parsing LightGBM request with custom regularization."""
        body = b"""{
            "dataset": "us",
            "backend": "lightgbm",
            "learning_rate": 0.05,
            "max_depth": 8,
            "n_estimators": 200,
            "num_leaves": 63,
            "min_child_samples": 10,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "random_state": 123,
            "device": "cuda",
            "early_stopping_rounds": 20,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0
        }"""
        result = parse_external_train_request(body)

        if result["backend"] != "lightgbm":
            raise AssertionError("Expected lightgbm backend")
        assert result["config"]["device"] == "cuda"
        assert result["config"]["early_stopping_rounds"] == 20
        assert result["config"]["reg_alpha"] == 1.0
        assert result["config"]["reg_lambda"] == 5.0

    def test_invalid_dataset_raises_value_error(self) -> None:
        """Test that invalid dataset raises ValueError."""
        body = b"""{
            "dataset": "invalid_dataset",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        with pytest.raises(ValueError, match="dataset must be one of"):
            parse_external_train_request(body)

    def test_missing_dataset_raises_json_type_error(self) -> None:
        """Test that missing dataset raises JSONTypeError."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'dataset'"):
            parse_external_train_request(body)

    def test_invalid_ratios_sum_raises_value_error(self) -> None:
        """Test that split ratios not summing to 1.0 raises ValueError."""
        body = b"""{
            "dataset": "taiwan",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": 0.5,
            "val_ratio": 0.2,
            "test_ratio": 0.2
        }"""
        with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
            parse_external_train_request(body)

    def test_invalid_precision_raises_json_type_error(self) -> None:
        """Test that invalid precision raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, 32],
            "precision": "invalid",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="precision must be fp32, fp16, bf16, or auto"):
            parse_external_train_request(body)

    def test_invalid_optimizer_raises_json_type_error(self) -> None:
        """Test that invalid optimizer raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, 32],
            "precision": "fp32",
            "optimizer": "invalid",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="optimizer must be adamw, adam, or sgd"):
            parse_external_train_request(body)

    def test_invalid_hidden_sizes_not_list_raises_json_type_error(self) -> None:
        """Test that hidden_sizes not being a list raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": 64,
            "precision": "fp32",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="hidden_sizes must be list of ints"):
            parse_external_train_request(body)

    def test_invalid_hidden_sizes_contains_non_int_raises_json_type_error(self) -> None:
        """Test that hidden_sizes containing non-int raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "backend": "mlp",
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "hidden_sizes": [64, "invalid"],
            "precision": "fp32",
            "optimizer": "adamw",
            "random_state": 42,
            "early_stopping_patience": 10
        }"""
        with pytest.raises(JSONTypeError, match="hidden_sizes must be list of ints"):
            parse_external_train_request(body)

    def test_xgboost_invalid_scale_pos_weight_raises_json_type_error(self) -> None:
        """Test that invalid scale_pos_weight raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "scale_pos_weight": "heavy"
        }"""
        with pytest.raises(JSONTypeError, match="scale_pos_weight must be a number"):
            parse_external_train_request(body)


class TestParseOptimizeRequest:
    """Tests for parse_optimize_request."""

    def test_valid_optimize_request_minimal(self) -> None:
        """Test parsing valid optimize request with minimal fields."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50
        }"""
        result = parse_optimize_request(body)

        assert result["dataset"] == "taiwan"
        assert result["config"]["n_trials"] == 50
        assert result["config"]["timeout_seconds"] is None
        assert result["device"] == "auto"
        assert result["feature_preset"] == "none"
        assert result["config"]["random_state"] == 42

    def test_valid_optimize_request_full(self) -> None:
        """Test parsing valid optimize request with all fields."""
        body = b"""{
            "dataset": "us",
            "n_trials": 100,
            "timeout_seconds": 3600,
            "device": "cuda",
            "space_profile": "categorical",
            "feature_preset": "full",
            "random_state": 123
        }"""
        result = parse_optimize_request(body)

        assert result["dataset"] == "us"
        assert result["config"]["n_trials"] == 100
        assert result["config"]["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["feature_preset"] == "full"
        assert result["config"]["random_state"] == 123

    def test_valid_optimize_request_polish_dataset(self) -> None:
        """Test parsing optimize request for polish dataset."""
        body = b"""{
            "dataset": "polish",
            "n_trials": 25
        }"""
        result = parse_optimize_request(body)

        assert result["dataset"] == "polish"
        assert result["config"]["n_trials"] == 25

    def test_valid_optimize_request_categorical_space(self) -> None:
        """Test parsing optimize request with categorical space profile."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "space_profile": "categorical"
        }"""
        result = parse_optimize_request(body)

        assert result["dataset"] == "taiwan"
        # Verify categorical space has categorical param types
        lr_spec = result["search_space"]["learning_rate"]
        assert lr_spec["param_type"] == "categorical_float"

    def test_valid_optimize_request_default_space(self) -> None:
        """Test parsing optimize request with default space profile."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "space_profile": "default"
        }"""
        result = parse_optimize_request(body)

        # Verify default space has float param types with log_scale
        lr_spec = result["search_space"]["learning_rate"]
        assert lr_spec["param_type"] == "float"

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

    def test_invalid_space_profile_raises_json_type_error(self) -> None:
        """Test that invalid space_profile raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "space_profile": "invalid"
        }"""
        with pytest.raises(JSONTypeError, match="space_profile must be one of"):
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

    def test_non_string_space_profile_raises_json_type_error(self) -> None:
        """Test that non-string space_profile raises JSONTypeError."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "space_profile": 123
        }"""
        with pytest.raises(JSONTypeError, match="space_profile must be a string"):
            parse_optimize_request(body)

    def test_null_timeout_allowed(self) -> None:
        """Test that null timeout is allowed and results in None."""
        body = b"""{
            "dataset": "taiwan",
            "n_trials": 50,
            "timeout_seconds": null
        }"""
        result = parse_optimize_request(body)

        assert result["config"]["timeout_seconds"] is None

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
        backends = ["xgboost", "mlp", "lstm", "lightgbm"]
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
